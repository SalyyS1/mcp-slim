"""Schema compaction, $ref inlining, and lazy meta-tool broker for MCP JSON-RPC traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import fnmatch
import json
import math
import re
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")

_NOISY_HYPERMEDIA_KEYS = frozenset({
    "followers_url",
    "following_url",
    "gists_url",
    "starred_url",
    "subscriptions_url",
    "organizations_url",
    "repos_url",
    "events_url",
    "received_events_url",
    "gravatar_id",
    "node_id",
})

_ABBREV_EXPANSIONS: dict[str, set[str]] = {
    "repo": {"repo", "repository"},
    "dir": {"dir", "directory", "folder"},
    "max": {"max", "maximum", "limit"},
    "min": {"min", "minimum"},
    "num": {"num", "number", "count"},
    "msg": {"msg", "message"},
    "id": {"id", "identifier"},
    "str": {"str", "string"},
    "int": {"int", "integer"},
    "bool": {"bool", "boolean"},
    "cfg": {"cfg", "config", "configuration"},
    "env": {"env", "environment"},
    "cmd": {"cmd", "command"},
    "arg": {"arg", "argument"},
    "args": {"args", "arguments"},
    "param": {"param", "parameter"},
    "params": {"params", "parameters"},
}

_STOP_WORDS = frozenset({
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "value",
    "name",
    "string",
    "number",
    "integer",
    "boolean",
    "git",
    "github",
    "target",
    "current",
    "specified",
    "given",
    "optional",
    "whether",
})

_REDUNDANT_SCHEMA_KEYS = frozenset({
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "definitions",
    "additionalProperties",
    "discriminator",
    "readOnly",
    "writeOnly",
    "deprecated",
    "title",
    "examples",
    "default",
})


def _is_redundant_param_description(prop_name: str, desc: str) -> bool:
  """Returns True when `desc` only restates `prop_name` using filler words."""
  norm_desc_words = re.findall(r"[a-z0-9]+", desc.lower())
  if not norm_desc_words or len(norm_desc_words) > 8:
    return False
  name_parts = [p for p in re.split(r"[^a-z0-9]+", prop_name.lower()) if p]
  allowed_words = set(_STOP_WORDS)
  for part in name_parts:
    allowed_words.add(part)
    allowed_words.add(part + "s")
    allowed_words.update(_ABBREV_EXPANSIONS.get(part, ()))
  return all(w in allowed_words for w in norm_desc_words)


def _matches_any_pattern(name: str, patterns: set[str]) -> bool:
  """Matches tool name against exact strings or wildcard glob patterns (e.g. 'get_*')."""
  if name in patterns:
    return True
  for pat in patterns:
    if ("*" in pat or "?" in pat or "[" in pat) and fnmatch.fnmatchcase(name, pat):
      return True
  return False


def estimate_tokens(obj: Any) -> int:
  """Fast ~4 chars/token heuristic aligned with cl100k_base / Claude tokenizer on JSON."""
  if isinstance(obj, str):
    raw = obj
  else:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
  return max(1, (len(raw) + 3) // 4)


def compact_description(
    text: str, max_chars: int = 140, keep_first_sentence: bool = True
) -> str:
  """Condenses verbose multi-line tool/property docstrings into a tight summary."""
  if not text:
    return ""
  cleaned = " ".join(text.strip().split())
  if keep_first_sentence:
    match = re.search(r"(?<![e]\.[g])(?<![i]\.[e])(?<!\d)\.\s+", cleaned)
    if match and match.start() + 1 >= 16:
      cleaned = cleaned[: match.start() + 1]
  if len(cleaned) > max_chars:
    cleaned = cleaned[: max_chars - 1].rstrip(" ,;:-") + "…"
  return cleaned


@dataclass
class SlimStats:
  """Cumulative token and byte savings across an MCP session."""

  schema_tokens_before: int = 0
  schema_tokens_after: int = 0
  output_tokens_before: int = 0
  output_tokens_after: int = 0
  wire_bytes_before: int = 0
  wire_bytes_after: int = 0
  tools_total: int = 0
  tools_kept: int = 0
  refs_inlined: int = 0
  cache_hits: int = 0
  args_coerced: int = 0
  blocked_calls: int = 0

  @property
  def total_tokens_before(self) -> int:
    return self.schema_tokens_before + self.output_tokens_before

  @property
  def total_tokens_after(self) -> int:
    return self.schema_tokens_after + self.output_tokens_after

  @property
  def tokens_saved(self) -> int:
    return max(0, self.total_tokens_before - self.total_tokens_after)

  @property
  def wire_bytes_saved(self) -> int:
    return max(0, self.wire_bytes_before - self.wire_bytes_after)

  @property
  def reduction_pct(self) -> float:
    if self.total_tokens_before == 0:
      return 0.0
    return round((self.tokens_saved / self.total_tokens_before) * 100.0, 1)


@dataclass(frozen=True)
class BranchContract:
  """Original object branch requirements captured before root-union flattening."""

  required: tuple[str, ...]
  allowed_props: frozenset[str]
  forbid_extra: bool = False


@dataclass(frozen=True)
class UnionContract:
  """Semantic branch contract preserved when flattening root `anyOf` or `oneOf` schemas."""

  kind: str  # "anyOf" or "oneOf"
  branches: tuple[BranchContract, ...]

  def describe_branches(self) -> str:
    parts: list[str] = []
    for b in self.branches:
      if b.required:
        parts.append("{" + ", ".join(b.required) + "}")
      elif b.allowed_props:
        parts.append("{" + ", ".join(sorted(b.allowed_props)) + "}")
    return " | ".join(parts) if parts else self.kind

  def compact_hint(self, common_required: set[str]) -> str:
    labels: list[str] = []
    for b in self.branches:
      residual = [r for r in b.required if r not in common_required]
      if not residual:
        return ""
      label = "+".join(residual)
      if label not in labels:
        labels.append(label)
    if len(labels) < 2:
      return ""
    joined = " | ".join(labels)
    if self.kind == "oneOf":
      return f"[Requires one of (exclusive): {joined}]"
    return f"[Requires one of: {joined}]"


class SchemaSlimmer:
  """Minifies MCP `tools/list` schemas and inlines `$defs`/`$ref` for cross-LLM compatibility."""

  def __init__(
      self,
      max_desc_chars: int = 120,
      max_param_desc_chars: int = 80,
      allow_tools: set[str] | None = None,
      deny_tools: set[str] | None = None,
      inline_refs: bool = True,
      canonical_sort: bool = True,
      preserve_root_unions: bool = False,
  ) -> None:
    self.max_desc_chars = max_desc_chars
    self.max_param_desc_chars = max_param_desc_chars
    self.allow_tools = allow_tools
    self.deny_tools = deny_tools or set()
    self.inline_refs = inline_refs
    self.canonical_sort = canonical_sort
    self.preserve_root_unions = preserve_root_unions
    self._tool_param_types: dict[str, dict[str, str]] = {}
    self._tool_union_contracts: dict[str, UnionContract] = {}
    self.policy_warnings: list[str] = []
    self._pending_union_hint: str = ""
    self._pending_union_contract: UnionContract | None = None

  def _resolve_ref(
      self,
      ref_str: str,
      defs: dict[str, Any],
      visited: frozenset[str],
      stats: SlimStats | None = None,
  ) -> dict[str, Any] | None:
    for prefix in ("#/$defs/", "#/definitions/"):
      if ref_str.startswith(prefix):
        def_name = ref_str[len(prefix) :]
        if def_name in visited:
          return {"type": "object"}
        target = defs.get(def_name)
        if isinstance(target, dict):
          if stats is not None:
            stats.refs_inlined += 1
          return self.slim_schema_node(
              target,
              is_root=False,
              defs=defs,
              visited=visited | {def_name},
              stats=stats,
          )
    return None

  def slim_schema_node(
      self,
      node: Any,
      is_root: bool = False,
      defs: dict[str, Any] | None = None,
      visited: frozenset[str] = frozenset(),
      stats: SlimStats | None = None,
  ) -> Any:
    if not isinstance(node, dict):
      return node

    if defs is None:
      defs = {}
      if isinstance(node.get("$defs"), dict):
        defs.update(node["$defs"])
      if isinstance(node.get("definitions"), dict):
        defs.update(node["definitions"])

    if self.inline_refs and "$ref" in node and isinstance(node["$ref"], str):
      resolved = self._resolve_ref(node["$ref"], defs, visited, stats=stats)
      if resolved is not None:
        merged = dict(resolved)
        for k, v in node.items():
          if k != "$ref" and k not in _REDUNDANT_SCHEMA_KEYS:
            if k == "description" and isinstance(v, str):
              compact = compact_description(v, max_chars=self.max_param_desc_chars)
              if compact:
                merged[k] = compact
            else:
              merged[k] = v
        return merged

    out: dict[str, Any] = {}
    for k, v in node.items():
      if k in _REDUNDANT_SCHEMA_KEYS or (isinstance(k, str) and k.startswith("x-")):
        continue
      if k == "description" and isinstance(v, str):
        compact = compact_description(v, max_chars=self.max_param_desc_chars)
        if compact:
          out[k] = compact
        continue
      if k == "properties" and isinstance(v, dict):
        props: dict[str, Any] = {}
        prop_items = sorted(v.items()) if self.canonical_sort else v.items()
        for prop_name, prop_schema in prop_items:
          slimmed_prop = self.slim_schema_node(
              prop_schema, is_root=False, defs=defs, visited=visited, stats=stats
          )
          if isinstance(slimmed_prop, dict):
            desc = slimmed_prop.get("description", "")
            if desc and _is_redundant_param_description(prop_name, desc):
              slimmed_prop.pop("description", None)
          props[prop_name] = slimmed_prop
        out[k] = props
        continue
      if k == "items" and isinstance(v, dict):
        out[k] = self.slim_schema_node(
            v, is_root=False, defs=defs, visited=visited, stats=stats
        )
        continue
      if k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
        non_null = [
            branch
            for branch in v
            if not (isinstance(branch, dict) and branch.get("type") == "null")
        ]
        if len(non_null) == 1 and isinstance(non_null[0], dict):
          merged = self.slim_schema_node(
              non_null[0], is_root=False, defs=defs, visited=visited, stats=stats
          )
          if isinstance(merged, dict):
            for mk, mv in merged.items():
              out.setdefault(mk, mv)
          continue
        slimmed_branches = [
            self.slim_schema_node(
                branch, is_root=False, defs=defs, visited=visited, stats=stats
            )
            for branch in non_null
        ]
        if (
            is_root
            and (k == "allOf" or not self.preserve_root_unions)
            and slimmed_branches
            and all(
                isinstance(b, dict)
                and b.get("type", "object") == "object"
                and isinstance(b.get("properties"), dict)
                for b in slimmed_branches
            )
        ):
          merged_props: dict[str, Any] = dict(out.get("properties") or {})
          branch_req_sets: list[list[str]] = []
          branch_contracts: list[BranchContract] = []
          for raw_b, b in zip(non_null, slimmed_branches):
            for pk, pv in b["properties"].items():
              merged_props.setdefault(pk, pv)
            b_req = b.get("required")
            req_items = (
                [r for r in b_req if isinstance(r, str)]
                if isinstance(b_req, list)
                else []
            )
            branch_req_sets.append(req_items)
            forbid_extra = (
                isinstance(raw_b, dict)
                and raw_b.get("additionalProperties") is False
            )
            branch_contracts.append(
                BranchContract(
                    required=tuple(req_items),
                    allowed_props=frozenset(b["properties"].keys()),
                    forbid_extra=forbid_extra,
                )
            )
          if self.canonical_sort:
            merged_props = dict(sorted(merged_props.items()))
          out["properties"] = merged_props
          existing_req = [
              r for r in (out.get("required") or []) if isinstance(r, str)
          ]
          if k == "allOf":
            combined_req = list(existing_req)
            for req_list in branch_req_sets:
              for r in req_list:
                if r not in combined_req:
                  combined_req.append(r)
          else:
            common = (
                set(branch_req_sets[0]).intersection(*branch_req_sets[1:])
                if branch_req_sets
                else set()
            )
            combined_req = list(existing_req)
            if branch_req_sets:
              for r in branch_req_sets[0]:
                if r in common and r not in combined_req:
                  combined_req.append(r)
            if len(branch_contracts) >= 2:
              contract = UnionContract(kind=k, branches=tuple(branch_contracts))
              self._pending_union_contract = contract
              self._pending_union_hint = contract.compact_hint(set(combined_req))
          if combined_req:
            out["required"] = combined_req
          continue
        out[k] = (
            slimmed_branches
            if slimmed_branches
            else [
                self.slim_schema_node(
                    branch, is_root=False, defs=defs, visited=visited, stats=stats
                )
                for branch in v
            ]
        )
        continue
      if k == "type" and isinstance(v, list):
        non_null_types = [t for t in v if isinstance(t, str) and t != "null"]
        if len(non_null_types) == 1:
          out[k] = non_null_types[0]
          continue
      if k == "required" and isinstance(v, list):
        if not v:
          continue
        out[k] = list(v)
        continue
      out[k] = v
    if is_root and "type" not in out:
      out["type"] = "object"
    return out

  def is_tool_allowed(self, name: str) -> bool:
    if self.allow_tools is not None and not _matches_any_pattern(name, self.allow_tools):
      return False
    if self.deny_tools and _matches_any_pattern(name, self.deny_tools):
      return False
    return True

  def check_policy_typos(self, upstream_tool_names: list[str]) -> list[str]:
    """Detects exact allow/deny tool patterns that match zero upstream tools and suggests close matches."""
    warnings: list[str] = []
    name_set = set(upstream_tool_names)
    for flag_label, pat_set in (
        ("--allow-tools", self.allow_tools or set()),
        ("--deny-tools", self.deny_tools),
    ):
      for pat in sorted(pat_set):
        if "*" in pat or "?" in pat or "[" in pat:
          if not any(fnmatch.fnmatchcase(n, pat) for n in upstream_tool_names):
            warnings.append(
                f"Policy {flag_label} pattern '{pat}' matched 0 upstream tools."
            )
          continue
        if pat not in name_set:
          close = difflib.get_close_matches(pat, upstream_tool_names, n=1, cutoff=0.6)
          suggestion = f" (did you mean '{close[0]}'?)" if close else ""
          warnings.append(
              f"Policy {flag_label} entry '{pat}' matched 0 upstream tools{suggestion}."
          )
    self.policy_warnings = warnings
    return warnings

  def slim_tool(
      self, tool: dict[str, Any], stats: SlimStats | None = None
  ) -> dict[str, Any]:
    slimmed: dict[str, Any] = {"name": tool["name"]}
    desc = tool.get("description")
    if isinstance(desc, str) and desc.strip():
      slimmed["description"] = compact_description(
          desc, max_chars=self.max_desc_chars
      )
    self._pending_union_contract = None
    self._pending_union_hint = ""
    input_schema = tool.get("inputSchema")
    if isinstance(input_schema, dict):
      slim_schema = self.slim_schema_node(
          input_schema, is_root=True, stats=stats
      )
      slimmed["inputSchema"] = slim_schema
      if self._pending_union_contract is not None:
        self._tool_union_contracts[tool["name"]] = self._pending_union_contract
        if self._pending_union_hint:
          base_desc = slimmed.get("description", "")
          slimmed["description"] = (
              f"{base_desc} {self._pending_union_hint}".strip()
          )
      else:
        self._tool_union_contracts.pop(tool["name"], None)
      self._pending_union_contract = None
      self._pending_union_hint = ""
      if isinstance(slim_schema, dict) and isinstance(
          slim_schema.get("properties"), dict
      ):
        param_map: dict[str, str] = {}
        for p_name, p_spec in slim_schema["properties"].items():
          if isinstance(p_spec, dict) and isinstance(p_spec.get("type"), str):
            param_map[p_name] = p_spec["type"]
        self._tool_param_types[tool["name"]] = param_map
    output_schema = tool.get("outputSchema")
    if isinstance(output_schema, dict):
      self._pending_union_contract = None
      self._pending_union_hint = ""
      slim_out_schema = self.slim_schema_node(
          output_schema, is_root=True, stats=stats
      )
      self._pending_union_contract = None
      self._pending_union_hint = ""
      if isinstance(slim_out_schema, dict):
        out_props = slim_out_schema.get("properties")
        is_void_schema = slim_out_schema.get("type") == "null" or (
            isinstance(out_props, dict)
            and len(out_props) == 1
            and isinstance(out_props.get("result"), dict)
            and out_props["result"].get("type") == "null"
        )
        if not is_void_schema:
          slimmed["outputSchema"] = slim_out_schema
    return slimmed

  def validate_union_arguments(
      self, tool_name: str, arguments: dict[str, Any]
  ) -> str | None:
    """Validates call arguments against original root `anyOf`/`oneOf` branch contracts."""
    if not isinstance(arguments, dict):
      return None
    contract = self._tool_union_contracts.get(tool_name)
    if contract is None:
      return None
    provided = {k for k, v in arguments.items() if v is not None}
    matched = 0
    for branch in contract.branches:
      if not all(req_k in provided for req_k in branch.required):
        continue
      if branch.forbid_extra and not (provided <= branch.allowed_props):
        continue
      matched += 1
    if contract.kind == "anyOf":
      if matched >= 1:
        return None
      return (
          f"[mcp-slim] Invalid arguments for '{tool_name}': flattened root anyOf "
          f"requires satisfying at least one branch among {contract.describe_branches()}."
      )
    if contract.kind == "oneOf":
      if matched == 1:
        return None
      if matched == 0:
        return (
            f"[mcp-slim] Invalid arguments for '{tool_name}': flattened root oneOf "
            f"requires satisfying exactly one branch among {contract.describe_branches()}."
        )
      return (
          f"[mcp-slim] Invalid arguments for '{tool_name}': arguments matched {matched} "
          f"mutually exclusive root oneOf branches; satisfy only one of {contract.describe_branches()}."
      )
    return None

  def coerce_arguments(
      self,
      tool_name: str,
      arguments: dict[str, Any],
      stats: SlimStats | None = None,
  ) -> dict[str, Any]:
    """Coerces stringified scalar/JSON arguments using the cached tool inputSchema."""
    if not isinstance(arguments, dict):
      return arguments
    expected_map = self._tool_param_types.get(tool_name)
    if not expected_map:
      return arguments

    out: dict[str, Any] = {}
    used_keys = set(arguments.keys())
    for orig_k, orig_v in arguments.items():
      if orig_k in expected_map:
        out[orig_k] = orig_v
        continue
      norm_k = re.sub(r"[_-]", "", orig_k).lower()
      if len(norm_k) > 4 and norm_k.endswith("ies"):
        norm_k_singular = norm_k[:-3] + "y"
      elif len(norm_k) > 3 and norm_k.endswith("s"):
        norm_k_singular = norm_k[:-1]
      else:
        norm_k_singular = norm_k
      candidates = [
          decl_k
          for decl_k in expected_map
          if decl_k not in used_keys
          and (
              re.sub(r"[_-]", "", decl_k).lower() in (norm_k, norm_k_singular)
              or (
                  len(decl_k) > 3
                  and re.sub(r"[_-]", "", decl_k).lower().rstrip("s") == norm_k_singular
              )
          )
      ]
      if len(candidates) == 1:
        healed_k = candidates[0]
        out[healed_k] = orig_v
        used_keys.add(healed_k)
        if stats is not None:
          stats.args_coerced += 1
      else:
        out[orig_k] = orig_v

    for k, v in list(out.items()):
      if not isinstance(v, str):
        continue
      expected = expected_map.get(k)
      if expected == "integer":
        stripped = v.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
          out[k] = int(stripped)
          if stats is not None:
            stats.args_coerced += 1
      elif expected == "number":
        stripped = v.strip()
        try:
          val = float(stripped)
          if math.isfinite(val):
            out[k] = val
            if stats is not None:
              stats.args_coerced += 1
        except ValueError:
          pass
      elif expected == "boolean":
        lower = v.strip() .lower()
        if lower in ("true", "false"):
          out[k] = lower == "true"
          if stats is not None:
            stats.args_coerced += 1
      elif expected in ("array", "object"):
        stripped = v.strip()
        if (expected == "array" and stripped.startswith("[") and stripped.endswith("]")) or (
            expected == "object" and stripped.startswith("{") and stripped.endswith("}")
        ):
          try:
            parsed = json.loads(stripped)
            if (expected == "array" and isinstance(parsed, list)) or (
                expected == "object" and isinstance(parsed, dict)
            ):
              out[k] = parsed
              if stats is not None:
                stats.args_coerced += 1
          except json.JSONDecodeError:
            pass
    return out

  def slim_tools_list(
      self, tools: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    before_tokens = estimate_tokens(tools) if stats is not None else 0
    upstream_names = [
        t["name"] for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)
    ]
    self.check_policy_typos(upstream_names)
    filtered: list[dict[str, Any]] = []
    for tool in tools:
      name = tool.get("name", "")
      if not self.is_tool_allowed(name):
        continue
      filtered.append(self.slim_tool(tool, stats=stats))
    if self.canonical_sort:
      filtered.sort(key=lambda t: t.get("name", ""))
    if stats is not None:
      after_tokens = estimate_tokens(filtered)
      stats.tools_total += len(tools)
      stats.tools_kept += len(filtered)
      stats.schema_tokens_before += before_tokens
      stats.schema_tokens_after += after_tokens
    return filtered


class LazyToolBroker:
  """On-demand MCP meta-tool discovery broker (`--lazy` mode).

  Replaces large `tools/list` catalogs (40-100+ tools) with two compact meta-tools
  (`mcp_search_tools` and `mcp_call_tool`), cutting static schema context consumption
  by ~98% while allowing the LLM to search and invoke any upstream tool on demand.
  """

  SEARCH_TOOL_NAME = "mcp_search_tools"
  CALL_TOOL_NAME = "mcp_call_tool"

  def __init__(self, schema_slimmer: SchemaSlimmer | None = None) -> None:
    self.schema_slimmer = schema_slimmer or SchemaSlimmer()
    self._tools_by_name: dict[str, dict[str, Any]] = {}
    self._doc_tokens: dict[str, list[str]] = {}
    self._df: dict[str, int] = {}

  @staticmethod
  def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]

  def ingest_tools(
      self, raw_tools: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    """Indexes upstream tools and returns the 2 compact meta-tool definitions."""
    before_tokens = estimate_tokens(raw_tools) if stats is not None else 0
    self._tools_by_name.clear()
    self._doc_tokens.clear()
    self._df.clear()

    for tool in raw_tools:
      name = tool.get("name", "")
      if not name or not self.schema_slimmer.is_tool_allowed(name):
        continue
      slimmed = self.schema_slimmer.slim_tool(tool, stats=stats)
      self._tools_by_name[name] = slimmed
      props = (
          slimmed.get("inputSchema", {}).get("properties", {})
          if isinstance(slimmed.get("inputSchema"), dict)
          else {}
      )
      searchable = " ".join([
          name.replace("_", " ").replace("-", " "),
          name,
          slimmed.get("description", ""),
          " ".join(props.keys()),
      ])
      tokens = self._tokenize(searchable)
      self._doc_tokens[name] = tokens
      for unique_tok in set(tokens):
        self._df[unique_tok] = self._df.get(unique_tok, 0) + 1

    meta_tools = self.meta_tool_definitions()
    if stats is not None:
      after_tokens = estimate_tokens(meta_tools)
      stats.tools_total += len(raw_tools)
      stats.tools_kept += len(self._tools_by_name)
      stats.schema_tokens_before += before_tokens
      stats.schema_tokens_after += after_tokens
    return meta_tools

  def meta_tool_definitions(self) -> list[dict[str, Any]]:
    count = len(self._tools_by_name)
    return [
        {
            "name": self.SEARCH_TOOL_NAME,
            "description": (
                f"Search {count} available MCP tools by keyword and return their"
                " input schemas. Pass query='' to list all tool names."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords matching tool name or capability.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max tools to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": self.CALL_TOOL_NAME,
            "description": (
                "Invoke any upstream MCP tool discovered via mcp_search_tools by"
                " passing its exact tool_name and arguments object."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["tool_name"],
            },
        },
    ]

  def search(self, query: str, limit: int = 5) -> dict[str, Any]:
    """BM25-ranked search over indexed upstream tools."""
    limit = max(1, min(limit, 25))
    q_tokens = self._tokenize(query.replace("_", " ").replace("-", " "))
    if not q_tokens:
      catalog = [
          {"name": name, "description": t.get("description", "")}
          for name, t in sorted(self._tools_by_name.items())
      ]
      return {"total_indexed": len(catalog), "tools": catalog[: max(limit, 20)]}

    n_docs = max(1, len(self._tools_by_name))
    avgdl = (
        sum(len(toks) for toks in self._doc_tokens.values()) / n_docs
        if self._doc_tokens
        else 1.0
    )
    scores: list[tuple[float, str]] = []
    for name, doc_toks in self._doc_tokens.items():
      score = 0.0
      dl = len(doc_toks)
      if query.lower() in name.lower():
        score += 4.0
      for qt in q_tokens:
        tf = doc_toks.count(qt)
        if tf == 0:
          for dt in doc_toks:
            if dt.startswith(qt) or qt.startswith(dt):
              tf += 1
        if tf > 0:
          df = self._df.get(qt, 1)
          idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
          score += idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (dl / avgdl))))
      if score > 0.0:
        scores.append((score, name))

    scores.sort(key=lambda item: (-item[0], item[1]))
    matched = [self._tools_by_name[name] for _, name in scores[:limit]]
    return {
        "query": query,
        "matched_count": len(matched),
        "total_indexed": len(self._tools_by_name),
        "tools": matched,
    }


class OutputSlimmer:
  """Compacts verbose `tools/call` outputs (JSON blobs, logs, ANSI sequences)."""

  def __init__(
      self,
      max_list_items: int = 50,
      max_string_chars: int = 4000,
      strip_nulls: bool = True,
      strip_hypermedia: bool = True,
      flatten_resources: bool = True,
  ) -> None:
    self.max_list_items = max_list_items
    self.max_string_chars = max_string_chars
    self.strip_nulls = strip_nulls
    self.strip_hypermedia = strip_hypermedia
    self.flatten_resources = flatten_resources

  def compact_json_value(self, value: Any) -> Any:
    if isinstance(value, dict):
      out: dict[str, Any] = {}
      for k, v in value.items():
        if self.strip_hypermedia and k in _NOISY_HYPERMEDIA_KEYS:
          continue
        if self.strip_nulls and (v is None or v == "" or v == [] or v == {}):
          continue
        compacted = self.compact_json_value(v)
        if self.strip_nulls and (
            compacted is None
            or compacted == ""
            or compacted == []
            or compacted == {}
        ):
          continue
        out[k] = compacted
      return out
    if isinstance(value, list):
      total = len(value)
      sliced = value[: self.max_list_items]
      compacted_list = [self.compact_json_value(item) for item in sliced]
      if total > self.max_list_items:
        compacted_list.append(
            f"... ({total - self.max_list_items} more items truncated by"
            " mcp-slim)"
        )
      return compacted_list
    if isinstance(value, str):
      cleaned_val = _LONE_SURROGATE.sub("\ufffd", value)
      if len(cleaned_val) > self.max_string_chars:
        return (
            cleaned_val[: self.max_string_chars]
            + f"... [truncated {len(cleaned_val) - self.max_string_chars} chars]"
        )
      return cleaned_val
    return value

  def _fit_json_within_budget(self, compacted: Any) -> str:
    """Ensures serialized JSON stays valid even when a list of objects exceeds max_string_chars."""
    serialized = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) <= self.max_string_chars:
      return serialized

    if isinstance(compacted, list) and len(compacted) > 1:
      kept: list[Any] = []
      for item in compacted:
        candidate = [*kept, item, f"... ({len(compacted) - len(kept) - 1} more items truncated by mcp-slim)"]
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) > self.max_string_chars and kept:
          break
        kept.append(item)
      rem = len(compacted) - len(kept)
      if rem > 0:
        kept.append(f"... ({rem} more items truncated by mcp-slim)")
      return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))

    if isinstance(compacted, dict):
      for list_key in ("items", "results", "data", "rows", "issues", "commits", "nodes"):
        arr = compacted.get(list_key)
        if isinstance(arr, list) and len(arr) > 1:
          kept_arr: list[Any] = []
          for item in arr:
            trial = {**compacted, list_key: [*kept_arr, item]}
            if len(json.dumps(trial, ensure_ascii=False, separators=(",", ":"))) > self.max_string_chars and kept_arr:
              break
            kept_arr.append(item)
          rem = len(arr) - len(kept_arr)
          if rem > 0:
            kept_arr.append(f"... ({rem} more items truncated by mcp-slim)")
          return json.dumps({**compacted, list_key: kept_arr}, ensure_ascii=False, separators=(",", ":"))

    return serialized

  def slim_text(self, text: str) -> str:
    if not text:
      return text
    cleaned = _LONE_SURROGATE.sub("\ufffd", _ANSI_ESCAPE.sub("", text))
    stripped = cleaned.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
      try:
        parsed = json.loads(stripped)
        compacted = self.compact_json_value(parsed)
        return self._fit_json_within_budget(compacted)
      except (json.JSONDecodeError, TypeError):
        pass
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
    # Compact NDJSON / JSON-Lines payloads where each line is a JSON object
    if "\n" in stripped and stripped.startswith("{") and stripped.endswith("}"):
      raw_lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
      if len(raw_lines) > 1 and all(
          ln.startswith("{") and ln.endswith("}") for ln in raw_lines
      ):
        try:
          jsonl_items = [
              self.compact_json_value(json.loads(ln))
              for ln in raw_lines[: self.max_list_items]
          ]
          out_lines = [
              json.dumps(item, ensure_ascii=False, separators=(",", ":"))
              for item in jsonl_items
          ]
          if len(raw_lines) > self.max_list_items:
            out_lines.append(
                f"... ({len(raw_lines) - self.max_list_items} more JSONL lines"
                " truncated by mcp-slim)"
            )
          cleaned = "\n".join(out_lines)
        except (json.JSONDecodeError, TypeError):
          pass
    cleaned = re.sub(r"\|[\s:-]*-{3,}[\s:-]*(?=\|)", "|---", cleaned)
    cleaned = re.sub(r"\|\s+([^|\n]+?)\s+(?=\|)", r"|\1", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    if len(cleaned) > self.max_string_chars:
      omitted = len(cleaned) - self.max_string_chars
      if "\n" in cleaned and self.max_string_chars >= 120:
        head_budget = int(self.max_string_chars * 0.6)
        tail_budget = self.max_string_chars - head_budget
        cleaned = (
            cleaned[:head_budget]
            + f"\n... [truncated {omitted} chars by mcp-slim] ...\n"
            + cleaned[-tail_budget:]
        )
      else:
        cleaned = (
            cleaned[: self.max_string_chars]
            + f"\n... [truncated {omitted} chars by mcp-slim]"
        )
    return cleaned

  def slim_call_result(
      self, result: dict[str, Any], stats: SlimStats | None = None
  ) -> dict[str, Any]:
    before_tokens = estimate_tokens(result) if stats is not None else 0
    out = dict(result)
    content = out.get("content")
    if isinstance(content, list):
      new_content: list[Any] = []
      for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
          text = item.get("text")
          if isinstance(text, str):
            item = {**item, "text": self.slim_text(text)}
        elif (
            isinstance(item, dict)
            and item.get("type") == "resource"
            and isinstance(item.get("resource"), dict)
        ):
          res = item["resource"]
          res_text = res.get("text")
          if isinstance(res_text, str):
            slimmed_res = self.slim_text(res_text)
            if self.flatten_resources:
              uri = res.get("uri")
              header = f"[Resource: {uri}]\n" if isinstance(uri, str) and uri else ""
              item = {"type": "text", "text": f"{header}{slimmed_res}"}
            else:
              item = {**item, "resource": {**res, "text": slimmed_res}}
        new_content.append(item)
      out["content"] = new_content
    if "structuredContent" in out and isinstance(
        out["structuredContent"], (dict, list)
    ):
      compacted_sc = self.compact_json_value(out["structuredContent"])
      if compacted_sc in (None, {}, []):
        out.pop("structuredContent", None)
      else:
        out["structuredContent"] = compacted_sc
    if stats is not None:
      after_tokens = estimate_tokens(out)
      stats.output_tokens_before += before_tokens
      stats.output_tokens_after += after_tokens
    return out
