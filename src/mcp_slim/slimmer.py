"""Schema compaction, $ref inlining, and lazy meta-tool broker for MCP JSON-RPC traffic."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import difflib
import fnmatch
import hashlib
import json
import math
import re
from typing import Any
import urllib.parse

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
  refs_deduped: int = 0
  cycles_broken: int = 0
  sampling_guarded: int = 0
  cache_hits: int = 0
  args_coerced: int = 0
  blocked_calls: int = 0
  rug_pulls_blocked: int = 0
  resources_blocked: int = 0
  tables_compacted: int = 0

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


_READ_ONLY_PREFIXES = (
    "get_",
    "read_",
    "list_",
    "search_",
    "fetch_",
    "describe_",
    "inspect_",
    "find_",
    "query_",
)

_ANNOTATION_BOOL_KEYS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)


@dataclass
class _RefContext:
  """Per-schema traversal state for cycle-aware DAG memoization and JSON Pointer resolution."""

  root_doc: dict[str, Any]
  memo: dict[str, tuple[Any, int]] = field(default_factory=dict)
  total_inlined: int = 0
  cycle_tainted: int = 0


def _decode_pointer_segment(segment: str) -> str:
  """Decodes a single RFC 6901 JSON Pointer segment (including URL-escaped fragments)."""
  return urllib.parse.unquote(segment).replace("~1", "/").replace("~0", "~")


def _build_cycle_stub(target: dict[str, Any]) -> dict[str, Any]:
  """Synthesizes a compact, cycle-safe stub preserving the target's normalized JSON Schema type."""
  raw_type = target.get("type")
  if isinstance(raw_type, list):
    non_null = [t for t in raw_type if isinstance(t, str) and t != "null"]
    norm_type = non_null[0] if len(non_null) == 1 else "object"
  elif isinstance(raw_type, str) and raw_type:
    norm_type = raw_type
  else:
    norm_type = "object"
  return {"type": norm_type}


def _resolve_pointer_target(
    ref_str: str, root_doc: dict[str, Any], defs: dict[str, Any]
) -> tuple[str, dict[str, Any], bool] | None:
  """Resolves an internal JSON Pointer (`#`, `#/$defs/...`, `#/definitions/...`, `#/properties/...`)."""
  if ref_str == "#":
    return ("#", root_doc, False)
  if not ref_str.startswith("#/"):
    return None
  segments = [_decode_pointer_segment(s) for s in ref_str[2:].split("/")]
  if len(segments) >= 2 and segments[0] in ("$defs", "definitions"):
    def_name = segments[1]
    base_def = defs.get(def_name)
    if isinstance(base_def, dict):
      root_defs = root_doc.get(segments[0])
      is_local = not (
          isinstance(root_defs, dict) and root_defs.get(def_name) is base_def
      )
      cur: Any = base_def
      for extra_seg in segments[2:]:
        if isinstance(cur, dict) and extra_seg in cur:
          cur = cur[extra_seg]
        elif (
            isinstance(cur, list)
            and extra_seg.isdigit()
            and int(extra_seg) < len(cur)
        ):
          cur = cur[int(extra_seg)]
        else:
          cur = None
          break
      if isinstance(cur, dict):
        tail = "/".join(segments[1:])
        canon = (
            f"#/$local/{id(base_def)}/{tail}"
            if is_local
            else f"#/$defs/{tail}"
        )
        return (canon, cur, is_local)

  cur = root_doc
  for seg in segments:
    if isinstance(cur, dict) and seg in cur:
      cur = cur[seg]
    elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
      cur = cur[int(seg)]
    else:
      return None
  if isinstance(cur, dict):
    canon = "#/" + "/".join(segments)
    return (canon, cur, False)
  return None


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
      read_only_only: bool = False,
  ) -> None:
    self.max_desc_chars = max_desc_chars
    self.max_param_desc_chars = max_param_desc_chars
    self.allow_tools = allow_tools
    self.deny_tools = deny_tools or set()
    self.inline_refs = inline_refs
    self.canonical_sort = canonical_sort
    self.preserve_root_unions = preserve_root_unions
    self.read_only_only = read_only_only
    self._tool_param_types: dict[str, dict[str, str]] = {}
    self._tool_union_contracts: dict[str, UnionContract] = {}
    self._tool_annotations: dict[str, dict[str, bool]] = {}
    self.policy_warnings: list[str] = []
    self._pending_union_hint: str = ""
    self._pending_union_contract: UnionContract | None = None

  def _resolve_ref(
      self,
      ref_str: str,
      defs: dict[str, Any],
      visited: frozenset[str],
      stats: SlimStats | None = None,
      ctx: _RefContext | None = None,
  ) -> dict[str, Any] | None:
    effective_ctx = ctx or _RefContext(root_doc={"$defs": defs})
    resolved_target = _resolve_pointer_target(
        ref_str, effective_ctx.root_doc, defs
    )
    if resolved_target is None:
      return None
    canon_key, target, is_local = resolved_target

    if not is_local and canon_key in effective_ctx.memo:
      cached_node, nested_inlined = effective_ctx.memo[canon_key]
      effective_ctx.total_inlined += 1 + nested_inlined
      if stats is not None:
        stats.refs_inlined += 1 + nested_inlined
        stats.refs_deduped += 1
      return copy.deepcopy(cached_node)

    if canon_key in visited:
      effective_ctx.cycle_tainted += 1
      if stats is not None:
        stats.cycles_broken += 1
      return _build_cycle_stub(target)

    effective_ctx.total_inlined += 1
    if stats is not None:
      stats.refs_inlined += 1
    inlined_after_self = effective_ctx.total_inlined
    taint_before = effective_ctx.cycle_tainted

    slimmed_target = self._slim_schema_node_impl(
        target,
        is_root=False,
        defs=defs,
        visited=visited | {canon_key},
        stats=stats,
        ctx=effective_ctx,
    )
    nested_inlined = effective_ctx.total_inlined - inlined_after_self
    if (
        not is_local
        and effective_ctx.cycle_tainted == taint_before
        and isinstance(slimmed_target, dict)
    ):
      effective_ctx.memo[canon_key] = (
          copy.deepcopy(slimmed_target),
          nested_inlined,
      )
    return slimmed_target if isinstance(slimmed_target, dict) else None

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
    initial_visited = visited | {"#"} if is_root and not visited else visited
    ctx = _RefContext(root_doc=node)
    return self._slim_schema_node_impl(
        node,
        is_root=is_root,
        defs=defs,
        visited=initial_visited,
        stats=stats,
        ctx=ctx,
    )

  def _slim_schema_node_impl(
      self,
      node: Any,
      is_root: bool,
      defs: dict[str, Any] | None,
      visited: frozenset[str],
      stats: SlimStats | None,
      ctx: _RefContext,
  ) -> Any:
    if not isinstance(node, dict):
      return node

    if defs is None:
      scoped_defs: dict[str, Any] = {}
      if isinstance(node.get("$defs"), dict):
        scoped_defs.update(node["$defs"])
      if isinstance(node.get("definitions"), dict):
        scoped_defs.update(node["definitions"])
    elif isinstance(node.get("$defs"), dict) or isinstance(
        node.get("definitions"), dict
    ):
      scoped_defs = dict(defs)
      if isinstance(node.get("$defs"), dict):
        scoped_defs.update(node["$defs"])
      if isinstance(node.get("definitions"), dict):
        scoped_defs.update(node["definitions"])
    else:
      scoped_defs = defs

    if self.inline_refs and "$ref" in node and isinstance(node["$ref"], str):
      resolved = self._resolve_ref(
          node["$ref"], scoped_defs, visited, stats=stats, ctx=ctx
      )
      if resolved is not None:
        merged = dict(resolved)
        for k, v in node.items():
          if k != "$ref" and k not in _REDUNDANT_SCHEMA_KEYS:
            if k == "description" and isinstance(v, str):
              compact = compact_description(v, max_chars=self.max_param_desc_chars)
              if compact:
                merged[k] = compact
            elif isinstance(v, (dict, list)):
              merged[k] = self._slim_schema_node_impl(
                  v,
                  is_root=False,
                  defs=scoped_defs,
                  visited=visited,
                  stats=stats,
                  ctx=ctx,
              )
            else:
              merged[k] = v
        return merged

    out: dict[str, Any] = {}
    for k, v in node.items():
      if (
          k == "additionalProperties"
          and isinstance(v, dict)
          and v
          and "properties" not in node
      ):
        slimmed_ap = self._slim_schema_node_impl(
            v,
            is_root=False,
            defs=scoped_defs,
            visited=visited,
            stats=stats,
            ctx=ctx,
        )
        if isinstance(slimmed_ap, dict) and slimmed_ap:
          out[k] = slimmed_ap
        continue
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
          slimmed_prop = self._slim_schema_node_impl(
              prop_schema,
              is_root=False,
              defs=scoped_defs,
              visited=visited,
              stats=stats,
              ctx=ctx,
          )
          if isinstance(slimmed_prop, dict):
            desc = slimmed_prop.get("description", "")
            if desc and _is_redundant_param_description(prop_name, desc):
              slimmed_prop.pop("description", None)
          props[prop_name] = slimmed_prop
        out[k] = props
        continue
      if k == "items":
        if isinstance(v, dict):
          out[k] = self._slim_schema_node_impl(
              v,
              is_root=False,
              defs=scoped_defs,
              visited=visited,
              stats=stats,
              ctx=ctx,
          )
          continue
        if isinstance(v, list):
          out[k] = [
              self._slim_schema_node_impl(
                  elem,
                  is_root=False,
                  defs=scoped_defs,
                  visited=visited,
                  stats=stats,
                  ctx=ctx,
              )
              for elem in v
          ]
          continue
      if k == "prefixItems" and isinstance(v, list):
        out[k] = [
            self._slim_schema_node_impl(
                elem,
                is_root=False,
                defs=scoped_defs,
                visited=visited,
                stats=stats,
                ctx=ctx,
            )
            for elem in v
        ]
        continue
      if k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
        non_null = [
            branch
            for branch in v
            if not (isinstance(branch, dict) and branch.get("type") == "null")
        ]
        if len(non_null) == 1 and isinstance(non_null[0], dict):
          merged = self._slim_schema_node_impl(
              non_null[0],
              is_root=False,
              defs=scoped_defs,
              visited=visited,
              stats=stats,
              ctx=ctx,
          )
          if isinstance(merged, dict):
            for mk, mv in merged.items():
              out.setdefault(mk, mv)
          continue
        raw_slimmed_branches = [
            self._slim_schema_node_impl(
                branch,
                is_root=False,
                defs=scoped_defs,
                visited=visited,
                stats=stats,
                ctx=ctx,
            )
            for branch in non_null
        ]
        deduped_pairs: list[tuple[Any, Any]] = []
        seen_sigs: set[str] = set()
        for raw_b, slim_b in zip(non_null, raw_slimmed_branches):
          sig = json.dumps(
              slim_b, sort_keys=True, ensure_ascii=False, separators=(",", ":")
          )
          if sig in seen_sigs:
            if stats is not None:
              stats.refs_deduped += 1
            continue
          seen_sigs.add(sig)
          deduped_pairs.append((raw_b, slim_b))

        if len(deduped_pairs) == 1 and isinstance(deduped_pairs[0][1], dict):
          for mk, mv in deduped_pairs[0][1].items():
            out.setdefault(mk, mv)
          continue

        non_null = [p[0] for p in deduped_pairs]
        slimmed_branches = [p[1] for p in deduped_pairs]
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
                self._slim_schema_node_impl(
                    branch,
                    is_root=False,
                    defs=scoped_defs,
                    visited=visited,
                    stats=stats,
                    ctx=ctx,
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

  @staticmethod
  def extract_annotations(tool: dict[str, Any]) -> dict[str, bool]:
    """Extracts MCP ToolAnnotations boolean hints from `tool['annotations']` or top-level keys."""
    hints: dict[str, bool] = {}
    raw_ann = tool.get("annotations")
    if isinstance(raw_ann, dict):
      for k in _ANNOTATION_BOOL_KEYS:
        if isinstance(raw_ann.get(k), bool):
          hints[k] = raw_ann[k]
    for k in _ANNOTATION_BOOL_KEYS:
      if k not in hints and isinstance(tool.get(k), bool):
        hints[k] = tool[k]
    return hints

  def is_read_only_tool(
      self, name: str, tool: dict[str, Any] | None = None
  ) -> bool:
    """Determines whether a tool is read-only using MCP ToolAnnotations first, then prefix fallback."""
    hints = (
        self.extract_annotations(tool)
        if isinstance(tool, dict)
        else self._tool_annotations.get(name, {})
    )
    if hints.get("destructiveHint") is True:
      return False
    if "readOnlyHint" in hints:
      return hints["readOnlyHint"]
    return name.lower().startswith(_READ_ONLY_PREFIXES)

  def is_tool_allowed(
      self, name: str, tool: dict[str, Any] | None = None
  ) -> bool:
    if self.allow_tools is not None and not _matches_any_pattern(name, self.allow_tools):
      return False
    if self.deny_tools and _matches_any_pattern(name, self.deny_tools):
      return False
    if self.read_only_only and not self.is_read_only_tool(name, tool=tool):
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
    tool_name = str(tool["name"])
    ann = self.extract_annotations(tool)
    if ann:
      self._tool_annotations[tool_name] = ann
    else:
      self._tool_annotations.pop(tool_name, None)
    slimmed: dict[str, Any] = {"name": tool_name}
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
      if isinstance(name, str) and name:
        ann = self.extract_annotations(tool)
        if ann:
          self._tool_annotations[name] = ann
        else:
          self._tool_annotations.pop(name, None)
      if not self.is_tool_allowed(name, tool=tool):
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
      if isinstance(name, str) and name:
        ann = self.schema_slimmer.extract_annotations(tool)
        if ann:
          self.schema_slimmer._tool_annotations[name] = ann
        else:
          self.schema_slimmer._tool_annotations.pop(name, None)
      if not name or not self.schema_slimmer.is_tool_allowed(name, tool=tool):
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
      tabular_arrays: bool = False,
      min_table_rows: int = 4,
  ) -> None:
    self.max_list_items = max_list_items
    self.max_string_chars = max_string_chars
    self.strip_nulls = strip_nulls
    self.strip_hypermedia = strip_hypermedia
    self.flatten_resources = flatten_resources
    self.tabular_arrays = tabular_arrays
    self.min_table_rows = max(2, min_table_rows)

  @staticmethod
  def _is_scalar_cell(val: Any) -> bool:
    return val is None or isinstance(val, (str, int, float, bool))

  def _maybe_tabularize_list(
      self, items: list[Any], stats: SlimStats | None = None
  ) -> Any:
    """Converts homogeneous arrays of flat dicts into a compact columnar table when smaller."""
    if not self.tabular_arrays or len(items) < self.min_table_rows:
      return items
    truncated_note: str | None = None
    dict_rows = items
    if (
        isinstance(items[-1], str)
        and items[-1].startswith("... (")
        and items[-1].endswith("truncated by mcp-slim)")
    ):
      truncated_note = items[-1]
      dict_rows = items[:-1]
    if len(dict_rows) < self.min_table_rows:
      return items
    if not all(isinstance(r, dict) and r for r in dict_rows):
      return items

    columns: list[str] = []
    seen_cols: set[str] = set()
    for row in dict_rows:
      for k, v in row.items():
        if not isinstance(k, str) or not self._is_scalar_cell(v):
          return items
        if k not in seen_cols:
          seen_cols.add(k)
          columns.append(k)

    if len(columns) < 2 or len(columns) > 32:
      return items

    min_overlap = math.ceil(len(columns) * 0.75)
    for row in dict_rows:
      if len(row) < min_overlap:
        return items

    matrix = [[row.get(col) for col in columns] for row in dict_rows]
    table_obj: dict[str, Any] = {
        "_table": True,
        "columns": columns,
        "rows": matrix,
    }
    if truncated_note is not None:
      table_obj["truncated"] = truncated_note

    raw_bytes = len(
        json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    tab_bytes = len(
        json.dumps(table_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if tab_bytes < raw_bytes:
      if stats is not None:
        stats.tables_compacted += 1
      return table_obj
    return items

  def compact_json_value(
      self, value: Any, stats: SlimStats | None = None
  ) -> Any:
    if isinstance(value, dict):
      out: dict[str, Any] = {}
      for k, v in value.items():
        if self.strip_hypermedia and k in _NOISY_HYPERMEDIA_KEYS:
          continue
        if self.strip_nulls and (v is None or v == "" or v == [] or v == {}):
          continue
        compacted = self.compact_json_value(v, stats=stats)
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
      compacted_list = [
          self.compact_json_value(item, stats=stats) for item in sliced
      ]
      if total > self.max_list_items:
        compacted_list.append(
            f"... ({total - self.max_list_items} more items truncated by"
            " mcp-slim)"
        )
      return self._maybe_tabularize_list(compacted_list, stats=stats)
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

  def slim_text(
      self,
      text: str,
      stats: SlimStats | None = None,
      guard: Any | None = None,
      *,
      force_redact: bool = False,
  ) -> str:
    if not text:
      return text
    cleaned = _LONE_SURROGATE.sub("\ufffd", _ANSI_ESCAPE.sub("", text))
    stripped = cleaned.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
      try:
        parsed = json.loads(stripped)
        if guard is not None:
          parsed = guard.redact_value(parsed, force_redact=force_redact)
        compacted = self.compact_json_value(parsed, stats=stats)
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
          jsonl_items = []
          for ln in raw_lines[: self.max_list_items]:
            obj = json.loads(ln)
            if guard is not None:
              obj = guard.redact_value(obj, force_redact=force_redact)
            jsonl_items.append(self.compact_json_value(obj))
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
          return (
              cleaned[: self.max_string_chars]
              + f"\n... [truncated {len(cleaned) - self.max_string_chars} chars by mcp-slim]"
              if len(cleaned) > self.max_string_chars
              else cleaned
          )
        except (json.JSONDecodeError, TypeError):
          pass
    if guard is not None:
      cleaned = guard.redact_text(cleaned, force_redact=force_redact)
    cleaned = re.sub(r"\|[\s:-]*-{3,}[\s:-]*(?=\|)", "|---", cleaned)
    cleaned = re.sub(r"\|\s+([^|\n]+?)\s+(?=\|)", r"|\1", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    if len(cleaned) > self.max_string_chars:
      marker_spans = [
          (m.start(), m.end())
          for m in re.finditer(r"\[(?:SECRET_REF|REDACTED):[^\]]+\]", cleaned)
      ]
      if "\n" in cleaned and self.max_string_chars >= 120:
        head_end = int(self.max_string_chars * 0.6)
        tail_start = len(cleaned) - (self.max_string_chars - head_end)
        for ms, me in marker_spans:
          if ms < head_end < me:
            head_end = me
          if ms < tail_start < me:
            tail_start = max(head_end, ms)
        if tail_start <= head_end:
          return cleaned
        omitted = tail_start - head_end
        cleaned = (
            cleaned[:head_end]
            + f"\n... [truncated {omitted} chars by mcp-slim] ...\n"
            + cleaned[tail_start:]
        )
      else:
        head_end = self.max_string_chars
        for ms, me in marker_spans:
          if ms < head_end < me:
            head_end = me
        if head_end >= len(cleaned):
          return cleaned
        omitted = len(cleaned) - head_end
        cleaned = (
            cleaned[:head_end]
            + f"\n... [truncated {omitted} chars by mcp-slim]"
        )
    return cleaned

  def slim_call_result(
      self,
      result: dict[str, Any],
      stats: SlimStats | None = None,
      guard: Any | None = None,
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
            item = {
                **item,
                "text": self.slim_text(text, stats=stats, guard=guard),
            }
        elif (
            isinstance(item, dict)
            and item.get("type") == "resource"
            and isinstance(item.get("resource"), dict)
        ):
          res = item["resource"]
          if guard is not None and isinstance(res.get("blob"), str):
            res = guard.redact_value(res)
          res_text = res.get("text")
          if isinstance(res_text, str):
            slimmed_res = self.slim_text(res_text, stats=stats, guard=guard)
            if self.flatten_resources:
              uri = res.get("uri")
              header = f"[Resource: {uri}]\n" if isinstance(uri, str) and uri else ""
              item = {"type": "text", "text": f"{header}{slimmed_res}"}
            else:
              item = {**item, "resource": {**res, "text": slimmed_res}}
          elif res is not item["resource"]:
            item = {**item, "resource": res}
        new_content.append(item)
      out["content"] = new_content
    if "structuredContent" in out and isinstance(
        out["structuredContent"], (dict, list)
    ):
      sc_val = (
          guard.redact_value(out["structuredContent"])
          if guard is not None
          else out["structuredContent"]
      )
      compacted_sc = self.compact_json_value(sc_val, stats=stats)
      if compacted_sc in (None, {}, []):
        out.pop("structuredContent", None)
      else:
        out["structuredContent"] = compacted_sc
    if stats is not None:
      after_tokens = estimate_tokens(out)
      stats.output_tokens_before += before_tokens
      stats.output_tokens_after += after_tokens
    return out

  def slim_resource_result(
      self,
      result: dict[str, Any],
      stats: SlimStats | None = None,
      guard: Any | None = None,
  ) -> dict[str, Any]:
    """Compacts MCP `resources/read` payloads (`result.contents[*].text` and UTF-8 `blob`)."""
    contents = result.get("contents")
    if not isinstance(contents, list):
      return result
    before_tokens = estimate_tokens(result) if stats is not None else 0
    new_contents: list[Any] = []
    for item in contents:
      if isinstance(item, dict):
        cur = dict(item)
        if guard is not None and isinstance(cur.get("blob"), str):
          cur["blob"] = guard.redact_base64_blob(
              cur["blob"],
              mime_type=cur.get("mimeType") if isinstance(cur.get("mimeType"), str) else None,
          )
        if isinstance(cur.get("text"), str):
          cur["text"] = self.slim_text(cur["text"], stats=stats, guard=guard)
        item = cur
      new_contents.append(item)
    out = {**result, "contents": new_contents}
    if stats is not None:
      stats.output_tokens_before += before_tokens
      stats.output_tokens_after += estimate_tokens(out)
    return out

  def _slim_prompt_part(
      self,
      part: Any,
      stats: SlimStats | None = None,
      guard: Any | None = None,
      *,
      force_redact: bool = False,
  ) -> Any:
    if not isinstance(part, dict):
      return part
    ptype = part.get("type")
    if ptype == "text" and isinstance(part.get("text"), str):
      return {
          **part,
          "text": self.slim_text(
              part["text"], stats=stats, guard=guard, force_redact=force_redact
          ),
      }
    if ptype == "resource" and isinstance(part.get("resource"), dict):
      res = dict(part["resource"])
      if guard is not None and isinstance(res.get("blob"), str):
        res["blob"] = guard.redact_base64_blob(
            res["blob"],
            mime_type=res.get("mimeType") if isinstance(res.get("mimeType"), str) else None,
            force_redact=force_redact,
        )
      if isinstance(res.get("text"), str):
        res["text"] = self.slim_text(
            res["text"], stats=stats, guard=guard, force_redact=force_redact
        )
      return {**part, "resource": res}
    return part

  def slim_prompt_result(
      self,
      result: dict[str, Any],
      stats: SlimStats | None = None,
      guard: Any | None = None,
      *,
      force_redact: bool = False,
  ) -> dict[str, Any]:
    """Compacts MCP `prompts/get` payloads (`result.messages[*].content`)."""
    messages = result.get("messages")
    if not isinstance(messages, list):
      return result
    before_tokens = estimate_tokens(result) if stats is not None else 0
    new_messages: list[Any] = []
    for msg in messages:
      if not isinstance(msg, dict):
        new_messages.append(msg)
        continue
      content = msg.get("content")
      if isinstance(content, dict):
        msg = {
            **msg,
            "content": self._slim_prompt_part(
                content, stats=stats, guard=guard, force_redact=force_redact
            ),
        }
      elif isinstance(content, list):
        msg = {
            **msg,
            "content": [
                self._slim_prompt_part(
                    p, stats=stats, guard=guard, force_redact=force_redact
                )
                for p in content
            ],
        }
      new_messages.append(msg)
    out = {**result, "messages": new_messages}
    if stats is not None:
      stats.output_tokens_before += before_tokens
      stats.output_tokens_after += estimate_tokens(out)
    return out


AUDIT_SARIF_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "MS001",
        "name": "ToolPoisoningOrSmuggling",
        "shortDescription": {
            "text": "MCP tool schema contains prompt injection directives or invisible Unicode smuggling."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS002",
        "name": "SchemaEmbeddedSecret",
        "shortDescription": {
            "text": "MCP tool schema embeds a live credential, token, or secret in description/default/examples."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS003",
        "name": "UninlinedJsonSchemaRef",
        "shortDescription": {
            "text": "MCP tool inputSchema uses $defs/$ref pointers rejected by Gemini, Vertex AI, and strict endpoints."
        },
        "defaultConfiguration": {"level": "warning"},
    },
    {
        "id": "MS004",
        "name": "RootSchemaUnion",
        "shortDescription": {
            "text": "MCP tool inputSchema uses root anyOf/oneOf object unions rejected by strict function-calling APIs."
        },
        "defaultConfiguration": {"level": "note"},
    },
    {
        "id": "MS005",
        "name": "ToolRugPullMutation",
        "shortDescription": {
            "text": "MCP tool definition mutated description, inputSchema, or safety annotations relative to pinned baseline."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS006",
        "name": "UnsafeOrPoisonedResource",
        "shortDescription": {
            "text": "MCP resource violates URI prefix policy, contains path traversal, or embeds secrets/prompt injection."
        },
        "defaultConfiguration": {"level": "error"},
    },
)


def _canonical_tool_components(tool: dict[str, Any]) -> dict[str, Any]:
  """Extracts deterministic fields used for SHA-256 tool schema pinning."""
  name = str(tool.get("name", ""))
  desc = str(tool.get("description", ""))
  schema = tool.get("inputSchema")
  norm_schema = schema if isinstance(schema, dict) else {}
  annotations = SchemaSlimmer.extract_annotations(tool)
  return {
      "name": name,
      "description": desc,
      "inputSchema": norm_schema,
      "annotations": annotations,
  }


def compute_tool_fingerprint(tool: dict[str, Any]) -> str:
  """Computes a canonical SHA-256 hex digest over a tool's name, description, schema, and annotations."""
  canonical = json.dumps(
      _canonical_tool_components(tool),
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_privilege_escalation(
    before_ann: dict[str, bool], after_ann: dict[str, bool]
) -> list[str]:
  """Detects MCP ToolAnnotations changes that widen execution authority or side-effect risk."""
  escalations: list[str] = []
  if before_ann.get("readOnlyHint") is True and after_ann.get("readOnlyHint") is not True:
    escalations.append("readOnlyHint:True->False")
  if before_ann.get("destructiveHint") is False and after_ann.get("destructiveHint") is True:
    escalations.append("destructiveHint:False->True")
  elif "destructiveHint" not in before_ann and after_ann.get("destructiveHint") is True:
    escalations.append("destructiveHint:unset->True")
  if before_ann.get("openWorldHint") is False and after_ann.get("openWorldHint") is True:
    escalations.append("openWorldHint:False->True")
  return escalations


@dataclass(frozen=True)
class ToolMutation:
  """Describes a detected mid-session or cross-version MCP tool mutation (rug-pull hazard)."""

  tool: str
  kind: str  # "privilege_escalation", "modified", "added", "removed"
  changed_fields: tuple[str, ...]
  baseline_sha256: str
  current_sha256: str
  level: str  # "error", "warning", "note"
  message: str

  def to_dict(self) -> dict[str, Any]:
    return {
        "rule_id": "MS005",
        "level": self.level,
        "tool": self.tool,
        "kind": self.kind,
        "changed_fields": list(self.changed_fields),
        "baseline_sha256": self.baseline_sha256,
        "current_sha256": self.current_sha256,
        "message": self.message,
    }


def _compare_tool_pair(
    before_tool: dict[str, Any], after_tool: dict[str, Any]
) -> ToolMutation | None:
  before_hash = compute_tool_fingerprint(before_tool)
  after_hash = compute_tool_fingerprint(after_tool)
  if before_hash == after_hash:
    return None

  b_comp = _canonical_tool_components(before_tool)
  a_comp = _canonical_tool_components(after_tool)
  name = a_comp["name"] or b_comp["name"]
  changed: list[str] = []
  for key in ("description", "inputSchema", "annotations"):
    if json.dumps(b_comp[key], sort_keys=True) != json.dumps(a_comp[key], sort_keys=True):
      changed.append(key)

  escalations = _has_privilege_escalation(b_comp["annotations"], a_comp["annotations"])
  if escalations:
    esc_str = ", ".join(escalations)
    return ToolMutation(
        tool=name,
        kind="privilege_escalation",
        changed_fields=tuple(changed or ["annotations"]),
        baseline_sha256=before_hash,
        current_sha256=after_hash,
        level="error",
        message=(
            f"Tool '{name}' escalated MCP safety annotations ({esc_str}) "
            f"across fields [{', '.join(changed)}]."
        ),
    )

  return ToolMutation(
      tool=name,
      kind="modified",
      changed_fields=tuple(changed),
      baseline_sha256=before_hash,
      current_sha256=after_hash,
      level="error",
      message=(
          f"Tool '{name}' mutated pinned definition fields [{', '.join(changed)}] "
          f"(baseline {before_hash[:12]} -> {after_hash[:12]})."
      ),
  )


class ToolPinGuard:
  """Session-scoped MCP rug-pull detector (`--pin-schemas`).

  Pins canonical SHA-256 digests of every tool on the initial `tools/list` response and
  quarantines any tool whose `description`, `inputSchema`, or `annotations` mutate on a
  subsequent `tools/list` turn (or is injected unannounced mid-session).
  """

  def __init__(
      self, enabled: bool = False, allow_new_tools: bool = False
  ) -> None:
    self.enabled = enabled
    self.allow_new_tools = allow_new_tools
    self._baselined: bool = False
    self._baseline_tools: dict[str, dict[str, Any]] = {}
    self._baseline_hashes: dict[str, str] = {}
    self.quarantined: dict[str, ToolMutation] = {}
    self.mutations: list[ToolMutation] = []

  @property
  def is_baselined(self) -> bool:
    return self._baselined

  def baseline_hashes(self) -> dict[str, str]:
    return dict(self._baseline_hashes)

  def get_quarantine(self, tool_name: str) -> ToolMutation | None:
    if not self.enabled:
      return None
    return self.quarantined.get(tool_name)

  def verify_or_pin(
      self, tools: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> tuple[list[dict[str, Any]], list[ToolMutation]]:
    """Pins baseline on first `tools/list`, then filters & quarantines mutated tools."""
    if not self.enabled:
      return tools, []

    if not self._baselined:
      for t in tools:
        if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]:
          name = t["name"]
          self._baseline_tools[name] = copy.deepcopy(t)
          self._baseline_hashes[name] = compute_tool_fingerprint(t)
      self._baselined = True
      return tools, []

    turn_mutations: list[ToolMutation] = []
    verified: list[dict[str, Any]] = []
    for t in tools:
      if not isinstance(t, dict):
        continue
      name = t.get("name")
      if not isinstance(name, str) or not name:
        continue
      if name in self.quarantined:
        continue
      if name not in self._baseline_tools:
        cur_hash = compute_tool_fingerprint(t)
        if self.allow_new_tools:
          self._baseline_tools[name] = copy.deepcopy(t)
          self._baseline_hashes[name] = cur_hash
          verified.append(t)
          continue
        mut = ToolMutation(
            tool=name,
            kind="added",
            changed_fields=("tool_added",),
            baseline_sha256="",
            current_sha256=cur_hash,
            level="error",
            message=(
                f"Tool '{name}' was injected mid-session after initial tools/list "
                f"pinning (sha256 {cur_hash[:12]})."
            ),
        )
        self.quarantined[name] = mut
        self.mutations.append(mut)
        turn_mutations.append(mut)
        if stats is not None:
          stats.rug_pulls_blocked += 1
        continue

      diff_mut = _compare_tool_pair(self._baseline_tools[name], t)
      if diff_mut is not None:
        self.quarantined[name] = diff_mut
        self.mutations.append(diff_mut)
        turn_mutations.append(diff_mut)
        if stats is not None:
          stats.rug_pulls_blocked += 1
        continue

      verified.append(t)

    return verified, turn_mutations


def diff_tools_manifests(
    before_tools: list[dict[str, Any]],
    after_tools: list[dict[str, Any]],
    *,
    allow_new_tools: bool = False,
) -> dict[str, Any]:
  """Compares two MCP `tools/list` manifests for rug-pull mutations, privilege escalations, and poisoning."""
  before_map: dict[str, dict[str, Any]] = {
      str(t["name"]): t
      for t in before_tools
      if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]
  }
  after_map: dict[str, dict[str, Any]] = {
      str(t["name"]): t
      for t in after_tools
      if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]
  }

  mutations: list[dict[str, Any]] = []
  unchanged_count = 0

  for name in sorted(set(before_map) | set(after_map)):
    b_tool = before_map.get(name)
    a_tool = after_map.get(name)
    if b_tool is not None and a_tool is None:
      b_hash = compute_tool_fingerprint(b_tool)
      mutations.append(
          ToolMutation(
              tool=name,
              kind="removed",
              changed_fields=("tool_removed",),
              baseline_sha256=b_hash,
              current_sha256="",
              level="note",
              message=f"Tool '{name}' was removed since baseline ({b_hash[:12]}).",
          ).to_dict()
      )
    elif b_tool is None and a_tool is not None:
      a_hash = compute_tool_fingerprint(a_tool)
      lvl = "note" if allow_new_tools else "warning"
      mutations.append(
          ToolMutation(
              tool=name,
              kind="added",
              changed_fields=("tool_added",),
              baseline_sha256="",
              current_sha256=a_hash,
              level=lvl,
              message=f"Tool '{name}' was added since baseline ({a_hash[:12]}).",
          ).to_dict()
      )
    elif b_tool is not None and a_tool is not None:
      mut = _compare_tool_pair(b_tool, a_tool)
      if mut is not None:
        mutations.append(mut.to_dict())
      else:
        unchanged_count += 1

  after_audit = audit_tools_manifest(list(after_map.values()))
  poison_or_secret = [
      f for f in after_audit["findings"] if f["rule_id"] in ("MS001", "MS002")
  ]
  all_findings = [*mutations, *poison_or_secret]

  blocking_count = sum(
      1
      for m in mutations
      if m["kind"] in ("privilege_escalation", "modified")
      or (m["kind"] == "added" and not allow_new_tools)
  ) + len(poison_or_secret)

  return {
      "tools_before": len(before_map),
      "tools_after": len(after_map),
      "unchanged": unchanged_count,
      "modified": sum(1 for m in mutations if m["kind"] == "modified"),
      "privilege_escalations": sum(
          1 for m in mutations if m["kind"] == "privilege_escalation"
      ),
      "added": sum(1 for m in mutations if m["kind"] == "added"),
      "removed": sum(1 for m in mutations if m["kind"] == "removed"),
      "security_findings": len(poison_or_secret),
      "blocking_mutations": blocking_count,
      "mutations": mutations,
      "findings": all_findings,
  }


def audit_tools_manifest(tools: list[dict[str, Any]]) -> dict[str, Any]:
  """Profiles token footprint, dialect hazards, and tool-poisoning / secret risks across MCP tools."""
  from .guard import SecretGuard

  guard_full = SecretGuard(enabled=True)
  sanitized_tools = guard_full.redact_value(tools)

  slimmer = SchemaSlimmer()
  stats_std = SlimStats()
  slimmed_tools = slimmer.slim_tools_list(sanitized_tools, stats=stats_std)
  stats_std.schema_tokens_before = estimate_tokens(tools)

  broker = LazyToolBroker(schema_slimmer=slimmer)
  stats_lazy = SlimStats()
  broker.ingest_tools(sanitized_tools, stats=stats_lazy)
  stats_lazy.schema_tokens_before = stats_std.schema_tokens_before

  slim_by_name = {t.get("name", ""): t for t in slimmed_tools}
  findings: list[dict[str, Any]] = []
  per_tool: list[dict[str, Any]] = []

  for idx, raw_t in enumerate(tools):
    if not isinstance(raw_t, dict):
      continue
    name = str(raw_t.get("name", f"tool_{idx}"))
    slim_t = slim_by_name.get(name, raw_t)
    raw_tok = estimate_tokens(raw_t)
    slim_tok = estimate_tokens(slim_t)
    raw_dump = json.dumps(raw_t, ensure_ascii=False)
    has_ref = '"$defs"' in raw_dump or '"$ref"' in raw_dump or '"definitions"' in raw_dump
    in_schema = raw_t.get("inputSchema")
    has_root_union = isinstance(in_schema, dict) and any(
        u in in_schema for u in ("anyOf", "oneOf")
    )

    tool_guard = SecretGuard(enabled=True)
    tool_guard.redact_value(raw_t)
    cats = dict(tool_guard.report.by_category)
    poison_count = (
        cats.get("prompt_injection", 0)
        + cats.get("unicode_smuggling", 0)
        + cats.get("markdown_exfil_beacon", 0)
    )
    secret_cats = {
        k: v
        for k, v in cats.items()
        if k not in ("prompt_injection", "unicode_smuggling", "markdown_exfil_beacon")
        and v > 0
    }

    if poison_count > 0:
      findings.append({
          "rule_id": "MS001",
          "level": "error",
          "tool": name,
          "message": (
              f"Tool '{name}' contains prompt injection or invisible Unicode "
              f"smuggling ({poison_count} occurrence(s))."
          ),
      })
    if secret_cats:
      cats_str = ", ".join(sorted(secret_cats.keys()))
      findings.append({
          "rule_id": "MS002",
          "level": "error",
          "tool": name,
          "message": (
              f"Tool '{name}' embeds credential material in schema metadata ({cats_str})."
          ),
      })
    if has_ref:
      findings.append({
          "rule_id": "MS003",
          "level": "warning",
          "tool": name,
          "message": (
              f"Tool '{name}' uses $defs/$ref pointers that require inlining for "
              "Gemini/Vertex compatibility."
          ),
      })
    if has_root_union:
      findings.append({
          "rule_id": "MS004",
          "level": "note",
          "tool": name,
          "message": (
              f"Tool '{name}' declares root anyOf/oneOf object unions."
          ),
      })

    per_tool.append({
        "name": name,
        "raw_tokens": raw_tok,
        "slim_tokens": slim_tok,
        "saved_pct": round(((raw_tok - slim_tok) / max(1, raw_tok)) * 100.0, 1),
        "has_refs": has_ref,
        "has_root_union": has_root_union,
        "read_only": slimmer.is_read_only_tool(name, tool=raw_t),
        "security_categories": cats,
    })

  std_pct = round(
      (
          (stats_std.schema_tokens_before - stats_std.schema_tokens_after)
          / max(1, stats_std.schema_tokens_before)
      )
      * 100.0,
      1,
  )
  lazy_pct = round(
      (
          (stats_lazy.schema_tokens_before - stats_lazy.schema_tokens_after)
          / max(1, stats_lazy.schema_tokens_before)
      )
      * 100.0,
      1,
  )
  security_findings = sum(
      1 for f in findings if f["rule_id"] in ("MS001", "MS002")
  )
  return {
      "tool_count": len(per_tool),
      "tokens_raw": stats_std.schema_tokens_before,
      "tokens_slim": stats_std.schema_tokens_after,
      "standard_reduction_pct": std_pct,
      "tokens_lazy": stats_lazy.schema_tokens_after,
      "lazy_reduction_pct": lazy_pct,
      "refs_inlined": stats_std.refs_inlined,
      "refs_deduped": stats_std.refs_deduped,
      "cycles_broken": stats_std.cycles_broken,
      "security_findings": security_findings,
      "findings": findings,
      "tools": per_tool,
  }


def _unquote_iterative(value: str, max_passes: int = 3) -> str:
  cur = value
  for _ in range(max_passes):
    nxt = urllib.parse.unquote(cur)
    if nxt == cur:
      break
    cur = nxt
  return cur


def canonicalize_resource_uri(uri: str) -> tuple[str, str | None]:
  """Canonicalizes an MCP resource URI and detects null-byte or `..` path traversal.

  Returns `(canonical_uri, violation_reason_or_None)`.
  """
  if not isinstance(uri, str) or not uri.strip():
    return ("", "Empty or non-string resource URI")
  stripped = uri.strip()
  decoded_full = _unquote_iterative(stripped)
  if "\x00" in stripped or "\x00" in decoded_full or any(ord(ch) < 32 for ch in stripped):
    return (stripped, "Null-byte or control-character smuggling in resource URI")

  parts = urllib.parse.urlsplit(stripped)
  scheme = parts.scheme.lower()
  netloc = parts.netloc.lower()
  dec_path = _unquote_iterative(parts.path).replace("\\", "/")

  segments = dec_path.split("/")
  has_dotdot = any(seg == ".." for seg in segments)
  norm_segs: list[str] = []
  escaped_root = False
  for seg in segments:
    if seg in ("", "."):
      continue
    if seg == "..":
      if not norm_segs:
        escaped_root = True
      else:
        norm_segs.pop()
      continue
    norm_segs.append(seg)

  if dec_path.startswith("/"):
    canon_path = "/" + "/".join(norm_segs)
    if dec_path.endswith("/") and canon_path != "/" and not has_dotdot:
      canon_path += "/"
  else:
    canon_path = "/".join(norm_segs)
    if dec_path.endswith("/") and canon_path and not has_dotdot:
      canon_path += "/"

  canon_uri = urllib.parse.urlunsplit((scheme, netloc, canon_path, parts.query, ""))
  if has_dotdot or escaped_root:
    return (
        canon_uri,
        f"Path traversal segment ('..') detected in resource URI '{stripped}'",
    )
  return (canon_uri, None)


def _normalize_uri_prefix(prefix: str) -> str:
  stripped = prefix.strip()
  if not stripped:
    return ""
  if any(w in stripped for w in ("*", "?", "[")):
    if "://" in stripped:
      sch, rest = stripped.split("://", 1)
      if "/" in rest:
        host, tail = rest.split("/", 1)
        return f"{sch.lower()}://{host.lower()}/{tail}"
      return f"{sch.lower()}://{rest.lower()}"
    if ":" in stripped:
      sch, rest = stripped.split(":", 1)
      return f"{sch.lower()}:{rest}"
    return stripped
  if stripped.endswith("://"):
    return stripped[:-3].lower() + "://"
  if stripped.endswith(":") and "/" not in stripped:
    return stripped.lower()
  canon, _ = canonicalize_resource_uri(stripped)
  return canon or stripped


def _uri_matches_prefix(canon_uri: str, norm_prefix: str) -> bool:
  """Matches a canonical URI against a prefix or glob without sibling-directory confusion."""
  if not norm_prefix:
    return False
  if any(w in norm_prefix for w in ("*", "?", "[")):
    return fnmatch.fnmatchcase(canon_uri, norm_prefix)
  if norm_prefix.endswith((":", "://")):
    return canon_uri.startswith(norm_prefix)
  if norm_prefix.endswith("/"):
    if canon_uri.startswith(norm_prefix):
      return True
    trimmed = norm_prefix[:-1]
    if trimmed and not trimmed.endswith(":/"):
      return canon_uri == trimmed
    return False
  return canon_uri == norm_prefix or canon_uri.startswith(norm_prefix + "/")


class ResourceUriGuard:
  """Enforces URI prefix allowlists/denylists and traversal guards on MCP resources (`--allow-resource-prefixes`)."""

  def __init__(
      self,
      allow_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      deny_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      block_traversal: bool = True,
  ) -> None:
    self.allow_prefixes: tuple[str, ...] | None = (
        tuple(
            p
            for p in (_normalize_uri_prefix(x) for x in sorted(set(allow_prefixes)))
            if p
        )
        if allow_prefixes is not None
        else None
    )
    self.deny_prefixes: tuple[str, ...] = (
        tuple(
            p
            for p in (_normalize_uri_prefix(x) for x in sorted(set(deny_prefixes)))
            if p
        )
        if deny_prefixes
        else ()
    )
    self.block_traversal = block_traversal

  @property
  def has_prefix_policy(self) -> bool:
    return self.allow_prefixes is not None or bool(self.deny_prefixes)

  def validate_uri(self, uri: Any) -> str | None:
    """Returns `None` when `uri` is permitted, or an actionable error message when blocked."""
    canon_uri, trav_err = canonicalize_resource_uri(uri if isinstance(uri, str) else "")
    if trav_err is not None:
      if self.block_traversal or not canon_uri:
        return f"[mcp-slim] Resource access blocked: {trav_err}."
    if self.deny_prefixes and any(
        _uri_matches_prefix(canon_uri, dp) for dp in self.deny_prefixes
    ):
      return (
          f"[mcp-slim] Resource URI '{uri}' is blocked by "
          "--deny-resource-prefixes policy."
      )
    if self.allow_prefixes is not None:
      if not self.allow_prefixes or not any(
          _uri_matches_prefix(canon_uri, ap) for ap in self.allow_prefixes
      ):
        return (
            f"[mcp-slim] Resource URI '{uri}' is outside allowed prefixes "
            "(--allow-resource-prefixes)."
        )
    return None

  def is_uri_allowed(self, uri: Any) -> bool:
    return self.validate_uri(uri) is None

  def is_template_allowed(self, uri_template: Any) -> bool:
    """Checks an RFC 6570 `uriTemplate` static prefix against configured allow/deny rules."""
    if not isinstance(uri_template, str) or not uri_template.strip():
      return False
    static_head = uri_template.strip().split("{", 1)[0]
    if not static_head:
      return self.allow_prefixes is None
    canon_head, trav_err = canonicalize_resource_uri(static_head)
    if trav_err is not None and self.block_traversal:
      return False
    if self.deny_prefixes and any(
        _uri_matches_prefix(canon_head, dp) for dp in self.deny_prefixes
    ):
      return False
    if self.allow_prefixes is not None:
      return any(_uri_matches_prefix(canon_head, ap) for ap in self.allow_prefixes)
    return True

  def filter_resources_list(
      self, resources: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for res in resources:
      if not isinstance(res, dict):
        continue
      uri = res.get("uri")
      if self.validate_uri(uri) is not None:
        if stats is not None:
          stats.resources_blocked += 1
        continue
      out_res = dict(res)
      if isinstance(out_res.get("description"), str) and out_res["description"].strip():
        out_res["description"] = compact_description(out_res["description"], max_chars=120)
      kept.append(out_res)
    return kept

  def filter_resource_templates(
      self, templates: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for tmpl in templates:
      if not isinstance(tmpl, dict):
        continue
      uri_tmpl = tmpl.get("uriTemplate")
      if not self.is_template_allowed(uri_tmpl):
        if stats is not None:
          stats.resources_blocked += 1
        continue
      out_tmpl = dict(tmpl)
      if isinstance(out_tmpl.get("description"), str) and out_tmpl["description"].strip():
        out_tmpl["description"] = compact_description(out_tmpl["description"], max_chars=120)
      kept.append(out_tmpl)
    return kept


def audit_resources_manifest(
    resources: list[dict[str, Any]],
    *,
    allow_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
    deny_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
  """Audits MCP `resources/list` or `resources/read` entries for URI traversal, policy violations, secrets, and indirect prompt injection."""
  from .guard import SecretGuard

  uri_guard = ResourceUriGuard(
      allow_prefixes=allow_prefixes,
      deny_prefixes=deny_prefixes,
      block_traversal=True,
  )
  findings: list[dict[str, Any]] = []
  per_resource: list[dict[str, Any]] = []

  for idx, item in enumerate(resources):
    if not isinstance(item, dict):
      continue
    uri = str(item.get("uri") or item.get("uriTemplate") or f"resource_{idx}")
    uri_err = (
        uri_guard.validate_uri(item["uri"])
        if isinstance(item.get("uri"), str)
        else (
            None
            if isinstance(item.get("uriTemplate"), str)
            and uri_guard.is_template_allowed(item["uriTemplate"])
            else f"[mcp-slim] Resource template '{uri}' violates URI policy."
        )
    )
    sec_guard = SecretGuard(enabled=True)
    sec_guard.redact_value(item)
    cats = dict(sec_guard.report.by_category)

    if uri_err is not None:
      findings.append({
          "rule_id": "MS006",
          "level": "error",
          "tool": uri,
          "kind": "uri_policy_violation",
          "message": uri_err,
      })
    if cats:
      cats_str = ", ".join(sorted(f"{k}:{v}" for k, v in cats.items()))
      findings.append({
          "rule_id": "MS006",
          "level": "error",
          "tool": uri,
          "kind": "resource_poison_or_secret",
          "message": (
              f"Resource '{uri}' contains secret material or indirect prompt "
              f"injection ({cats_str})."
          ),
      })

    per_resource.append({
        "uri": uri,
        "allowed": uri_err is None,
        "uri_error": uri_err,
        "security_categories": cats,
    })

  return {
      "resource_count": len(per_resource),
      "blocked_uris": sum(1 for r in per_resource if not r["allowed"]),
      "poisoned_or_secret_resources": sum(
          1 for r in per_resource if r["security_categories"]
      ),
      "security_findings": len(findings),
      "findings": findings,
      "resources": per_resource,
  }


def build_audit_sarif(
    report: dict[str, Any],
    source_uri: str = "tools.json",
    version: str = "1.1.0",
) -> dict[str, Any]:
  """Formats an `audit_tools_manifest`, `diff_tools_manifests`, or `audit_resources_manifest` report as OASIS SARIF v2.1.0."""
  results: list[dict[str, Any]] = []
  for f in report.get("findings", []):
    loc_name = str(f.get("tool") or f.get("uri") or "mcp")
    loc_kind = "resource" if f.get("rule_id") == "MS006" else "function"
    results.append({
        "ruleId": f["rule_id"],
        "level": f["level"],
        "message": {"text": f["message"]},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": source_uri},
                },
                "logicalLocations": [
                    {
                        "name": loc_name,
                        "kind": loc_kind,
                    }
                ],
            }
        ],
    })
  return {
      "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
      "version": "2.1.0",
      "runs": [
          {
              "tool": {
                  "driver": {
                      "name": "mcp-slim",
                      "version": version,
                      "informationUri": "https://github.com/SalyyS1/mcp-slim",
                      "rules": list(AUDIT_SARIF_RULES),
                  }
              },
              "results": results,
          }
      ],
  }
