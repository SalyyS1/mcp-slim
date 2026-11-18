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
    if len(part) > 3 and part.endswith("s"):
      allowed_words.add(part[:-1])
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
  budget_throttles: int = 0
  stream_chunks_sanitized: int = 0
  elicitations_guarded: int = 0
  roots_filtered: int = 0
  ref_hints_preserved: int = 0

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
  synthesized_hint_nodes: set[int] = field(default_factory=set)


_CLEAN_DEF_IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _extract_ref_def_name(ref_str: str) -> str | None:
  """Returns the decoded definition name for top-level `#/$defs/<Name>` or `#/definitions/<Name>` pointers."""
  if not isinstance(ref_str, str) or not ref_str.startswith("#/"):
    return None
  segments = [_decode_pointer_segment(s) for s in ref_str[2:].split("/")]
  if len(segments) != 2 or segments[0] not in ("$defs", "definitions"):
    return None
  def_name = segments[1]
  if not _CLEAN_DEF_IDENT_RE.match(def_name):
    return None
  return def_name


def _humanize_def_identifier(def_name: str, *, require_compound: bool = True) -> str:
  """Converts a PascalCase/snake_case definition identifier into a concise semantic hint."""
  spaced = re.sub(
      r"([a-z0-9])([A-Z])",
      r"\1 \2",
      re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", def_name),
  )
  words = [w for w in re.split(r"[_\-\s]+", spaced.strip()) if w]
  if not words:
    return ""
  if require_compound and (len(words) < 2 or any(len(w) < 2 for w in words)):
    return ""
  if not require_compound and len(words[0]) < 2:
    return ""
  formatted: list[str] = []
  for idx, w in enumerate(words):
    is_acronym = len(w) >= 2 and sum(1 for ch in w if ch.isupper()) >= 2
    if idx == 0:
      formatted.append(w if is_acronym else (w[0].upper() + w[1:].lower()))
    else:
      formatted.append(w if is_acronym else w.lower())
  return " ".join(formatted)


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
      preserve_ref_hints: bool = True,
  ) -> None:
    self.max_desc_chars = max_desc_chars
    self.max_param_desc_chars = max_param_desc_chars
    self.allow_tools = allow_tools
    self.deny_tools = deny_tools or set()
    self.inline_refs = inline_refs
    self.canonical_sort = canonical_sort
    self.preserve_root_unions = preserve_root_unions
    self.read_only_only = read_only_only
    self.preserve_ref_hints = preserve_ref_hints
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
        if (
            self.preserve_ref_hints
            and "description" not in merged
            and (merged.get("type") == "object" or "properties" in merged)
        ):
          def_name = _extract_ref_def_name(node["$ref"])
          if def_name:
            hint = _humanize_def_identifier(def_name, require_compound=True)
            if hint:
              compact_hint = compact_description(
                  hint, max_chars=self.max_param_desc_chars
              )
              if compact_hint:
                merged["description"] = compact_hint
                ctx.synthesized_hint_nodes.add(id(merged))
                if stats is not None:
                  stats.ref_hints_preserved += 1
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
              if id(slimmed_prop) in ctx.synthesized_hint_nodes and stats is not None:
                stats.ref_hints_preserved = max(0, stats.ref_hints_preserved - 1)
            items_node = slimmed_prop.get("items")
            if (
                slimmed_prop.get("type") == "array"
                and isinstance(items_node, dict)
                and id(items_node) in ctx.synthesized_hint_nodes
            ):
              item_desc = items_node.get("description", "")
              if item_desc and _is_redundant_param_description(prop_name, item_desc):
                items_node.pop("description", None)
                if stats is not None:
                  stats.ref_hints_preserved = max(0, stats.ref_hints_preserved - 1)
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
            had_own_desc = "description" in out
            for mk, mv in merged.items():
              out.setdefault(mk, mv)
            if not had_own_desc and id(merged) in ctx.synthesized_hint_nodes:
              ctx.synthesized_hint_nodes.add(id(out))
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
        if self.preserve_ref_hints and k in ("anyOf", "oneOf") and len(non_null) >= 2:
          branch_def_names = {
              name
              for b in non_null
              if isinstance(b, dict)
              for name in [_extract_ref_def_name(str(b.get("$ref", "")))]
              if name is not None
          }
          if len(branch_def_names) >= 2:
            for raw_b, slim_b in zip(non_null, raw_slimmed_branches):
              if (
                  isinstance(raw_b, dict)
                  and isinstance(slim_b, dict)
                  and "description" not in slim_b
                  and (slim_b.get("type") == "object" or "properties" in slim_b)
              ):
                b_def = _extract_ref_def_name(str(raw_b.get("$ref", "")))
                if b_def:
                  b_hint = compact_description(
                      _humanize_def_identifier(b_def, require_compound=False),
                      max_chars=self.max_param_desc_chars,
                  )
                  if b_hint:
                    slim_b["description"] = b_hint
                    ctx.synthesized_hint_nodes.add(id(slim_b))
                    if stats is not None:
                      stats.ref_hints_preserved += 1
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


@dataclass(frozen=True)
class WindowBudgetDecision:
  """Per-turn compaction limits computed from the sliding exponential token window."""

  max_string_chars: int
  max_list_items: int
  window_tokens_before: float
  budget_tokens: int
  pressure: float
  scale_factor: float
  throttled: bool


class TokenWindowGovernor:
  """Exponentially decaying sliding token-window budget governor (`--token-budget`).

  Maintains `W_t = decay * W_{t-1} + emitted_{t-1}` across sequential MCP outputs and
  smoothly tightens `OutputSlimmer` character and list budgets as `W_t / budget_tokens`
  rises, preventing context-window blowouts on long-horizon agent loops while naturally
  recovering capacity when subsequent turns emit light outputs.
  """

  def __init__(
      self,
      budget_tokens: int = 0,
      *,
      decay: float = 0.85,
      min_scale: float = 0.15,
      min_string_chars: int = 240,
      min_list_items: int = 3,
      initial_window_tokens: float = 0.0,
  ) -> None:
    self.budget_tokens = max(0, int(budget_tokens))
    self.enabled = self.budget_tokens > 0
    self.decay = min(1.0, max(0.0, float(decay)))
    self.min_scale = min(1.0, max(0.05, float(min_scale)))
    self.min_string_chars = max(80, int(min_string_chars))
    self.min_list_items = max(1, int(min_list_items))
    self.window_tokens: float = max(0.0, float(initial_window_tokens))
    self.turns: int = 0
    self.throttled_turns: int = 0
    self.last_decision: WindowBudgetDecision | None = None

  def reset(self) -> None:
    self.window_tokens = 0.0
    self.turns = 0
    self.throttled_turns = 0
    self.last_decision = None

  def plan_turn(
      self, base_max_chars: int, base_max_items: int
  ) -> WindowBudgetDecision:
    if not self.enabled or self.budget_tokens <= 0:
      decision = WindowBudgetDecision(
          max_string_chars=base_max_chars,
          max_list_items=base_max_items,
          window_tokens_before=self.window_tokens,
          budget_tokens=0,
          pressure=0.0,
          scale_factor=1.0,
          throttled=False,
      )
      self.last_decision = decision
      return decision

    pressure = self.window_tokens / float(self.budget_tokens)
    if pressure <= 0.4:
      scale = 1.0
    elif pressure >= 1.0:
      scale = self.min_scale
    else:
      norm_x = (pressure - 0.4) / 0.6
      scale = math.exp(math.log(self.min_scale) * norm_x)

    eff_min_chars = min(base_max_chars, self.min_string_chars)
    eff_min_items = min(base_max_items, self.min_list_items)
    if pressure <= 0.4:
      headroom_chars = max(eff_min_chars, self.budget_tokens * 4)
    else:
      remaining_tokens = max(0, self.budget_tokens - int(round(self.window_tokens)))
      headroom_chars = max(eff_min_chars, remaining_tokens * 4)

    scaled_chars = int(round(base_max_chars * scale))
    target_chars = max(eff_min_chars, min(base_max_chars, scaled_chars, headroom_chars))

    if pressure >= 1.0:
      target_items = eff_min_items
    else:
      target_items = max(
          eff_min_items, min(base_max_items, int(round(base_max_items * scale)))
      )

    throttled = (target_chars < base_max_chars) or (target_items < base_max_items)
    decision = WindowBudgetDecision(
        max_string_chars=target_chars,
        max_list_items=target_items,
        window_tokens_before=round(self.window_tokens, 2),
        budget_tokens=self.budget_tokens,
        pressure=round(pressure, 4),
        scale_factor=round(scale, 4),
        throttled=throttled,
    )
    self.last_decision = decision
    return decision

  def record_turn(
      self,
      emitted_tokens: int,
      *,
      throttled: bool = False,
      stats: SlimStats | None = None,
  ) -> float:
    if not self.enabled:
      return 0.0
    self.turns += 1
    if throttled:
      self.throttled_turns += 1
      if stats is not None:
        stats.budget_throttles += 1
    self.window_tokens = (self.window_tokens * self.decay) + max(0, int(emitted_tokens))
    return self.window_tokens

  def snapshot(self) -> dict[str, Any]:
    pressure = (
        round(self.window_tokens / float(self.budget_tokens), 4)
        if self.enabled and self.budget_tokens > 0
        else 0.0
    )
    return {
        "enabled": self.enabled,
        "budget_tokens": self.budget_tokens,
        "decay": self.decay,
        "window_tokens": round(self.window_tokens, 2),
        "pressure": pressure,
        "turns": self.turns,
        "throttled_turns": self.throttled_turns,
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
      window_governor: TokenWindowGovernor | None = None,
  ) -> None:
    self.max_list_items = max_list_items
    self.max_string_chars = max_string_chars
    self.strip_nulls = strip_nulls
    self.strip_hypermedia = strip_hypermedia
    self.flatten_resources = flatten_resources
    self.tabular_arrays = tabular_arrays
    self.min_table_rows = max(2, min_table_rows)
    self.window_governor = window_governor

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
      *,
      governor: TokenWindowGovernor | None = None,
  ) -> dict[str, Any]:
    eff_gov = governor if governor is not None else self.window_governor
    orig_chars = self.max_string_chars
    orig_items = self.max_list_items
    decision = (
        eff_gov.plan_turn(orig_chars, orig_items)
        if eff_gov is not None and eff_gov.enabled
        else None
    )
    if decision is not None:
      self.max_string_chars = decision.max_string_chars
      self.max_list_items = decision.max_list_items
    try:
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
      after_tokens = (
          estimate_tokens(out)
          if (stats is not None or decision is not None)
          else 0
      )
      if stats is not None:
        stats.output_tokens_before += before_tokens
        stats.output_tokens_after += after_tokens
      if eff_gov is not None and decision is not None:
        eff_gov.record_turn(after_tokens, throttled=decision.throttled, stats=stats)
      return out
    finally:
      self.max_string_chars = orig_chars
      self.max_list_items = orig_items

  def slim_resource_result(
      self,
      result: dict[str, Any],
      stats: SlimStats | None = None,
      guard: Any | None = None,
      *,
      governor: TokenWindowGovernor | None = None,
  ) -> dict[str, Any]:
    """Compacts MCP `resources/read` payloads (`result.contents[*].text` and UTF-8 `blob`)."""
    contents = result.get("contents")
    if not isinstance(contents, list):
      return result
    eff_gov = governor if governor is not None else self.window_governor
    orig_chars = self.max_string_chars
    orig_items = self.max_list_items
    decision = (
        eff_gov.plan_turn(orig_chars, orig_items)
        if eff_gov is not None and eff_gov.enabled
        else None
    )
    if decision is not None:
      self.max_string_chars = decision.max_string_chars
      self.max_list_items = decision.max_list_items
    try:
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
      after_tokens = (
          estimate_tokens(out)
          if (stats is not None or decision is not None)
          else 0
      )
      if stats is not None:
        stats.output_tokens_before += before_tokens
        stats.output_tokens_after += after_tokens
      if eff_gov is not None and decision is not None:
        eff_gov.record_turn(after_tokens, throttled=decision.throttled, stats=stats)
      return out
    finally:
      self.max_string_chars = orig_chars
      self.max_list_items = orig_items

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
      governor: TokenWindowGovernor | None = None,
  ) -> dict[str, Any]:
    """Compacts MCP `prompts/get` payloads (`result.messages[*].content`)."""
    messages = result.get("messages")
    if not isinstance(messages, list):
      return result
    eff_gov = governor if governor is not None else self.window_governor
    orig_chars = self.max_string_chars
    orig_items = self.max_list_items
    decision = (
        eff_gov.plan_turn(orig_chars, orig_items)
        if eff_gov is not None and eff_gov.enabled
        else None
    )
    if decision is not None:
      self.max_string_chars = decision.max_string_chars
      self.max_list_items = decision.max_list_items
    try:
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
      after_tokens = (
          estimate_tokens(out)
          if (stats is not None or decision is not None)
          else 0
      )
      if stats is not None:
        stats.output_tokens_before += before_tokens
        stats.output_tokens_after += after_tokens
      if eff_gov is not None and decision is not None:
        eff_gov.record_turn(after_tokens, throttled=decision.throttled, stats=stats)
      return out
    finally:
      self.max_string_chars = orig_chars
      self.max_list_items = orig_items


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
    {
        "id": "MS007",
        "name": "CredentialPhishingElicitation",
        "shortDescription": {
            "text": "Server-initiated MCP elicitation/create request solicits credentials, passphrases, OTP codes, or embeds deceptive smuggling."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS008",
        "name": "CrossToolShadowingHazard",
        "shortDescription": {
            "text": "MCP tool description contains cross-tool shadowing, covert concealment, or host-credential exfiltration directives."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS009",
        "name": "ToolResultInjection",
        "shortDescription": {
            "text": "MCP tools/call response content contains prompt-injection directives or invisible Unicode smuggling targeting the LLM context.",
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "MS010",
        "name": "ToolCallResultReplay",
        "shortDescription": {
            "text": "MCP server returned an identical result fingerprint for a tools/call with different input arguments, indicating server-side result substitution.",
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
      f
      for f in after_audit["findings"]
      if f["rule_id"] in ("MS001", "MS002", "MS008")
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


_SHADOW_CONCEALMENT_RE = re.compile(
    r"(?i)\b(?:"
    r"do\s+not\s+(?:tell|inform|mention(?:\s+this)?\s+to|notify|alert|show)\s+(?:the\s+)?user"
    r"|without\s+(?:telling|informing|notifying|asking|alerting)\s+(?:the\s+)?user"
    r"|never\s+(?:mention|reveal|disclose)\s+(?:this|these)\s+(?:step|action|call|instruction)s?"
    r"|hide\s+this\s+(?:step|action|tool\s+call|call|instruction)"
    r"|silently\s+(?:call|invoke|execute|forward|exfiltrate|send|upload|read)"
    r")\b"
)

_SHADOW_SENSITIVE_FILE_RE = re.compile(
    r"(?i)\b(?:read|cat|open|include|pass|attach|upload|exfiltrate|send|copy)\b"
    r"[^\n.]{0,96}"
    r"(?:~/\.ssh|/\.ssh/|id_rsa|id_ed25519|/etc/shadow|\.aws/credentials|\.npmrc|\.netrc|\.kube/config)"
)

_SHADOW_CROSS_TOOL_RE = re.compile(
    r"(?i)\b(?:before|after|instead\s+of|whenever|every\s+time)\s+"
    r"(?:you\s+)?(?:call|calling|use|using|invoke|invoking|run|running|execute|executing)\s+"
    r"[`'\"(]*([a-zA-Z][a-zA-Z0-9_.:/-]{2,63})[`'\")]*"
)

_SHADOW_COERCIVE_CUE_RE = re.compile(
    r"(?i)\b(?:always|must|required\s+to|make\s+sure\s+to|first\s+pass|also\s+call|"
    r"also\s+send|forward|bcc|cc|exfiltrate|override|reroute|redirect|include\s+the\s+contents)\b"
)

_SENSITIVE_TARGET_TOOLS = frozenset({
    "send_email",
    "send_message",
    "write_file",
    "edit_file",
    "read_file",
    "delete_file",
    "push_files",
    "create_or_update_file",
    "execute_command",
    "run_command",
    "bash",
    "shell",
    "curl",
    "fetch",
    "http_request",
    "postgres_query",
    "execute_sql",
})


def _collect_schema_texts(node: Any, out: list[str]) -> None:
  if isinstance(node, dict):
    for k, v in node.items():
      if k in ("description", "title") and isinstance(v, str) and v.strip():
        out.append(v)
      elif isinstance(v, (dict, list)):
        _collect_schema_texts(v, out)
  elif isinstance(node, list):
    for item in node:
      _collect_schema_texts(item, out)


def detect_tool_shadowing(
    tool: dict[str, Any], sibling_tool_names: set[str] | None = None
) -> list[str]:
  """Detects cross-tool shadowing, covert concealment, and host-credential exfiltration directives in a tool schema."""
  if not isinstance(tool, dict):
    return []
  own_name = str(tool.get("name", "")).strip().lower()
  siblings = {
      s.strip().lower()
      for s in (sibling_tool_names or set())
      if isinstance(s, str) and s.strip() and s.strip().lower() != own_name
  }
  texts: list[str] = []
  if isinstance(tool.get("description"), str) and tool["description"].strip():
    texts.append(tool["description"])
  if isinstance(tool.get("inputSchema"), dict):
    _collect_schema_texts(tool["inputSchema"], texts)
  if not texts:
    return []

  joined = "\n".join(texts)
  reasons: list[str] = []

  conceal_match = _SHADOW_CONCEALMENT_RE.search(joined)
  if conceal_match is not None:
    reasons.append(f"covert user-concealment directive ('{conceal_match.group(0).strip()}')")

  file_match = _SHADOW_SENSITIVE_FILE_RE.search(joined)
  if file_match is not None:
    reasons.append("host credential file exfiltration instruction")

  for m in _SHADOW_CROSS_TOOL_RE.finditer(joined):
    target_raw = m.group(1).strip("`'\".,;:()")
    target_low = target_raw.lower()
    if not target_low or target_low == own_name or target_low in ("this", "endpoint", "tool", "api", "function"):
      continue
    window_start = max(0, m.start() - 40)
    window_end = min(len(joined), m.end() + 140)
    context_slice = joined[window_start:window_end]
    is_known_target = target_low in siblings or target_low in _SENSITIVE_TARGET_TOOLS
    if is_known_target and _SHADOW_COERCIVE_CUE_RE.search(context_slice):
      reasons.append(f"coercive cross-tool shadowing targeting '{target_raw}'")
      break

  return reasons


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
  sibling_names = {
      str(t.get("name"))
      for t in tools
      if isinstance(t, dict) and isinstance(t.get("name"), str)
  }
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
    shadow_reasons = detect_tool_shadowing(raw_t, sibling_tool_names=sibling_names)

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
    if shadow_reasons:
      findings.append({
          "rule_id": "MS008",
          "level": "error",
          "tool": name,
          "message": (
              f"Tool '{name}' exhibits cross-tool shadowing hazard: "
              f"{'; '.join(shadow_reasons)}."
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
        "shadowing_hazards": shadow_reasons,
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
      1 for f in findings if f["rule_id"] in ("MS001", "MS002", "MS008")
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


_ELICITATION_CREDENTIAL_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:"
    r"password|passphrase|passwd"
    r"|api[_\s-]?key|secret[_\s-]?key|client[_\s-]?secret"
    r"|access[_\s-]?token|refresh[_\s-]?token|bearer[_\s-]?token|auth[_\s-]?token"
    r"|session[_\s-]?token|github[_\s-]?token|hf[_\s-]?token|slack[_\s-]?token"
    r"|openai[_\s-]?key|anthropic[_\s-]?key|aws[_\s-]?secret"
    r"|private[_\s-]?key|ssh[_\s-]?key"
    r"|seed[_\s-]?phrase|recovery[_\s-]?phrase|mnemonic"
    r"|totp|otp(?:\s+code|[_-]code)?|2fa(?:\s+code|[_-]code)?|two[_\s-]?factor"
    r"|card[_\s-]?number|credit[_\s-]?card|cvv|cvc|ssn"
    r")(?:$|[^a-z0-9])"
)

_ELICITATION_PHISHING_URL_RE = re.compile(
    r"(?i)https?://[^\s)\"'>]+(?:/|\b)(?:login|signin|sign-in|oauth|authorize|verify|reauth|credentials|passwd|token)\b[^\s)\"'>]*"
)


def _scan_elicitation_schema_fields(
    schema: Any, prefix: str, credential_fields: list[str]
) -> None:
  if not isinstance(schema, dict):
    return
  props = schema.get("properties")
  if isinstance(props, dict):
    for prop_name, prop_def in props.items():
      if not isinstance(prop_name, str):
        continue
      field_path = f"{prefix}.{prop_name}" if prefix else prop_name
      matched = bool(_ELICITATION_CREDENTIAL_RE.search(prop_name))
      if not matched and isinstance(prop_def, dict):
        fmt = str(prop_def.get("format", "")).strip().lower()
        if fmt in ("password", "secret", "credential"):
          matched = True
        else:
          for meta_key in ("title", "description"):
            meta_val = prop_def.get(meta_key)
            if isinstance(meta_val, str) and _ELICITATION_CREDENTIAL_RE.search(meta_val):
              matched = True
              break
      if matched and field_path not in credential_fields:
        credential_fields.append(field_path)
      if isinstance(prop_def, dict):
        _scan_elicitation_schema_fields(prop_def, field_path, credential_fields)


def audit_elicitation_request(params: dict[str, Any]) -> dict[str, Any]:
  """Audits a server-initiated MCP `elicitation/create` request for credential phishing and smuggling (`MS007`)."""
  from .guard import SecretGuard

  if not isinstance(params, dict):
    return {
        "allowed": False,
        "risk_level": "error",
        "credential_fields": [],
        "phishing_urls": [],
        "security_categories": {},
        "security_findings": 1,
        "findings": [{
            "rule_id": "MS007",
            "level": "error",
            "tool": "elicitation/create",
            "kind": "malformed_elicitation",
            "message": "Server-initiated elicitation/create params must be a JSON object.",
        }],
    }

  message = str(params.get("message", "")) if isinstance(params.get("message"), str) else ""
  req_schema = params.get("requestedSchema")
  credential_fields: list[str] = []
  if isinstance(req_schema, dict):
    _scan_elicitation_schema_fields(req_schema, "", credential_fields)

  if message and _ELICITATION_CREDENTIAL_RE.search(message) and "__message__" not in credential_fields:
    credential_fields.append("__message__")

  phishing_urls = (
      [m.group(0) for m in _ELICITATION_PHISHING_URL_RE.finditer(message)]
      if message
      else []
  )

  sec_guard = SecretGuard(enabled=True)
  sec_guard.redact_value(params, force_redact=True)
  cats = dict(sec_guard.report.by_category)

  findings: list[dict[str, Any]] = []
  reasons: list[str] = []
  if credential_fields:
    shown_fields = [f for f in credential_fields if f != "__message__"] or ["message"]
    reasons.append(f"credential solicitation fields ({', '.join(shown_fields)})")
  if phishing_urls:
    reasons.append(f"external auth/phishing URL ({phishing_urls[0]})")
  if cats:
    cats_str = ", ".join(sorted(f"{k}:{v}" for k, v in cats.items()))
    reasons.append(f"smuggling/injection/secret markers ({cats_str})")

  if reasons:
    findings.append({
        "rule_id": "MS007",
        "level": "error",
        "tool": "elicitation/create",
        "kind": "credential_phishing_or_smuggling",
        "message": (
            "Server-initiated elicitation/create blocked for security hazard: "
            f"{'; '.join(reasons)}."
        ),
    })

  slimmer = SchemaSlimmer()
  compacted_schema = (
      slimmer.slim_schema_node(req_schema, is_root=True)
      if isinstance(req_schema, dict)
      else {}
  )

  return {
      "allowed": len(findings) == 0,
      "risk_level": "error" if findings else "clean",
      "credential_fields": credential_fields,
      "phishing_urls": phishing_urls,
      "security_categories": cats,
      "security_findings": len(findings),
      "findings": findings,
      "compacted_schema": compacted_schema,
  }


class ElicitationGuard:
  """Firewall for server-initiated MCP `elicitation/create` user-prompt dialogs."""

  def __init__(
      self,
      *,
      block_all: bool = False,
      block_credential_phishing: bool = True,
  ) -> None:
    self.block_all = block_all
    self.block_credential_phishing = block_credential_phishing

  def evaluate(self, params: Any) -> dict[str, Any]:
    if self.block_all:
      return {
          "allowed": False,
          "mode": "policy_error",
          "reason": (
              "[mcp-slim] Server-initiated elicitation/create is blocked by "
              "proxy policy (--block-elicitation)."
          ),
          "audit": None,
      }
    audit = audit_elicitation_request(params if isinstance(params, dict) else {})
    if self.block_credential_phishing and not audit["allowed"]:
      reason_msg = (
          audit["findings"][0]["message"]
          if audit["findings"]
          else "Server-initiated elicitation/create blocked by ElicitationGuard."
      )
      return {
          "allowed": False,
          "mode": "decline",
          "reason": f"[mcp-slim] {reason_msg}",
          "audit": audit,
      }
    return {
        "allowed": True,
        "mode": "forward",
        "reason": None,
        "audit": audit,
    }


class RootsPrivacyGuard:
  """Filters client `roots/list` workspace directory disclosures sent to upstream MCP servers."""

  def __init__(
      self,
      *,
      block_all: bool = False,
      allow_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      deny_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
  ) -> None:
    self.block_all = block_all
    self.uri_guard = ResourceUriGuard(
        allow_prefixes=allow_prefixes,
        deny_prefixes=deny_prefixes,
        block_traversal=True,
    )

  @property
  def enabled(self) -> bool:
    return self.block_all or self.uri_guard.has_prefix_policy

  def filter_roots(
      self, roots: list[Any], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    if self.block_all:
      if stats is not None:
        stats.roots_filtered += sum(1 for r in roots if isinstance(r, dict))
      return []
    kept: list[dict[str, Any]] = []
    for item in roots:
      if not isinstance(item, dict):
        continue
      uri = item.get("uri")
      if not isinstance(uri, str) or self.uri_guard.validate_uri(uri) is not None:
        if stats is not None:
          stats.roots_filtered += 1
        continue
      canon_uri, _ = canonicalize_resource_uri(uri)
      clean_root = dict(item)
      if canon_uri:
        clean_root["uri"] = canon_uri
      kept.append(clean_root)
    return kept



def scan_tool_result_for_injection(
    result: dict[str, Any],
    secret_guard: "Any | None" = None,
) -> list[dict[str, Any]]:
  """Scans a tools/call result dict for prompt-injection patterns in content text items (MS009).

  Returns a list of finding dicts with keys: rule_id, level, message, item_index, text_snippet.
  """
  from .guard import _SECRET_RULES, _UNICODE_SMUGGLING_RE, _ASCII_ZW_SPLIT_RE  # noqa: PLC0415

  findings: list[dict[str, Any]] = []
  content = result.get("content")
  if not isinstance(content, list):
    return findings

  for idx, item in enumerate(content):
    if not isinstance(item, dict):
      continue
    item_type = item.get("type", "text")
    # Only scan text-typed content items; skip binary/image blobs.
    if item_type not in ("text", "markdown", "html"):
      continue
    text = item.get("text", "")
    if not isinstance(text, str) or len(text) < 8:
      continue

    # Strip BiDi / Plane-14 smuggling to expose concealed keywords.
    stripped = _UNICODE_SMUGGLING_RE.sub("", text)
    stripped = _ASCII_ZW_SPLIT_RE.sub("", stripped)
    lower = stripped.lower()

    triggered_cats: list[str] = []
    for cat, triggers, case_insensitive, pattern, _replace in _SECRET_RULES:
      if cat not in ("prompt_injection", "unicode_smuggling", "markdown_exfil_beacon"):
        continue
      haystack = lower if case_insensitive else stripped
      if any(t in haystack for t in triggers):
        if pattern.search(stripped):
          triggered_cats.append(cat)

    # Also check for raw BiDi / Plane-14 smuggling markers in the original text.
    if _UNICODE_SMUGGLING_RE.search(text) or _ASCII_ZW_SPLIT_RE.search(text):
      if "unicode_smuggling" not in triggered_cats:
        triggered_cats.append("unicode_smuggling")

    if triggered_cats:
      snippet = stripped[:120].replace("\n", " ")
      findings.append({
          "rule_id": "MS009",
          "level": "error",
          "message": (
              f"tools/call result content[{idx}] contains "
              f"prompt-injection or smuggling markers "
              f"({', '.join(triggered_cats)}): {snippet!r}"
          ),
          "item_index": idx,
          "categories": triggered_cats,
          "text_snippet": snippet,
      })

  return findings


class ToolCallReplayGuard:
  """Detects server-side tools/call result substitution attacks (MS010).

  Tracks (tool_name, SHA-256(canonical_args)) -> SHA-256(result_text) across
  the session. When the same tool is called with *different* argument fingerprints
  but the server returns the *same* result fingerprint, it raises MS010.

  This catches a class of attacks where a malicious MCP server caches a poisoned
  tool result and injects it on subsequent calls with different arguments to
  silently substitute data returned to the LLM.
  """

  def __init__(self, *, enabled: bool = False) -> None:
    self.enabled = enabled
    # Maps tool_name -> list of (args_hash, result_hash) pairs seen.
    self._seen: dict[str, list[tuple[str, str]]] = {}

  def _fingerprint_args(self, arguments: Any) -> str:
    """Returns the SHA-256 hex of canonically serialized arguments."""
    import hashlib  # noqa: PLC0415
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

  def _fingerprint_result(self, result: dict[str, Any]) -> str:
    """Returns the SHA-256 hex of the result content text concatenation."""
    import hashlib  # noqa: PLC0415
    parts: list[str] = []
    for item in result.get("content", []):
      if isinstance(item, dict) and isinstance(item.get("text"), str):
        parts.append(item["text"])
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

  def check(
      self,
      tool_name: str,
      arguments: Any,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Records the call and returns an MS010 finding if replay is detected, else None."""
    if not self.enabled:
      return None
    args_hash = self._fingerprint_args(arguments)
    result_hash = self._fingerprint_result(result)
    # Empty results (e.g., always-empty responses) are not conclusive evidence of replay.
    if not result_hash or result_hash == hashlib.sha256(b"").hexdigest()[:16]:
      return None

    seen_for_tool = self._seen.setdefault(tool_name, [])
    for prev_args_hash, prev_result_hash in seen_for_tool:
      if prev_args_hash != args_hash and prev_result_hash == result_hash:
        return {
            "rule_id": "MS010",
            "level": "error",
            "message": (
                f"Tool '{tool_name}' returned identical result fingerprint "
                f"({result_hash}) for different argument fingerprints "
                f"(prev={prev_args_hash}, curr={args_hash}), indicating "
                f"server-side result substitution."
            ),
            "tool": tool_name,
            "args_hash": args_hash,
            "result_hash": result_hash,
        }
    seen_for_tool.append((args_hash, result_hash))
    return None


_DEFAULT_AMBIGUOUS_REF_CASES: tuple[dict[str, Any], ...] = (
    {
        "tool": {
            "name": "route_shipment",
            "description": (
                "Create an outbound logistics shipment between two physical addresses. "
                "Validates street, city, and postal code before booking carrier pickup."
            ),
            "inputSchema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "RouteShipmentInput",
                "type": "object",
                "additionalProperties": False,
                "$defs": {
                    "ShippingAddress": {
                        "title": "ShippingAddress",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "street": {"title": "Street", "type": "string", "description": "The street"},
                            "city": {"title": "City", "type": "string", "description": "The city"},
                            "postal_code": {"title": "Postal Code", "type": "string", "description": "The postal code"},
                        },
                        "required": ["street", "city", "postal_code"],
                    },
                    "BillingAddress": {
                        "title": "BillingAddress",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "street": {"title": "Street", "type": "string", "description": "The street"},
                            "city": {"title": "City", "type": "string", "description": "The city"},
                            "postal_code": {"title": "Postal Code", "type": "string", "description": "The postal code"},
                        },
                        "required": ["street", "city", "postal_code"],
                    },
                },
                "properties": {
                    "primary": {"$ref": "#/$defs/ShippingAddress"},
                    "secondary": {"$ref": "#/$defs/BillingAddress"},
                    "shipping_address": {"$ref": "#/$defs/ShippingAddress", "default": None},
                },
                "required": ["primary", "secondary"],
            },
        },
        "expected_roles": {
            "primary": "shipping",
            "secondary": "billing",
            "shipping_address": "shipping",
        },
    },
    {
        "tool": {
            "name": "transfer_funds",
            "description": (
                "Transfer funds between two ledger accounts with IBAN and ISO currency "
                "validation. Supports direct account references and union counterparty targets."
            ),
            "inputSchema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "TransferFundsInput",
                "type": "object",
                "additionalProperties": False,
                "$defs": {
                    "SourceAccount": {
                        "title": "SourceAccount",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "iban": {"title": "Iban", "type": "string", "description": "The iban"},
                            "currency": {"title": "Currency", "type": "string", "description": "The currency"},
                        },
                        "required": ["iban", "currency"],
                    },
                    "DestinationAccount": {
                        "title": "DestinationAccount",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "iban": {"title": "Iban", "type": "string", "description": "The iban"},
                            "currency": {"title": "Currency", "type": "string", "description": "The currency"},
                        },
                        "required": ["iban", "currency"],
                    },
                },
                "properties": {
                    "first": {"$ref": "#/$defs/SourceAccount"},
                    "second": {"$ref": "#/$defs/DestinationAccount"},
                    "target_union": {
                        "anyOf": [
                            {"$ref": "#/$defs/SourceAccount"},
                            {"$ref": "#/$defs/DestinationAccount"},
                        ]
                    },
                },
                "required": ["first", "second"],
            },
        },
        "expected_roles": {
            "first": "source",
            "second": "destination",
        },
        "expected_union_branches": {"target_union": 2},
    },
)


def evaluate_ref_disambiguation(
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  """Evaluates semantic role retention and token delta between structural-only and hint-preserving `$ref` inlining."""
  eval_cases = list(cases) if cases is not None else list(_DEFAULT_AMBIGUOUS_REF_CASES)
  raw_tools = [c["tool"] for c in eval_cases]

  structural_slimmer = SchemaSlimmer(preserve_ref_hints=False)
  semantic_slimmer = SchemaSlimmer(preserve_ref_hints=True)

  structural_stats = SlimStats()
  semantic_stats = SlimStats()

  structural_tools = structural_slimmer.slim_tools_list(raw_tools, stats=structural_stats)
  semantic_tools = semantic_slimmer.slim_tools_list(raw_tools, stats=semantic_stats)

  def _score(slimmed_list: list[dict[str, Any]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for case, slim_tool in zip(eval_cases, slimmed_list):
      props = slim_tool.get("inputSchema", {}).get("properties", {})
      for prop_name, role_kw in case.get("expected_roles", {}).items():
        total += 1
        prop_node = props.get(prop_name, {})
        haystack = f"{prop_name} {prop_node.get('description', '')}".lower()
        if role_kw.lower() in haystack:
          correct += 1
      for union_prop, expected_cnt in case.get("expected_union_branches", {}).items():
        total += 1
        u_node = props.get(union_prop, {})
        branches = u_node.get("anyOf") or u_node.get("oneOf") or []
        if len(branches) == expected_cnt:
          correct += 1
    return correct, total

  struct_ok, total_checks = _score(structural_tools)
  sem_ok, _ = _score(semantic_tools)

  return {
      "total_checks": total_checks,
      "structural_only_accuracy": round(struct_ok / max(1, total_checks), 4),
      "semantic_hint_accuracy": round(sem_ok / max(1, total_checks), 4),
      "raw_tokens": semantic_stats.schema_tokens_before,
      "structural_only_tokens": structural_stats.schema_tokens_after,
      "semantic_hint_tokens": semantic_stats.schema_tokens_after,
      "extra_hint_tokens": semantic_stats.schema_tokens_after - structural_stats.schema_tokens_after,
      "ref_hints_preserved": semantic_stats.ref_hints_preserved,
  }


def build_audit_sarif(
    report: dict[str, Any],
    source_uri: str = "tools.json",
    version: str = "1.5.0",
) -> dict[str, Any]:
  """Formats an `audit_tools_manifest`, `diff_tools_manifests`, `audit_resources_manifest`, or `audit_elicitation_request` report as OASIS SARIF v2.1.0."""
  results: list[dict[str, Any]] = []
  for f in report.get("findings", []):
    loc_name = str(f.get("tool") or f.get("uri") or "mcp")
    if f.get("rule_id") == "MS006":
      loc_kind = "resource"
    elif f.get("rule_id") == "MS007":
      loc_kind = "elicitation"
    else:
      loc_kind = "function"
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


# ---------------------------------------------------------------------------
# v1.6.0 --- SseChunkThrottle
# ---------------------------------------------------------------------------

@dataclass
class SseChunkDecision:
  """Result returned by SseChunkThrottle.evaluate()."""

  allowed: bool
  """Whether the chunk should be forwarded to the client."""

  chunk_tokens: int
  """Estimated token count for this chunk."""

  window_tokens: float
  """Running exponentially-decayed window total after this chunk."""

  budget_tokens: int
  """Configured per-window burst budget."""

  throttled: bool
  """True when the chunk was dropped or will be suppressed."""

  drop_reason: str
  """Human-readable reason when throttled is True, else empty string."""


class SseChunkThrottle:
  """Real-time SSE stream chunk budget throttle (MS011).

  Prevents token burst depletion across SSE transports by tracking a sliding
  exponentially-decaying window of chunk tokens delivered per session.  When
  cumulative window pressure exceeds `burst_budget_tokens`, excess chunks are
  suppressed and an MS011 alert is emitted.

  The window decays by `decay` on each chunk (multiplicative). With the
  default ``decay=0.95`` and a 60-token chunk every second, the window
  stabilises around ``chunk_tokens / (1 - decay) = 1200`` tokens, giving a
  natural steady-state ceiling.

  Args:
      burst_budget_tokens: Maximum cumulative (decayed) tokens allowed in the
          rolling window before throttling activates.  0 disables throttling.
      decay: Per-chunk multiplicative decay factor applied to the window
          before adding the new chunk (0 < decay < 1).
      min_passthrough: Minimum number of chunks always forwarded before
          throttling can engage (default 1 so the first chunk is never
          dropped).
  """

  def __init__(
      self,
      burst_budget_tokens: int = 0,
      *,
      decay: float = 0.95,
      min_passthrough: int = 1,
  ) -> None:
    self.burst_budget_tokens: int = max(0, int(burst_budget_tokens))
    self.enabled: bool = self.burst_budget_tokens > 0
    if not (0.0 < decay < 1.0):
      raise ValueError(f"SseChunkThrottle: decay must be in (0, 1), got {decay!r}")
    self.decay: float = decay
    self.min_passthrough: int = max(0, int(min_passthrough))
    self._window_tokens: float = 0.0
    self._chunks_seen: int = 0
    self._chunks_throttled: int = 0

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def evaluate(self, chunk_text: str, *, stats: "SlimStats | None" = None) -> SseChunkDecision:
    """Process one SSE chunk and return a throttle decision.

    The window is decayed *before* adding the new chunk so that an idle
    session recovers its full budget immediately.

    Args:
        chunk_text: Raw SSE chunk text (the ``data:`` payload).
        stats: Optional :class:`SlimStats` accumulator; ``budget_throttles``
            is incremented when this chunk is suppressed.

    Returns:
        :class:`SseChunkDecision` with ``allowed=True`` when the chunk
        should be forwarded, ``allowed=False`` when it is suppressed.
    """
    chunk_tokens = estimate_tokens(chunk_text)
    self._window_tokens = self._window_tokens * self.decay + chunk_tokens
    self._chunks_seen += 1

    if not self.enabled or self._chunks_seen <= self.min_passthrough:
      return SseChunkDecision(
          allowed=True,
          chunk_tokens=chunk_tokens,
          window_tokens=self._window_tokens,
          budget_tokens=self.burst_budget_tokens,
          throttled=False,
          drop_reason="",
      )

    over_budget = self._window_tokens > self.burst_budget_tokens
    if over_budget:
      self._chunks_throttled += 1
      if stats is not None:
        stats.budget_throttles += 1
      excess = round(self._window_tokens - self.burst_budget_tokens, 1)
      return SseChunkDecision(
          allowed=False,
          chunk_tokens=chunk_tokens,
          window_tokens=self._window_tokens,
          budget_tokens=self.burst_budget_tokens,
          throttled=True,
          drop_reason=(
              f"MS011: SSE chunk suppressed; window_tokens={self._window_tokens:.1f} "
              f"exceeds burst_budget_tokens={self.burst_budget_tokens} "
              f"(excess {excess} tokens, decay={self.decay})"
          ),
      )

    return SseChunkDecision(
        allowed=True,
        chunk_tokens=chunk_tokens,
        window_tokens=self._window_tokens,
        budget_tokens=self.burst_budget_tokens,
        throttled=False,
        drop_reason="",
    )

  def reset(self) -> None:
    """Reset the window; call between SSE sessions."""
    self._window_tokens = 0.0
    self._chunks_seen = 0
    self._chunks_throttled = 0

  def snapshot(self) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the current throttle state."""
    return {
        "enabled": self.enabled,
        "burst_budget_tokens": self.burst_budget_tokens,
        "decay": self.decay,
        "window_tokens": round(self._window_tokens, 3),
        "chunks_seen": self._chunks_seen,
        "chunks_throttled": self._chunks_throttled,
        "throttle_rate": (
            round(self._chunks_throttled / self._chunks_seen, 4)
            if self._chunks_seen > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# v1.6.0 --- LatencyHistogram
# ---------------------------------------------------------------------------

class LatencyHistogram:
  """JSON-RPC call latency histogram with Prometheus text-format export (MS012).

  Records per-method round-trip latency samples (in milliseconds), computes
  percentile statistics (p50/p90/p95/p99), and can render a Prometheus
  ``# TYPE``/``# HELP`` text-format metrics page consumable by any scraper.

  Buckets follow the standard Prometheus histogram convention: each bucket
  counts observations ``<= upper_bound``.  The default bucket boundaries
  cover sub-millisecond to 60-second RPC spans.

  Args:
      method_allowlist: When non-empty, only methods in this set are tracked.
          Pass ``None`` or an empty set to track all methods.
      buckets: Explicit upper-bound values in milliseconds (must be sorted
          ascending).  Defaults to a sensible set for JSON-RPC over stdio.
  """

  _DEFAULT_BUCKETS: tuple[float, ...] = (
      1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
      1_000.0, 2_500.0, 5_000.0, 10_000.0, 30_000.0, 60_000.0,
  )

  def __init__(
      self,
      method_allowlist: set[str] | list[str] | None = None,
      *,
      buckets: tuple[float, ...] | list[float] | None = None,
  ) -> None:
    self._allowlist: frozenset[str] | None = (
        frozenset(method_allowlist) if method_allowlist else None
    )
    raw_buckets = list(buckets) if buckets is not None else list(self._DEFAULT_BUCKETS)
    if sorted(raw_buckets) != raw_buckets:
      raise ValueError("LatencyHistogram: buckets must be sorted ascending")
    self._buckets: tuple[float, ...] = tuple(raw_buckets)
    # method -> list of latency samples (ms)
    self._samples: dict[str, list[float]] = {}
    # method -> bucket_counts (len == len(buckets))
    self._bucket_counts: dict[str, list[int]] = {}
    self._total_count: dict[str, int] = {}
    self._total_sum_ms: dict[str, float] = {}

  # ------------------------------------------------------------------
  # Recording
  # ------------------------------------------------------------------

  def record(self, method: str, latency_ms: float) -> None:
    """Record one latency observation for a JSON-RPC method.

    Args:
        method: JSON-RPC method name, e.g. ``"tools/call"``.
        latency_ms: Round-trip latency in milliseconds (must be >= 0).
    """
    if latency_ms < 0:
      latency_ms = 0.0
    if self._allowlist is not None and method not in self._allowlist:
      return
    if method not in self._samples:
      self._samples[method] = []
      self._bucket_counts[method] = [0] * len(self._buckets)
      self._total_count[method] = 0
      self._total_sum_ms[method] = 0.0
    self._samples[method].append(latency_ms)
    self._total_count[method] += 1
    self._total_sum_ms[method] += latency_ms
    for idx, upper in enumerate(self._buckets):
      if latency_ms <= upper:
        self._bucket_counts[method][idx] += 1

  # ------------------------------------------------------------------
  # Statistics
  # ------------------------------------------------------------------

  def _percentile(self, samples: list[float], pct: float) -> float:
    """Return the p-th percentile from a sorted or unsorted sample list."""
    if not samples:
      return 0.0
    s = sorted(samples)
    n = len(s)
    if n == 1:
      return s[0]
    # Nearest-rank method (inclusive).
    rank = max(1, int(math.ceil(pct / 100.0 * n)))
    return s[rank - 1]

  def stats(self, method: str) -> dict[str, Any]:
    """Return summary statistics for one method.

    Returns:
        Dict with keys: ``count``, ``sum_ms``, ``mean_ms``, ``min_ms``,
        ``max_ms``, ``p50_ms``, ``p90_ms``, ``p95_ms``, ``p99_ms``.
        All values are ``0`` when no samples have been recorded.
    """
    samples = self._samples.get(method, [])
    if not samples:
      return {
          "count": 0,
          "sum_ms": 0.0,
          "mean_ms": 0.0,
          "min_ms": 0.0,
          "max_ms": 0.0,
          "p50_ms": 0.0,
          "p90_ms": 0.0,
          "p95_ms": 0.0,
          "p99_ms": 0.0,
      }
    total = self._total_count[method]
    total_sum = self._total_sum_ms[method]
    return {
        "count": total,
        "sum_ms": round(total_sum, 3),
        "mean_ms": round(total_sum / total, 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "p50_ms": round(self._percentile(samples, 50.0), 3),
        "p90_ms": round(self._percentile(samples, 90.0), 3),
        "p95_ms": round(self._percentile(samples, 95.0), 3),
        "p99_ms": round(self._percentile(samples, 99.0), 3),
    }

  def all_stats(self) -> dict[str, dict[str, Any]]:
    """Return statistics for every tracked method."""
    return {method: self.stats(method) for method in sorted(self._samples)}

  def methods(self) -> list[str]:
    """Return sorted list of methods for which samples have been recorded."""
    return sorted(self._samples.keys())

  def total_observations(self) -> int:
    """Return total number of observations across all methods."""
    return sum(self._total_count.values())

  # ------------------------------------------------------------------
  # Prometheus text-format export
  # ------------------------------------------------------------------

  def prometheus_text(self, *, prefix: str = "mcp_slim_jsonrpc") -> str:
    """Render a Prometheus text-format metrics page.

    Produces standard ``_bucket``, ``_count``, and ``_sum`` series for each
    method plus ``_percentile`` gauges (p50/p90/p95/p99).

    Args:
        prefix: Metric name prefix (default ``mcp_slim_jsonrpc``).

    Returns:
        A string in Prometheus exposition format v0.0.4, ending with ``\\n``.
    """
    lines: list[str] = []
    hist_name = f"{prefix}_latency_ms"
    pct_name = f"{prefix}_latency_percentile_ms"

    lines.append(f"# HELP {hist_name} JSON-RPC method round-trip latency in milliseconds.")
    lines.append(f"# TYPE {hist_name} histogram")
    for method in sorted(self._samples):
      safe_method = method.replace("/", "_").replace(".", "_").replace("-", "_")
      label = f'method="{method}"'
      samples = self._samples[method]
      counts = self._bucket_counts[method]
      # counts[idx] is already cumulative (incremented for every bucket
      # whose upper_bound >= the observed latency), matching Prometheus convention.
      for idx, upper in enumerate(self._buckets):
        lines.append(
            f'{hist_name}_bucket{{le="{upper}",{label}}} {counts[idx]}'
        )
      lines.append(f'{hist_name}_bucket{{le="+Inf",{label}}} {self._total_count[method]}')
      lines.append(f'{hist_name}_count{{{label}}} {self._total_count[method]}')
      lines.append(f'{hist_name}_sum{{{label}}} {self._total_sum_ms[method]:.3f}')

    lines.append(f"# HELP {pct_name} JSON-RPC latency percentiles (gauge, milliseconds).")
    lines.append(f"# TYPE {pct_name} gauge")
    for method in sorted(self._samples):
      s = self.stats(method)
      label = f'method="{method}"'
      for pct_label, key in (("0.5", "p50_ms"), ("0.9", "p90_ms"), ("0.95", "p95_ms"), ("0.99", "p99_ms")):
        lines.append(
            f'{pct_name}{{quantile="{pct_label}",{label}}} {s[key]}'
        )
    lines.append("")
    return "\n".join(lines)

  def reset(self) -> None:
    """Clear all recorded samples."""
    self._samples.clear()
    self._bucket_counts.clear()
    self._total_count.clear()
    self._total_sum_ms.clear()

  def snapshot(self) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of all histogram data."""
    return {
        "buckets_ms": list(self._buckets),
        "methods": self.all_stats(),
        "total_observations": self.total_observations(),
    }


# ---------------------------------------------------------------------------
# MS013 / MS014 -- imported from dedicated module
# ---------------------------------------------------------------------------
from ._ms013_ms014 import (  # noqa: E402
    CollisionReport,
    NamespaceRoutingResult,
    RefDepthGuard,
    RefDepthReport,
    ToolNamespaceRouter,
)

__all__ = [
    "CollisionReport",
    "NamespaceRoutingResult",
    "RefDepthGuard",
    "RefDepthReport",
    "ToolNamespaceRouter",
]
