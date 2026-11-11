"""Schema and tool-output compaction engine for MCP JSON-RPC traffic."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")

# Redundant JSON Schema keys that LLMs do not need to invoke a tool.
_REDUNDANT_SCHEMA_KEYS = frozenset({
    "$schema",
    "$id",
    "additionalProperties",
    "title",
    "examples",
    "default",
})


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
    # Split at first period/newline that ends a sentence (ignoring e.g. / i.e. / v1.0).
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
  tools_total: int = 0
  tools_kept: int = 0

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
  def reduction_pct(self) -> float:
    if self.total_tokens_before == 0:
      return 0.0
    return round((self.tokens_saved / self.total_tokens_before) * 100.0, 1)


class SchemaSlimmer:
  """Minifies MCP `tools/list` schemas without altering tool names or required parameters."""

  def __init__(
      self,
      max_desc_chars: int = 120,
      max_param_desc_chars: int = 80,
      allow_tools: set[str] | None = None,
      deny_tools: set[str] | None = None,
  ) -> None:
    self.max_desc_chars = max_desc_chars
    self.max_param_desc_chars = max_param_desc_chars
    self.allow_tools = allow_tools
    self.deny_tools = deny_tools or set()

  def slim_schema_node(self, node: Any, is_root: bool = False) -> Any:
    if not isinstance(node, dict):
      return node
    out: dict[str, Any] = {}
    for k, v in node.items():
      if k in _REDUNDANT_SCHEMA_KEYS:
        continue
      if k == "description" and isinstance(v, str):
        compact = compact_description(v, max_chars=self.max_param_desc_chars)
        # Drop trivial descriptions that merely repeat the property name.
        if compact:
          out[k] = compact
        continue
      if k == "properties" and isinstance(v, dict):
        props: dict[str, Any] = {}
        for prop_name, prop_schema in v.items():
          slimmed_prop = self.slim_schema_node(prop_schema, is_root=False)
          if isinstance(slimmed_prop, dict):
            desc = slimmed_prop.get("description", "")
            norm_name = prop_name.lower().replace("_", " ").strip()
            norm_desc = desc.lower().rstrip(".").strip()
            if norm_desc in (
                norm_name,
                f"the {norm_name}",
                f"{norm_name} value",
            ):
              slimmed_prop.pop("description", None)
          props[prop_name] = slimmed_prop
        out[k] = props
        continue
      if k == "items" and isinstance(v, dict):
        out[k] = self.slim_schema_node(v, is_root=False)
        continue
      if k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
        # Simplify nullable union `anyOf: [{type: X}, {type: null}]` -> `type: X`
        non_null = [
            branch
            for branch in v
            if not (isinstance(branch, dict) and branch.get("type") == "null")
        ]
        if len(non_null) == 1 and isinstance(non_null[0], dict):
          merged = self.slim_schema_node(non_null[0], is_root=False)
          if isinstance(merged, dict):
            for mk, mv in merged.items():
              out.setdefault(mk, mv)
          continue
        out[k] = [self.slim_schema_node(branch, is_root=False) for branch in v]
        continue
      if k == "required" and isinstance(v, list) and not v:
        continue
      out[k] = v
    if is_root and "type" not in out:
      out["type"] = "object"
    return out

  def slim_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
    slimmed: dict[str, Any] = {"name": tool["name"]}
    desc = tool.get("description")
    if isinstance(desc, str) and desc.strip():
      slimmed["description"] = compact_description(
          desc, max_chars=self.max_desc_chars
      )
    input_schema = tool.get("inputSchema")
    if isinstance(input_schema, dict):
      slimmed["inputSchema"] = self.slim_schema_node(input_schema, is_root=True)
    return slimmed

  def slim_tools_list(
      self, tools: list[dict[str, Any]], stats: SlimStats | None = None
  ) -> list[dict[str, Any]]:
    before_tokens = estimate_tokens(tools)
    filtered: list[dict[str, Any]] = []
    for tool in tools:
      name = tool.get("name", "")
      if self.allow_tools is not None and name not in self.allow_tools:
        continue
      if name in self.deny_tools:
        continue
      filtered.append(self.slim_tool(tool))
    after_tokens = estimate_tokens(filtered)
    if stats is not None:
      stats.tools_total += len(tools)
      stats.tools_kept += len(filtered)
      stats.schema_tokens_before += before_tokens
      stats.schema_tokens_after += after_tokens
    return filtered


class OutputSlimmer:
  """Compacts verbose `tools/call` outputs (JSON blobs, logs, ANSI sequences)."""

  def __init__(
      self,
      max_list_items: int = 50,
      max_string_chars: int = 4000,
      strip_nulls: bool = True,
  ) -> None:
    self.max_list_items = max_list_items
    self.max_string_chars = max_string_chars
    self.strip_nulls = strip_nulls

  def compact_json_value(self, value: Any) -> Any:
    if isinstance(value, dict):
      out: dict[str, Any] = {}
      for k, v in value.items():
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
      if len(value) > self.max_string_chars:
        return (
            value[: self.max_string_chars]
            + f"... [truncated {len(value) - self.max_string_chars} chars]"
        )
      return value
    return value

  def slim_text(self, text: str) -> str:
    if not text:
      return text
    cleaned = _ANSI_ESCAPE.sub("", text)
    stripped = cleaned.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
      try:
        parsed = json.loads(stripped)
        compacted = self.compact_json_value(parsed)
        return json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
      except (json.JSONDecodeError, TypeError):
        pass
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    if len(cleaned) > self.max_string_chars:
      cleaned = (
          cleaned[: self.max_string_chars]
          + f"\n... [truncated {len(cleaned) - self.max_string_chars} chars by"
          " mcp-slim]"
      )
    return cleaned

  def slim_call_result(
      self, result: dict[str, Any], stats: SlimStats | None = None
  ) -> dict[str, Any]:
    before_tokens = estimate_tokens(result)
    out = copy.deepcopy(result)
    content = out.get("content")
    if isinstance(content, list):
      for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
          text = item.get("text")
          if isinstance(text, str):
            item["text"] = self.slim_text(text)
    if "structuredContent" in out and isinstance(
        out["structuredContent"], (dict, list)
    ):
      out["structuredContent"] = self.compact_json_value(
          out["structuredContent"]
      )
    after_tokens = estimate_tokens(out)
    if stats is not None:
      stats.output_tokens_before += before_tokens
      stats.output_tokens_after += after_tokens
    return out
