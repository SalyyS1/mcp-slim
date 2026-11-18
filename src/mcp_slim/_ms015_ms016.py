"""MS015: Streaming tool-call structural JSON repair for truncated/malformed token streams.
MS016: Multi-server composite schema merger with collision and shared-$defs resolution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

_JSON_REPAIR_RULE_ID = "MS015"
_COMPOSITE_MERGE_RULE_ID = "MS016"

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_VALID_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_VALID_NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_TRUNCATED_NUMBER_PREFIX_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?")
_CODE_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?")
_CODE_FENCE_CLOSE_RE = re.compile(r"\r?\n?[ \t]*`{1,3}[ \t]*$")


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass
class _ContainerFrame:
    kind: str  # "{" or "["
    state: str
    member_rollback_pos: int


@dataclass
class JsonRepairResult:
    """Outcome of structural JSON repair on a truncated or lightly malformed payload."""

    repaired: bool
    is_valid: bool
    repaired_json: str
    value: Any
    repairs_applied: list[str] = field(default_factory=list)
    error: str | None = None

    def to_finding(self, tool_name: str = "tools/call") -> dict[str, Any] | None:
        """Convert a repair or failure into an audit finding dict (MS015)."""
        if not self.is_valid:
            return {
                "rule_id": _JSON_REPAIR_RULE_ID,
                "level": "error",
                "tool": tool_name,
                "kind": "unrepairable_json",
                "repairs_applied": list(self.repairs_applied),
                "message": (
                    f"Tool-call JSON for '{tool_name}' could not be structurally repaired: "
                    f"{self.error or 'invalid JSON'}."
                ),
            }
        if not self.repaired:
            return None
        ops = ", ".join(self.repairs_applied)
        return {
            "rule_id": _JSON_REPAIR_RULE_ID,
            "level": "warning",
            "tool": tool_name,
            "kind": "repaired_truncated_json",
            "repairs_applied": list(self.repairs_applied),
            "message": (
                f"Repaired truncated or malformed tool-call JSON for '{tool_name}' "
                f"({ops})."
            ),
        }

    def to_sarif_results(self, tool_name: str = "tools/call") -> list[dict[str, Any]]:
        """Convert repair activity or failure to SARIF v2.1.0 result objects."""
        finding = self.to_finding(tool_name=tool_name)
        if finding is None:
            return []
        return [
            {
                "ruleId": _JSON_REPAIR_RULE_ID,
                "level": finding["level"],
                "message": {"text": finding["message"]},
                "locations": [
                    {
                        "logicalLocations": [
                            {"name": tool_name, "kind": "function"}
                        ]
                    }
                ],
                "properties": {
                    "kind": finding["kind"],
                    "repairs_applied": list(self.repairs_applied),
                },
            }
        ]


def _repair_atom_token(atom: str) -> tuple[str | None, str | None]:
    """Repairs a truncated JSON primitive token at EOF.

    Returns ``(replacement_token, repair_label)`` or ``(None, None)`` if unrecoverable.
    """
    token = atom.strip()
    if not token:
        return None, None
    if token in ("true", "false", "null"):
        return token, None
    if _VALID_NUMBER_RE.match(token):
        return token, None

    for kw in ("true", "false", "null"):
        if kw.startswith(token) and len(token) < len(kw):
            return kw, "completed_keyword_literal"

    num_match = _TRUNCATED_NUMBER_PREFIX_RE.match(token)
    if num_match is not None:
        prefix = num_match.group(0)
        rem = token[len(prefix) :]
        if rem in (".", "e", "E", "e+", "e-", "E+", "E-") or (
            rem.startswith(".") and rem[1:] in ("e", "E", "e+", "e-", "E+", "E-")
        ):
            if prefix == "-0":
                prefix = "0"
            return prefix, "repaired_truncated_number"

    return None, None


class ToolCallJsonRepairer:
    """MS015: Deterministic O(N) lexical state-machine repairer for truncated tool-call JSON.

    Designed for streaming LLM tool invocations that terminate mid-token due to
    ``max_tokens`` cutoffs, SSE stream drops, trailing commas, unescaped string
    control characters, or Markdown code-fence wrappers.
    """

    def __init__(
        self,
        *,
        require_object_root: bool = True,
        strip_code_fences: bool = True,
        complete_keywords: bool = True,
        max_input_chars: int = 262_144,
    ) -> None:
        if max_input_chars < 64:
            raise ValueError("max_input_chars must be >= 64")
        self.require_object_root = require_object_root
        self.strip_code_fences = strip_code_fences
        self.complete_keywords = complete_keywords
        self.max_input_chars = max_input_chars

    def repair(self, raw_json: str) -> JsonRepairResult:
        """Attempt structural JSON repair and verify with ``json.loads``."""
        if not isinstance(raw_json, str):
            return JsonRepairResult(
                repaired=False,
                is_valid=False,
                repaired_json="",
                value=None,
                repairs_applied=[],
                error="Input must be a string.",
            )
        if len(raw_json) > self.max_input_chars:
            return JsonRepairResult(
                repaired=False,
                is_valid=False,
                repaired_json="",
                value=None,
                repairs_applied=[],
                error=f"Input length {len(raw_json)} exceeds max_input_chars={self.max_input_chars}.",
            )

        text = raw_json.strip()
        if not text:
            if self.require_object_root:
                return JsonRepairResult(
                    repaired=False,
                    is_valid=True,
                    repaired_json="{}",
                    value={},
                    repairs_applied=[],
                    error=None,
                )
            return JsonRepairResult(
                repaired=False,
                is_valid=False,
                repaired_json="",
                value=None,
                repairs_applied=[],
                error="Empty JSON input.",
            )

        # Fast-path: valid JSON requires zero repair
        try:
            parsed_fast = json.loads(text)
        except json.JSONDecodeError:
            parsed_fast = None
        else:
            if self.require_object_root and not isinstance(parsed_fast, dict):
                return JsonRepairResult(
                    repaired=False,
                    is_valid=False,
                    repaired_json=text,
                    value=None,
                    repairs_applied=[],
                    error="Tool-call arguments must decode to a JSON object.",
                )
            canonical = json.dumps(parsed_fast, ensure_ascii=False, separators=(",", ":"))
            return JsonRepairResult(
                repaired=False,
                is_valid=True,
                repaired_json=canonical,
                value=parsed_fast,
                repairs_applied=[],
                error=None,
            )

        repairs: list[str] = []
        working = text
        if self.strip_code_fences and working.startswith("```"):
            working = _CODE_FENCE_OPEN_RE.sub("", working, count=1)
            working = _CODE_FENCE_CLOSE_RE.sub("", working, count=1).strip()
            repairs.append("stripped_markdown_fence")
            if not working:
                if self.require_object_root:
                    return JsonRepairResult(
                        repaired=True,
                        is_valid=True,
                        repaired_json="{}",
                        value={},
                        repairs_applied=["stripped_markdown_fence", "closed_object"],
                        error=None,
                    )
                return JsonRepairResult(
                    repaired=False,
                    is_valid=False,
                    repaired_json="",
                    value=None,
                    repairs_applied=repairs,
                    error="Empty content inside Markdown code fence.",
                )
            try:
                parsed_fenced = json.loads(working)
            except json.JSONDecodeError:
                pass
            else:
                if self.require_object_root and not isinstance(parsed_fenced, dict):
                    return JsonRepairResult(
                        repaired=False,
                        is_valid=False,
                        repaired_json=working,
                        value=None,
                        repairs_applied=repairs,
                        error="Tool-call arguments must decode to a JSON object.",
                    )
                canonical = json.dumps(parsed_fenced, ensure_ascii=False, separators=(",", ":"))
                return JsonRepairResult(
                    repaired=True,
                    is_valid=True,
                    repaired_json=canonical,
                    value=parsed_fenced,
                    repairs_applied=repairs,
                    error=None,
                )

        # Single-pass lexical state machine
        out: list[str] = []
        stack: list[_ContainerFrame] = []
        in_string = False
        string_role = "ROOT"  # "KEY" or "VALUE" or "ROOT"
        in_escape = False
        escape_start_pos = 0
        unicode_hex_count: int | None = None
        in_atom = False
        atom_start_pos = 0

        def _advance_parent_after_value() -> None:
            if not stack:
                return
            top = stack[-1]
            if top.kind == "{":
                top.state = "EXPECT_COMMA_OR_END"
            elif top.kind == "[":
                top.state = "EXPECT_COMMA_OR_END"

        def _finish_atom_midstream() -> bool:
            nonlocal in_atom
            if not in_atom:
                return True
            atom_text = "".join(out[atom_start_pos:])
            in_atom = False
            if atom_text in ("true", "false", "null") or _VALID_NUMBER_RE.match(atom_text):
                _advance_parent_after_value()
                return True
            fixed, label = _repair_atom_token(atom_text)
            if fixed is not None and (label != "completed_keyword_literal" or self.complete_keywords):
                out[atom_start_pos:] = list(fixed)
                if label:
                    repairs.append(label)
                _advance_parent_after_value()
                return True
            return False

        for ch in working:
            if in_string:
                if in_escape:
                    in_escape = False
                    if ch == "u":
                        out.append(ch)
                        unicode_hex_count = 0
                    elif ch in _VALID_SIMPLE_ESCAPES:
                        out.append(ch)
                    else:
                        # Unescaped backslash such as Windows path C:\dev -> C:\\dev
                        out.append("\\")
                        out.append(ch)
                        repairs.append("repaired_invalid_escape")
                    continue

                if unicode_hex_count is not None:
                    if ch in _HEX_CHARS:
                        out.append(ch)
                        unicode_hex_count += 1
                        if unicode_hex_count == 4:
                            unicode_hex_count = None
                        continue
                    # Incomplete \u escape mid-string (e.g. "C:\users"): escape initial backslash
                    out.insert(escape_start_pos, "\\")
                    unicode_hex_count = None
                    repairs.append("repaired_invalid_escape")
                    # Fall through to handle `ch` normally inside string

                if ch == "\\":
                    escape_start_pos = len(out)
                    out.append("\\")
                    in_escape = True
                elif ch == '"':
                    out.append('"')
                    in_string = False
                    if stack:
                        top = stack[-1]
                        if string_role == "KEY" and top.kind == "{":
                            top.state = "EXPECT_COLON"
                        else:
                            _advance_parent_after_value()
                elif ord(ch) < 0x20:
                    # Raw control character inside JSON string
                    if ch == "\n":
                        out.append("\\n")
                    elif ch == "\r":
                        out.append("\\r")
                    elif ch == "\t":
                        out.append("\\t")
                    else:
                        out.append(f"\\u{ord(ch):04x}")
                    repairs.append("escaped_control_chars")
                else:
                    out.append(ch)
                continue

            # Outside string
            if in_atom and ch in (" ", "\t", "\r", "\n", ",", "}", "]", ":"):
                if not _finish_atom_midstream():
                    break

            if ch in (" ", "\t", "\r", "\n"):
                continue

            if ch == '"':
                if stack:
                    top = stack[-1]
                    if top.kind == "{" and top.state == "EXPECT_KEY_OR_END":
                        string_role = "KEY"
                        top.state = "IN_KEY"
                    else:
                        string_role = "VALUE"
                        if top.kind == "{":
                            top.state = "IN_VALUE"
                        else:
                            top.state = "IN_ELEM"
                else:
                    string_role = "ROOT"
                in_string = True
                out.append('"')
                continue

            if ch == "{":
                if stack:
                    stack[-1].state = "IN_SUBCONTAINER"
                out.append("{")
                stack.append(
                    _ContainerFrame(
                        kind="{",
                        state="EXPECT_KEY_OR_END",
                        member_rollback_pos=len(out),
                    )
                )
                continue

            if ch == "[":
                if stack:
                    stack[-1].state = "IN_SUBCONTAINER"
                out.append("[")
                stack.append(
                    _ContainerFrame(
                        kind="[",
                        state="EXPECT_ELEM_OR_END",
                        member_rollback_pos=len(out),
                    )
                )
                continue

            if ch == ":":
                if stack and stack[-1].kind == "{" and stack[-1].state == "EXPECT_COLON":
                    out.append(":")
                    stack[-1].state = "EXPECT_VALUE"
                continue

            if ch == ",":
                if stack:
                    top = stack[-1]
                    if top.state == "EXPECT_COMMA_OR_END":
                        comma_pos = len(out)
                        out.append(",")
                        top.member_rollback_pos = comma_pos
                        top.state = (
                            "EXPECT_KEY_OR_END" if top.kind == "{" else "EXPECT_ELEM_OR_END"
                        )
                    elif top.state in ("EXPECT_KEY_OR_END", "EXPECT_ELEM_OR_END"):
                        # Duplicate comma: ignore
                        repairs.append("removed_trailing_comma")
                continue

            if ch == "}":
                if stack and stack[-1].kind == "{":
                    top = stack[-1]
                    if top.state == "EXPECT_KEY_OR_END" and len(out) > top.member_rollback_pos:
                        del out[top.member_rollback_pos :]
                        repairs.append("removed_trailing_comma")
                    elif top.state in ("IN_KEY", "EXPECT_COLON", "EXPECT_VALUE"):
                        del out[top.member_rollback_pos :]
                        repairs.append("dropped_dangling_property")
                    stack.pop()
                    out.append("}")
                    _advance_parent_after_value()
                continue

            if ch == "]":
                if stack and stack[-1].kind == "[":
                    top = stack[-1]
                    if top.state == "EXPECT_ELEM_OR_END" and len(out) > top.member_rollback_pos:
                        del out[top.member_rollback_pos :]
                        repairs.append("removed_trailing_comma")
                    stack.pop()
                    out.append("]")
                    _advance_parent_after_value()
                continue

            # Atom character (digits, -, ., e, true/false/null)
            if not in_atom:
                in_atom = True
                atom_start_pos = len(out)
                if stack:
                    top = stack[-1]
                    if top.kind == "{":
                        top.state = "IN_VALUE_ATOM"
                    else:
                        top.state = "IN_ELEM_ATOM"
            out.append(ch)

        # EOF repair phase
        if in_string:
            if stack and stack[-1].kind == "{" and string_role == "KEY":
                # Truncated inside an object key -> drop the incomplete member
                del out[stack[-1].member_rollback_pos :]
                in_string = False
                in_escape = False
                unicode_hex_count = None
                stack[-1].state = "EXPECT_COMMA_OR_END"
                repairs.append("dropped_dangling_property")
            else:
                if in_escape or unicode_hex_count is not None:
                    del out[escape_start_pos :]
                    in_escape = False
                    unicode_hex_count = None
                    repairs.append("trimmed_dangling_escape")
                out.append('"')
                in_string = False
                repairs.append("closed_string")
                _advance_parent_after_value()

        elif in_atom:
            atom_text = "".join(out[atom_start_pos:])
            in_atom = False
            fixed, label = _repair_atom_token(atom_text)
            if fixed is not None and (label != "completed_keyword_literal" or self.complete_keywords):
                out[atom_start_pos:] = list(fixed)
                if label:
                    repairs.append(label)
                _advance_parent_after_value()
            elif stack:
                top = stack[-1]
                del out[top.member_rollback_pos :]
                if top.kind == "{":
                    repairs.append("dropped_dangling_property")
                else:
                    repairs.append("dropped_dangling_element")
                top.state = "EXPECT_COMMA_OR_END"

        # Unwind any unclosed containers on the stack in LIFO order
        while stack:
            top = stack.pop()
            if top.kind == "{":
                if top.state == "EXPECT_KEY_OR_END" and len(out) > top.member_rollback_pos:
                    del out[top.member_rollback_pos :]
                    repairs.append("removed_trailing_comma")
                elif top.state in ("IN_KEY", "EXPECT_COLON", "EXPECT_VALUE"):
                    del out[top.member_rollback_pos :]
                    repairs.append("dropped_dangling_property")
                out.append("}")
                repairs.append("closed_object")
            else:
                if top.state == "EXPECT_ELEM_OR_END" and len(out) > top.member_rollback_pos:
                    del out[top.member_rollback_pos :]
                    repairs.append("removed_trailing_comma")
                out.append("]")
                repairs.append("closed_array")
            _advance_parent_after_value()

        candidate = "".join(out)
        deduped_repairs = _dedup_preserve_order(repairs)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return JsonRepairResult(
                repaired=False,
                is_valid=False,
                repaired_json=raw_json,
                value=None,
                repairs_applied=deduped_repairs,
                error=f"Malformed tool-call arguments JSON: {exc.msg}",
            )

        if self.require_object_root and not isinstance(parsed, dict):
            return JsonRepairResult(
                repaired=False,
                is_valid=False,
                repaired_json=candidate,
                value=None,
                repairs_applied=deduped_repairs,
                error="Tool-call arguments must decode to a JSON object.",
            )

        canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return JsonRepairResult(
            repaired=True,
            is_valid=True,
            repaired_json=canonical,
            value=parsed,
            repairs_applied=deduped_repairs or ["normalized_json"],
            error=None,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "rule": _JSON_REPAIR_RULE_ID,
            "require_object_root": self.require_object_root,
            "strip_code_fences": self.strip_code_fences,
            "complete_keywords": self.complete_keywords,
            "max_input_chars": self.max_input_chars,
        }


def repair_truncated_json(
    raw_json: str,
    *,
    require_object_root: bool = True,
    strip_code_fences: bool = True,
    complete_keywords: bool = True,
) -> JsonRepairResult:
    """Convenience function to structurally repair truncated or lightly malformed JSON."""
    repairer = ToolCallJsonRepairer(
        require_object_root=require_object_root,
        strip_code_fences=strip_code_fences,
        complete_keywords=complete_keywords,
    )
    return repairer.repair(raw_json)


# ---------------------------------------------------------------------------
# MS016 -- CompositeSchemaMerger
# ---------------------------------------------------------------------------

_VALID_MERGE_STRATEGIES = frozenset({
    "prefix_colliding",
    "prefix_all",
    "merge_identical",
    "first_wins",
})


def _safe_server_prefix(server_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(server_id))
    return safe.strip("_") or "srv"


def _canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tool_equivalence_payload(tool: dict[str, Any]) -> dict[str, Any]:
    """Canonical payload excluding tool name so cross-server identical tools compare cleanly."""
    schema = tool.get("inputSchema")
    annotations = tool.get("annotations")
    return {
        "description": str(tool.get("description", "")),
        "inputSchema": schema if isinstance(schema, dict) else {},
        "annotations": annotations if isinstance(annotations, dict) else {},
    }


def _rewrite_local_ref_pointers(node: Any, rename_map: dict[str, str]) -> Any:
    """Recursively rewrites local ``#/$defs/Old`` and ``#/definitions/Old`` pointers."""
    if not rename_map:
        return node
    if isinstance(node, dict):
        updated: dict[str, Any] = {}
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                rewritten_ref = v
                for prefix in ("#/$defs/", "#/definitions/"):
                    if v.startswith(prefix):
                        tail = v[len(prefix) :]
                        def_key, sep, rest = tail.partition("/")
                        if def_key in rename_map:
                            new_key = rename_map[def_key]
                            rewritten_ref = f"{prefix}{new_key}{sep}{rest}" if sep else f"{prefix}{new_key}"
                        break
                updated[k] = rewritten_ref
            elif k in ("$defs", "definitions") and isinstance(v, dict):
                new_defs: dict[str, Any] = {}
                for def_name, def_schema in v.items():
                    mapped_name = rename_map.get(def_name, def_name)
                    new_defs[mapped_name] = _rewrite_local_ref_pointers(def_schema, rename_map)
                updated[k] = new_defs
            else:
                updated[k] = _rewrite_local_ref_pointers(v, rename_map)
        return updated
    if isinstance(node, list):
        return [_rewrite_local_ref_pointers(elem, rename_map) for elem in node]
    return node


@dataclass
class CompositeMergeResult:
    """Output of merging multiple upstream MCP server tool manifests (MS016)."""

    tools: list[dict[str, Any]]
    routing_table: dict[str, dict[str, Any]]
    composite_defs: dict[str, dict[str, Any]]
    conflicts: list[dict[str, Any]]
    servers_merged: int
    tools_input: int
    tools_output: int
    tool_collisions: int
    deduplicated_tools: int
    shared_defs_deduped: int
    def_conflicts_resolved: int
    tokens_before: int
    tokens_after: int

    @property
    def has_conflicts(self) -> bool:
        return any(c.get("level") in ("warning", "error") for c in self.conflicts)

    @property
    def reduction_pct(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        saved = max(0, self.tokens_before - self.tokens_after)
        return round((saved / self.tokens_before) * 100.0, 1)

    def resolve_call(self, exposed_tool_name: str) -> tuple[str, str] | None:
        """Resolve an exposed composite tool name back to ``(server_id, upstream_name)``."""
        entry = self.routing_table.get(exposed_tool_name)
        if entry is None:
            return None
        return (str(entry["server_id"]), str(entry["upstream_name"]))

    def to_audit_report(self) -> dict[str, Any]:
        """Return a report dict directly consumable by ``build_audit_sarif``."""
        blocking = sum(1 for c in self.conflicts if c.get("level") in ("warning", "error"))
        return {
            "servers_merged": self.servers_merged,
            "tools_input": self.tools_input,
            "tools_output": self.tools_output,
            "tool_collisions": self.tool_collisions,
            "deduplicated_tools": self.deduplicated_tools,
            "shared_defs_deduped": self.shared_defs_deduped,
            "def_conflicts_resolved": self.def_conflicts_resolved,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": self.reduction_pct,
            "security_findings": blocking,
            "findings": list(self.conflicts),
            "routing_table": dict(self.routing_table),
            "composite_defs": dict(self.composite_defs),
            "tools": list(self.tools),
        }

    def to_sarif_results(self) -> list[dict[str, Any]]:
        """Convert merge conflicts/notes into OASIS SARIF v2.1.0 result objects."""
        results: list[dict[str, Any]] = []
        for c in self.conflicts:
            loc_name = str(c.get("tool") or "composite_manifest")
            props: dict[str, Any] = {"kind": c.get("kind", "merge_conflict")}
            if "servers" in c:
                props["servers"] = c["servers"]
            if "resolved_names" in c:
                props["resolved_names"] = c["resolved_names"]
            if "def_name" in c:
                props["def_name"] = c["def_name"]
            results.append(
                {
                    "ruleId": _COMPOSITE_MERGE_RULE_ID,
                    "level": c.get("level", "warning"),
                    "message": {"text": c["message"]},
                    "locations": [
                        {
                            "logicalLocations": [
                                {"name": loc_name, "kind": "function"}
                            ]
                        }
                    ],
                    "properties": props,
                }
            )
        return results


class CompositeSchemaMerger:
    """MS016: Merges multiple MCP ``tools/list`` manifests into a unified, collision-free catalog.

    Resolves both cross-server tool-name collisions and conflicting ``$defs``/``definitions``
    sub-schemas while building an O(1) reverse routing table for ``tools/call`` dispatch.
    """

    def __init__(
        self,
        *,
        strategy: str = "prefix_colliding",
        separator: str = "__",
        slim_schemas: bool = False,
        hoist_shared_defs: bool = True,
        sort_tools: bool = True,
    ) -> None:
        if strategy not in _VALID_MERGE_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {strategy!r}; expected one of {sorted(_VALID_MERGE_STRATEGIES)}"
            )
        if not separator:
            raise ValueError("separator must be a non-empty string")
        self.strategy = strategy
        self.separator = separator
        self.slim_schemas = slim_schemas
        self.hoist_shared_defs = hoist_shared_defs
        self.sort_tools = sort_tools

    def merge(
        self,
        server_manifests: dict[str, list[dict[str, Any]]],
        *,
        schema_slimmer: Any | None = None,
    ) -> CompositeMergeResult:
        """Merge ``{server_id: [tool, ...]}`` into a single composite manifest."""
        from .slimmer import SchemaSlimmer, estimate_tokens  # noqa: PLC0415

        all_raw_tools: list[dict[str, Any]] = []
        server_entries: list[tuple[str, str, dict[str, Any]]] = []
        for server_id, tools in server_manifests.items():
            if not isinstance(tools, list):
                continue
            safe_srv = _safe_server_prefix(server_id)
            for t in tools:
                if not isinstance(t, dict):
                    continue
                t_name = str(t.get("name", "")).strip()
                if not t_name:
                    continue
                cloned = copy.deepcopy(t)
                all_raw_tools.append(cloned)
                server_entries.append((str(server_id), safe_srv, cloned))

        tokens_before = estimate_tokens(all_raw_tools) if all_raw_tools else 0
        conflicts: list[dict[str, Any]] = []

        # Phase 1: Analyze and disambiguate cross-server $defs / definitions
        composite_defs: dict[str, dict[str, Any]] = {}
        shared_defs_deduped = 0
        def_conflicts_resolved = 0

        if self.hoist_shared_defs:
            # Map def_name -> {sha256 -> (schema_dict, list[(server_id, safe_srv, tool_ref)])}
            def_variants: dict[str, dict[str, tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]]] = {}
            for server_id, safe_srv, tool in server_entries:
                in_schema = tool.get("inputSchema")
                if not isinstance(in_schema, dict):
                    continue
                for defs_key in ("$defs", "definitions"):
                    defs_obj = in_schema.get(defs_key)
                    if not isinstance(defs_obj, dict):
                        continue
                    for def_name, def_schema in defs_obj.items():
                        if not isinstance(def_name, str) or not isinstance(def_schema, dict):
                            continue
                        digest = _canonical_json_sha256(def_schema)
                        variants = def_variants.setdefault(def_name, {})
                        if digest not in variants:
                            variants[digest] = (copy.deepcopy(def_schema), [])
                        variants[digest][1].append((server_id, safe_srv, tool))

            # Build rename maps per tool id() when same def_name has >= 2 distinct schemas
            per_tool_renames: dict[int, dict[str, str]] = {}
            for def_name in sorted(def_variants):
                variants = def_variants[def_name]
                total_usages = sum(len(usages) for _, usages in variants.values())
                if len(variants) == 1:
                    only_schema, _ = next(iter(variants.values()))
                    composite_defs[def_name] = only_schema
                    if total_usages > 1:
                        shared_defs_deduped += total_usages - 1
                else:
                    def_conflicts_resolved += 1
                    involved_servers = sorted({
                        srv for _, usages in variants.values() for srv, _, _ in usages
                    })
                    resolved_def_names: dict[str, str] = {}
                    used_def_keys: set[str] = set(composite_defs.keys())
                    for digest, (def_schema, usages) in sorted(variants.items()):
                        first_srv, first_safe, _ = usages[0]
                        candidate_name = f"{first_safe}_{def_name}"
                        if candidate_name in used_def_keys:
                            candidate_name = f"{first_safe}_{def_name}_{digest[:6]}"
                        used_def_keys.add(candidate_name)
                        composite_defs[candidate_name] = def_schema
                        resolved_def_names[first_srv] = candidate_name
                        if len(usages) > 1:
                            shared_defs_deduped += len(usages) - 1
                        for _, _, tool_ref in usages:
                            per_tool_renames.setdefault(id(tool_ref), {})[def_name] = candidate_name

                    conflicts.append({
                        "rule_id": _COMPOSITE_MERGE_RULE_ID,
                        "level": "warning",
                        "tool": f"$defs/{def_name}",
                        "kind": "conflicting_shared_def",
                        "def_name": def_name,
                        "servers": involved_servers,
                        "resolved_names": resolved_def_names,
                        "message": (
                            f"Shared JSON Schema definition '{def_name}' has {len(variants)} "
                            f"incompatible schemas across servers {involved_servers}; "
                            f"disambiguated as {sorted(resolved_def_names.values())}."
                        ),
                    })

            for _, _, tool in server_entries:
                rename_map = per_tool_renames.get(id(tool))
                if rename_map and isinstance(tool.get("inputSchema"), dict):
                    tool["inputSchema"] = _rewrite_local_ref_pointers(
                        tool["inputSchema"], rename_map
                    )

        # Phase 2: Resolve tool-name collisions across servers
        by_tool_name: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for entry in server_entries:
            t_name = str(entry[2]["name"])
            by_tool_name.setdefault(t_name, []).append(entry)

        colliding_names = {
            name for name, group in by_tool_name.items() if len(group) >= 2
        }

        merged_tools: list[dict[str, Any]] = []
        routing_table: dict[str, dict[str, Any]] = {}
        used_exposed_names: set[str] = set()
        deduplicated_tools = 0

        def _alloc_unique_name(desired: str, fallback_srv: str) -> str:
            if desired not in used_exposed_names:
                used_exposed_names.add(desired)
                return desired
            cand = f"{fallback_srv}{self.separator}{desired}"
            idx = 2
            while cand in used_exposed_names:
                cand = f"{fallback_srv}_{idx}{self.separator}{desired}"
                idx += 1
            used_exposed_names.add(cand)
            return cand

        # Record collision diagnostics in deterministic name order
        handled_identical: set[str] = set()
        first_wins_winner: dict[str, int] = {}

        for t_name in sorted(colliding_names):
            group = by_tool_name[t_name]
            srv_list = [srv for srv, _, _ in group]
            digests = {_canonical_json_sha256(_tool_equivalence_payload(t)) for _, _, t in group}

            if self.strategy == "merge_identical" and len(digests) == 1:
                handled_identical.add(t_name)
                deduplicated_tools += len(group) - 1
                conflicts.append({
                    "rule_id": _COMPOSITE_MERGE_RULE_ID,
                    "level": "note",
                    "tool": t_name,
                    "kind": "deduplicated_identical_tool",
                    "servers": srv_list,
                    "message": (
                        f"Identical tool '{t_name}' exposed by {srv_list} was deduplicated "
                        f"into a single composite definition."
                    ),
                })
            elif self.strategy == "first_wins":
                first_wins_winner[t_name] = id(group[0][2])
                conflicts.append({
                    "rule_id": _COMPOSITE_MERGE_RULE_ID,
                    "level": "warning",
                    "tool": t_name,
                    "kind": "shadowed_tool_dropped",
                    "servers": srv_list,
                    "resolved_names": {group[0][0]: t_name},
                    "message": (
                        f"Tool '{t_name}' collides across {srv_list}; kept '{group[0][0]}' "
                        f"and dropped {len(group) - 1} shadowed definition(s) (first_wins)."
                    ),
                })
            else:
                resolved_map = {
                    srv: f"{safe_srv}{self.separator}{t_name}"
                    for srv, safe_srv, _ in group
                }
                conflicts.append({
                    "rule_id": _COMPOSITE_MERGE_RULE_ID,
                    "level": "warning",
                    "tool": t_name,
                    "kind": "tool_name_collision",
                    "servers": srv_list,
                    "resolved_names": resolved_map,
                    "message": (
                        f"Tool '{t_name}' collides across {srv_list}; rewritten to "
                        f"{sorted(resolved_map.values())}."
                    ),
                })

        emitted_identical: set[str] = set()
        for server_id, safe_srv, tool in server_entries:
            upstream_name = str(tool["name"])
            fp = _canonical_json_sha256(_tool_equivalence_payload(tool))

            if upstream_name in handled_identical:
                if upstream_name in emitted_identical:
                    routing_table[upstream_name]["alternate_servers"].append(server_id)
                    continue
                emitted_identical.add(upstream_name)
                exposed = _alloc_unique_name(upstream_name, safe_srv)
                out_tool = dict(tool)
                out_tool["name"] = exposed
                merged_tools.append(out_tool)
                routing_table[exposed] = {
                    "server_id": server_id,
                    "upstream_name": upstream_name,
                    "fingerprint": fp,
                    "alternate_servers": [],
                }
                continue

            if upstream_name in first_wins_winner:
                if id(tool) != first_wins_winner[upstream_name]:
                    continue
                exposed = _alloc_unique_name(upstream_name, safe_srv)
                out_tool = dict(tool)
                out_tool["name"] = exposed
                merged_tools.append(out_tool)
                routing_table[exposed] = {
                    "server_id": server_id,
                    "upstream_name": upstream_name,
                    "fingerprint": fp,
                    "alternate_servers": [
                        srv for srv, _, t_ref in by_tool_name[upstream_name] if id(t_ref) != id(tool)
                    ],
                }
                continue

            needs_prefix = self.strategy == "prefix_all" or (
                upstream_name in colliding_names
                and self.strategy in ("prefix_colliding", "merge_identical")
            )
            desired = f"{safe_srv}{self.separator}{upstream_name}" if needs_prefix else upstream_name
            exposed = _alloc_unique_name(desired, safe_srv)
            out_tool = dict(tool)
            out_tool["name"] = exposed
            merged_tools.append(out_tool)
            routing_table[exposed] = {
                "server_id": server_id,
                "upstream_name": upstream_name,
                "fingerprint": fp,
                "alternate_servers": [],
            }

        if self.slim_schemas:
            eff_slimmer = schema_slimmer if schema_slimmer is not None else SchemaSlimmer()
            merged_tools = eff_slimmer.slim_tools_list(merged_tools)
        elif self.sort_tools:
            merged_tools.sort(key=lambda item: str(item.get("name", "")))

        tokens_after = estimate_tokens(merged_tools) if merged_tools else 0

        return CompositeMergeResult(
            tools=merged_tools,
            routing_table=routing_table,
            composite_defs=composite_defs,
            conflicts=conflicts,
            servers_merged=len(server_manifests),
            tools_input=len(server_entries),
            tools_output=len(merged_tools),
            tool_collisions=len(colliding_names),
            deduplicated_tools=deduplicated_tools,
            shared_defs_deduped=shared_defs_deduped,
            def_conflicts_resolved=def_conflicts_resolved,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "rule": _COMPOSITE_MERGE_RULE_ID,
            "strategy": self.strategy,
            "separator": self.separator,
            "slim_schemas": self.slim_schemas,
            "hoist_shared_defs": self.hoist_shared_defs,
            "sort_tools": self.sort_tools,
        }


def merge_server_manifests(
    server_manifests: dict[str, list[dict[str, Any]]],
    *,
    strategy: str = "prefix_colliding",
    separator: str = "__",
    slim_schemas: bool = False,
    hoist_shared_defs: bool = True,
) -> CompositeMergeResult:
    """Convenience function to merge multiple MCP ``tools/list`` server manifests."""
    merger = CompositeSchemaMerger(
        strategy=strategy,
        separator=separator,
        slim_schemas=slim_schemas,
        hoist_shared_defs=hoist_shared_defs,
    )
    return merger.merge(server_manifests)
