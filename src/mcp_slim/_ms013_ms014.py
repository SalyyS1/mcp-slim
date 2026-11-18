"""MS013: JSON Schema $ref depth guard.
MS014: Cross-server tool-name collision detector with namespace prefixing router.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# MS013 -- RefDepthGuard
# ---------------------------------------------------------------------------

_REF_DEPTH_RULE_ID = "MS013"
_TOOL_NS_RULE_ID = "MS014"


def _walk_ref_depth(
    node: Any,
    root: dict[str, Any],
    visited: frozenset[str],
    depth: int,
    max_depth: int,
    violations: list[dict[str, Any]],
    path: str,
) -> None:
    """Recursively walk a JSON Schema node, tracking $ref resolution depth."""
    if not isinstance(node, dict):
        return
    if depth > max_depth:
        violations.append(
            {
                "rule": _REF_DEPTH_RULE_ID,
                "path": path,
                "depth": depth,
                "message": (
                    f"$ref chain at '{path}' reached depth {depth}, "
                    f"exceeding max_ref_depth={max_depth}. "
                    "Recursive schema may cause unbounded LLM token inflation."
                ),
            }
        )
        return

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        if ref in visited:
            # Already seen -- cycle detected at any depth >= 1
            violations.append(
                {
                    "rule": _REF_DEPTH_RULE_ID,
                    "path": path,
                    "depth": depth,
                    "message": (
                        f"Cyclic $ref '{ref}' revisited at depth {depth} "
                        f"(path: {path}). Self-referential schema risks "
                        "token explosion without a depth cap."
                    ),
                }
            )
            return
        # Resolve pointer
        target = _resolve_simple_pointer(ref, root)
        if target is not None:
            _walk_ref_depth(
                target,
                root,
                visited | {ref},
                depth + 1,
                max_depth,
                violations,
                path + f" -> {ref}",
            )
        return

    # Recurse into sub-schemas
    for key in ("items", "additionalProperties", "not"):
        child = node.get(key)
        if isinstance(child, dict):
            _walk_ref_depth(
                child, root, visited, depth, max_depth, violations, f"{path}/{key}"
            )
    for key in ("properties", "$defs", "definitions"):
        mapping = node.get(key)
        if isinstance(mapping, dict):
            for prop_name, prop_node in mapping.items():
                _walk_ref_depth(
                    prop_node,
                    root,
                    visited,
                    depth,
                    max_depth,
                    violations,
                    f"{path}/{key}/{prop_name}",
                )
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        lst = node.get(key)
        if isinstance(lst, list):
            for idx, elem in enumerate(lst):
                _walk_ref_depth(
                    elem,
                    root,
                    visited,
                    depth,
                    max_depth,
                    violations,
                    f"{path}/{key}/{idx}",
                )


def _resolve_simple_pointer(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a simple internal JSON Pointer (#/...) without full RFC 6901 decoding."""
    if not ref.startswith("#/"):
        return None
    segments = ref[2:].split("/")
    cur: Any = root
    for seg in segments:
        seg = seg.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur if isinstance(cur, dict) else None


@dataclass
class RefDepthReport:
    """Result of a RefDepthGuard audit on a single tool schema."""

    tool_name: str
    max_depth: int
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_sarif_results(self) -> list[dict[str, Any]]:
        """Convert violations to SARIF v2.1.0 result objects."""
        results = []
        for v in self.violations:
            results.append(
                {
                    "ruleId": v["rule"],
                    "level": "warning",
                    "message": {"text": v["message"]},
                    "locations": [
                        {
                            "logicalLocations": [
                                {
                                    "name": self.tool_name,
                                    "kind": "function",
                                    "decoratedName": v["path"],
                                }
                            ]
                        }
                    ],
                    "properties": {"depth": v["depth"]},
                }
            )
        return results


class RefDepthGuard:
    """MS013: Audits JSON Schema tool input schemas for excessive $ref recursion depth.

    Detects both cyclic $ref chains and deeply-nested linear chains that would
    cause token explosion when inlined by SchemaSlimmer or the LLM client.

    Args:
        max_ref_depth: Maximum allowed $ref resolution depth (default 20).
            A chain A -> B -> C -> ... of length > max_ref_depth is flagged.

    Example::

        guard = RefDepthGuard(max_ref_depth=10)
        reports = guard.audit_tools(tools_list)
        for report in reports:
            if report.has_violations:
                print(report.tool_name, report.violations)
    """

    def __init__(self, max_ref_depth: int = 20) -> None:
        if max_ref_depth < 1:
            raise ValueError("max_ref_depth must be >= 1")
        self.max_ref_depth = max_ref_depth

    def audit_schema(
        self, schema: dict[str, Any], tool_name: str = "<anonymous>"
    ) -> RefDepthReport:
        """Audit a single JSON Schema object for $ref depth violations.

        Args:
            schema: A JSON Schema dict (the ``inputSchema`` of an MCP tool).
            tool_name: Label used in the returned report.

        Returns:
            A RefDepthReport listing all violations found.
        """
        violations: list[dict[str, Any]] = []
        _walk_ref_depth(
            schema,
            root=schema,
            visited=frozenset(),
            depth=0,
            max_depth=self.max_ref_depth,
            violations=violations,
            path="#",
        )
        # Deduplicate by (path, depth) in case multiple traversal paths hit the same ref
        seen: set[tuple[str, int]] = set()
        unique: list[dict[str, Any]] = []
        for v in violations:
            key = (v["path"], v["depth"])
            if key not in seen:
                seen.add(key)
                unique.append(v)
        return RefDepthReport(tool_name=tool_name, max_depth=self.max_ref_depth, violations=unique)

    def audit_tools(self, tools: list[dict[str, Any]]) -> list[RefDepthReport]:
        """Audit all tools in an MCP tools/list response.

        Args:
            tools: List of tool dicts, each with ``name`` and ``inputSchema``.

        Returns:
            One RefDepthReport per tool that has violations (empty tools skipped).
        """
        reports = []
        for tool in tools:
            name = tool.get("name", "<unknown>")
            schema = tool.get("inputSchema") or tool.get("input_schema")
            if not isinstance(schema, dict):
                continue
            report = self.audit_schema(schema, tool_name=name)
            if report.has_violations:
                reports.append(report)
        return reports

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable configuration snapshot."""
        return {"rule": _REF_DEPTH_RULE_ID, "max_ref_depth": self.max_ref_depth}


# ---------------------------------------------------------------------------
# MS014 -- ToolNamespaceRouter
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DOUBLE_UNDERSCORE = "__"


@dataclass
class CollisionReport:
    """Records a tool-name collision found across two or more servers."""

    tool_name: str
    servers: list[str]
    prefixed_names: dict[str, str]  # server_id -> prefixed tool name


@dataclass
class NamespaceRoutingResult:
    """Result of ToolNamespaceRouter.route()."""

    collisions: list[CollisionReport]
    routed_tools: dict[str, list[dict[str, Any]]]  # server_id -> rewritten tool list
    total_tools: int
    total_collisions: int

    @property
    def has_collisions(self) -> bool:
        return self.total_collisions > 0

    def to_sarif_results(self) -> list[dict[str, Any]]:
        """Convert collisions to SARIF v2.1.0 result objects."""
        results = []
        for c in self.collisions:
            prefixed = ", ".join(
                f"{srv} -> {name}" for srv, name in sorted(c.prefixed_names.items())
            )
            results.append(
                {
                    "ruleId": _TOOL_NS_RULE_ID,
                    "level": "warning",
                    "message": {
                        "text": (
                            f"Tool name '{c.tool_name}' collides across servers "
                            f"{c.servers}. Prefixed as: {prefixed}."
                        )
                    },
                    "locations": [
                        {
                            "logicalLocations": [
                                {"name": c.tool_name, "kind": "function"}
                            ]
                        }
                    ],
                    "properties": {"servers": c.servers, "prefixed_names": c.prefixed_names},
                }
            )
        return results


class ToolNamespaceRouter:
    """MS014: Detects cross-server tool-name collisions and rewrites tool names.

    When an MCP proxy aggregates tools from multiple upstream servers, identical
    tool names from different servers cause silent ambiguity. This router:
    1. Detects collisions (same tool name appears in >= 2 servers).
    2. Optionally rewrites every colliding tool's name to ``{server_id}__{tool_name}``.
    3. Reports all collisions as a SARIF-compatible audit result.

    Args:
        apply_prefix: If True (default), rewrite conflicting tool names with the
            ``{server_id}__{name}`` prefix. If False, only report collisions.
        prefix_all: If True, prefix ALL tool names from every server regardless
            of collision. Useful when a strict namespace isolation is desired.
        separator: String inserted between server_id and tool_name. Default ``"__"``.

    Example::

        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route({
            "fileserver": [{"name": "read_file", ...}],
            "gitserver": [{"name": "read_file", ...}, {"name": "commit", ...}],
        })
        # result.routed_tools["fileserver"][0]["name"] == "fileserver__read_file"
        # result.routed_tools["gitserver"][0]["name"] == "gitserver__read_file"
        # result.routed_tools["gitserver"][1]["name"] == "commit"  (no collision)
    """

    def __init__(
        self,
        apply_prefix: bool = True,
        prefix_all: bool = False,
        separator: str = _DOUBLE_UNDERSCORE,
    ) -> None:
        if not separator:
            raise ValueError("separator must be a non-empty string")
        self.apply_prefix = apply_prefix
        self.prefix_all = prefix_all
        self.separator = separator

    def _make_safe_server_id(self, server_id: str) -> str:
        """Sanitize a server_id so it is safe as a tool-name prefix component."""
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", server_id)
        return safe.strip("_") or "srv"

    def route(
        self, server_tools: dict[str, list[dict[str, Any]]]
    ) -> NamespaceRoutingResult:
        """Detect collisions and optionally rewrite tool names.

        Args:
            server_tools: Mapping of server_id -> list of tool dicts.
                Each tool dict must have at minimum a ``"name"`` key.

        Returns:
            A NamespaceRoutingResult with collision reports and rewritten tool lists.
        """
        # 1. Tally tool names -> servers
        name_to_servers: dict[str, list[str]] = {}
        for server_id, tools in server_tools.items():
            for tool in tools:
                tool_name = tool.get("name", "")
                if not tool_name:
                    continue
                name_to_servers.setdefault(tool_name, []).append(server_id)

        colliding_names: set[str] = {
            name for name, servers in name_to_servers.items() if len(servers) >= 2
        }

        # 2. Build CollisionReport objects
        collision_reports: list[CollisionReport] = []
        for name, servers in sorted(name_to_servers.items()):
            if name not in colliding_names:
                continue
            prefixed: dict[str, str] = {}
            for srv in servers:
                safe_srv = self._make_safe_server_id(srv)
                prefixed[srv] = f"{safe_srv}{self.separator}{name}"
            collision_reports.append(
                CollisionReport(tool_name=name, servers=servers, prefixed_names=prefixed)
            )

        # 3. Rewrite tool lists
        routed: dict[str, list[dict[str, Any]]] = {}
        for server_id, tools in server_tools.items():
            safe_srv = self._make_safe_server_id(server_id)
            rewritten: list[dict[str, Any]] = []
            for tool in tools:
                tool_name = tool.get("name", "")
                is_colliding = tool_name in colliding_names
                should_prefix = self.prefix_all or (self.apply_prefix and is_colliding)
                if should_prefix and tool_name:
                    new_tool = dict(tool)
                    new_tool["name"] = f"{safe_srv}{self.separator}{tool_name}"
                    rewritten.append(new_tool)
                else:
                    rewritten.append(dict(tool))
            routed[server_id] = rewritten

        return NamespaceRoutingResult(
            collisions=collision_reports,
            routed_tools=routed,
            total_tools=sum(len(t) for t in server_tools.values()),
            total_collisions=len(colliding_names),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable configuration snapshot."""
        return {
            "rule": _TOOL_NS_RULE_ID,
            "apply_prefix": self.apply_prefix,
            "prefix_all": self.prefix_all,
            "separator": self.separator,
        }
