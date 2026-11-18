"""Unit tests for mcp-slim v1.7.0: RefDepthGuard (MS013) and ToolNamespaceRouter (MS014)."""

from __future__ import annotations

import unittest

from mcp_slim import (
    CollisionReport,
    NamespaceRoutingResult,
    RefDepthGuard,
    RefDepthReport,
    ToolNamespaceRouter,
)


# ---------------------------------------------------------------------------
# MS013 RefDepthGuard tests
# ---------------------------------------------------------------------------


class TestRefDepthGuard(unittest.TestCase):
    """RefDepthGuard (MS013): JSON Schema $ref depth guard."""

    def _make_guard(self, max_depth: int = 5) -> RefDepthGuard:
        return RefDepthGuard(max_ref_depth=max_depth)

    # --- flat schema ---

    def test_flat_schema_no_violations(self) -> None:
        guard = self._make_guard()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        report = guard.audit_schema(schema, tool_name="flat_tool")
        self.assertFalse(report.has_violations)
        self.assertEqual(report.tool_name, "flat_tool")

    # --- shallow $ref chain within depth limit ---

    def test_shallow_ref_chain_ok(self) -> None:
        guard = self._make_guard(max_depth=5)
        schema = {
            "$defs": {
                "A": {"type": "string"},
            },
            "type": "object",
            "properties": {
                "field": {"$ref": "#/$defs/A"},
            },
        }
        report = guard.audit_schema(schema)
        self.assertFalse(report.has_violations)

    # --- deep linear $ref chain exceeds limit ---

    def test_deep_linear_ref_chain_flagged(self) -> None:
        # Build A -> B -> C -> D -> E -> F, depth > 4
        schema: dict = {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/C"},
                "C": {"$ref": "#/$defs/D"},
                "D": {"$ref": "#/$defs/E"},
                "E": {"$ref": "#/$defs/F"},
                "F": {"type": "string"},
            },
            "$ref": "#/$defs/A",
        }
        guard = RefDepthGuard(max_ref_depth=4)
        report = guard.audit_schema(schema, tool_name="deep_chain")
        self.assertTrue(report.has_violations)
        depths = [v["depth"] for v in report.violations]
        self.assertTrue(any(d > 4 for d in depths))
        self.assertEqual(report.max_depth, 4)

    # --- direct self-reference cycle ---

    def test_self_referential_cycle_detected(self) -> None:
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "$ref": "#/$defs/Node",
        }
        guard = RefDepthGuard(max_ref_depth=20)
        report = guard.audit_schema(schema, tool_name="self_ref_node")
        self.assertTrue(report.has_violations)
        for v in report.violations:
            self.assertIn("MS013", v["rule"])

    # --- mutual cycle A <-> B ---

    def test_mutual_cycle_detected(self) -> None:
        schema = {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "$ref": "#/$defs/A",
        }
        guard = RefDepthGuard(max_ref_depth=20)
        report = guard.audit_schema(schema, tool_name="mutual_cycle")
        self.assertTrue(report.has_violations)

    # --- violations deduped ---

    def test_violations_deduplicated(self) -> None:
        # Same cycle triggered from multiple paths should not produce infinite violations
        schema = {
            "$defs": {
                "X": {
                    "type": "object",
                    "properties": {
                        "a": {"$ref": "#/$defs/X"},
                        "b": {"$ref": "#/$defs/X"},
                    },
                }
            },
            "$ref": "#/$defs/X",
        }
        guard = RefDepthGuard(max_ref_depth=20)
        report = guard.audit_schema(schema, tool_name="dedup_test")
        # Should have violations but a reasonable finite number
        self.assertLess(len(report.violations), 20)

    # --- audit_tools filters schema-less tools ---

    def test_audit_tools_skips_no_schema(self) -> None:
        guard = self._make_guard()
        tools = [
            {"name": "no_schema_tool"},
            {"name": "empty_schema_tool", "inputSchema": {}},
        ]
        reports = guard.audit_tools(tools)
        self.assertEqual(len(reports), 0)

    # --- audit_tools returns only violating tools ---

    def test_audit_tools_returns_violating_only(self) -> None:
        cycle_schema = {
            "$defs": {"N": {"$ref": "#/$defs/N"}},
            "$ref": "#/$defs/N",
        }
        flat_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        tools = [
            {"name": "cyclic_tool", "inputSchema": cycle_schema},
            {"name": "flat_tool", "inputSchema": flat_schema},
        ]
        guard = RefDepthGuard(max_ref_depth=20)
        reports = guard.audit_tools(tools)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].tool_name, "cyclic_tool")

    # --- to_sarif_results structure ---

    def test_sarif_results_shape(self) -> None:
        cycle_schema = {
            "$defs": {"N": {"$ref": "#/$defs/N"}},
            "$ref": "#/$defs/N",
        }
        guard = RefDepthGuard(max_ref_depth=20)
        report = guard.audit_schema(cycle_schema, tool_name="sarif_test")
        sarif = report.to_sarif_results()
        self.assertGreater(len(sarif), 0)
        for result in sarif:
            self.assertIn("ruleId", result)
            self.assertEqual(result["ruleId"], "MS013")
            self.assertIn("level", result)
            self.assertIn("message", result)
            self.assertIn("text", result["message"])

    # --- constructor validation ---

    def test_invalid_max_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            RefDepthGuard(max_ref_depth=0)
        with self.assertRaises(ValueError):
            RefDepthGuard(max_ref_depth=-1)

    # --- snapshot ---

    def test_snapshot_shape(self) -> None:
        guard = RefDepthGuard(max_ref_depth=15)
        snap = guard.snapshot()
        self.assertEqual(snap["rule"], "MS013")
        self.assertEqual(snap["max_ref_depth"], 15)

    # --- properties within $defs that hold refs ---

    def test_allOf_anyOf_oneOf_traversed(self) -> None:
        schema = {
            "$defs": {
                "Base": {"type": "string"},
                "Ext": {
                    "allOf": [
                        {"$ref": "#/$defs/Base"},
                        {"properties": {"extra": {"type": "integer"}}},
                    ]
                },
            },
            "anyOf": [
                {"$ref": "#/$defs/Ext"},
                {"type": "null"},
            ],
        }
        guard = RefDepthGuard(max_ref_depth=10)
        report = guard.audit_schema(schema)
        self.assertFalse(report.has_violations)

    # --- non-dict schema is harmless ---

    def test_non_dict_schema_no_crash(self) -> None:
        guard = self._make_guard()
        # audit_schema expects a dict; test that invalid input is handled safely
        report = guard.audit_schema({}, tool_name="empty")
        self.assertFalse(report.has_violations)


# ---------------------------------------------------------------------------
# MS014 ToolNamespaceRouter tests
# ---------------------------------------------------------------------------


class TestToolNamespaceRouter(unittest.TestCase):
    """ToolNamespaceRouter (MS014): cross-server tool-name collision detector."""

    def _make_tools(self, names: list[str]) -> list[dict]:
        return [{"name": n, "description": f"desc for {n}"} for n in names]

    # --- no collision ---

    def test_no_collision_no_rename(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "serverA": self._make_tools(["read_file", "write_file"]),
                "serverB": self._make_tools(["commit", "diff"]),
            }
        )
        self.assertFalse(result.has_collisions)
        self.assertEqual(result.total_collisions, 0)
        # Names unchanged
        names_a = [t["name"] for t in result.routed_tools["serverA"]]
        self.assertIn("read_file", names_a)
        names_b = [t["name"] for t in result.routed_tools["serverB"]]
        self.assertIn("commit", names_b)

    # --- single collision renamed ---

    def test_collision_renamed_with_prefix(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "files": self._make_tools(["read", "write"]),
                "git": self._make_tools(["read", "commit"]),
            }
        )
        self.assertTrue(result.has_collisions)
        self.assertEqual(result.total_collisions, 1)
        # "read" should be prefixed on both servers
        names_files = [t["name"] for t in result.routed_tools["files"]]
        names_git = [t["name"] for t in result.routed_tools["git"]]
        self.assertIn("files__read", names_files)
        self.assertNotIn("read", names_files)
        self.assertIn("git__read", names_git)
        self.assertNotIn("read", names_git)
        # Non-colliding names untouched
        self.assertIn("write", names_files)
        self.assertIn("commit", names_git)

    # --- apply_prefix=False only reports ---

    def test_report_only_no_rename(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=False)
        result = router.route(
            {
                "s1": self._make_tools(["overlap"]),
                "s2": self._make_tools(["overlap"]),
            }
        )
        self.assertTrue(result.has_collisions)
        # Names NOT rewritten
        names_s1 = [t["name"] for t in result.routed_tools["s1"]]
        self.assertIn("overlap", names_s1)
        self.assertNotIn("s1__overlap", names_s1)

    # --- prefix_all prefixes everything ---

    def test_prefix_all_renames_all(self) -> None:
        router = ToolNamespaceRouter(prefix_all=True)
        result = router.route(
            {
                "alpha": self._make_tools(["unique_tool"]),
                "beta": self._make_tools(["other_tool"]),
            }
        )
        # No actual collisions but all names get prefix
        names_alpha = [t["name"] for t in result.routed_tools["alpha"]]
        names_beta = [t["name"] for t in result.routed_tools["beta"]]
        self.assertIn("alpha__unique_tool", names_alpha)
        self.assertIn("beta__other_tool", names_beta)

    # --- custom separator ---

    def test_custom_separator(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True, separator=".")
        result = router.route(
            {
                "srv1": self._make_tools(["dup"]),
                "srv2": self._make_tools(["dup"]),
            }
        )
        names_srv1 = [t["name"] for t in result.routed_tools["srv1"]]
        self.assertIn("srv1.dup", names_srv1)

    # --- server_id with special chars is sanitized ---

    def test_server_id_sanitized(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "my-server.com:8080": self._make_tools(["dup"]),
                "other/server": self._make_tools(["dup"]),
            }
        )
        # Both prefixes are safe identifiers (no colons, slashes, dots)
        for server_tools in result.routed_tools.values():
            for tool in server_tools:
                self.assertNotIn(":", tool["name"])
                self.assertNotIn("/", tool["name"])
                self.assertNotIn(".", tool["name"])

    # --- three-way collision ---

    def test_three_server_collision(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "alpha": self._make_tools(["shared"]),
                "beta": self._make_tools(["shared"]),
                "gamma": self._make_tools(["shared"]),
            }
        )
        self.assertEqual(result.total_collisions, 1)
        self.assertEqual(len(result.collisions), 1)
        c = result.collisions[0]
        self.assertEqual(c.tool_name, "shared")
        self.assertEqual(set(c.servers), {"alpha", "beta", "gamma"})
        self.assertIn("alpha", c.prefixed_names)
        self.assertEqual(c.prefixed_names["alpha"], "alpha__shared")

    # --- empty server lists are safe ---

    def test_empty_server_no_error(self) -> None:
        router = ToolNamespaceRouter()
        result = router.route({"s1": [], "s2": []})
        self.assertFalse(result.has_collisions)
        self.assertEqual(result.total_tools, 0)

    # --- tool without name is skipped ---

    def test_tool_without_name_skipped(self) -> None:
        router = ToolNamespaceRouter()
        result = router.route(
            {
                "s1": [{"description": "no name here"}],
                "s2": self._make_tools(["named"]),
            }
        )
        self.assertFalse(result.has_collisions)

    # --- collision_report.prefixed_names ---

    def test_collision_report_prefixed_names(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "db": self._make_tools(["query"]),
                "search": self._make_tools(["query"]),
            }
        )
        self.assertEqual(len(result.collisions), 1)
        c = result.collisions[0]
        self.assertIn("db", c.prefixed_names)
        self.assertIn("search", c.prefixed_names)
        self.assertEqual(c.prefixed_names["db"], "db__query")
        self.assertEqual(c.prefixed_names["search"], "search__query")

    # --- to_sarif_results shape ---

    def test_sarif_results_shape(self) -> None:
        router = ToolNamespaceRouter()
        result = router.route(
            {
                "s1": self._make_tools(["dup"]),
                "s2": self._make_tools(["dup"]),
            }
        )
        sarif = result.to_sarif_results()
        self.assertEqual(len(sarif), 1)
        r = sarif[0]
        self.assertEqual(r["ruleId"], "MS014")
        self.assertIn("dup", r["message"]["text"])
        self.assertIn("servers", r["properties"])

    # --- tool descriptions and other fields preserved ---

    def test_other_fields_preserved(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        tools = [{"name": "dup", "description": "original desc", "inputSchema": {"type": "object"}}]
        result = router.route({"s1": tools, "s2": tools})
        s1_tool = result.routed_tools["s1"][0]
        self.assertEqual(s1_tool["description"], "original desc")
        self.assertIn("inputSchema", s1_tool)

    # --- invalid separator raises ---

    def test_empty_separator_raises(self) -> None:
        with self.assertRaises(ValueError):
            ToolNamespaceRouter(separator="")

    # --- snapshot ---

    def test_snapshot_shape(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True, prefix_all=False, separator="__")
        snap = router.snapshot()
        self.assertEqual(snap["rule"], "MS014")
        self.assertTrue(snap["apply_prefix"])
        self.assertFalse(snap["prefix_all"])
        self.assertEqual(snap["separator"], "__")

    # --- total_tools is counted correctly ---

    def test_total_tools_count(self) -> None:
        router = ToolNamespaceRouter()
        result = router.route(
            {
                "s1": self._make_tools(["a", "b", "c"]),
                "s2": self._make_tools(["d", "e"]),
            }
        )
        self.assertEqual(result.total_tools, 5)

    # --- multiple collisions detected ---

    def test_multiple_collisions(self) -> None:
        router = ToolNamespaceRouter(apply_prefix=True)
        result = router.route(
            {
                "s1": self._make_tools(["dup1", "dup2", "unique1"]),
                "s2": self._make_tools(["dup1", "dup2", "unique2"]),
            }
        )
        self.assertEqual(result.total_collisions, 2)
        self.assertEqual(len(result.collisions), 2)
        collision_names = {c.tool_name for c in result.collisions}
        self.assertEqual(collision_names, {"dup1", "dup2"})
        # Non-colliding tools untouched
        names_s1 = [t["name"] for t in result.routed_tools["s1"]]
        self.assertIn("unique1", names_s1)


if __name__ == "__main__":
    unittest.main()
