"""Unit and integration tests for mcp-slim v1.8.0:
- MS015: ToolCallJsonRepairer / streaming tool-call structural JSON repair
- MS016: CompositeSchemaMerger / multi-server composite schema merging and $defs resolution
- OASIS SARIF v2.1.0 report enhancements (ruleIndex, partialFingerprints, help, MS013 audit integration)
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import mcp_slim
from mcp_slim import (
    AUDIT_SARIF_RULES,
    CompositeMergeResult,
    CompositeSchemaMerger,
    JsonRepairResult,
    MCPProxy,
    SchemaSlimmer,
    SlimStats,
    ToolCallJsonRepairer,
    ToolCallStreamSanitizer,
    audit_tools_manifest,
    build_audit_sarif,
    handle_server_rpc,
    merge_server_manifests,
    repair_truncated_json,
)
from mcp_slim.cli import main, wrap_config


class TestToolCallJsonRepairerMS015(unittest.TestCase):
    """MS015: Deterministic lexical state-machine JSON repair for truncated tool calls."""

    def test_valid_json_fast_path_no_repair(self) -> None:
        res = repair_truncated_json('{"query": "mcp", "limit": 10, "active": true}')
        self.assertTrue(res.is_valid)
        self.assertFalse(res.repaired)
        self.assertEqual(res.repairs_applied, [])
        self.assertEqual(res.value, {"query": "mcp", "limit": 10, "active": True})
        self.assertIsNone(res.to_finding("search_repos"))
        self.assertEqual(res.to_sarif_results("search_repos"), [])

    def test_empty_input_behavior(self) -> None:
        res_obj = repair_truncated_json("   ", require_object_root=True)
        self.assertTrue(res_obj.is_valid)
        self.assertFalse(res_obj.repaired)
        self.assertEqual(res_obj.value, {})

        res_any = repair_truncated_json("   ", require_object_root=False)
        self.assertFalse(res_any.is_valid)
        self.assertIn("Empty JSON input", str(res_any.error))

    def test_truncated_string_value_and_nested_containers(self) -> None:
        raw = '{"repo": "SalyyS1/mcp-slim", "filters": [{"tag": "security", "note": "cut mid-tok'
        res = repair_truncated_json(raw)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.repaired)
        self.assertEqual(
            res.value,
            {
                "repo": "SalyyS1/mcp-slim",
                "filters": [{"tag": "security", "note": "cut mid-tok"}],
            },
        )
        self.assertIn("closed_string", res.repairs_applied)
        self.assertIn("closed_array", res.repairs_applied)
        self.assertIn("closed_object", res.repairs_applied)

    def test_truncated_object_key_and_dangling_colon_dropped(self) -> None:
        raw_key = '{"query": "mcp", "trunc_ke'
        res_key = repair_truncated_json(raw_key)
        self.assertTrue(res_key.is_valid)
        self.assertTrue(res_key.repaired)
        self.assertEqual(res_key.value, {"query": "mcp"})
        self.assertIn("dropped_dangling_property", res_key.repairs_applied)

        raw_colon = '{"query": "mcp", "limit": '
        res_colon = repair_truncated_json(raw_colon)
        self.assertTrue(res_colon.is_valid)
        self.assertEqual(res_colon.value, {"query": "mcp"})
        self.assertIn("dropped_dangling_property", res_colon.repairs_applied)

    def test_truncated_escape_at_eof_and_windows_paths_midstream(self) -> None:
        raw_eof_slash = '{"path": "C:\\\\work\\\\file\\'
        res_eof = repair_truncated_json(raw_eof_slash)
        self.assertTrue(res_eof.is_valid)
        self.assertEqual(res_eof.value, {"path": "C:\\work\\file"})
        self.assertIn("trimmed_dangling_escape", res_eof.repairs_applied)

        raw_eof_u = '{"text": "abc\\u00'
        res_u = repair_truncated_json(raw_eof_u)
        self.assertTrue(res_u.is_valid)
        self.assertEqual(res_u.value, {"text": "abc"})
        self.assertIn("trimmed_dangling_escape", res_u.repairs_applied)

        # Unescaped backslashes such as C:\dev and C:\users mid-string
        raw_win = '{"dir": "C:\\dev\\users", "home": "C:\\users\\alice"}'
        res_win = repair_truncated_json(raw_win)
        self.assertTrue(res_win.is_valid)
        self.assertTrue(res_win.repaired)
        self.assertEqual(
            res_win.value,
            {"dir": "C:\\dev\\users", "home": "C:\\users\\alice"},
        )
        self.assertIn("repaired_invalid_escape", res_win.repairs_applied)

    def test_raw_control_characters_inside_strings_escaped(self) -> None:
        raw = '{"script": "line1\nline2\tcol2\x07"}'
        res = repair_truncated_json(raw)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.repaired)
        self.assertEqual(res.value, {"script": "line1\nline2\tcol2\x07"})
        self.assertIn("escaped_control_chars", res.repairs_applied)

    def test_truncated_numbers_and_keyword_literals(self) -> None:
        res_kw = repair_truncated_json('{"enabled": tru, "deleted": fal, "extra": nul')
        self.assertTrue(res_kw.is_valid)
        self.assertEqual(
            res_kw.value, {"enabled": True, "deleted": False, "extra": None}
        )
        self.assertIn("completed_keyword_literal", res_kw.repairs_applied)

        # With complete_keywords=False, truncated keyword at EOF drops the dangling property
        res_no_kw = repair_truncated_json(
            '{"keep": 1, "enabled": tru', complete_keywords=False
        )
        self.assertTrue(res_no_kw.is_valid)
        self.assertEqual(res_no_kw.value, {"keep": 1})
        self.assertIn("dropped_dangling_property", res_no_kw.repairs_applied)

        res_num = repair_truncated_json('{"a": 42., "b": -0., "c": 1.5e-')
        self.assertTrue(res_num.is_valid)
        self.assertEqual(res_num.value, {"a": 42, "b": 0, "c": 1.5})
        self.assertIn("repaired_truncated_number", res_num.repairs_applied)

        # Bare minus sign at EOF drops dangling property rather than inventing 0
        res_minus = repair_truncated_json('{"ok": 7, "delta": -')
        self.assertTrue(res_minus.is_valid)
        self.assertEqual(res_minus.value, {"ok": 7})

    def test_trailing_and_duplicate_commas_and_markdown_fences(self) -> None:
        raw = '```json\n{"items": [1, 2, ], "meta": {"k": "v",, }, }\n```'
        res = repair_truncated_json(raw)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.repaired)
        self.assertEqual(res.value, {"items": [1, 2], "meta": {"k": "v"}})
        self.assertIn("stripped_markdown_fence", res.repairs_applied)
        self.assertIn("removed_trailing_comma", res.repairs_applied)

    def test_non_object_root_and_unrepairable_sarif_reporting(self) -> None:
        res_arr = repair_truncated_json("[1, 2, 3]", require_object_root=True)
        self.assertFalse(res_arr.is_valid)
        self.assertIn("JSON object", str(res_arr.error))

        res_arr_ok = repair_truncated_json("[1, 2, ", require_object_root=False)
        self.assertTrue(res_arr_ok.is_valid)
        self.assertEqual(res_arr_ok.value, [1, 2])

        res_bad = repair_truncated_json("{unquoted_key: 123}")
        self.assertFalse(res_bad.is_valid)
        finding = res_bad.to_finding("run_query")
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["rule_id"], "MS015")
        self.assertEqual(finding["level"], "error")
        sarif_items = res_bad.to_sarif_results("run_query")
        self.assertEqual(len(sarif_items), 1)
        self.assertEqual(sarif_items[0]["ruleId"], "MS015")

        repairer_small = ToolCallJsonRepairer(max_input_chars=64)
        too_big = repairer_small.repair('{"a": "' + ("x" * 100) + '"}')
        self.assertFalse(too_big.is_valid)
        self.assertIn("max_input_chars", str(too_big.error))
        with self.assertRaises(ValueError):
            ToolCallJsonRepairer(max_input_chars=16)

    def test_streaming_sanitizer_and_proxy_repair_integration(self) -> None:
        sanitizer = ToolCallStreamSanitizer(repair_truncated_json=True)
        stats = SlimStats()
        leaked_tok = "ghp_" + ("A" * 36)
        chunks = [
            '{"repo": "SalyyS1/mcp-slim", "token": "',
            leaked_tok[:12],
            leaked_tok[12:],
            '", "count": "25", "partial_key',
        ]
        for ch in chunks:
            sanitizer.push_delta("call-1", ch, tool_name="search_repos")

        slimmer = SchemaSlimmer()
        slimmer.slim_tools_list([
            {
                "name": "search_repos",
                "description": "Search repositories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "token": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            }
        ])
        final = sanitizer.finish_call(
            "call-1", schema_slimmer=slimmer, stats=stats
        )
        self.assertTrue(final["is_valid_json"])
        self.assertTrue(final["repaired"])
        self.assertEqual(stats.json_repairs, 1)
        self.assertEqual(sanitizer.json_repairs, 1)
        # Integer coerced AND split secret redacted!
        self.assertEqual(final["arguments"]["count"], 25)
        self.assertNotIn(leaked_tok, final["sanitized_json"])
        self.assertIn("REDACTED", final["arguments"]["token"])

        # MCPProxy --repair-truncated-json with truncated string arguments
        proxy = MCPProxy(
            schema_slimmer=slimmer,
            repair_truncated_json=True,
            quiet=True,
            no_stats=True,
        )
        raw_req = json.dumps({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "search_repos",
                "arguments": '{"repo": "SalyyS1/mcp-slim", "count": "12", "dangling": ',
            },
        })
        forwarded = proxy.process_client_message(raw_req + "\n")
        forwarded_obj = json.loads(forwarded)
        self.assertEqual(
            forwarded_obj["params"]["arguments"],
            {"repo": "SalyyS1/mcp-slim", "count": 12},
        )
        self.assertEqual(proxy.stats.json_repairs, 1)


class TestCompositeSchemaMergerMS016(unittest.TestCase):
    """MS016: Multi-server composite manifest merger, collision router, and $defs resolver."""

    def _sample_manifests(self) -> dict[str, list[dict]]:
        return {
            "git": [
                {
                    "name": "read_file",
                    "description": "Read file at revision from git tree.",
                    "inputSchema": {
                        "$defs": {
                            "RefSpec": {
                                "type": "object",
                                "properties": {"rev": {"type": "string"}},
                            },
                            "SharedPage": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                        },
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "ref": {"$ref": "#/$defs/RefSpec"},
                            "page": {"$ref": "#/$defs/SharedPage"},
                        },
                    },
                },
                {
                    "name": "ping",
                    "description": "Health check.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ],
            "filesystem": [
                {
                    "name": "read_file",
                    "description": "Read file from local disk.",
                    "inputSchema": {
                        "$defs": {
                            # Conflicting RefSpec definition vs git server!
                            "RefSpec": {
                                "type": "object",
                                "properties": {
                                    "inode": {"type": "integer"},
                                    "follow_symlinks": {"type": "boolean"},
                                },
                            },
                            # Byte-identical SharedPage definition!
                            "SharedPage": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                        },
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "ref": {"$ref": "#/$defs/RefSpec"},
                            "page": {"$ref": "#/$defs/SharedPage"},
                        },
                    },
                },
                {
                    "name": "ping",
                    "description": "Health check.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "list_dir",
                    "description": "List directory entries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}},
                    },
                },
            ],
        }

    def test_prefix_colliding_and_conflicting_defs_disambiguation(self) -> None:
        manifests = self._sample_manifests()
        res: CompositeMergeResult = merge_server_manifests(
            manifests, strategy="prefix_colliding"
        )
        self.assertEqual(res.servers_merged, 2)
        self.assertEqual(res.tools_input, 5)
        self.assertEqual(res.tools_output, 5)
        self.assertEqual(res.tool_collisions, 2)  # read_file and ping
        self.assertEqual(res.shared_defs_deduped, 1)  # SharedPage
        self.assertEqual(res.def_conflicts_resolved, 1)  # RefSpec
        self.assertTrue(res.has_conflicts)

        exposed_names = {t["name"] for t in res.tools}
        self.assertEqual(
            exposed_names,
            {
                "git__read_file",
                "filesystem__read_file",
                "git__ping",
                "filesystem__ping",
                "list_dir",
            },
        )
        self.assertEqual(res.resolve_call("git__read_file"), ("git", "read_file"))
        self.assertEqual(
            res.resolve_call("filesystem__read_file"), ("filesystem", "read_file")
        )
        self.assertEqual(res.resolve_call("list_dir"), ("filesystem", "list_dir"))
        self.assertIsNone(res.resolve_call("nonexistent"))

        # Verify conflicting $defs/RefSpec pointers were rewritten in each tool schema
        by_name = {t["name"]: t for t in res.tools}
        git_ref_ptr = by_name["git__read_file"]["inputSchema"]["properties"]["ref"]["$ref"]
        fs_ref_ptr = by_name["filesystem__read_file"]["inputSchema"]["properties"]["ref"]["$ref"]
        self.assertNotEqual(git_ref_ptr, fs_ref_ptr)
        self.assertEqual(git_ref_ptr, "#/$defs/git_RefSpec")
        self.assertEqual(fs_ref_ptr, "#/$defs/filesystem_RefSpec")
        # Identical SharedPage stayed unprefixed
        self.assertEqual(
            by_name["git__read_file"]["inputSchema"]["properties"]["page"]["$ref"],
            "#/$defs/SharedPage",
        )
        self.assertIn("SharedPage", res.composite_defs)
        self.assertIn("git_RefSpec", res.composite_defs)
        self.assertIn("filesystem_RefSpec", res.composite_defs)

    def test_merge_identical_prefix_all_first_wins_and_slim(self) -> None:
        manifests = self._sample_manifests()

        # merge_identical deduplicates identical `ping`, prefixes divergent `read_file`
        res_ident = merge_server_manifests(manifests, strategy="merge_identical")
        self.assertEqual(res_ident.deduplicated_tools, 1)
        self.assertEqual(res_ident.tools_output, 4)
        names_ident = {t["name"] for t in res_ident.tools}
        self.assertIn("ping", names_ident)
        self.assertIn("git__read_file", names_ident)
        self.assertIn("filesystem__read_file", names_ident)
        self.assertEqual(
            res_ident.routing_table["ping"]["alternate_servers"], ["filesystem"]
        )

        # first_wins keeps git's read_file & ping, drops filesystem's duplicates
        res_first = merge_server_manifests(manifests, strategy="first_wins")
        self.assertEqual(res_first.tools_output, 3)
        self.assertEqual(res_first.resolve_call("read_file"), ("git", "read_file"))

        # prefix_all namespaces every tool including list_dir, and slim_schemas compacts $defs
        res_all = merge_server_manifests(
            manifests, strategy="prefix_all", separator=".", slim_schemas=True
        )
        self.assertEqual(
            {t["name"] for t in res_all.tools},
            {
                "git.read_file",
                "filesystem.read_file",
                "git.ping",
                "filesystem.ping",
                "filesystem.list_dir",
            },
        )
        self.assertLess(res_all.tokens_after, res_all.tokens_before)
        for t in res_all.tools:
            self.assertNotIn("$defs", t["inputSchema"])

        with self.assertRaises(ValueError):
            CompositeSchemaMerger(strategy="unknown_mode")
        with self.assertRaises(ValueError):
            CompositeSchemaMerger(separator="")


class TestCliServeAndSarifV180(unittest.TestCase):
    """CLI `mcp-slim merge`, `wrap --repair-truncated-json`, `serve` tools, and SARIF v2.1.0."""

    def test_cli_merge_sarif_and_fail_on_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = tmp / "alpha.json"
            s2 = tmp / "beta.json"
            sarif_out = tmp / "merge.sarif"

            s1.write_text(
                json.dumps([
                    {
                        "name": "search",
                        "description": "Search alpha index.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ]),
                encoding="utf-8",
            )
            s2.write_text(
                json.dumps([
                    {
                        "name": "search",
                        "description": "Search beta vector store.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"vector": {"type": "array"}},
                        },
                    }
                ]),
                encoding="utf-8",
            )

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main([
                    "merge",
                    f"alpha={s1}",
                    f"beta={s2}",
                    "--slim",
                    "--json",
                    "--sarif",
                    str(sarif_out),
                    "--fail-on-conflict",
                ])
            self.assertEqual(rc, 2)
            report = json.loads(buf.getvalue())
            self.assertEqual(report["servers_merged"], 2)
            self.assertEqual(report["tool_collisions"], 1)

            sarif = json.loads(sarif_out.read_text(encoding="utf-8"))
            self.assertEqual(sarif["version"], "2.1.0")
            run0 = sarif["runs"][0]
            self.assertEqual(run0["tool"]["driver"]["version"], mcp_slim.__version__)
            rule_ids = [r["id"] for r in run0["tool"]["driver"]["rules"]]
            self.assertEqual(len(rule_ids), 18)
            self.assertIn("MS015", rule_ids)
            self.assertIn("MS016", rule_ids)

            # Verify enhanced SARIF v2.1.0 fields: ruleIndex, partialFingerprints, help, invocations
            for rule_meta in run0["tool"]["driver"]["rules"]:
                self.assertIn("help", rule_meta)
                self.assertIn("properties", rule_meta)
            self.assertEqual(len(run0["results"]), 1)
            res0 = run0["results"][0]
            self.assertEqual(res0["ruleId"], "MS016")
            self.assertEqual(res0["ruleIndex"], rule_ids.index("MS016"))
            self.assertIn(
                "primaryLocationLineHash", res0["partialFingerprints"]
            )
            self.assertTrue(run0["invocations"][0]["executionSuccessful"])
            self.assertEqual(
                run0["invocations"][0]["properties"]["servers_merged"], 2
            )

    def test_audit_tools_manifest_surfaces_ms013_and_extra_sarif_results(self) -> None:
        cyclic_tools = [
            {
                "name": "tree_walk",
                "description": "Walk recursive AST nodes.",
                "inputSchema": {
                    "$defs": {
                        "AstNode": {
                            "type": "object",
                            "properties": {"next": {"$ref": "#/$defs/AstNode"}},
                        }
                    },
                    "type": "object",
                    "properties": {"root": {"$ref": "#/$defs/AstNode"}},
                },
            }
        ]
        audit_rep = audit_tools_manifest(cyclic_tools)
        found_rules = {f["rule_id"] for f in audit_rep["findings"]}
        self.assertIn("MS003", found_rules)
        self.assertIn("MS013", found_rules)

        rep_res = repair_truncated_json('{"root": {"next": ')
        sarif = build_audit_sarif(
            audit_rep,
            source_uri="cyclic.json",
            extra_sarif_results=rep_res.to_sarif_results("tree_walk"),
        )
        emitted_ids = {r["ruleId"] for r in sarif["runs"][0]["results"]}
        self.assertIn("MS013", emitted_ids)
        self.assertIn("MS015", emitted_ids)

    def test_wrap_config_and_mcp_serve_v180_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "mcp.json"
            cfg.write_text(
                json.dumps({
                    "mcpServers": {
                        "fs": {
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        }
                    }
                }),
                encoding="utf-8",
            )
            rc = wrap_config(cfg, in_place=True, repair_truncated_json=True)
            self.assertEqual(rc, 0)
            wrapped = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn(
                "--repair-truncated-json", wrapped["mcpServers"]["fs"]["args"]
            )

        # Standalone MCP server tools: repair_tool_call_json & merge_mcp_manifests
        secret = "sk-ant-api03-" + ("B" * 40)
        rpc_repair = handle_server_rpc({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "repair_tool_call_json",
                "arguments": {
                    "raw_json": f'{{"api_key": "{secret}", "max_tokens": 512, "dangling": '
                },
            },
        })
        assert rpc_repair is not None
        payload = json.loads(rpc_repair["result"]["content"][0]["text"])
        self.assertTrue(payload["is_valid"])
        self.assertTrue(payload["repaired"])
        self.assertGreaterEqual(payload["secrets_redacted"], 1)
        self.assertNotIn(secret, payload["repaired_json"])
        self.assertEqual(payload["value"]["max_tokens"], 512)

        rpc_merge = handle_server_rpc({
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "merge_mcp_manifests",
                "arguments": {
                    "servers": {
                        "s1": [{"name": "echo", "description": "Echo A", "inputSchema": {"type": "object"}}],
                        "s2": [{"name": "echo", "description": "Echo B", "inputSchema": {"type": "object"}}],
                    },
                    "strategy": "prefix_colliding",
                },
            },
        })
        assert rpc_merge is not None
        merge_payload = json.loads(rpc_merge["result"]["content"][0]["text"])
        self.assertEqual(merge_payload["tool_collisions"], 1)
        self.assertEqual(
            {t["name"] for t in merge_payload["tools"]},
            {"s1__echo", "s2__echo"},
        )


if __name__ == "__main__":
    unittest.main()
