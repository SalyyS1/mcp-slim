"""Unit and integration tests for mcp-slim v1.9.0:
- MS017: ToolCatalogPruner / multi-field BM25+ intent-aware tool catalog pruning
- MS018: JsonRpcResponseFirewall / stateful JSON-RPC 2.0 response correlation & batch firewall
- CLI `mcp-slim prune`, `wrap`, `MCPProxy`, `mcp-slim serve`, and OASIS SARIF v2.1.0 integration
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
    JsonRpcResponseFirewall,
    MCPProxy,
    OTelSpanExporter,
    SecretGuard,
    ToolCatalogPruner,
    audit_rpc_transcript,
    audit_tools_manifest,
    build_audit_sarif,
    filter_tools_for_intent,
    handle_server_rpc,
    tokenize_mcp_text,
)
from mcp_slim.cli import main, wrap_config


def _build_multi_domain_catalog() -> list[dict]:
    return [
        {
            "name": "github__createPullRequest",
            "description": (
                "Open a pull request between head and base branches in a repository "
                "with reviewers, labels, and draft state."
            ),
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Account owner"},
                    "repository": {"type": "string", "description": "Target repository"},
                    "headBranch": {"type": "string"},
                    "baseBranch": {"type": "string"},
                    "draft": {"type": "boolean"},
                },
                "required": ["owner", "repository", "headBranch", "baseBranch"],
            },
        },
        {
            "name": "github__listPullRequestDiffs",
            "description": "Retrieve unified patch diffs for a pull request.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
        {
            "name": "postgres__executeSqlQuery",
            "description": "Run a read-only SQL query against PostgreSQL analytics tables.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "dialect": {"type": "string", "enum": ["postgresql", "ansi"]},
                    "timeout_ms": {"type": "integer"},
                },
                "required": ["sql"],
            },
        },
        {
            "name": "slack__postChannelMessage",
            "description": "Publish a formatted notification to a Slack channel.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["channel_id", "text"],
            },
        },
        {
            "name": "k8s__rolloutRestartDeployment",
            "description": "Trigger a rolling pod restart on a Kubernetes cluster deployment.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "deployment": {"type": "string"},
                },
                "required": ["namespace", "deployment"],
            },
        },
        {
            "name": "jira__transitionIssueStatus",
            "description": "Move a Jira ticket across workflow transitions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "issue_key": {"type": "string"},
                    "status": {"type": "string"},
                },
                "required": ["issue_key", "status"],
            },
        },
        {
            "name": "system__health_ping",
            "description": "Lightweight liveness probe.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


class TestToolCatalogPrunerMS017(unittest.TestCase):
    """MS017: Subword + synonym BM25+ intent-aware MCP tool catalog pruner."""

    def test_subword_and_synonym_tokenization(self) -> None:
        toks = tokenize_mcp_text("github__createPullRequestsV2")
        self.assertIn("github", toks)
        self.assertIn("create", toks)
        self.assertIn("pull", toks)
        self.assertIn("request", toks)

        # Protected stems like postgres / status / k8s remain intact
        prot = tokenize_mcp_text("postgres status k8s redis")
        self.assertIn("postgres", prot)
        self.assertIn("status", prot)
        self.assertIn("k8s", prot)
        self.assertIn("redis", prot)

        # Synonym expansion maps 'db' -> 'database'/'sql' and 'pr' -> 'merge'
        expanded = tokenize_mcp_text("db pr", expand_synonyms=True)
        self.assertIn("database", expanded)
        self.assertIn("sql", expanded)
        self.assertIn("merge", expanded)

    def test_bm25_plus_intent_ranking_and_token_reduction(self) -> None:
        catalog = _build_multi_domain_catalog()
        res = filter_tools_for_intent(
            catalog,
            "open a PR and inspect database SQL tables",
            top_k=3,
            always_keep=["*health_ping"],
            slim_schemas=True,
        )
        self.assertEqual(res.tools_before, 7)
        self.assertEqual(res.tools_after, 4)  # 3 matched + 1 pinned
        kept_names = {t["name"] for t in res.tools}
        self.assertIn("github__createPullRequest", kept_names)
        self.assertIn("github__listPullRequestDiffs", kept_names)
        self.assertIn("postgres__executeSqlQuery", kept_names)
        self.assertIn("system__health_ping", kept_names)
        self.assertNotIn("slack__postChannelMessage", kept_names)
        self.assertNotIn("jira__transitionIssueStatus", kept_names)
        self.assertGreater(res.reduction_pct, 25.0)
        # SchemaSlimmer also stripped $schema from github__createPullRequest
        pr_tool = next(t for t in res.tools if t["name"] == "github__createPullRequest")
        self.assertNotIn("$schema", pr_tool["inputSchema"])

    def test_typo_tolerance_and_enum_schema_matching(self) -> None:
        catalog = _build_multi_domain_catalog()
        # 'databse' has 1-char deletion vs 'database', 'kubernetez' has 1-char substitution
        res = filter_tools_for_intent(catalog, "databse kubernetez pod", top_k=2)
        names = {t["name"] for t in res.tools}
        self.assertIn("postgres__executeSqlQuery", names)
        self.assertIn("k8s__rolloutRestartDeployment", names)

    def test_zero_overlap_strict_vs_fallback_modes(self) -> None:
        catalog = _build_multi_domain_catalog()
        strict_res = filter_tools_for_intent(
            catalog, "quantum waveform superconductor", top_k=2, fallback_to_all=False
        )
        self.assertEqual(strict_res.tools_after, 0)
        self.assertFalse(strict_res.fallback_used)

        fb_res = filter_tools_for_intent(
            catalog, "quantum waveform superconductor", top_k=2, fallback_to_all=True
        )
        self.assertEqual(fb_res.tools_after, 2)
        self.assertTrue(fb_res.fallback_used)
        finding = fb_res.to_finding()
        assert finding is not None
        self.assertEqual(finding["rule_id"], "MS017")
        self.assertEqual(finding["kind"], "intent_prune_fallback")

    def test_audit_tools_manifest_flags_oversized_catalog_ms017(self) -> None:
        big_catalog = [
            {
                "name": f"service_tool_{i:02d}",
                "description": f"Execute domain operation #{i}.",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
            }
            for i in range(42)
        ]
        rep = audit_tools_manifest(big_catalog)
        ms017_findings = [f for f in rep["findings"] if f["rule_id"] == "MS017"]
        self.assertEqual(len(ms017_findings), 1)
        self.assertEqual(ms017_findings[0]["tools_before"], 42)

    def test_cli_prune_json_and_sarif_output(self) -> None:
        catalog = _build_multi_domain_catalog()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "catalog.json"
            sarif_path = Path(tmpdir) / "prune.sarif"
            manifest_path.write_text(json.dumps({"tools": catalog}), encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main([
                    "prune",
                    str(manifest_path),
                    "--intent",
                    "kubernetes deployment restart",
                    "--top-k",
                    "2",
                    "--always-keep",
                    "system__health_ping",
                    "--slim",
                    "--json",
                    "--sarif",
                    str(sarif_path),
                ])
            self.assertEqual(rc, 0)
            rep = json.loads(buf.getvalue())
            self.assertEqual(rep["tools_input"], 7)
            self.assertEqual(rep["tools_output"], 2)  # k8s + pinned health_ping
            sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
            self.assertEqual(sarif["version"], "2.1.0")
            self.assertEqual(sarif["runs"][0]["tool"]["driver"]["version"], "1.9.0")
            rule_ids = [r["id"] for r in sarif["runs"][0]["tool"]["driver"]["rules"]]
            self.assertIn("MS017", rule_ids)
            self.assertIn("MS018", rule_ids)
            self.assertEqual(sarif["runs"][0]["results"][0]["ruleId"], "MS017")


class TestJsonRpcResponseFirewallMS018(unittest.TestCase):
    """MS018: Stateful JSON-RPC 2.0 ID correlation, replay, and batch-smuggling firewall."""

    def test_unsolicited_type_confused_and_boolean_id_spoofing(self) -> None:
        fw = JsonRpcResponseFirewall(enabled=True)
        fw.register_client_payload({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

        # 1. Boolean id: true MUST NOT match integer id: 1 in Python!
        dec_bool = fw.inspect_server_payload({"jsonrpc": "2.0", "id": True, "result": {"tools": []}})
        self.assertFalse(dec_bool.allowed)
        self.assertEqual(dec_bool.violations[0].kind, "invalid_id_type")

        # 2. String id: "1" MUST NOT match pending integer id: 1!
        dec_str = fw.inspect_server_payload({"jsonrpc": "2.0", "id": "1", "result": {"tools": []}})
        self.assertFalse(dec_str.allowed)
        self.assertEqual(dec_str.violations[0].kind, "unsolicited_response_id")
        self.assertIn("type-confused", dec_str.violations[0].message)

        # 3. Unknown integer id: 999 rejected
        dec_unk = fw.inspect_server_payload({"jsonrpc": "2.0", "id": 999, "result": {}})
        self.assertFalse(dec_unk.allowed)
        self.assertEqual(dec_unk.violations[0].kind, "unsolicited_response_id")

        # 4. Legitimate integer id: 1 succeeds!
        dec_ok = fw.inspect_server_payload({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
        self.assertTrue(dec_ok.allowed)
        self.assertEqual(dec_ok.violations, [])

        # 5. Replaying integer id: 1 a second time is caught as duplicate_response_id!
        dec_dup = fw.inspect_server_payload({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
        self.assertFalse(dec_dup.allowed)
        self.assertEqual(dec_dup.violations[0].kind, "duplicate_response_id")

    def test_null_id_and_dual_role_envelopes(self) -> None:
        fw = JsonRpcResponseFirewall(enabled=True)

        # Result with id=null is always illegal in JSON-RPC 2.0
        dec_res_null = fw.inspect_server_payload({"jsonrpc": "2.0", "id": None, "result": {"ok": True}})
        self.assertFalse(dec_res_null.allowed)
        self.assertEqual(dec_res_null.violations[0].kind, "null_response_id")

        # Error with id=null when client sent no malformed frame is unsolicited
        dec_err_unsol = fw.inspect_server_payload({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        })
        self.assertFalse(dec_err_unsol.allowed)
        self.assertEqual(dec_err_unsol.violations[0].kind, "null_response_id")

        # When client actually sent unparseable JSON, exactly one id=null error is allowed
        fw.note_unparseable_client_frame()
        dec_err_allowed = fw.inspect_server_payload({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        })
        self.assertTrue(dec_err_allowed.allowed)

        # Dual-role envelope containing both method and result is rejected
        dec_dual = fw.inspect_server_payload({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "roots/list",
            "result": {"ok": True},
        })
        self.assertFalse(dec_dual.allowed)
        self.assertEqual(dec_dual.violations[0].kind, "ambiguous_dual_role_envelope")

    def test_batch_correlation_and_request_smuggling(self) -> None:
        fw = JsonRpcResponseFirewall(enabled=True, max_batch_size=4)
        # Single-frame client request -> server batch response is rejected!
        fw.register_client_payload({"jsonrpc": "2.0", "id": 10, "method": "ping"})
        dec_unreq_batch = fw.inspect_server_payload([{"jsonrpc": "2.0", "id": 10, "result": {}}])
        self.assertFalse(dec_unreq_batch.allowed)
        self.assertEqual(dec_unreq_batch.violations[0].kind, "unrequested_batch_response")

        # Legitimate client batch of 2 requests -> server batch smuggling an extra request + dup ID
        fw.register_client_payload([
            {"jsonrpc": "2.0", "id": "b1", "method": "ping"},
            {"jsonrpc": "2.0", "id": "b2", "method": "ping"},
        ])
        dec_mixed = fw.inspect_server_payload([
            {"jsonrpc": "2.0", "id": "b1", "result": {"ok": 1}},
            {"jsonrpc": "2.0", "id": "b1", "result": {"ok": "dup"}},
            {"jsonrpc": "2.0", "id": 77, "method": "sampling/createMessage", "params": {}},
            {"jsonrpc": "2.0", "id": "b2", "result": {"ok": 2}},
        ])
        self.assertTrue(dec_mixed.allowed)
        self.assertEqual(dec_mixed.dropped_count, 2)
        self.assertEqual(len(dec_mixed.sanitized_payload), 2)
        kinds = {v.kind for v in dec_mixed.violations}
        self.assertEqual(kinds, {"duplicate_batch_id", "batch_request_smuggling"})

        # Oversized batch rejected cleanly
        dec_over = fw.inspect_server_payload([
            {"jsonrpc": "2.0", "id": i, "result": {}} for i in range(5)
        ])
        self.assertFalse(dec_over.allowed)
        self.assertEqual(dec_over.violations[0].kind, "oversized_batch")


class TestProxyServeAndWrapV190(unittest.TestCase):
    """End-to-end integration across MCPProxy, mcp-slim serve, and wrap_config."""

    def test_mcp_proxy_prune_tools_and_strict_rpc_ids_with_otel(self) -> None:
        catalog = _build_multi_domain_catalog()
        with tempfile.TemporaryDirectory() as tmpdir:
            otel_path = Path(tmpdir) / "spans.jsonl"
            proxy = MCPProxy(
                secret_guard=SecretGuard(enabled=True),
                prune_tools_intent="execute postgres sql query",
                prune_top_k=1,
                strict_rpc_ids=True,
                quiet=True,
                no_stats=True,
                otel_exporter=OTelSpanExporter(path=otel_path, server_label="composite"),
            )

            # 1. Client sends tools/list (id=1)
            c_out = proxy.process_client_message(
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
            )
            self.assertTrue(c_out.endswith("\n"))

            # 2. Malicious server sends unsolicited id=999 response first -> blocked!
            spoof_out = proxy.process_server_message(
                json.dumps({"jsonrpc": "2.0", "id": 999, "result": {"tools": []}}) + "\n"
            )
            self.assertEqual(spoof_out, "")
            self.assertEqual(proxy.stats.rpc_spoofs_blocked, 1)

            # 3. Server sends legitimate id=1 response -> pruned from 7 tools to 1!
            srv_out = proxy.process_server_message(
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": catalog}}) + "\n"
            )
            parsed_srv = json.loads(srv_out)
            exposed_tools = parsed_srv["result"]["tools"]
            self.assertEqual(len(exposed_tools), 1)
            self.assertEqual(exposed_tools[0]["name"], "postgres__executeSqlQuery")
            self.assertEqual(proxy.stats.tools_pruned, 6)
            self.assertGreater(proxy.stats.schema_tokens_before, proxy.stats.schema_tokens_after)

            # 4. Duplicate replay of id=1 is also dropped!
            dup_out = proxy.process_server_message(
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": catalog}}) + "\n"
            )
            self.assertEqual(dup_out, "")
            self.assertEqual(proxy.stats.rpc_spoofs_blocked, 2)

            spans = [
                json.loads(line)
                for line in otel_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            fw_spans = [s for s in spans if s["name"] == "mcp.rpc_firewall"]
            self.assertEqual(len(fw_spans), 2)
            self.assertTrue(fw_spans[0]["attributes"]["mcp_slim.rpc_spoof_blocked"])

    def test_server_batch_secret_redaction_never_leaks_credentials(self) -> None:
        proxy = MCPProxy(secret_guard=SecretGuard(enabled=True), strict_rpc_ids=True, quiet=True, no_stats=True)
        req_batch = [
            {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "read_env"}},
            {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "read_key"}},
        ]
        proxy.process_client_message(json.dumps(req_batch) + "\n")

        leaked_gh = "ghp_" + ("A" * 36)
        resp_batch = [
            {"jsonrpc": "2.0", "id": 11, "result": {"content": [{"type": "text", "text": f"TOKEN={leaked_gh}"}]}},
            {"jsonrpc": "2.0", "id": 12, "result": {"content": [{"type": "text", "text": "ok"}]}},
        ]
        out_line = proxy.process_server_message(json.dumps(resp_batch) + "\n")
        self.assertNotIn(leaked_gh, out_line)
        parsed_batch = json.loads(out_line)
        self.assertEqual(len(parsed_batch), 2)
        self.assertGreaterEqual(proxy.secret_guard.report.total_redactions, 1)

    def test_mcp_serve_v190_tools_and_wrap_config(self) -> None:
        self.assertEqual(mcp_slim.__version__, "1.9.0")
        self.assertEqual(len(AUDIT_SARIF_RULES), 18)

        # tools/list exposes filter_tools_for_intent and audit_rpc_transcript
        list_resp = handle_server_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        self.assertIn("filter_tools_for_intent", tool_names)
        self.assertIn("audit_rpc_transcript", tool_names)

        # Call filter_tools_for_intent
        prune_rpc = handle_server_rpc({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "filter_tools_for_intent",
                "arguments": {
                    "tools": _build_multi_domain_catalog(),
                    "intent": "send slack notification message",
                    "top_k": 1,
                },
            },
        })
        assert prune_rpc is not None
        prune_data = json.loads(prune_rpc["result"]["content"][0]["text"])
        self.assertEqual(prune_data["tools_output"], 1)
        self.assertEqual(prune_data["tools"][0]["name"], "slack__postChannelMessage")

        # Call audit_rpc_transcript
        rpc_audit = handle_server_rpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "audit_rpc_transcript",
                "arguments": {
                    "events": [
                        {"direction": "client", "message": {"jsonrpc": "2.0", "id": 1, "method": "ping"}},
                        {"direction": "server", "message": {"jsonrpc": "2.0", "id": "1", "result": {}}},
                    ]
                },
            },
        })
        assert rpc_audit is not None
        audit_data = json.loads(rpc_audit["result"]["content"][0]["text"])
        self.assertEqual(audit_data["frames_blocked"], 1)
        self.assertEqual(audit_data["findings"][0]["rule_id"], "MS018")

        sarif_doc = build_audit_sarif(audit_data, source_uri="rpc.jsonl")
        self.assertEqual(sarif_doc["runs"][0]["results"][0]["ruleId"], "MS018")

        # wrap_config supports --prune-tools, --top-k-tools, and --strict-rpc-ids
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "mcp.json"
            cfg.write_text(
                json.dumps({
                    "mcpServers": {
                        "git": {"command": "uvx", "args": ["mcp-server-git"]}
                    }
                }),
                encoding="utf-8",
            )
            rc = wrap_config(
                cfg,
                in_place=True,
                prune_tools="git commit and diff",
                top_k_tools=6,
                strict_rpc_ids=True,
            )
            self.assertEqual(rc, 0)
            wrapped_args = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["git"]["args"]
            self.assertIn("--prune-tools", wrapped_args)
            self.assertIn("git commit and diff", wrapped_args)
            self.assertIn("--top-k-tools", wrapped_args)
            self.assertIn("6", wrapped_args)
            self.assertIn("--strict-rpc-ids", wrapped_args)


if __name__ == "__main__":
    unittest.main()
