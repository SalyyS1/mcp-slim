"""Unit tests for mcp-slim v0.7.0 features: root-union semantics, vault handles, typo checks, and serve mode."""

from __future__ import annotations

import json
import unittest

from mcp_slim import (
    MCPProxy,
    SchemaSlimmer,
    SecretGuard,
    handle_server_rpc,
)


class TestMcpSlimV070(unittest.TestCase):

  def _make_union_tool(
      self,
      union_key: str = "anyOf",
      forbid_extra: bool = False,
  ) -> dict:
    branch_extra = {"additionalProperties": False} if forbid_extra else {}
    return {
        "name": "lookup_target",
        "description": "Lookup target by id or slug.",
        "inputSchema": {
            "type": "object",
            union_key: [
                {
                    "type": "object",
                    **branch_extra,
                    "properties": {
                        "id": {"type": "string"},
                        "org": {"type": "string"},
                    },
                    "required": ["id", "org"],
                },
                {
                    "type": "object",
                    **branch_extra,
                    "properties": {
                        "slug": {"type": "string"},
                        "org": {"type": "string"},
                    },
                    "required": ["slug", "org"],
                },
            ],
        },
    }

  def test_root_anyof_and_oneof_paired_compatibility_matrix(self) -> None:
    """Covers the 4-request compatibility matrix (org-only, id+org, slug+org, id+slug+org)."""
    # 1. Root anyOf flattened: org-only rejected, id+org accepted, slug+org accepted, id+slug+org accepted
    anyof_proxy = MCPProxy(quiet=True, no_stats=True)
    anyof_proxy.process_client_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    )
    slimmed_resp = json.loads(
        anyof_proxy.process_server_message(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"tools": [self._make_union_tool("anyOf")]},
            })
        )
    )
    tool_def = slimmed_resp["result"]["tools"][0]
    self.assertIn("[Requires one of: id | slug]", tool_def["description"])
    self.assertEqual(tool_def["inputSchema"]["required"], ["org"])

    # Case A: {"org": "acme"} -> satisfies neither branch -> blocked locally
    out_org_only = anyof_proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "lookup_target", "arguments": {"org": "acme"}},
        })
    )
    self.assertEqual(out_org_only, "")
    local_err = json.loads(anyof_proxy.pop_local_responses()[0])
    self.assertTrue(local_err["result"]["isError"])
    self.assertIn("anyOf", local_err["result"]["content"][0]["text"])

    # Case B & C: {"id": "1", "org": "acme"} and {"slug": "s", "org": "acme"} -> forwarded
    for req_id, valid_args in (
        (3, {"id": "1", "org": "acme"}),
        (4, {"slug": "s", "org": "acme"}),
        (5, {"id": "1", "slug": "s", "org": "acme"}),
    ):
      forwarded = anyof_proxy.process_client_message(
          json.dumps({
              "jsonrpc": "2.0",
              "id": req_id,
              "method": "tools/call",
              "params": {"name": "lookup_target", "arguments": valid_args},
          })
      )
      self.assertNotEqual(forwarded, "")
      self.assertEqual(anyof_proxy.pop_local_responses(), [])

    # 2. Root oneOf flattened: org-only rejected, id+org & slug+org accepted, id+slug+org rejected!
    oneof_proxy = MCPProxy(quiet=True, no_stats=True)
    oneof_proxy.process_client_message(
        json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}})
    )
    oneof_resp = json.loads(
        oneof_proxy.process_server_message(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 10,
                "result": {"tools": [self._make_union_tool("oneOf")]},
            })
        )
    )
    self.assertIn(
        "[Requires one of (exclusive): id | slug]",
        oneof_resp["result"]["tools"][0]["description"],
    )

    both_branches_out = oneof_proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "lookup_target",
                "arguments": {"id": "1", "slug": "s", "org": "acme"},
            },
        })
    )
    self.assertEqual(both_branches_out, "")
    oneof_err = json.loads(oneof_proxy.pop_local_responses()[0])
    self.assertTrue(oneof_err["result"]["isError"])
    self.assertIn("mutually exclusive root oneOf", oneof_err["result"]["content"][0]["text"])

    # 3. preserve_root_unions=True keeps root anyOf intact
    strict_slimmer = SchemaSlimmer(preserve_root_unions=True)
    preserved = strict_slimmer.slim_tool(self._make_union_tool("anyOf"))
    self.assertIn("anyOf", preserved["inputSchema"])

  def test_vault_handles_roundtrip_structured_arguments(self) -> None:
    """Verifies raw secrets never enter client context while outbound structured handles resolve."""
    guard = SecretGuard(enabled=True, vault_handles=True)
    proxy = MCPProxy(secret_guard=guard, quiet=True, no_stats=True)

    raw_gh_token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_ci_env", "arguments": {}},
        })
    )
    server_msg = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": f"Found deploy token: {raw_gh_token}"}
            ]
        },
    })
    sanitized_to_llm = json.loads(proxy.process_server_message(server_msg))
    llm_text = sanitized_to_llm["result"]["content"][0]["text"]
    self.assertNotIn(raw_gh_token, llm_text)
    self.assertIn("[SECRET_REF:github_token:", llm_text)

    handle = llm_text.split("Found deploy token: ")[1].strip()
    # Now the LLM passes the opaque handle inside nested structured tool arguments
    outbound_call = proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "publish_release",
                "arguments": {
                    "headers": {"Authorization": f"Bearer {handle}"},
                    "mirrors": [handle, "[SECRET_REF:github_token:deadbeef00]"],
                },
            },
        })
    )
    wire_to_server = json.loads(outbound_call)
    args_on_wire = wire_to_server["params"]["arguments"]
    self.assertEqual(
        args_on_wire["headers"]["Authorization"], f"Bearer {raw_gh_token}"
    )
    self.assertEqual(args_on_wire["mirrors"][0], raw_gh_token)
    # Forged/unknown handle remains unexpanded
    self.assertEqual(
        args_on_wire["mirrors"][1], "[SECRET_REF:github_token:deadbeef00]"
    )
    self.assertEqual(guard.report.handles_resolved, 2)

  def test_policy_typo_fuzzy_suggestions(self) -> None:
    slimmer = SchemaSlimmer(
        allow_tools={"create_isue", "get_*"},
        deny_tools={"delet_repo"},
    )
    tools = [
        {"name": "create_issue", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_issue", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "delete_repo", "inputSchema": {"type": "object", "properties": {}}},
    ]
    kept = slimmer.slim_tools_list(tools)
    self.assertEqual([t["name"] for t in kept], ["get_issue"])
    self.assertEqual(len(slimmer.policy_warnings), 2)
    joined = "\n".join(slimmer.policy_warnings)
    self.assertIn("did you mean 'create_issue'?", joined)
    self.assertIn("did you mean 'delete_repo'?", joined)

  def test_standalone_serve_mcp_protocol(self) -> None:
    init_resp = handle_server_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    })
    assert init_resp is not None
    self.assertEqual(init_resp["result"]["serverInfo"]["name"], "mcp-slim")

    # Notification returns None
    self.assertIsNone(
        handle_server_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
    )

    tools_resp = handle_server_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    self.assertEqual(
        tool_names,
        {"compact_mcp_tools", "audit_mcp_schema", "redact_secrets", "get_slim_stats"},
    )

    redact_resp = handle_server_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "redact_secrets",
            "arguments": {"text": "token=ghp_1234567890abcdefghijklmnopqrstuvwx"},
        },
    })
    assert redact_resp is not None
    body = json.loads(redact_resp["result"]["content"][0]["text"])
    self.assertEqual(body["total_redactions"], 1)
    self.assertIn("[REDACTED:GITHUB_TOKEN]", body["sanitized"])


if __name__ == "__main__":
  unittest.main()
