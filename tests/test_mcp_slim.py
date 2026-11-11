"""Comprehensive unit and integration tests for mcp-slim."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mcp_slim.cli import SAMPLE_BENCHMARK_OUTPUT, SAMPLE_BENCHMARK_TOOLS, main, wrap_config
from mcp_slim.guard import SecretGuard
from mcp_slim.proxy import MCPProxy
from mcp_slim.slimmer import OutputSlimmer, SchemaSlimmer, SlimStats


class TestSchemaSlimmer(unittest.TestCase):
  def test_slim_tools_reduces_tokens_and_strips_bloat(self) -> None:
    slimmer = SchemaSlimmer()
    stats = SlimStats()
    slimmed = slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS, stats=stats)

    self.assertEqual(len(slimmed), 2)
    self.assertLess(stats.schema_tokens_after, stats.schema_tokens_before)
    self.assertGreater(stats.reduction_pct, 45.0)

    first_schema = slimmed[0]["inputSchema"]
    self.assertNotIn("$schema", first_schema)
    self.assertNotIn("title", first_schema)
    self.assertNotIn("additionalProperties", first_schema)

    props = first_schema["properties"]
    # Redundant descriptions ("The repo", "The content", "The branch") should be stripped
    self.assertNotIn("description", props["repo"])
    self.assertNotIn("description", props["content"])
    self.assertNotIn("description", props["branch"])
    # Meaningful description kept & condensed to first sentence
    self.assertIn("description", props["owner"])
    self.assertEqual(props["owner"]["description"], "The owner of the repository.")
    # Nullable anyOf simplified to direct type
    self.assertEqual(props["sha"]["type"], "string")
    self.assertNotIn("anyOf", props["sha"])

  def test_allow_and_deny_tool_filters(self) -> None:
    allow_slimmer = SchemaSlimmer(allow_tools={"search_repositories"})
    allowed = allow_slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS)
    self.assertEqual([t["name"] for t in allowed], ["search_repositories"])

    deny_slimmer = SchemaSlimmer(deny_tools={"search_repositories"})
    denied = deny_slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS)
    self.assertEqual([t["name"] for t in denied], ["create_or_update_file"])


class TestOutputSlimmer(unittest.TestCase):
  def test_compacts_json_and_strips_null_empty_values(self) -> None:
    out_slimmer = OutputSlimmer()
    stats = SlimStats()
    slimmed = out_slimmer.slim_call_result(SAMPLE_BENCHMARK_OUTPUT, stats=stats)

    text = slimmed["content"][0]["text"]
    parsed = json.loads(text)
    self.assertEqual(parsed["total_count"], 2)
    item0 = parsed["items"][0]
    self.assertNotIn("homepage", item0)
    self.assertNotIn("mirror_url", item0)
    self.assertNotIn("topics", item0)
    self.assertNotIn("permissions", item0)
    self.assertLess(stats.output_tokens_after, stats.output_tokens_before)

  def test_truncates_large_lists_and_strips_ansi(self) -> None:
    out_slimmer = OutputSlimmer(max_list_items=5)
    raw = {
        "content": [
            {
                "type": "text",
                "text": "\x1b[32m" + json.dumps({"numbers": list(range(20))}) + "\x1b[0m",
            }
        ]
    }
    slimmed = out_slimmer.slim_call_result(raw)
    parsed = json.loads(slimmed["content"][0]["text"])
    self.assertEqual(len(parsed["numbers"]), 6)
    self.assertIn("more items truncated", parsed["numbers"][-1])


class TestSecretGuard(unittest.TestCase):
  def test_redacts_multiple_secret_providers(self) -> None:
    guard = SecretGuard()
    payload = {
        "anthropic": "sk-ant-api03-" + "abcdefghijklmnopqrstuvwxyz0123456789",
        "github": "ghp_" + "1234567890abcdefghijklmnopqrstuvwx",
        "aws": "AKIA" + "IOSFODNN7EXAMPLE",
        "slack": "xoxb-" + "123456789012-123456789012-abcdefghijklmnopqrstuvwx",
        "url": "https://admin:superSecretPass123@db.internal.example.com:5432/prod",
        "env": "OPENAI_API_KEY=" + "my_custom_secret_token_value_12345",
        "private_key": (
            "-----BEGIN OPENSSH "
            + "PRIVATE KEY-----\n"
            + "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n"
            + "-----END OPENSSH PRIVATE KEY-----"
        ),
    }
    redacted = guard.redact_value(payload)
    self.assertIn("[REDACTED:ANTHROPIC_KEY]", redacted["anthropic"])
    self.assertIn("[REDACTED:GITHUB_TOKEN]", redacted["github"])
    self.assertIn("[REDACTED:AWS_ACCESS_KEY]", redacted["aws"])
    self.assertIn("[REDACTED:SLACK_TOKEN]", redacted["slack"])
    self.assertIn("[REDACTED:URL_SECRET]", redacted["url"])
    self.assertNotIn("superSecretPass123", redacted["url"])
    self.assertIn("[REDACTED:ENV_SECRET]", redacted["env"])
    self.assertIn("[REDACTED:PRIVATE_KEY_BLOCK]", redacted["private_key"])
    self.assertGreaterEqual(guard.report.total_redactions, 7)


class TestMCPProxyAndCLI(unittest.TestCase):
  def test_proxy_end_to_end_jsonrpc_flow(self) -> None:
    proxy = MCPProxy(quiet=True)

    # 1. Client -> Server: tools/list request
    req_line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    processed_req = proxy.process_client_message(req_line)
    self.assertIn('"tools/list"', processed_req)

    # 2. Server -> Client: tools/list response
    resp_line = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": SAMPLE_BENCHMARK_TOOLS}}
    )
    processed_resp = json.loads(proxy.process_server_message(resp_line))
    self.assertEqual(len(processed_resp["result"]["tools"]), 2)
    self.assertNotIn("$schema", processed_resp["result"]["tools"][0]["inputSchema"])

    # 3. Server -> Client: tools/call response with secret
    call_req = json.dumps(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "search"}}
    )
    proxy.process_client_message(call_req)
    call_resp = json.dumps(
        {"jsonrpc": "2.0", "id": 2, "result": SAMPLE_BENCHMARK_OUTPUT}
    )
    processed_call_resp = proxy.process_server_message(call_resp)
    self.assertIn("[REDACTED:GITHUB_TOKEN]", processed_call_resp)
    self.assertNotIn("ghp_1234567890abcdefghijklmnopqrstuvwx", processed_call_resp)

  def test_wrap_config_command(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = Path(tmpdir) / "mcp.json"
      cfg_path.write_text(
          json.dumps(
              {
                  "mcpServers": {
                      "github": {
                          "command": "npx",
                          "args": ["-y", "@modelcontextprotocol/server-github"],
                      }
                  }
              }
          ),
          encoding="utf-8",
      )
      rc = wrap_config(cfg_path, in_place=True)
      self.assertEqual(rc, 0)
      updated = json.loads(cfg_path.read_text(encoding="utf-8"))
      gh = updated["mcpServers"]["github"]
      self.assertEqual(gh["command"], "mcp-slim")
      self.assertEqual(
          gh["args"], ["run", "--", "npx", "-y", "@modelcontextprotocol/server-github"]
      )

  def test_cli_benchmark_exits_zero(self) -> None:
    self.assertEqual(main(["benchmark"]), 0)


if __name__ == "__main__":
  unittest.main()
