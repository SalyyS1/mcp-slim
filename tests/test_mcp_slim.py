"""Comprehensive unit and integration tests for mcp-slim."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mcp_slim.cli import SAMPLE_BENCHMARK_OUTPUT, SAMPLE_BENCHMARK_TOOLS, main, wrap_config
from mcp_slim.guard import SecretGuard
from mcp_slim.proxy import MCPProxy
from mcp_slim.slimmer import LazyToolBroker, OutputSlimmer, SchemaSlimmer, SlimStats


class TestSchemaSlimmer(unittest.TestCase):
  def test_slim_tools_reduces_tokens_and_strips_bloat(self) -> None:
    slimmer = SchemaSlimmer()
    stats = SlimStats()
    slimmed = slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS, stats=stats)

    self.assertEqual(len(slimmed), 2)
    self.assertLess(stats.schema_tokens_after, stats.schema_tokens_before)
    self.assertGreater(stats.reduction_pct, 45.0)
    self.assertGreaterEqual(stats.refs_inlined, 1)

    first_schema = slimmed[0]["inputSchema"]
    self.assertNotIn("$schema", first_schema)
    self.assertNotIn("title", first_schema)
    self.assertNotIn("additionalProperties", first_schema)
    self.assertNotIn("$defs", first_schema)

    props = first_schema["properties"]
    self.assertNotIn("description", props["repo"])
    self.assertNotIn("description", props["content"])
    self.assertNotIn("description", props["branch"])
    self.assertIn("description", props["owner"])
    self.assertEqual(props["owner"]["description"], "The owner of the repository.")
    self.assertEqual(props["sha"]["type"], "string")
    self.assertNotIn("anyOf", props["sha"])

    # Verify $ref was inlined into properties["author"] for Gemini/Vertex compatibility
    self.assertEqual(props["author"]["type"], "object")
    self.assertIn("name", props["author"]["properties"])
    self.assertNotIn("$ref", props["author"])

  def test_allow_and_deny_tool_filters_and_globs(self) -> None:
    allow_slimmer = SchemaSlimmer(allow_tools={"search_*"})
    allowed = allow_slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS)
    self.assertEqual([t["name"] for t in allowed], ["search_repositories"])

    deny_slimmer = SchemaSlimmer(deny_tools={"create_*"})
    denied = deny_slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS)
    self.assertEqual([t["name"] for t in denied], ["search_repositories"])


class TestLazyToolBroker(unittest.TestCase):
  def test_lazy_broker_replaces_catalog_and_searches_tools(self) -> None:
    broker = LazyToolBroker()
    stats = SlimStats()
    meta_tools = broker.ingest_tools(SAMPLE_BENCHMARK_TOOLS, stats=stats)
    self.assertEqual(len(meta_tools), 2)
    self.assertEqual(meta_tools[0]["name"], "mcp_search_tools")
    self.assertEqual(meta_tools[1]["name"], "mcp_call_tool")

    res = broker.search("commit branch file")
    self.assertGreaterEqual(res["matched_count"], 1)
    self.assertEqual(res["tools"][0]["name"], "create_or_update_file")

    # Empty query lists catalog summaries
    all_res = broker.search("")
    self.assertEqual(all_res["total_indexed"], 2)


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

    req_line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    processed_req = proxy.process_client_message(req_line)
    self.assertIn('"tools/list"', processed_req)

    resp_line = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": SAMPLE_BENCHMARK_TOOLS}}
    )
    processed_resp = json.loads(proxy.process_server_message(resp_line))
    self.assertEqual(len(processed_resp["result"]["tools"]), 2)
    self.assertNotIn("$schema", processed_resp["result"]["tools"][0]["inputSchema"])

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

  def test_proxy_lazy_mode_search_and_call_rewrite(self) -> None:
    proxy = MCPProxy(lazy=True, quiet=True)
    proxy.process_client_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    )
    list_resp = json.loads(
        proxy.process_server_message(
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": SAMPLE_BENCHMARK_TOOLS}})
        )
    )
    self.assertEqual(
        [t["name"] for t in list_resp["result"]["tools"]],
        ["mcp_search_tools", "mcp_call_tool"],
    )

    # Local search request is answered directly without forwarding upstream
    forwarded = proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "mcp_search_tools", "arguments": {"query": "search repositories"}},
        })
    )
    self.assertEqual(forwarded, "")
    local_replies = proxy.pop_local_responses()
    self.assertEqual(len(local_replies), 1)
    search_payload = json.loads(
        json.loads(local_replies[0])["result"]["content"][0]["text"]
    )
    self.assertEqual(search_payload["tools"][0]["name"], "search_repositories")

    # mcp_call_tool is rewritten to the upstream tool name
    rewritten_str = proxy.process_client_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "mcp_call_tool",
                "arguments": {
                    "tool_name": "search_repositories",
                    "arguments": {"query": "mcp-slim"},
                },
            },
        })
    )
    rewritten = json.loads(rewritten_str)
    self.assertEqual(rewritten["params"]["name"], "search_repositories")
    self.assertEqual(rewritten["params"]["arguments"], {"query": "mcp-slim"})

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

  def test_cli_benchmark_and_audit_exit_zero(self) -> None:
    self.assertEqual(main(["benchmark"]), 0)
    self.assertEqual(main(["audit"]), 0)

  def test_canonical_sort_stabilizes_kv_cache_prefix(self) -> None:
    slimmer = SchemaSlimmer(canonical_sort=True)
    tools_order_a = list(reversed(SAMPLE_BENCHMARK_TOOLS))
    tools_order_b = list(SAMPLE_BENCHMARK_TOOLS)
    out_a = json.dumps(slimmer.slim_tools_list(tools_order_a), separators=(",", ":"))
    out_b = json.dumps(slimmer.slim_tools_list(tools_order_b), separators=(",", ":"))
    self.assertEqual(out_a, out_b)

  def test_hypermedia_stripping_and_valid_json_budget_truncation(self) -> None:
    out_slimmer = OutputSlimmer(max_list_items=50, max_string_chars=260)
    raw = {
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "total_count": 10,
                    "items": [
                        {
                            "id": i,
                            "name": f"repo_{i}",
                            "description": "A repository with some detailed text payload " + ("x" * 30),
                            "followers_url": "https://api.github.com/users/u/followers",
                            "node_id": "MDEwOlJlcG9zaXRvcnk=",
                        }
                        for i in range(10)
                    ],
                }),
            }
        ]
    }
    slimmed = out_slimmer.slim_call_result(raw)
    slim_text = slimmed["content"][0]["text"]
    # Output must remain 100% valid parseable JSON even when truncated by character budget
    parsed = json.loads(slim_text)
    self.assertNotIn("followers_url", parsed["items"][0])
    self.assertNotIn("node_id", parsed["items"][0])
    self.assertLess(len(parsed["items"]), 10)
    self.assertIn("more items truncated", parsed["items"][-1])

  def test_read_only_tool_call_ttl_cache_and_mutation_invalidation(self) -> None:
    proxy = MCPProxy(cache_ttl=60.0, quiet=True)
    req1 = json.dumps({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "get_issue", "arguments": {"repo": "a/b", "number": 1}},
    })
    self.assertNotEqual(proxy.process_client_message(req1), "")
    proxy.process_server_message(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 10,
            "result": {"content": [{"type": "text", "text": '{"id":1,"title":"bug"}'}]},
        })
    )

    # Second identical read call hits local TTL cache
    req2 = json.dumps({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {"name": "get_issue", "arguments": {"number": 1, "repo": "a/b"}},
    })
    self.assertEqual(proxy.process_client_message(req2), "")
    self.assertEqual(proxy.stats.cache_hits, 1)
    cached_replies = proxy.pop_local_responses()
    self.assertEqual(len(cached_replies), 1)
    self.assertEqual(json.loads(cached_replies[0])["id"], 11)

    # Mutating tool call invalidates read cache
    mut_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {"name": "create_issue", "arguments": {"title": "new"}},
    })
    self.assertNotEqual(proxy.process_client_message(mut_req), "")
    self.assertNotEqual(proxy.process_client_message(req2), "")

  def test_live_stdio_server_inspect_command(self) -> None:
    mock_server_code = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "  msg = json.loads(line)\n"
        "  if msg.get('method') == 'initialize':\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}}), flush=True)\n"
        "  elif msg.get('method') == 'tools/list':\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'tools':[{'name':'read_file','description':'Read file content from disk.','inputSchema':{'type':'object','properties':{'path':{'type':'string','description':'The path'}}}}]}}), flush=True)\n"
        "    break\n"
    )
    rc = main(["inspect", "--", "python3", "-c", mock_server_code])
    self.assertEqual(rc, 0)


if __name__ == "__main__":
  unittest.main()
