"""Command-line interface for mcp-slim."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .guard import SecretGuard
from .proxy import MCPProxy
from .server import run_serve
from .slimmer import (
    LazyToolBroker,
    OutputSlimmer,
    SchemaSlimmer,
    SlimStats,
    audit_tools_manifest,
    build_audit_sarif,
    estimate_tokens,
)
from .telemetry import StatsLedger

SAMPLE_BENCHMARK_TOOLS: list[dict[str, Any]] = [
    {
        "name": "create_or_update_file",
        "description": (
            "Create a new file or update an existing file in a GitHub repository. "
            "This tool allows you to push content directly to a branch. Note that if "
            "you are updating an existing file, you must provide the current SHA blob "
            "hash of the file to avoid conflicts. Always make sure the branch exists "
            "before calling this endpoint."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "CreateOrUpdateFileInput",
            "type": "object",
            "additionalProperties": False,
            "$defs": {
                "CommitAuthor": {
                    "title": "CommitAuthor",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"title": "Name", "type": "string", "description": "The name"},
                        "email": {"title": "Email", "type": "string", "description": "The email"},
                    },
                    "required": ["name", "email"],
                }
            },
            "properties": {
                "owner": {
                    "title": "Owner",
                    "type": "string",
                    "description": (
                        "The owner of the repository. This can be a user or an "
                        "organization account name on GitHub."
                    ),
                },
                "repo": {
                    "title": "Repo",
                    "type": "string",
                    "description": "The repo",
                },
                "path": {
                    "title": "Path",
                    "type": "string",
                    "description": (
                        "The file path inside the repository where the file will be "
                        "created or updated (e.g. 'src/index.ts')."
                    ),
                },
                "content": {
                    "title": "Content",
                    "type": "string",
                    "description": "The content",
                },
                "message": {
                    "title": "Commit Message",
                    "type": "string",
                    "description": (
                        "The commit message describing the change. Should follow "
                        "conventional commits when possible."
                    ),
                },
                "branch": {
                    "title": "Branch",
                    "type": "string",
                    "description": "The branch",
                },
                "sha": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": (
                        "The blob SHA of the file being replaced. Required only when "
                        "updating an existing file."
                    ),
                },
                "author": {
                    "$ref": "#/$defs/CommitAuthor",
                    "description": "Optional commit author metadata.",
                },
            },
            "required": ["owner", "repo", "path", "content", "message", "branch"],
        },
    },
    {
        "name": "search_repositories",
        "description": (
            "Search for GitHub repositories matching a query string. Supports all "
            "standard GitHub search qualifiers such as language:, stars:, topic:, and "
            "org:. Returns paginated repository metadata."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "SearchRepositoriesParams",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": (
                        "Search query string using GitHub search syntax. Examples: "
                        "'mcp language:python stars:>100'."
                    ),
                    "examples": ["machine learning", "mcp server"],
                },
                "page": {
                    "title": "Page",
                    "type": "integer",
                    "default": 1,
                    "description": "Page number for pagination (starts at 1).",
                },
                "perPage": {
                    "title": "PerPage",
                    "type": "integer",
                    "default": 30,
                    "description": "Number of results per page (1-100, default 30).",
                },
            },
            "required": ["query"],
        },
    },
]

SAMPLE_BENCHMARK_OUTPUT: dict[str, Any] = {
    "content": [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "total_count": 2,
                    "incomplete_results": False,
                    "items": [
                        {
                            "id": 101,
                            "name": "mcp-slim",
                            "full_name": "SalyyS1/mcp-slim",
                            "description": "Zero-config MCP token optimizer and secret guard",
                            "homepage": None,
                            "mirror_url": None,
                            "archived": False,
                            "disabled": False,
                            "topics": [],
                            "permissions": {},
                            "leaked_token_example": "ghp_" + "1234567890abcdefghijklmnopqrstuvwx",
                        },
                        {
                            "id": 102,
                            "name": "omnivm",
                            "full_name": "SalyyS1/omnivm",
                            "description": "Scale-to-zero microVMs for AI workloads",
                            "homepage": "",
                            "mirror_url": None,
                            "topics": [],
                            "permissions": {},
                        },
                    ],
                },
                indent=2,
            ),
        }
    ]
}


def run_benchmark() -> int:
  slimmer = SchemaSlimmer()
  out_slimmer = OutputSlimmer()
  guard = SecretGuard()
  stats = SlimStats()

  slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS, stats=stats)
  slimmed_out = out_slimmer.slim_call_result(SAMPLE_BENCHMARK_OUTPUT, stats=stats)
  guard.redact_value(slimmed_out)

  schema_saved = stats.schema_tokens_before - stats.schema_tokens_after
  schema_pct = round((schema_saved / stats.schema_tokens_before) * 100.0, 1)
  out_saved = stats.output_tokens_before - stats.output_tokens_after
  out_pct = round((out_saved / stats.output_tokens_before) * 100.0, 1)

  print("mcp-slim Benchmark Report")
  print("=" * 64)
  print(f"{'Stage':<24} | {'Before':>10} | {'After':>10} | {'Reduction':>12}")
  print("-" * 64)
  print(
      f"{'tools/list Schema':<24} | {stats.schema_tokens_before:>10} |"
      f" {stats.schema_tokens_after:>10} | {f'-{schema_pct}%':>12}"
  )
  print(
      f"{'tools/call JSON Output':<24} | {stats.output_tokens_before:>10} |"
      f" {stats.output_tokens_after:>10} | {f'-{out_pct}%':>12}"
  )
  print("-" * 64)
  print(
      f"{'Total (1 Turn)':<24} | {stats.total_tokens_before:>10} |"
      f" {stats.total_tokens_after:>10} | {f'-{stats.reduction_pct}%':>12}"
  )
  print("=" * 64)
  print(f"Secrets intercepted & redacted: {guard.report.total_redactions}")
  return 0


def _print_tools_audit(tools: list[dict[str, Any]], label: str = "") -> int:
  slimmer = SchemaSlimmer()
  stats_std = SlimStats()
  slimmed_tools = slimmer.slim_tools_list(tools, stats=stats_std)

  broker = LazyToolBroker(schema_slimmer=slimmer)
  stats_lazy = SlimStats()
  broker.ingest_tools(tools, stats=stats_lazy)

  title_suffix = f" - {label}" if label else ""
  print(f"mcp-slim Schema Audit ({len(tools)} tools{title_suffix})")
  print("=" * 78)
  print(f"{'TOOL NAME':<32} | {'RAW TOK':>9} | {'SLIM TOK':>9} | {'SAVED':>9} | {'$REF':>6}")
  print("-" * 78)
  slim_by_name = {t.get("name", ""): t for t in slimmed_tools}
  for raw_t in tools[:20]:
    name = raw_t.get("name", "")
    slim_t = slim_by_name.get(name, raw_t)
    r_tok = estimate_tokens(raw_t)
    s_tok = estimate_tokens(slim_t)
    pct = round(((r_tok - s_tok) / max(1, r_tok)) * 100.0, 1)
    has_ref = "$defs" in json.dumps(raw_t) or "$ref" in json.dumps(raw_t)
    print(
        f"{name:<32.32} | {r_tok:>9} | {s_tok:>9} |"
        f" {f'-{pct}%':>9} | {'FIXED' if has_ref else 'ok':>6}"
    )
  if len(tools) > 20:
    print(f"... ({len(tools) - 20} additional tools omitted from row listing)")
  print("-" * 78)
  std_pct = round(
      ((stats_std.schema_tokens_before - stats_std.schema_tokens_after)
       / max(1, stats_std.schema_tokens_before))
      * 100.0,
      1,
  )
  lazy_pct = round(
      ((stats_lazy.schema_tokens_before - stats_lazy.schema_tokens_after)
       / max(1, stats_lazy.schema_tokens_before))
      * 100.0,
      1,
  )
  print(
      f"Standard SchemaSlimmer : {stats_std.schema_tokens_before} ->"
      f" {stats_std.schema_tokens_after} tokens (-{std_pct}%,"
      f" inlined {stats_std.refs_inlined} $ref definitions)"
  )
  print(
      f"Lazy Discovery (--lazy): {stats_lazy.schema_tokens_before} ->"
      f" {stats_lazy.schema_tokens_after} tokens (-{lazy_pct}% via 2 on-demand"
      " meta-tools)"
  )
  print("=" * 78)
  return 0


def run_audit(
    path: Path | None = None,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_security: bool = False,
) -> int:
  if path is not None:
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, list):
      tools = raw_data
    elif isinstance(raw_data, dict) and isinstance(raw_data.get("tools"), list):
      tools = raw_data["tools"]
    elif isinstance(raw_data, dict) and isinstance(raw_data.get("servers"), dict):
      tools = [
          t
          for srv_tools in raw_data["servers"].values()
          if isinstance(srv_tools, list)
          for t in srv_tools
      ]
    else:
      sys.stderr.write(f"Could not locate a tools list inside {path}\n")
      return 1
  else:
    tools = SAMPLE_BENCHMARK_TOOLS

  report = audit_tools_manifest(tools)
  if sarif_path is not None:
    sarif_doc = build_audit_sarif(
        report,
        source_uri=str(path) if path is not None else "sample_tools.json",
        version=__version__,
    )
    sarif_path.parent.mkdir(parents=True, exist_ok=True)
    sarif_path.write_text(
        json.dumps(sarif_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

  if json_output:
    print(json.dumps(report, indent=2, ensure_ascii=False))
  else:
    _print_tools_audit(tools)
    if report["security_findings"] > 0:
      print(
          f"Security findings detected: {report['security_findings']} "
          "(prompt injection / Unicode smuggling / embedded secrets)"
      )
      for f in report["findings"]:
        if f["rule_id"] in ("MS001", "MS002"):
          print(f"  [{f['rule_id']}] {f['message']}")

  if fail_on_security and report["security_findings"] > 0:
    return 2
  return 0


async def _inspect_stdio_server(command: list[str], timeout: float = 15.0) -> list[dict[str, Any]]:
  """Spawns a live MCP stdio server, performs initialize + tools/list handshake, and returns tools."""
  proc = await asyncio.create_subprocess_exec(
      *command,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
  )
  assert proc.stdin is not None
  assert proc.stdout is not None
  try:
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-slim-inspector", "version": __version__},
        },
    }
    proc.stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
    await proc.stdin.drain()

    while True:
      line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
      if not line:
        raise RuntimeError("MCP server closed stdout during initialize handshake")
      msg = json.loads(line.decode("utf-8", errors="replace"))
      if isinstance(msg, dict) and msg.get("id") == 1:
        break

    notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    proc.stdin.write(
        (json.dumps(notif) + "\n" + json.dumps(list_req) + "\n").encode("utf-8")
    )
    await proc.stdin.drain()

    while True:
      line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
      if not line:
        raise RuntimeError("MCP server closed stdout before returning tools/list")
      msg = json.loads(line.decode("utf-8", errors="replace"))
      if isinstance(msg, dict) and msg.get("id") == 2:
        tools = msg.get("result", {}).get("tools", [])
        return tools if isinstance(tools, list) else []
  finally:
    try:
      proc.terminate()
      await asyncio.wait_for(proc.wait(), timeout=2.0)
    except Exception:
      pass


def run_inspect(command: list[str]) -> int:
  try:
    tools = asyncio.run(_inspect_stdio_server(command))
  except Exception as exc:
    sys.stderr.write(f"Failed to inspect MCP server {' '.join(command)}: {exc}\n")
    return 1
  return _print_tools_audit(tools, label=" ".join(command[:2]))


async def _send_bridge_command(socket_path: str, payload: dict[str, Any]) -> dict[str, Any]:
  reader, writer = await asyncio.open_unix_connection(socket_path)
  try:
    writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    await writer.drain()
    line = await reader.readline()
    if not line:
      raise RuntimeError("Empty response from mcp-slim bridge socket")
    return json.loads(line.decode("utf-8", errors="replace"))
  finally:
    writer.close()


def run_bridge_cli(
    socket_path: str,
    tool: str | None = None,
    args_json: str | None = None,
    show_schema: bool = False,
) -> int:
  try:
    if not tool:
      res = asyncio.run(_send_bridge_command(socket_path, {"action": "list"}))
      if not res.get("ok"):
        sys.stderr.write(f"Bridge error: {res.get('error')}\n")
        return 1
      for t in res.get("tools", []):
        print(f"{t.get('name', ''):<30} {t.get('description', '')}")
      return 0

    if show_schema:
      res = asyncio.run(
          _send_bridge_command(socket_path, {"action": "schema", "tool": tool})
      )
      if not res.get("ok"):
        sys.stderr.write(f"Bridge error: {res.get('error')}\n")
        return 1
      print(json.dumps(res.get("tool", {}), indent=2, ensure_ascii=False))
      return 0

    parsed_args = json.loads(args_json) if args_json else {}
    res = asyncio.run(
        _send_bridge_command(
            socket_path,
            {"action": "call", "tool": tool, "arguments": parsed_args},
        )
    )
    if not res.get("ok"):
      sys.stderr.write(f"Bridge error: {res.get('error')}\n")
      return 1
    rpc_resp = res.get("response", {})
    result = rpc_resp.get("result", {}) if isinstance(rpc_resp, dict) else {}
    content = result.get("content", []) if isinstance(result, dict) else []
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
      sys.stdout.write(str(content[0].get("text", "")) + "\n")
    else:
      print(json.dumps(result or rpc_resp, indent=2, ensure_ascii=False))
    return 1 if isinstance(result, dict) and result.get("isError") else 0
  except Exception as exc:
    sys.stderr.write(f"Bridge call failed: {exc}\n")
    return 1


def run_stats(
    json_output: bool = False,
    reset: bool = False,
    stats_file: Path | None = None,
) -> int:
  ledger = StatsLedger(path=stats_file)
  if reset:
    ledger.reset()
    if json_output:
      print(json.dumps(ledger.summary(), indent=2, ensure_ascii=False))
    else:
      print(f"Reset mcp-slim telemetry ledger at {ledger.path}")
    return 0
  if json_output:
    print(json.dumps(ledger.summary(), indent=2, ensure_ascii=False))
    return 0
  print(ledger.format_dashboard())
  return 0


def wrap_config(
    config_path: Path,
    in_place: bool = False,
    lazy: bool = False,
    runner: str = "mcp-slim",
    read_only: bool = False,
) -> int:
  data = json.loads(config_path.read_text(encoding="utf-8"))
  servers = data.get("mcpServers")
  if not isinstance(servers, dict):
    sys.stderr.write(f"No 'mcpServers' object found in {config_path}\n")
    return 1

  wrapped_count = 0
  for name, spec in servers.items():
    if not isinstance(spec, dict):
      continue
    cmd = spec.get("command")
    args = spec.get("args", [])
    arg_strs = [str(a) for a in args] if isinstance(args, list) else []
    if (
        not isinstance(cmd, str)
        or cmd == "mcp-slim"
        or any("SalyyS1/mcp-slim" in a or a == "mcp-slim" for a in arg_strs)
    ):
      continue
    prefix: list[str] = ["run"]
    if lazy:
      prefix.append("--lazy")
    if read_only:
      prefix.append("--read-only")
    prefix.append("--")
    if runner == "npx":
      spec["command"] = "npx"
      spec["args"] = ["-y", "github:SalyyS1/mcp-slim", *prefix, cmd, *arg_strs]
    elif runner == "uvx":
      spec["command"] = "uvx"
      spec["args"] = [
          "--from",
          "git+https://github.com/SalyyS1/mcp-slim",
          "mcp-slim",
          *prefix,
          cmd,
          *arg_strs,
      ]
    else:
      spec["command"] = "mcp-slim"
      spec["args"] = [*prefix, cmd, *arg_strs]
    wrapped_count += 1

  formatted = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
  if in_place:
    config_path.write_text(formatted, encoding="utf-8")
    print(f"Wrapped {wrapped_count} MCP server(s) in {config_path}")
  else:
    sys.stdout.write(formatted)
  return 0


def main(argv: list[str] | None = None) -> int:
  args_list = list(sys.argv[1:] if argv is None else argv)

  if args_list and args_list[0] == "--":
    args_list = ["run", *args_list]

  parser = argparse.ArgumentParser(
      prog="mcp-slim",
      description=(
          "Zero-dependency MCP proxy for schema compaction (-49% tokens), "
          "lazy meta-tool discovery (-98% tokens), and real-time secret redaction."
      ),
  )
  parser.add_argument(
      "--version", action="version", version=f"mcp-slim {__version__}"
  )
  parser.add_argument(
      "--benchmark",
      action="store_true",
      help="Run built-in token compression & security benchmark",
  )
  sub = parser.add_subparsers(dest="subcommand")

  run_p = sub.add_parser("run", help="Wrap and proxy an upstream MCP server over stdio")
  run_p.add_argument(
      "--max-desc",
      type=int,
      default=120,
      help="Max characters for tool descriptions (default: 120)",
  )
  run_p.add_argument(
      "--max-param-desc",
      type=int,
      default=80,
      help="Max characters for parameter descriptions (default: 80)",
  )
  run_p.add_argument(
      "--allow-tools",
      type=str,
      default=None,
      help="Comma-separated allowlist of tool names or globs (e.g. 'get_*,search_*')",
  )
  run_p.add_argument(
      "--deny-tools",
      type=str,
      default=None,
      help="Comma-separated denylist of tool names or globs (e.g. 'delete_*')",
  )
  run_p.add_argument(
      "--read-only",
      action="store_true",
      help="Restrict exposure and execution to read-only tools using MCP ToolAnnotations (readOnlyHint/destructiveHint) and prefix heuristics",
  )
  run_p.add_argument(
      "--lazy",
      action="store_true",
      help="Enable LazyToolBroker on-demand discovery (exposes 2 meta-tools, cutting static schema tokens by ~98%)",
  )
  run_p.add_argument(
      "--cache-ttl",
      type=float,
      default=0.0,
      help="TTL in seconds to cache deterministic read-only tool calls (get_*, read_*, list_*, search_*)",
  )
  run_p.add_argument(
      "--no-secret-guard",
      action="store_true",
      help="Disable bidirectional secret redaction",
  )
  run_p.add_argument(
      "--vault-handles",
      action="store_true",
      help="Tokenize intercepted secrets into session-scoped [SECRET_REF:...] handles and resolve them on outbound tool calls",
  )
  run_p.add_argument(
      "--preserve-root-unions",
      action="store_true",
      help="Keep root anyOf/oneOf JSON Schemas unflattened instead of adapting them for strict LLM providers",
  )
  run_p.add_argument(
      "--no-stats",
      action="store_true",
      help="Disable persistent token & cost telemetry for this session (or set MCP_SLIM_NO_STATS=1)",
  )
  run_p.add_argument(
      "--stats-file",
      type=Path,
      default=None,
      help="Custom path to telemetry ledger JSON file (default: ~/.mcp-slim/stats.json)",
  )
  run_p.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress session summary on stderr",
  )
  run_p.add_argument(
      "--bridge-socket",
      type=str,
      default=None,
      help="Expose a local Unix domain socket so CLI processes can inspect schemas and call tools over the live MCP session",
  )
  run_p.add_argument(
      "--cli-only",
      action="store_true",
      help="Return an empty tools/list to the MCP host so 0 schema tokens enter the LLM context while tools remain callable via --bridge-socket",
  )
  run_p.add_argument(
      "command",
      nargs=argparse.REMAINDER,
      help="Upstream command to execute (after '--')",
  )

  bridge_p = sub.add_parser(
      "bridge",
      help="List tools, inspect on-demand schemas, or call a tool over a running --bridge-socket session",
  )
  bridge_p.add_argument(
      "--socket",
      "-s",
      required=True,
      type=str,
      help="Path to the active mcp-slim Unix domain socket",
  )
  bridge_p.add_argument(
      "--schema",
      action="store_true",
      help="Print the compacted JSON Schema for the specified tool",
  )
  bridge_p.add_argument(
      "tool",
      nargs="?",
      default=None,
      help="Tool name to inspect or invoke (omit to list all tools)",
  )
  bridge_p.add_argument(
      "args_json",
      nargs="?",
      default=None,
      help="JSON object string of tool arguments (e.g. '{\"query\": \"mcp\"}')",
  )

  wrap_p = sub.add_parser(
      "wrap",
      help="Wrap all servers in a Claude / Cursor / Windsurf mcp.json config",
  )
  wrap_p.add_argument("config", type=Path, help="Path to MCP config JSON file")
  wrap_p.add_argument(
      "-i", "--in-place", action="store_true", help="Modify the config file in place"
  )
  wrap_p.add_argument(
      "--lazy",
      action="store_true",
      help="Wrap servers with --lazy on-demand discovery mode",
  )
  wrap_p.add_argument(
      "--read-only",
      action="store_true",
      help="Wrap servers with --read-only tool policy enforcement",
  )
  wrap_p.add_argument(
      "--runner",
      choices=("mcp-slim", "uvx", "npx"),
      default="mcp-slim",
      help="Launcher command written to mcpServers (default: mcp-slim)",
  )

  audit_p = sub.add_parser(
      "audit",
      help="Audit token footprint, $defs/$ref compatibility, and tool poisoning of a tools JSON file",
  )
  audit_p.add_argument(
      "file",
      type=Path,
      nargs="?",
      default=None,
      help="Optional path to tools JSON or benchmark fixture",
  )
  audit_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON audit report",
  )
  audit_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )
  audit_p.add_argument(
      "--fail-on-security",
      action="store_true",
      help="Exit with code 2 when MS001/MS002 tool-poisoning or embedded-secret findings are detected",
  )

  inspect_p = sub.add_parser(
      "inspect",
      help="Spawn a live MCP server over stdio, query tools/list, and print a token & schema audit",
  )
  inspect_p.add_argument(
      "command",
      nargs=argparse.REMAINDER,
      help="Live MCP server command to inspect (after '--')",
  )

  stats_p = sub.add_parser(
      "stats",
      help="Display cumulative token, wire-byte, and estimated API cost savings",
  )
  stats_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit cumulative telemetry metrics as JSON",
  )
  stats_p.add_argument(
      "--reset",
      action="store_true",
      help="Reset cumulative telemetry statistics",
  )
  stats_p.add_argument(
      "--stats-file",
      type=Path,
      default=None,
      help="Path to stats JSON file (overrides MCP_SLIM_STATS_PATH)",
  )

  sub.add_parser(
      "serve",
      help="Run mcp-slim as a standalone MCP stdio server exposing schema compaction & SecretGuard tools",
  )
  sub.add_parser("benchmark", help="Run built-in token compression & security benchmark")

  parsed = parser.parse_args(args_list)

  if parsed.benchmark or parsed.subcommand == "benchmark":
    return run_benchmark()

  if parsed.subcommand == "serve":
    return run_serve()

  if parsed.subcommand == "stats":
    return run_stats(
        json_output=parsed.json_output,
        reset=parsed.reset,
        stats_file=parsed.stats_file,
    )

  if parsed.subcommand == "audit":
    return run_audit(
        parsed.file,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
        fail_on_security=parsed.fail_on_security,
    )

  if parsed.subcommand == "inspect":
    cmd = list(parsed.command)
    if cmd and cmd[0] == "--":
      cmd = cmd[1:]
    if not cmd:
      inspect_p.error("missing server command after '--'")
    return run_inspect(cmd)

  if parsed.subcommand == "wrap":
    return wrap_config(
        parsed.config,
        in_place=parsed.in_place,
        lazy=parsed.lazy,
        runner=parsed.runner,
        read_only=parsed.read_only,
    )

  if parsed.subcommand == "bridge":
    return run_bridge_cli(
        socket_path=parsed.socket,
        tool=parsed.tool,
        args_json=parsed.args_json,
        show_schema=parsed.schema,
    )

  if parsed.subcommand == "run":
    cmd = list(parsed.command)
    if cmd and cmd[0] == "--":
      cmd = cmd[1:]
    if not cmd:
      run_p.error("missing upstream command after '--'")
    allow_set = (
        {t.strip() for t in parsed.allow_tools.split(",") if t.strip()}
        if parsed.allow_tools
        else None
    )
    deny_set = (
        {t.strip() for t in parsed.deny_tools.split(",") if t.strip()}
        if parsed.deny_tools
        else set()
    )
    proxy = MCPProxy(
        schema_slimmer=SchemaSlimmer(
            max_desc_chars=parsed.max_desc,
            max_param_desc_chars=parsed.max_param_desc,
            allow_tools=allow_set,
            deny_tools=deny_set,
            preserve_root_unions=parsed.preserve_root_unions,
            read_only_only=parsed.read_only,
        ),
        output_slimmer=OutputSlimmer(),
        secret_guard=SecretGuard(
            enabled=not parsed.no_secret_guard,
            vault_handles=parsed.vault_handles,
        ),
        lazy=parsed.lazy,
        cache_ttl=parsed.cache_ttl,
        quiet=parsed.quiet,
        cli_only=parsed.cli_only,
        bridge_socket=parsed.bridge_socket,
        no_stats=parsed.no_stats,
        stats_ledger=StatsLedger(
            path=parsed.stats_file, enabled=not parsed.no_stats
        ),
    )
    return asyncio.run(proxy.run_stdio(cmd))

  parser.print_help()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
