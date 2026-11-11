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
from .slimmer import OutputSlimmer, SchemaSlimmer, SlimStats

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


def wrap_config(config_path: Path, in_place: bool = False) -> int:
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
    if not isinstance(cmd, str) or cmd == "mcp-slim":
      continue
    spec["command"] = "mcp-slim"
    spec["args"] = ["run", "--", cmd, *(args if isinstance(args, list) else [])]
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

  # Allow shorthand `mcp-slim -- <cmd> [args...]`
  if args_list and args_list[0] == "--":
    args_list = ["run", *args_list]

  parser = argparse.ArgumentParser(
      prog="mcp-slim",
      description=(
          "Zero-config MCP proxy that cuts tool schema & output token usage by "
          "60-80% and redacts leaked secrets."
      ),
  )
  parser.add_argument(
      "--version", action="version", version=f"mcp-slim {__version__}"
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
      help="Comma-separated allowlist of tool names to expose",
  )
  run_p.add_argument(
      "--deny-tools",
      type=str,
      default=None,
      help="Comma-separated denylist of tool names to hide",
  )
  run_p.add_argument(
      "--no-secret-guard",
      action="store_true",
      help="Disable bidirectional secret redaction",
  )
  run_p.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress session summary on stderr",
  )
  run_p.add_argument(
      "command",
      nargs=argparse.REMAINDER,
      help="Upstream command to execute (after '--')",
  )

  wrap_p = sub.add_parser(
      "wrap",
      help="Wrap all servers in a Claude / Cursor / Windsurf mcp.json config",
  )
  wrap_p.add_argument("config", type=Path, help="Path to MCP config JSON file")
  wrap_p.add_argument(
      "-i", "--in-place", action="store_true", help="Modify the config file in place"
  )

  sub.add_parser("benchmark", help="Run built-in token compression & security benchmark")

  parsed = parser.parse_args(args_list)

  if parsed.subcommand == "benchmark":
    return run_benchmark()

  if parsed.subcommand == "wrap":
    return wrap_config(parsed.config, in_place=parsed.in_place)

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
        ),
        output_slimmer=OutputSlimmer(),
        secret_guard=SecretGuard(enabled=not parsed.no_secret_guard),
        quiet=parsed.quiet,
    )
    return asyncio.run(proxy.run_stdio(cmd))

  parser.print_help()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
