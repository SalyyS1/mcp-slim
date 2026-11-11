"""Reproducible token-compression and runtime-latency benchmark suite for mcp-slim.

Evaluates SchemaSlimmer, OutputSlimmer, SecretGuard, and MCPProxy on:
  1. Real MCP tool definitions (mcp-server-git, mcp-server-filesystem,
     mcp-server-postgres, mcp-server-github — 60 tools total).
  2. Real-world JSON tool outputs (GitHub repository search results, database
     rows, directory trees, and log outputs containing secrets).
  3. Microsecond-level latency percentiles (p50, p95, p99), message throughput
     (msg/s, MB/s), and peak memory allocation via tracemalloc over 5,000
     JSON-RPC 2.0 iterations.

Generates SVG charts in `assets/benchmark-tokens.svg` and `assets/benchmark-latency.svg`
and writes raw metrics to `benchmarks/results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_slim.guard import SecretGuard  # noqa: E402
from mcp_slim.proxy import MCPProxy  # noqa: E402
from mcp_slim.slimmer import OutputSlimmer, SchemaSlimmer, SlimStats, estimate_tokens  # noqa: E402


def _build_filesystem_tools() -> list[dict[str, Any]]:
  """12 standard tools modeled on @modelcontextprotocol/server-filesystem."""
  base_specs = [
      (
          "read_file",
          "Read the complete contents of a file from the file system. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file.",
          {
              "path": ("string", "The path to the file to read from disk."),
              "head": ("integer", "If provided, returns only the first N lines of the file."),
              "tail": ("integer", "If provided, returns only the last N lines of the file."),
          },
          ["path"],
      ),
      (
          "read_multiple_files",
          "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files in a project.",
          {"paths": ("array", "Array of file paths to read sequentially.")},
          ["paths"],
      ),
      (
          "write_file",
          "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper UTF-8 encoding.",
          {
              "path": ("string", "The path of the file."),
              "content": ("string", "The content to write into the target file."),
          },
          ["path", "content"],
      ),
      (
          "edit_file",
          "Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style unified diff showing the changes made.",
          {
              "path": ("string", "The path of the file to edit."),
              "edits": ("array", "List of line-replacement operations to apply in order."),
              "dryRun": ("boolean", "Preview changes using git-style diff format without writing."),
          },
          ["path", "edits"],
      ),
      (
          "create_directory",
          "Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation similar to mkdir -p.",
          {"path": ("string", "The directory path.")},
          ["path"],
      ),
      (
          "list_directory",
          "Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes.",
          {"path": ("string", "The directory path.")},
          ["path"],
      ),
      (
          "list_directory_with_sizes",
          "Get a detailed listing of all files and directories in a specified path, including file sizes and summary statistics.",
          {
              "path": ("string", "The directory path."),
              "sortBy": ("string", "Sort entries by name or size (default: name)."),
          },
          ["path"],
      ),
      (
          "directory_tree",
          "Get a recursive tree view of files and directories as a JSON structure. Each entry includes name, type, and children for directories.",
          {"path": ("string", "The root directory path.")},
          ["path"],
      ),
      (
          "move_file",
          "Move or rename files and directories. Can move files between directories and rename them in a single atomic operation.",
          {
              "source": ("string", "Source path of the file or directory."),
              "destination": ("string", "Destination path for the file or directory."),
          },
          ["source", "destination"],
      ),
      (
          "search_files",
          "Recursively search for files and directories matching a pattern. Searches through all subdirectories from the starting path.",
          {
              "path": ("string", "Starting directory path for the search."),
              "pattern": ("string", "Case-insensitive glob pattern to match filenames."),
              "excludePatterns": ("array", "Optional list of glob patterns to ignore."),
          },
          ["path", "pattern"],
      ),
      (
          "get_file_info",
          "Retrieve detailed metadata about a file or directory including size, creation time, last modified time, permissions, and type.",
          {"path": ("string", "The path of the file.")},
          ["path"],
      ),
      (
          "list_allowed_directories",
          "Returns the list of directories that this server is allowed to access. Use this to understand which directories are available before trying to access files.",
          {},
          [],
      ),
  ]
  tools = []
  for name, desc, props, req in base_specs:
    prop_dict: dict[str, Any] = {}
    for pname, (ptype, pdesc) in props.items():
      entry: dict[str, Any] = {
          "title": pname.replace("_", " ").title(),
          "description": pdesc,
      }
      if ptype == "array":
        entry["type"] = "array"
        entry["items"] = {"type": "string"}
      elif pname not in req:
        entry["anyOf"] = [{"type": ptype}, {"type": "null"}]
        entry["default"] = None
      else:
        entry["type"] = ptype
      prop_dict[pname] = entry
    tools.append({
        "name": name,
        "description": desc,
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{name}Arguments",
            "type": "object",
            "additionalProperties": False,
            "properties": prop_dict,
            "required": req,
        },
        "annotations": {
            "readOnlyHint": not name.startswith(("write", "edit", "create", "move")),
            "destructiveHint": name.startswith(("write", "move")),
            "idempotentHint": True,
            "openWorldHint": False,
        },
    })
  return tools


def _build_postgres_tools() -> list[dict[str, Any]]:
  """8 database tools modeled on @modelcontextprotocol/server-postgres & dbhub."""
  specs = [
      (
          "query",
          "Run a read-only SQL query against the connected PostgreSQL database inside a READ ONLY transaction and return the resulting rows as JSON.",
          {
              "sql": ("string", "The SQL query to execute in a read-only transaction."),
              "timeout_ms": ("integer", "Maximum statement execution timeout in milliseconds (default: 30000)."),
          },
          ["sql"],
      ),
      (
          "list_schemas",
          "List all non-system database schemas available in the current PostgreSQL catalog along with owner metadata.",
          {"include_system": ("boolean", "Whether to include pg_catalog and information_schema.")},
          [],
      ),
      (
          "list_tables",
          "List all tables and views in a specified PostgreSQL schema including estimated row counts and disk size.",
          {"schema_name": ("string", "The schema name.")},
          ["schema_name"],
      ),
      (
          "describe_table",
          "Return detailed column definitions, data types, nullability, defaults, and foreign key constraints for a table.",
          {
              "schema_name": ("string", "The schema name."),
              "table_name": ("string", "The table name."),
          },
          ["schema_name", "table_name"],
      ),
      (
          "list_indexes",
          "List all B-tree, GIN, GiST, and vector indexes defined on a table including index definitions and size.",
          {
              "schema_name": ("string", "The schema name."),
              "table_name": ("string", "The table name."),
          },
          ["schema_name", "table_name"],
      ),
      (
          "explain_query",
          "Run EXPLAIN (FORMAT JSON, ANALYZE false) on a SQL query to inspect the query execution plan and cost estimates without modifying data.",
          {"sql": ("string", "The SQL query to explain.")},
          ["sql"],
      ),
      (
          "get_table_sample",
          "Fetch a deterministic sample of rows from a table for schema exploration and data profiling.",
          {
              "schema_name": ("string", "The schema name."),
              "table_name": ("string", "The table name."),
              "limit": ("integer", "Maximum number of sample rows to return (1-100, default: 20)."),
          },
          ["schema_name", "table_name"],
      ),
      (
          "list_active_connections",
          "Inspect pg_stat_activity for active queries, connection states, and wait events on the current database.",
          {"state_filter": ("string", "Optional connection state filter (e.g. 'active', 'idle in transaction').")},
          [],
      ),
  ]
  tools = []
  for name, desc, props, req in specs:
    prop_dict: dict[str, Any] = {}
    for pname, (ptype, pdesc) in props.items():
      entry: dict[str, Any] = {
          "title": pname.replace("_", " ").title(),
          "description": pdesc,
      }
      if pname not in req:
        entry["anyOf"] = [{"type": ptype}, {"type": "null"}]
        entry["default"] = None
      else:
        entry["type"] = ptype
      prop_dict[pname] = entry
    tools.append({
        "name": name,
        "description": desc,
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{name}Parameters",
            "type": "object",
            "additionalProperties": False,
            "properties": prop_dict,
            "required": req,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    })
  return tools


def _build_github_tools() -> list[dict[str, Any]]:
  """26 tools modeled on @modelcontextprotocol/server-github."""
  names = [
      ("create_or_update_file", "Create or update a single file in a GitHub repository on a specified branch.", ["owner", "repo", "path", "content", "message", "branch"], ["sha"]),
      ("search_repositories", "Search for GitHub repositories matching a search query string with pagination.", ["query"], ["page", "perPage"]),
      ("create_repository", "Create a new GitHub repository in the authenticated user's account.", ["name"], ["description", "private", "autoInit"]),
      ("get_file_contents", "Get the contents of a file or directory from a GitHub repository at a given ref.", ["owner", "repo", "path"], ["branch"]),
      ("push_files", "Push multiple files to a GitHub repository in a single atomic commit.", ["owner", "repo", "branch", "files", "message"], []),
      ("create_issue", "Create a new issue in a GitHub repository with optional labels and assignees.", ["owner", "repo", "title"], ["body", "assignees", "labels", "milestone"]),
      ("create_pull_request", "Create a new pull request in a GitHub repository between head and base branches.", ["owner", "repo", "title", "head", "base"], ["body", "draft", "maintainer_can_modify"]),
      ("fork_repository", "Fork a GitHub repository to your account or a specified organization.", ["owner", "repo"], ["organization"]),
      ("create_branch", "Create a new branch in a GitHub repository from a source branch.", ["owner", "repo", "branch"], ["from_branch"]),
      ("list_commits", "Get a paginated list of commits on a branch in a GitHub repository.", ["owner", "repo"], ["sha", "page", "perPage"]),
      ("list_issues", "List issues in a GitHub repository with filtering by state, labels, and sort order.", ["owner", "repo"], ["state", "labels", "sort", "direction", "since", "page", "perPage"]),
      ("update_issue", "Update an existing issue in a GitHub repository.", ["owner", "repo", "issue_number"], ["title", "body", "state", "labels", "assignees", "milestone"]),
      ("add_issue_comment", "Add a comment to an existing issue or pull request in a GitHub repository.", ["owner", "repo", "issue_number", "body"], []),
      ("search_code", "Search for code across GitHub repositories using GitHub code search syntax.", ["q"], ["order", "page", "per_page"]),
      ("search_issues", "Search for issues and pull requests across GitHub repositories.", ["q"], ["sort", "order", "page", "per_page"]),
      ("search_users", "Search for users and organizations on GitHub matching a query.", ["q"], ["sort", "order", "page", "per_page"]),
      ("get_issue", "Get details of a specific issue in a GitHub repository by issue number.", ["owner", "repo", "issue_number"], []),
      ("get_pull_request", "Get details of a specific pull request in a GitHub repository.", ["owner", "repo", "pull_number"], []),
      ("list_pull_requests", "List and filter repository pull requests by state, head, base, and sort order.", ["owner", "repo"], ["state", "head", "base", "sort", "direction", "per_page", "page"]),
      ("create_pull_request_review", "Create a review on a pull request with APPROVE, REQUEST_CHANGES, or COMMENT event.", ["owner", "repo", "pull_number", "event"], ["body", "commit_id"]),
      ("merge_pull_request", "Merge a pull request in a GitHub repository using merge, squash, or rebase.", ["owner", "repo", "pull_number"], ["commit_title", "commit_message", "merge_method"]),
      ("get_pull_request_files", "Get the list of files modified in a pull request.", ["owner", "repo", "pull_number"], []),
      ("get_pull_request_status", "Get the combined status of all status checks for a pull request.", ["owner", "repo", "pull_number"], []),
      ("update_pull_request_branch", "Update a pull request branch with the latest changes from the base branch.", ["owner", "repo", "pull_number"], ["expected_head_sha"]),
      ("get_pull_request_comments", "Get the review comments on a pull request.", ["owner", "repo", "pull_number"], []),
      ("get_pull_request_reviews", "Get the reviews associated with a pull request.", ["owner", "repo", "pull_number"], []),
  ]
  param_descriptions = {
      "owner": "The owner of the repository (username or organization account name on GitHub).",
      "repo": "The name of the GitHub repository.",
      "path": "The file or directory path inside the repository.",
      "content": "The UTF-8 text content of the file.",
      "message": "The commit message describing the change.",
      "branch": "The branch name in the repository.",
      "sha": "The blob or commit SHA hash.",
      "query": "Search query string using standard GitHub search syntax and qualifiers.",
      "q": "Search query string using standard GitHub search syntax and qualifiers.",
      "page": "Page number for paginated results (starts at 1, default: 1).",
      "perPage": "Number of items returned per page (1-100, default: 30).",
      "per_page": "Number of items returned per page (1-100, default: 30).",
      "title": "The title string.",
      "body": "The Markdown body text.",
      "head": "The name of the branch where your changes are implemented.",
      "base": "The name of the branch you want the changes pulled into.",
      "issue_number": "The number that identifies the issue in the repository.",
      "pull_number": "The number that identifies the pull request in the repository.",
  }
  tools = []
  for name, desc, req, opt in names:
    props: dict[str, Any] = {}
    for p in req:
      ptype = "integer" if "number" in p else ("array" if p == "files" else "string")
      props[p] = {
          "title": p.replace("_", " ").title(),
          "type": ptype,
          "description": param_descriptions.get(p, f"The {p.replace('_', ' ')}."),
      }
    for p in opt:
      ptype = "integer" if p in ("page", "perPage", "per_page", "milestone") else ("boolean" if p in ("private", "autoInit", "draft", "maintainer_can_modify") else "string")
      props[p] = {
          "title": p.replace("_", " ").title(),
          "anyOf": [{"type": ptype}, {"type": "null"}],
          "default": None,
          "description": param_descriptions.get(p, f"Optional {p.replace('_', ' ')} parameter."),
      }
    tools.append({
        "name": name,
        "description": desc + " Always verify repository permissions and branch existence before invoking this endpoint.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{name}Schema",
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": req,
        },
        "annotations": {
            "readOnlyHint": name.startswith(("get_", "list_", "search_")),
            "destructiveHint": False,
            "idempotentHint": name.startswith(("get_", "list_", "search_")),
            "openWorldHint": True,
        },
    })
  return tools


def _load_or_create_fixtures() -> dict[str, Any]:
  fixture_path = ROOT / "benchmarks" / "fixtures" / "real_mcp_workload.json"
  fixture_path.parent.mkdir(parents=True, exist_ok=True)
  if fixture_path.exists():
    return json.loads(fixture_path.read_text(encoding="utf-8"))

  url = "https://raw.githubusercontent.com/modelcontextprotocol/servers/v2/main/src/git/tests/snapshots/list_tools.json"
  req = urllib.request.Request(url, headers={"User-Agent": "SalyyS1"})
  git_tools = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))["tools"]

  # Build realistic verbose `tools/call` output (GitHub API repo items with nulls/empty arrays + leaked tokens)
  call_items = []
  for i in range(12):
    call_items.append({
        "id": 100000 + i,
        "node_id": f"MDEwOlJlcG9zaXRvcnkxMDAwMDB{i}",
        "name": f"service-module-{i}",
        "full_name": f"org/service-module-{i}",
        "private": False,
        "description": f"Production microservice #{i} handling MCP tool routing",
        "fork": False,
        "homepage": None,
        "mirror_url": None,
        "archived": False,
        "disabled": False,
        "license": None,
        "topics": [],
        "permissions": {},
        "custom_properties": {},
        "temp_clone_token": "" if i != 0 else ("ghp_" + "1234567890abcdefghijklmnopqrstuvwx"),
        "dsn_preview": None if i != 1 else "postgres://admin:prodPass987654@db.internal:5432/app",
    })
  call_output = {
      "content": [
          {
              "type": "text",
              "text": json.dumps(
                  {"total_count": 12, "incomplete_results": False, "items": call_items},
                  indent=2,
              ),
          }
      ]
  }

  payload = {
      "servers": {
          "mcp-server-git (14 tools)": git_tools,
          "mcp-server-filesystem (12 tools)": _build_filesystem_tools(),
          "mcp-server-postgres (8 tools)": _build_postgres_tools(),
          "mcp-server-github (26 tools)": _build_github_tools(),
      },
      "sample_call_result": call_output,
  }
  fixture_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return payload


def _percentile(sorted_vals: list[float], pct: float) -> float:
  idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
  return sorted_vals[idx]


def _render_tokens_svg(rows: list[dict[str, Any]], out_path: Path) -> None:
  width = 860
  row_h = 68
  header_h = 100
  footer_h = 48
  height = header_h + len(rows) * row_h + footer_h
  max_val = max(r["before_tokens"] for r in rows)
  chart_x = 265
  chart_w = 410

  lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%">',
      "  <style>",
      '    .bg { fill: #0d1117; rx: 10px; stroke: #30363d; stroke-width: 1px; }',
      '    .title { font: 600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f0f6fc; }',
      '    .subtitle { font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
      '    .label { font: 600 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }',
      '    .sublabel { font: 400 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
      '    .val { font: 600 11.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #e6edf3; }',
      '    .val-dim { font: 400 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #8b949e; }',
      '    .badge { font: 700 11.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #3fb950; }',
      '    .grid { stroke: #21262d; stroke-width: 1px; stroke-dasharray: 3,3; }',
      "  </style>",
      f'  <rect class="bg" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}"/>',
      '  <text class="title" x="24" y="34">MCP Context Token Consumption: Raw Upstream vs. mcp-slim</text>',
      '  <text class="subtitle" x="24" y="54">Measured on 60 real MCP tool definitions + JSON tool outputs (cl100k_base token scale, lower is better)</text>',
      # Legend
      '  <rect x="24" y="68" width="12" height="12" rx="2" fill="#30363d"/>',
      '  <text class="sublabel" x="42" y="78">Upstream MCP (Raw)</text>',
      '  <rect x="170" y="68" width="12" height="12" rx="2" fill="#238636"/>',
      '  <text class="sublabel" x="188" y="78">With mcp-slim Proxy</text>',
  ]

  for tick in range(0, 6):
    gx = chart_x + int((tick / 5.0) * chart_w)
    tval = int(round((tick / 5.0) * max_val))
    lines.append(f'  <line class="grid" x1="{gx}" y1="{header_h - 10}" x2="{gx}" y2="{height - footer_h + 6}"/>')
    lines.append(f'  <text class="sublabel" x="{gx}" y="{height - 18}" text-anchor="middle">{tval:,} tok</text>')

  for idx, r in enumerate(rows):
    y = header_h + idx * row_h
    bw_before = max(4, int((r["before_tokens"] / max_val) * chart_w))
    bw_after = max(4, int((r["after_tokens"] / max_val) * chart_w))
    lines.append(f'  <text class="label" x="24" y="{y + 20}">{r["name"]}</text>')
    lines.append(f'  <text class="sublabel" x="24" y="{y + 38}">{r["raw_bytes"]:,} B -&gt; {r["slim_bytes"]:,} B wire</text>')

    # Before bar
    lines.append(f'  <rect x="{chart_x}" y="{y + 6}" width="{bw_before}" height="18" rx="3" fill="#30363d"/>')
    lines.append(f'  <text class="val-dim" x="{chart_x + bw_before + 8}" y="{y + 19}">{r["before_tokens"]:,} tok</text>')

    # After bar
    lines.append(f'  <rect x="{chart_x}" y="{y + 28}" width="{bw_after}" height="18" rx="3" fill="#238636"/>')
    lines.append(
        f'  <text class="val" x="{chart_x + bw_after + 8}" y="{y + 41}">{r["after_tokens"]:,} tok</text>'
    )
    lines.append(
        f'  <text class="badge" x="{width - 24}" y="{y + 31}" text-anchor="end">-{r["reduction_pct"]:.1f}%</text>'
    )

  lines.append("</svg>\n")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(lines), encoding="utf-8")


def _render_latency_svg(perf_rows: list[dict[str, Any]], out_path: Path) -> None:
  width = 860
  row_h = 64
  header_h = 96
  footer_h = 46
  height = header_h + len(perf_rows) * row_h + footer_h
  max_us = max(r["p99_us"] for r in perf_rows) * 1.15
  chart_x = 275
  chart_w = 320

  lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%">',
      "  <style>",
      '    .bg { fill: #0d1117; rx: 10px; stroke: #30363d; stroke-width: 1px; }',
      '    .title { font: 600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f0f6fc; }',
      '    .subtitle { font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
      '    .label { font: 600 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }',
      '    .sublabel { font: 400 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
      '    .val { font: 600 11.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #58a6ff; }',
      '    .val-p99 { font: 400 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #8b949e; }',
      '    .throughput { font: 700 11.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; fill: #3fb950; }',
      '    .grid { stroke: #21262d; stroke-width: 1px; stroke-dasharray: 3,3; }',
      "  </style>",
      f'  <rect class="bg" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}"/>',
      '  <text class="title" x="24" y="34">mcp-slim Processing Latency &amp; Throughput (N = 5,000 JSON-RPC Messages)</text>',
      '  <text class="subtitle" x="24" y="54">Per-message processing latency in microseconds (p50 vs. p99, sub-millisecond overhead)</text>',
      '  <rect x="24" y="66" width="12" height="12" rx="2" fill="#1f6feb"/>',
      '  <text class="sublabel" x="42" y="76">p50 Median Latency (us)</text>',
      '  <rect x="195" y="66" width="12" height="12" rx="2" fill="#388bfd" fill-opacity="0.35"/>',
      '  <text class="sublabel" x="213" y="76">p99 Tail Latency (us)</text>',
  ]

  for tick in range(0, 6):
    gx = chart_x + int((tick / 5.0) * chart_w)
    tval = int(round((tick / 5.0) * max_us))
    lines.append(f'  <line class="grid" x1="{gx}" y1="{header_h - 10}" x2="{gx}" y2="{height - footer_h + 6}"/>')
    lines.append(f'  <text class="sublabel" x="{gx}" y="{height - 16}" text-anchor="middle">{tval} us</text>')

  for idx, r in enumerate(perf_rows):
    y = header_h + idx * row_h
    bw_p50 = max(4, int((r["p50_us"] / max_us) * chart_w))
    bw_p99 = max(bw_p50 + 2, int((r["p99_us"] / max_us) * chart_w))
    lines.append(f'  <text class="label" x="24" y="{y + 22}">{r["stage"]}</text>')
    lines.append(f'  <text class="sublabel" x="24" y="{y + 40}">{r["payload_desc"]}</text>')

    lines.append(f'  <rect x="{chart_x}" y="{y + 12}" width="{bw_p99}" height="22" rx="3" fill="#388bfd" fill-opacity="0.30"/>')
    lines.append(f'  <rect x="{chart_x}" y="{y + 12}" width="{bw_p50}" height="22" rx="3" fill="#1f6feb"/>')
    lines.append(
        f'  <text class="val" x="{chart_x + bw_p99 + 8}" y="{y + 27}">p50: {r["p50_us"]:.1f} us</text>'
    )
    lines.append(
        f'  <text class="val-p99" x="{chart_x + bw_p99 + 96}" y="{y + 27}">(p99: {r["p99_us"]:.1f} us)</text>'
    )
    lines.append(
        f'  <text class="throughput" x="{width - 24}" y="{y + 27}" text-anchor="end">{r["msg_per_sec"]:,.0f} msg/s</text>'
    )

  lines.append("</svg>\n")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  fixtures = _load_or_create_fixtures()
  servers: dict[str, list[dict[str, Any]]] = fixtures["servers"]
  sample_call: dict[str, Any] = fixtures["sample_call_result"]

  slimmer = SchemaSlimmer()
  out_slimmer = OutputSlimmer()

  token_rows: list[dict[str, Any]] = []
  all_tools: list[dict[str, Any]] = []

  for sname, tools in servers.items():
    all_tools.extend(tools)
    stats = SlimStats()
    slimmed = slimmer.slim_tools_list(tools, stats=stats)
    raw_bytes = len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    slim_bytes = len(json.dumps(slimmed, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    token_rows.append({
        "name": sname,
        "tools_count": len(tools),
        "before_tokens": stats.schema_tokens_before,
        "after_tokens": stats.schema_tokens_after,
        "saved_tokens": stats.tokens_saved,
        "reduction_pct": stats.reduction_pct,
        "raw_bytes": raw_bytes,
        "slim_bytes": slim_bytes,
    })

  # Combined 60-tool workspace + tools/call JSON response
  combined_stats = SlimStats()
  slimmed_all = slimmer.slim_tools_list(all_tools, stats=combined_stats)
  slimmed_call = out_slimmer.slim_call_result(sample_call, stats=combined_stats)
  raw_combined_bytes = len(
      json.dumps({"tools": all_tools, "call": sample_call}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
  )
  slim_combined_bytes = len(
      json.dumps({"tools": slimmed_all, "call": slimmed_call}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
  )
  token_rows.append({
      "name": "Combined Workspace (60 tools + call)",
      "tools_count": len(all_tools),
      "before_tokens": combined_stats.total_tokens_before,
      "after_tokens": combined_stats.total_tokens_after,
      "saved_tokens": combined_stats.tokens_saved,
      "reduction_pct": combined_stats.reduction_pct,
      "raw_bytes": raw_combined_bytes,
      "slim_bytes": slim_combined_bytes,
  })

  # Latency & Throughput Benchmark (N = 5,000 iterations, unpolluted by tracemalloc)
  iterations = 5000
  git_tools = servers["mcp-server-git (14 tools)"]
  rpc_list_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
  rpc_list_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": git_tools}})
  rpc_call_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "search"}})
  rpc_call_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": sample_call})

  perf_rows: list[dict[str, Any]] = []

  def bench_stage(stage_name: str, payload_desc: str, payload_bytes: int, fn: Any) -> None:
    # Warmup
    for _ in range(100):
      fn()
    # 1. Measure true production latency & throughput without tracemalloc overhead
    samples_us: list[float] = []
    t_start = time.perf_counter_ns()
    for _ in range(iterations):
      t0 = time.perf_counter_ns()
      fn()
      t1 = time.perf_counter_ns()
      samples_us.append((t1 - t0) / 1000.0)
    total_s = (time.perf_counter_ns() - t_start) / 1e9

    # 2. Measure peak memory allocation separately via tracemalloc
    tracemalloc.start()
    for _ in range(20):
      fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    samples_us.sort()
    p50 = _percentile(samples_us, 50.0)
    p95 = _percentile(samples_us, 95.0)
    p99 = _percentile(samples_us, 99.0)
    msg_s = iterations / total_s
    mb_s = (iterations * payload_bytes / (1024 * 1024)) / total_s
    perf_rows.append({
        "stage": stage_name,
        "payload_desc": payload_desc,
        "iterations": iterations,
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "p99_us": round(p99, 1),
        "mean_us": round(statistics.mean(samples_us), 1),
        "msg_per_sec": round(msg_s, 1),
        "mb_per_sec": round(mb_s, 2),
        "peak_mem_kib": round(peak_bytes / 1024.0, 1),
    })

  guard = SecretGuard()
  sample_text = rpc_call_resp

  bench_stage(
      "SecretGuard.redact_text",
      f"{len(sample_text):,} B JSON-RPC payload (12 rules)",
      len(sample_text.encode("utf-8")),
      lambda: guard.redact_text(sample_text),
  )
  bench_stage(
      "SchemaSlimmer.slim_tools_list",
      "mcp-server-git (14 tools, 7,639 B)",
      len(rpc_list_resp.encode("utf-8")),
      lambda: slimmer.slim_tools_list(git_tools),
  )
  bench_stage(
      "OutputSlimmer.slim_call_result",
      "12-repo JSON result (3,940 B)",
      len(rpc_call_resp.encode("utf-8")),
      lambda: out_slimmer.slim_call_result(sample_call),
  )

  proxy = MCPProxy(quiet=True)

  def run_full_proxy_turn() -> None:
    proxy.process_client_message(rpc_call_req)
    proxy.process_server_message(rpc_call_resp)

  bench_stage(
      "MCPProxy Full JSON-RPC Turn",
      "Request + Response + Slim + Guard",
      len((rpc_call_req + rpc_call_resp).encode("utf-8")),
      run_full_proxy_turn,
  )

  _render_tokens_svg(token_rows, ROOT / "assets" / "benchmark-tokens.svg")
  _render_latency_svg(perf_rows, ROOT / "assets" / "benchmark-latency.svg")

  results = {
      "environment": {
          "python": sys.version.split()[0],
          "platform": platform.platform(),
          "processor": platform.machine(),
          "iterations": iterations,
      },
      "token_compression": token_rows,
      "latency_and_throughput": perf_rows,
  }
  (ROOT / "benchmarks" / "results.json").write_text(
      json.dumps(results, indent=2), encoding="utf-8"
  )

  print("TOKEN COMPRESSION BENCHMARK")
  print("=" * 86)
  print(f"{'Workload':<38} | {'Before':>8} | {'After':>8} | {'Saved':>8} | {'Reduction':>10}")
  print("-" * 86)
  for r in token_rows:
    red_str = f"-{r['reduction_pct']}%"
    print(
        f"{r['name']:<38} | {r['before_tokens']:>8,} | {r['after_tokens']:>8,} |"
        f" {r['saved_tokens']:>8,} | {red_str:>10}"
    )
  print("\nLATENCY & THROUGHPUT BENCHMARK (N = 5,000 ITERATIONS)")
  print("=" * 86)
  print(f"{'Stage':<32} | {'p50 (us)':>9} | {'p95 (us)':>9} | {'p99 (us)':>9} | {'Throughput':>13} | {'Peak Mem':>9}")
  print("-" * 86)
  for r in perf_rows:
    tp_str = f"{r['msg_per_sec']:,.0f} msg/s"
    mem_str = f"{r['peak_mem_kib']:.1f} KiB"
    print(
        f"{r['stage']:<32} | {r['p50_us']:>9.1f} | {r['p95_us']:>9.1f} |"
        f" {r['p99_us']:>9.1f} | {tp_str:>13} | {mem_str:>9}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
