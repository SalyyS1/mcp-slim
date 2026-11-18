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
    CompositeSchemaMerger,
    LazyToolBroker,
    LatencyHistogram,
    OutputSlimmer,
    SchemaSlimmer,
    SessionPolicy,
    SessionPolicyEngine,
    SlimStats,
    SseChunkThrottle,
    TemplateArgumentShield,
    ToolCatalogPruner,
    audit_session_policy,
    audit_tools_manifest,
    build_audit_sarif,
    diff_tools_manifests,
    estimate_tokens,
    shield_template_arguments,
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


def run_bench(
    turns: int = 10,
    *,
    sse_burst_budget: int = 0,
    json_output: bool = False,
) -> int:
  """Multi-turn compression simulation with latency histogram (mcp-slim bench).

  Simulates `turns` back-to-back tool-call + schema-discovery rounds using the
  built-in benchmark fixtures, records per-method latency samples, optionally
  exercises SseChunkThrottle with `sse_burst_budget`, and prints a detailed
  report.  With --json the output is machine-readable JSON.
  """
  import time as _time

  slimmer = SchemaSlimmer()
  out_slimmer = OutputSlimmer()
  guard = SecretGuard()
  histogram = LatencyHistogram(
      method_allowlist=["tools/list", "tools/call"],
  )
  throttle = SseChunkThrottle(burst_budget_tokens=sse_burst_budget) if sse_burst_budget > 0 else None

  total_before = 0
  total_after = 0
  total_secrets = 0
  chunks_seen = 0
  chunks_throttled = 0

  for _ in range(turns):
    # tools/list (schema slimming)
    stats_sl = SlimStats()
    t0 = _time.perf_counter()
    slimmer.slim_tools_list(SAMPLE_BENCHMARK_TOOLS, stats=stats_sl)
    t1 = _time.perf_counter()
    histogram.record("tools/list", (t1 - t0) * 1000.0)
    total_before += stats_sl.schema_tokens_before
    total_after += stats_sl.schema_tokens_after

    # tools/call (output slimming + secret redaction)
    stats_out = SlimStats()
    t2 = _time.perf_counter()
    slimmed = out_slimmer.slim_call_result(SAMPLE_BENCHMARK_OUTPUT, stats=stats_out)
    t3 = _time.perf_counter()
    histogram.record("tools/call", (t3 - t2) * 1000.0)
    redacted = guard.redact_value(slimmed)
    total_before += stats_out.output_tokens_before
    total_after += stats_out.output_tokens_after
    total_secrets += guard.report.total_redactions

    # SSE chunk throttle simulation: treat slimmed output as one SSE chunk
    if throttle is not None:
      chunk_text = str(redacted)
      decision = throttle.evaluate(chunk_text)
      chunks_seen += 1
      if not decision.allowed:
        chunks_throttled += 1

  saved = max(0, total_before - total_after)
  reduction_pct = round(saved / total_before * 100.0, 1) if total_before > 0 else 0.0
  sl_stats = histogram.stats("tools/list")
  tc_stats = histogram.stats("tools/call")
  del sl_stats, tc_stats

  pol_engine = SessionPolicyEngine(
      policy={
          "session_id": "bench",
          "allow_cross_server_delegation": False,
          "server_scopes": {
              "git": {"read_only": True, "sampling_enabled": False},
              "prod_db": {"allowed_tools": ["select_*"], "read_only": True},
          },
      },
      enabled=True,
  )
  tpl_shield = TemplateArgumentShield(enabled=True, max_depth=1, block_exfiltration=True)
  sec_hist = LatencyHistogram(method_allowlist=["session_policy", "template_shield"])
  pol_blocked = 0
  tpl_blocked = 0
  sample_pol_req = {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {"name": "prod_db__drop_users", "arguments": {}},
  }
  sample_tpl_args = {
      "query": "{{ outer {{ env.AWS_SECRET_ACCESS_KEY }} }}",
      "path": "../../etc/passwd",
  }
  for _ in range(turns):
    tp0 = _time.perf_counter()
    p_dec = pol_engine.evaluate_client_request(sample_pol_req, server_id="git")
    tp1 = _time.perf_counter()
    sec_hist.record("session_policy", (tp1 - tp0) * 1000.0)
    if not p_dec.allowed:
      pol_blocked += 1

    ts0 = _time.perf_counter()
    t_dec = tpl_shield.audit_arguments(
        sample_tpl_args,
        template_type="resource",
        uri_template="file:///workspace/{path}",
    )
    ts1 = _time.perf_counter()
    sec_hist.record("template_shield", (ts1 - ts0) * 1000.0)
    if not t_dec.allowed:
      tpl_blocked += 1

  security_bench = {
      "policy_evaluations": turns,
      "policy_escalations_blocked": pol_blocked,
      "template_evaluations": turns,
      "template_exfiltrations_blocked": tpl_blocked,
      "latency_ms": sec_hist.all_stats(),
  }

  if json_output:
    import json as _json
    out = {
        "turns": turns,
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "tokens_saved": saved,
        "reduction_pct": reduction_pct,
        "secrets_redacted": total_secrets,
        "latency_ms": histogram.all_stats(),
        "sse_throttle": throttle.snapshot() if throttle is not None else None,
        "security_bench": security_bench,
    }
    print(_json.dumps(out, indent=2))
    return 0

  print("mcp-slim Bench Report")
  print("=" * 76)
  print(f"Turns simulated : {turns}")
  print(f"Tokens before   : {total_before:,}")
  print(f"Tokens after    : {total_after:,}")
  print(f"Tokens saved    : {saved:,}  (-{reduction_pct}%)")
  print(f"Secrets redacted: {total_secrets}")
  print("-" * 76)
  print(f"{'Method':<16} | {'Count':>6} | {'Mean ms':>8} | {'p50 ms':>7} | {'p90 ms':>7} | {'p99 ms':>7}")
  print("-" * 76)
  for method, s in histogram.all_stats().items():
    print(
        f"{method:<16} | {s['count']:>6} | {s['mean_ms']:>8.3f} | "
        f"{s['p50_ms']:>7.3f} | {s['p90_ms']:>7.3f} | {s['p99_ms']:>7.3f}"
    )
  if throttle is not None:
    snap = throttle.snapshot()
    print("-" * 76)
    print(
        f"SSE throttle: burst_budget={snap['burst_budget_tokens']} tokens, "
        f"chunks={snap['chunks_seen']}, throttled={snap['chunks_throttled']} "
        f"({round(snap['throttle_rate'] * 100, 1)}%)"
    )
  print("=" * 76)
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


def _load_tools_file(path: Path) -> list[dict[str, Any]] | None:
  raw_data = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(raw_data, list):
    return raw_data
  if isinstance(raw_data, dict) and isinstance(raw_data.get("tools"), list):
    return raw_data["tools"]
  if isinstance(raw_data, dict) and isinstance(raw_data.get("result"), dict):
    inner = raw_data["result"].get("tools")
    if isinstance(inner, list):
      return inner
  if isinstance(raw_data, dict) and isinstance(raw_data.get("servers"), dict):
    return [
        t
        for srv_tools in raw_data["servers"].values()
        if isinstance(srv_tools, list)
        for t in srv_tools
    ]
  return None


def run_audit(
    path: Path | None = None,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_security: bool = False,
) -> int:
  if path is not None:
    loaded = _load_tools_file(path)
    if loaded is None:
      sys.stderr.write(f"Could not locate a tools list inside {path}\n")
      return 1
    tools = loaded
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
        if f["rule_id"] in ("MS001", "MS002", "MS008"):
          print(f"  [{f['rule_id']}] {f['message']}")

  if fail_on_security and report["security_findings"] > 0:
    return 2
  return 0


def run_diff(
    before_path: Path,
    after_path: Path,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_mutation: bool = False,
    allow_new_tools: bool = False,
) -> int:
  before_tools = _load_tools_file(before_path)
  if before_tools is None:
    sys.stderr.write(f"Could not locate a tools list inside {before_path}\n")
    return 1
  after_tools = _load_tools_file(after_path)
  if after_tools is None:
    sys.stderr.write(f"Could not locate a tools list inside {after_path}\n")
    return 1

  report = diff_tools_manifests(
      before_tools, after_tools, allow_new_tools=allow_new_tools
  )
  if sarif_path is not None:
    sarif_doc = build_audit_sarif(
        report,
        source_uri=str(after_path),
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
    print(
        f"mcp-slim Manifest Diff ({before_path.name} -> {after_path.name})"
    )
    print("=" * 78)
    print(
        f"Tools: {report['tools_before']} -> {report['tools_after']} | "
        f"Unchanged: {report['unchanged']} | Modified: {report['modified']} | "
        f"Escalations: {report['privilege_escalations']} | "
        f"Added: {report['added']} | Removed: {report['removed']}"
    )
    if report["findings"]:
      print("-" * 78)
      for f in report["findings"]:
        print(f"[{f['rule_id']}:{f['level'].upper()}] {f['message']}")
    print("=" * 78)

  if fail_on_mutation and report["blocking_mutations"] > 0:
    return 2
  return 0


def run_merge(
    inputs: list[str],
    *,
    strategy: str = "prefix_colliding",
    separator: str = "__",
    slim: bool = False,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_conflict: bool = False,
) -> int:
  server_manifests: dict[str, list[dict[str, Any]]] = {}
  for spec in inputs:
    if "=" in spec:
      srv_id, raw_path = spec.split("=", 1)
      path = Path(raw_path)
      explicit_id: str | None = srv_id.strip() or path.stem
    else:
      path = Path(spec)
      explicit_id = None

    if not path.exists():
      sys.stderr.write(f"Manifest file not found: {path}\n")
      return 1
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if (
        explicit_id is None
        and isinstance(raw_data, dict)
        and isinstance(raw_data.get("servers"), dict)
    ):
      for sub_srv, sub_tools in raw_data["servers"].items():
        if isinstance(sub_tools, list):
          server_manifests[str(sub_srv)] = sub_tools
      continue

    loaded = _load_tools_file(path)
    if loaded is None:
      sys.stderr.write(f"Could not locate a tools list inside {path}\n")
      return 1
    srv_key = explicit_id or path.stem
    if srv_key in server_manifests:
      idx = 2
      while f"{srv_key}_{idx}" in server_manifests:
        idx += 1
      srv_key = f"{srv_key}_{idx}"
    server_manifests[srv_key] = loaded

  merger = CompositeSchemaMerger(
      strategy=strategy,
      separator=separator,
      slim_schemas=slim,
  )
  merged = merger.merge(server_manifests)
  report = merged.to_audit_report()

  if sarif_path is not None:
    sarif_doc = build_audit_sarif(
        report,
        source_uri="composite_manifest.json",
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
    print(
        f"mcp-slim Composite Schema Merge ({merged.servers_merged} server(s), strategy={strategy})"
    )
    print("=" * 78)
    print(
        f"Tools: {merged.tools_input} -> {merged.tools_output} | "
        f"Collisions: {merged.tool_collisions} | "
        f"Deduped Tools: {merged.deduplicated_tools} | "
        f"Shared $defs Deduped: {merged.shared_defs_deduped} | "
        f"$defs Conflicts Resolved: {merged.def_conflicts_resolved}"
    )
    print(
        f"Tokens: {merged.tokens_before} -> {merged.tokens_after} "
        f"(-{merged.reduction_pct}%)"
    )
    if merged.conflicts:
      print("-" * 78)
      for c in merged.conflicts:
        print(f"[{c['rule_id']}:{c['level'].upper()}] {c['message']}")
    print("=" * 78)

  if fail_on_conflict and merged.has_conflicts:
    return 2
  return 0


def run_prune(
    path: Path | None,
    *,
    intent: str,
    top_k: int = 12,
    min_score_ratio: float = 0.15,
    always_keep: list[str] | None = None,
    slim: bool = False,
    json_output: bool = False,
    sarif_path: Path | None = None,
) -> int:
  if path is not None:
    if not path.exists():
      sys.stderr.write(f"Manifest file not found: {path}\n")
      return 1
    loaded = _load_tools_file(path)
    if loaded is None:
      sys.stderr.write(f"Could not locate a tools list inside {path}\n")
      return 1
    tools = loaded
  else:
    tools = SAMPLE_BENCHMARK_TOOLS

  pruner = ToolCatalogPruner(
      intent=intent,
      top_k=top_k,
      min_score_ratio=min_score_ratio,
      always_keep=always_keep,
      slim_schemas=slim,
  )
  res = pruner.prune(tools)
  report = res.to_audit_report()

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
    return 0

  print(f"mcp-slim BM25+ Intent Catalog Pruner (intent={intent!r}, top_k={top_k})")
  print("=" * 78)
  print(
      f"Tools: {res.tools_before} -> {res.tools_after} (pruned {res.tools_pruned}) | "
      f"Tokens: {res.tokens_before} -> {res.tokens_after} (-{res.reduction_pct}%)"
  )
  print("-" * 78)
  print(f"{'TOOL NAME':<34} | {'BM25+':>8} | {'STATUS':>8} | {'MATCHED TERMS'}")
  print("-" * 78)
  for row in res.ranked_scores[:25]:
    status_lbl = "PINNED" if row["pinned"] else ("KEPT" if row["kept"] else "PRUNED")
    terms_str = ", ".join(row["matched_terms"][:6])
    print(
        f"{row['name']:<34.34} | {row['score']:>8.3f} | {status_lbl:>8} | {terms_str}"
    )
  print("=" * 78)
  return 0


def run_policy(
    policy_path: Path,
    target_path: Path,
    *,
    server_id: str = "",
    target_server_id: str | None = None,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_violation: bool = False,
) -> int:
  if not policy_path.exists():
    sys.stderr.write(f"Policy file not found: {policy_path}\n")
    return 1
  if not target_path.exists():
    sys.stderr.write(f"Target JSON file not found: {target_path}\n")
    return 1

  pol_data = json.loads(policy_path.read_text(encoding="utf-8"))
  target_data = json.loads(target_path.read_text(encoding="utf-8"))
  engine = SessionPolicyEngine(policy=pol_data, enabled=True)

  if isinstance(target_data, list):
    report = engine.audit_session_transcript(target_data)
  elif isinstance(target_data, dict) and isinstance(target_data.get("events"), list):
    report = engine.audit_session_transcript(target_data["events"])
  elif isinstance(target_data, dict):
    dec = audit_session_policy(
        target_data,
        pol_data,
        server_id=server_id,
        target_server_id=target_server_id,
    )
    report = dec.to_dict()
  else:
    sys.stderr.write(f"Unsupported target payload in {target_path}\n")
    return 1

  if sarif_path is not None:
    sarif_doc = build_audit_sarif(
        report,
        source_uri=str(target_path),
        version=__version__,
    )
    sarif_path.parent.mkdir(parents=True, exist_ok=True)
    sarif_path.write_text(
        json.dumps(sarif_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

  has_violations = int(report.get("violation_count", 0)) > 0
  if json_output:
    print(json.dumps(report, indent=2, ensure_ascii=False))
  else:
    verdict = "BLOCKED" if has_violations else "ALLOWED"
    print(f"mcp-slim Session Policy Audit ({verdict}, violations={report.get('violation_count', 0)})")
    print("=" * 78)
    for v in report.get("violations", []):
      print(f"[{v.get('rule_id', 'MS019')}:{v.get('violation_code', '')}] {v.get('message', '')}")
    print("=" * 78)

  if fail_on_violation and has_violations:
    return 2
  return 0


def run_shield(
    args_path: Path,
    *,
    template_type: str = "prompt",
    uri_template: str = "",
    max_depth: int = 1,
    json_output: bool = False,
    sarif_path: Path | None = None,
    fail_on_violation: bool = False,
) -> int:
  if not args_path.exists():
    sys.stderr.write(f"Arguments JSON file not found: {args_path}\n")
    return 1

  raw_obj = json.loads(args_path.read_text(encoding="utf-8"))
  if isinstance(raw_obj, dict) and isinstance(raw_obj.get("arguments"), dict):
    eff_args = raw_obj["arguments"]
    eff_uri = uri_template or str(raw_obj.get("uri_template") or raw_obj.get("name") or "")
    eff_type = (
        str(raw_obj["template_type"])
        if isinstance(raw_obj.get("template_type"), str)
        else template_type
    )
  elif isinstance(raw_obj, dict):
    eff_args = raw_obj
    eff_uri = uri_template
    eff_type = template_type
  else:
    sys.stderr.write(f"Expected JSON object in {args_path}\n")
    return 1

  dec = shield_template_arguments(
      template_type=eff_type,
      arguments=eff_args,
      uri_template=eff_uri,
      max_depth=max_depth,
      block_exfiltration=True,
  )
  report = dec.to_dict()

  if sarif_path is not None:
    sarif_doc = build_audit_sarif(
        report,
        source_uri=str(args_path),
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
    verdict = "ALLOWED" if dec.allowed else "BLOCKED"
    print(f"mcp-slim Template Argument Shield ({verdict}, violations={len(dec.violations)})")
    print("=" * 78)
    for v in dec.violations:
      print(f"[{v.rule_id}:{v.violation_code}] ({v.argument_key}) {v.message}")
    print("=" * 78)

  if fail_on_violation and not dec.allowed:
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
    block_sampling: bool = False,
    block_elicitation: bool = False,
    block_roots: bool = False,
    allow_root_prefixes: str | None = None,
    pin_schemas: bool = False,
    tabular_arrays: bool = False,
    allow_resource_prefixes: str | None = None,
    deny_resource_prefixes: str | None = None,
    token_budget: int = 0,
    window_decay: float = 0.85,
    repair_truncated_json: bool = False,
    prune_tools: str | None = None,
    top_k_tools: int = 12,
    strict_rpc_ids: bool = False,
    session_policy: str | None = None,
    shield_templates: bool = False,
    max_template_depth: int = 1,
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
    if block_sampling:
      prefix.append("--block-sampling")
    if block_elicitation:
      prefix.append("--block-elicitation")
    if block_roots:
      prefix.append("--block-roots")
    if allow_root_prefixes:
      prefix.extend(["--allow-root-prefixes", allow_root_prefixes])
    if pin_schemas:
      prefix.append("--pin-schemas")
    if tabular_arrays:
      prefix.append("--tabular-arrays")
    if allow_resource_prefixes:
      prefix.extend(["--allow-resource-prefixes", allow_resource_prefixes])
    if deny_resource_prefixes:
      prefix.extend(["--deny-resource-prefixes", deny_resource_prefixes])
    if token_budget > 0:
      prefix.extend(["--token-budget", str(token_budget)])
      if window_decay != 0.85:
        prefix.extend(["--window-decay", str(window_decay)])
    if repair_truncated_json:
      prefix.append("--repair-truncated-json")
    if prune_tools:
      prefix.extend(["--prune-tools", prune_tools])
      if top_k_tools != 12:
        prefix.extend(["--top-k-tools", str(top_k_tools)])
    if strict_rpc_ids:
      prefix.append("--strict-rpc-ids")
    if session_policy:
      prefix.extend(["--session-policy", session_policy, "--server-id", str(name)])
    if shield_templates:
      prefix.append("--shield-templates")
      if max_template_depth != 1:
        prefix.extend(["--max-template-depth", str(max_template_depth)])
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
      "--block-sampling",
      action="store_true",
      help="Reject server-initiated sampling/createMessage requests locally with a JSON-RPC policy error",
  )
  run_p.add_argument(
      "--block-elicitation",
      action="store_true",
      help="Reject all server-initiated elicitation/create user-prompt dialogs locally with a JSON-RPC policy error",
  )
  run_p.add_argument(
      "--allow-credential-elicitation",
      action="store_true",
      help="Allow elicitation/create schemas that request passwords, API keys, or OTP codes (blocked by default when SecretGuard is active)",
  )
  run_p.add_argument(
      "--block-result-injection",
      action="store_true",
      help="Scan tools/call response content for prompt-injection directives (MS009) and replace poisoned items with a redaction notice",
  )
  run_p.add_argument(
      "--detect-replay",
      action="store_true",
      help="Detect server-side result substitution attacks (MS010): alert when the same tool returns identical result fingerprints for different arguments",
  )
  run_p.add_argument(
      "--repair-truncated-json",
      action="store_true",
      help="Structurally repair truncated or lightly malformed streaming tool-call JSON arguments (MS015)",
  )
  run_p.add_argument(
      "--prune-tools",
      type=str,
      default=None,
      help="Prune tools/list catalog using multi-field BM25+ ranking for the given task intent (MS017)",
  )
  run_p.add_argument(
      "--top-k-tools",
      type=int,
      default=12,
      help="Maximum tools retained when --prune-tools is active (default: 12)",
  )
  run_p.add_argument(
      "--strict-rpc-ids",
      action="store_true",
      help="Enforce stateful JSON-RPC 2.0 response ID correlation, replay blocking, and batch firewall (MS018)",
  )
  run_p.add_argument(
      "--session-policy",
      type=Path,
      default=None,
      help="Path to capability-scoped multi-upstream SessionPolicy JSON file (MS019)",
  )
  run_p.add_argument(
      "--server-id",
      type=str,
      default="",
      help="Upstream server identifier used when evaluating --session-policy scopes",
  )
  run_p.add_argument(
      "--shield-templates",
      action="store_true",
      help="Audit prompt, resource, and completion template arguments against recursive expansion and secret exfiltration (MS020)",
  )
  run_p.add_argument(
      "--max-template-depth",
      type=int,
      default=1,
      help="Maximum allowed template delimiter nesting depth when --shield-templates is active (default: 1)",
  )
  run_p.add_argument(
      "--sanitize-templates",
      action="store_true",
      help="Sanitize dangerous template tokens in-place instead of rejecting the request",
  )
  run_p.add_argument(
      "--block-roots",
      action="store_true",
      help="Intercept server-initiated roots/list requests locally and return an empty roots list to conceal host workspace paths",
  )
  run_p.add_argument(
      "--allow-root-prefixes",
      type=str,
      default=None,
      help="Comma-separated allowlist of workspace root URI prefixes disclosed on roots/list (e.g. 'file:///workspace/app')",
  )
  run_p.add_argument(
      "--deny-root-prefixes",
      type=str,
      default=None,
      help="Comma-separated denylist of workspace root URI prefixes concealed on roots/list",
  )
  run_p.add_argument(
      "--pin-schemas",
      action="store_true",
      help="Pin canonical SHA-256 tool hashes on initial tools/list and quarantine mid-session rug-pull mutations",
  )
  run_p.add_argument(
      "--allow-new-tools",
      action="store_true",
      help="Allow newly added tools during --pin-schemas sessions while still quarantining mutations to existing tools",
  )
  run_p.add_argument(
      "--allow-resource-prefixes",
      type=str,
      default=None,
      help="Comma-separated allowlist of MCP resource URI prefixes or globs (e.g. 'file:///workspace/,postgres://readonly/')",
  )
  run_p.add_argument(
      "--deny-resource-prefixes",
      type=str,
      default=None,
      help="Comma-separated denylist of MCP resource URI prefixes or globs (e.g. 'file:///etc/,file://*/.ssh/*')",
  )
  run_p.add_argument(
      "--tabular-arrays",
      action="store_true",
      help="Compact homogeneous arrays of flat JSON objects in tools/call outputs into columnar tables",
  )
  run_p.add_argument(
      "--token-budget",
      type=int,
      default=0,
      help="Sliding exponential token-window budget for MCP outputs; dynamically tightens output limits as recent load rises (0 disables)",
  )
  run_p.add_argument(
      "--window-decay",
      type=float,
      default=0.85,
      help="Exponential per-turn decay factor [0.0..1.0] for --token-budget sliding window (default: 0.85; 1.0 = cumulative cap)",
  )
  run_p.add_argument(
      "--otel-jsonl",
      type=Path,
      default=None,
      help="Append structured OpenTelemetry-compatible JSONL spans per MCP operation (or set MCP_SLIM_OTEL_JSONL)",
  )
  run_p.add_argument(
      "--preserve-root-unions",
      action="store_true",
      help="Keep root anyOf/oneOf JSON Schemas unflattened instead of adapting them for strict LLM providers",
  )
  run_p.add_argument(
      "--no-ref-hints",
      action="store_true",
      help="Disable semantic definition-name hints when inlining undocumented object $refs",
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
      "--block-sampling",
      action="store_true",
      help="Wrap servers with --block-sampling policy enforcement",
  )
  wrap_p.add_argument(
      "--block-elicitation",
      action="store_true",
      help="Wrap servers with --block-elicitation policy enforcement",
  )
  wrap_p.add_argument(
      "--block-roots",
      action="store_true",
      help="Wrap servers with --block-roots workspace path privacy",
  )
  wrap_p.add_argument(
      "--allow-root-prefixes",
      type=str,
      default=None,
      help="Wrap servers with --allow-root-prefixes workspace root boundary filtering",
  )
  wrap_p.add_argument(
      "--pin-schemas",
      action="store_true",
      help="Wrap servers with --pin-schemas mid-session rug-pull protection",
  )
  wrap_p.add_argument(
      "--tabular-arrays",
      action="store_true",
      help="Wrap servers with --tabular-arrays columnar output compaction",
  )
  wrap_p.add_argument(
      "--allow-resource-prefixes",
      type=str,
      default=None,
      help="Wrap servers with --allow-resource-prefixes URI allowlist enforcement",
  )
  wrap_p.add_argument(
      "--deny-resource-prefixes",
      type=str,
      default=None,
      help="Wrap servers with --deny-resource-prefixes URI denylist enforcement",
  )
  wrap_p.add_argument(
      "--token-budget",
      type=int,
      default=0,
      help="Wrap servers with --token-budget sliding exponential context-window budget compaction",
  )
  wrap_p.add_argument(
      "--window-decay",
      type=float,
      default=0.85,
      help="Decay factor [0.0..1.0] paired with --token-budget when wrapping mcpServers (default: 0.85)",
  )
  wrap_p.add_argument(
      "--repair-truncated-json",
      action="store_true",
      help="Wrap servers with --repair-truncated-json structural JSON repair",
  )
  wrap_p.add_argument(
      "--prune-tools",
      type=str,
      default=None,
      help="Wrap servers with --prune-tools BM25+ intent-aware catalog pruning",
  )
  wrap_p.add_argument(
      "--top-k-tools",
      type=int,
      default=12,
      help="Maximum tools retained when paired with --prune-tools (default: 12)",
  )
  wrap_p.add_argument(
      "--strict-rpc-ids",
      action="store_true",
      help="Wrap servers with --strict-rpc-ids JSON-RPC response correlation firewall",
  )
  wrap_p.add_argument(
      "--session-policy",
      type=str,
      default=None,
      help="Wrap servers with --session-policy <path> and per-entry --server-id isolation (MS019)",
  )
  wrap_p.add_argument(
      "--shield-templates",
      action="store_true",
      help="Wrap servers with --shield-templates recursive template exfiltration shield (MS020)",
  )
  wrap_p.add_argument(
      "--max-template-depth",
      type=int,
      default=1,
      help="Maximum template nesting depth paired with --shield-templates (default: 1)",
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

  diff_p = sub.add_parser(
      "diff",
      help="Compare two MCP tools JSON manifests for rug-pull mutations, privilege escalations, and poisoning",
  )
  diff_p.add_argument("before", type=Path, help="Baseline tools JSON manifest")
  diff_p.add_argument("after", type=Path, help="Updated tools JSON manifest")
  diff_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON diff report",
  )
  diff_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )
  diff_p.add_argument(
      "--fail-on-mutation",
      action="store_true",
      help="Exit with code 2 when blocking tool mutations, privilege escalations, or security findings exist",
  )
  diff_p.add_argument(
      "--allow-new-tools",
      action="store_true",
      help="Treat newly added tools as non-blocking notes",
  )

  merge_p = sub.add_parser(
      "merge",
      help="Merge multiple MCP tools JSON manifests into a unified composite catalog with collision and $defs resolution",
  )
  merge_p.add_argument(
      "inputs",
      nargs="+",
      help="Manifest paths or server_id=path specs (or a multi-server {'servers': ...} JSON file)",
  )
  merge_p.add_argument(
      "--strategy",
      choices=("prefix_colliding", "prefix_all", "merge_identical", "first_wins"),
      default="prefix_colliding",
      help="Cross-server tool-name collision resolution strategy (default: prefix_colliding)",
  )
  merge_p.add_argument(
      "--separator",
      type=str,
      default="__",
      help="Namespace separator between server_id and tool_name (default: __)",
  )
  merge_p.add_argument(
      "--slim",
      action="store_true",
      help="Compact merged tool schemas with SchemaSlimmer",
  )
  merge_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON composite merge report",
  )
  merge_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )
  merge_p.add_argument(
      "--fail-on-conflict",
      action="store_true",
      help="Exit with code 2 when MS016 tool collisions or conflicting $defs schemas are detected",
  )

  prune_p = sub.add_parser(
      "prune",
      help="Rank and prune an MCP tools JSON manifest for a task intent using BM25+ subword scoring (MS017)",
  )
  prune_p.add_argument(
      "file",
      type=Path,
      nargs="?",
      default=None,
      help="Optional path to tools JSON manifest (defaults to built-in benchmark fixture)",
  )
  prune_p.add_argument(
      "--intent",
      "-q",
      required=True,
      type=str,
      help="Task description or query keywords used to rank tools",
  )
  prune_p.add_argument(
      "--top-k",
      type=int,
      default=12,
      help="Maximum matched tools to retain (default: 12)",
  )
  prune_p.add_argument(
      "--min-score-ratio",
      type=float,
      default=0.15,
      help="Relative BM25+ score cutoff [0.0..1.0] vs top match (default: 0.15)",
  )
  prune_p.add_argument(
      "--always-keep",
      type=str,
      default=None,
      help="Comma-separated tool names or glob patterns always retained",
  )
  prune_p.add_argument(
      "--slim",
      action="store_true",
      help="Compact retained tool schemas with SchemaSlimmer",
  )
  prune_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON prune report",
  )
  prune_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )

  policy_p = sub.add_parser(
      "policy",
      help="Audit an MCP JSON-RPC frame or session transcript against a capability SessionPolicy (MS019)",
  )
  policy_p.add_argument("policy", type=Path, help="Path to SessionPolicy JSON file")
  policy_p.add_argument("target", type=Path, help="Path to JSON-RPC frame or transcript events JSON file")
  policy_p.add_argument("--server-id", type=str, default="", help="Origin upstream server ID")
  policy_p.add_argument("--target-server-id", type=str, default=None, help="Target upstream server ID")
  policy_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON policy evaluation report",
  )
  policy_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )
  policy_p.add_argument(
      "--fail-on-violation",
      action="store_true",
      help="Exit with code 2 when MS019 policy violations are detected",
  )

  shield_p = sub.add_parser(
      "shield",
      help="Audit prompt or resource template arguments for recursive expansion and exfiltration (MS020)",
  )
  shield_p.add_argument("file", type=Path, help="Path to template arguments JSON file")
  shield_p.add_argument(
      "--type",
      choices=("prompt", "resource", "completion"),
      default="prompt",
      dest="template_type",
      help="Template context type (default: prompt)",
  )
  shield_p.add_argument(
      "--uri-template",
      type=str,
      default="",
      help="Optional RFC 6570 URI template or prompt identifier",
  )
  shield_p.add_argument(
      "--max-depth",
      type=int,
      default=1,
      help="Maximum allowed delimiter nesting depth (default: 1)",
  )
  shield_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      help="Emit structured JSON template shield report",
  )
  shield_p.add_argument(
      "--sarif",
      type=Path,
      default=None,
      help="Write OASIS SARIF v2.1.0 report to the specified path",
  )
  shield_p.add_argument(
      "--fail-on-violation",
      action="store_true",
      help="Exit with code 2 when MS020 template violations are detected",
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

  bench_p = sub.add_parser(
      "bench",
      help="Multi-turn compression simulation with JSON-RPC latency histogram and SSE chunk throttle stats",
  )
  bench_p.add_argument(
      "--turns",
      type=int,
      default=10,
      metavar="N",
      help="Number of simulated tool-call turns (default: 10)",
  )
  bench_p.add_argument(
      "--sse-burst-budget",
      type=int,
      default=0,
      metavar="TOKENS",
      dest="sse_burst_budget",
      help="SSE chunk burst budget tokens; exercises SseChunkThrottle (0 disables)",
  )
  bench_p.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      default=False,
      help="Output bench results as JSON",
  )

  parsed = parser.parse_args(args_list)

  if parsed.benchmark or parsed.subcommand == "benchmark":
    return run_benchmark()

  if parsed.subcommand == "bench":
    return run_bench(
        turns=getattr(parsed, "turns", 10),
        sse_burst_budget=getattr(parsed, "sse_burst_budget", 0),
        json_output=getattr(parsed, "json_output", False),
    )

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

  if parsed.subcommand == "diff":
    return run_diff(
        parsed.before,
        parsed.after,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
        fail_on_mutation=parsed.fail_on_mutation,
        allow_new_tools=parsed.allow_new_tools,
    )

  if parsed.subcommand == "merge":
    return run_merge(
        list(parsed.inputs),
        strategy=parsed.strategy,
        separator=parsed.separator,
        slim=parsed.slim,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
        fail_on_conflict=parsed.fail_on_conflict,
    )

  if parsed.subcommand == "prune":
    keep_list = (
        [x.strip() for x in parsed.always_keep.split(",") if x.strip()]
        if parsed.always_keep
        else None
    )
    return run_prune(
        parsed.file,
        intent=parsed.intent,
        top_k=parsed.top_k,
        min_score_ratio=parsed.min_score_ratio,
        always_keep=keep_list,
        slim=parsed.slim,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
    )

  if parsed.subcommand == "policy":
    return run_policy(
        parsed.policy,
        parsed.target,
        server_id=parsed.server_id,
        target_server_id=parsed.target_server_id,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
        fail_on_violation=parsed.fail_on_violation,
    )

  if parsed.subcommand == "shield":
    return run_shield(
        parsed.file,
        template_type=parsed.template_type,
        uri_template=parsed.uri_template,
        max_depth=parsed.max_depth,
        json_output=parsed.json_output,
        sarif_path=parsed.sarif,
        fail_on_violation=parsed.fail_on_violation,
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
        block_sampling=parsed.block_sampling,
        block_elicitation=parsed.block_elicitation,
        block_roots=parsed.block_roots,
        allow_root_prefixes=parsed.allow_root_prefixes,
        pin_schemas=parsed.pin_schemas,
        tabular_arrays=parsed.tabular_arrays,
        allow_resource_prefixes=parsed.allow_resource_prefixes,
        deny_resource_prefixes=parsed.deny_resource_prefixes,
        token_budget=parsed.token_budget,
        window_decay=parsed.window_decay,
        repair_truncated_json=parsed.repair_truncated_json,
        prune_tools=parsed.prune_tools,
        top_k_tools=parsed.top_k_tools,
        strict_rpc_ids=parsed.strict_rpc_ids,
        session_policy=parsed.session_policy,
        shield_templates=parsed.shield_templates,
        max_template_depth=parsed.max_template_depth,
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
    allow_res_set = (
        {p.strip() for p in parsed.allow_resource_prefixes.split(",") if p.strip()}
        if parsed.allow_resource_prefixes
        else None
    )
    deny_res_set = (
        {p.strip() for p in parsed.deny_resource_prefixes.split(",") if p.strip()}
        if parsed.deny_resource_prefixes
        else None
    )
    allow_root_set = (
        {p.strip() for p in parsed.allow_root_prefixes.split(",") if p.strip()}
        if parsed.allow_root_prefixes
        else None
    )
    deny_root_set = (
        {p.strip() for p in parsed.deny_root_prefixes.split(",") if p.strip()}
        if parsed.deny_root_prefixes
        else None
    )
    proxy = MCPProxy(
        schema_slimmer=SchemaSlimmer(
            max_desc_chars=parsed.max_desc,
            max_param_desc_chars=parsed.max_param_desc,
            allow_tools=allow_set,
            deny_tools=deny_set,
            preserve_root_unions=parsed.preserve_root_unions,
            read_only_only=parsed.read_only,
            preserve_ref_hints=not parsed.no_ref_hints,
        ),
        output_slimmer=OutputSlimmer(tabular_arrays=parsed.tabular_arrays),
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
        block_sampling=parsed.block_sampling,
        block_elicitation=parsed.block_elicitation,
        allow_credential_elicitation=parsed.allow_credential_elicitation,
        block_result_injection=parsed.block_result_injection,
        detect_replay=parsed.detect_replay,
        repair_truncated_json=parsed.repair_truncated_json,
        prune_tools_intent=parsed.prune_tools,
        prune_top_k=parsed.top_k_tools,
        strict_rpc_ids=parsed.strict_rpc_ids,
        enforce_session_policy=parsed.session_policy is not None,
        session_policy=(
            json.loads(parsed.session_policy.read_text(encoding="utf-8"))
            if parsed.session_policy is not None
            else None
        ),
        shield_templates=parsed.shield_templates,
        max_template_depth=parsed.max_template_depth,
        sanitize_templates_in_place=parsed.sanitize_templates,
        server_id=parsed.server_id,
        block_roots=parsed.block_roots,
        allow_root_prefixes=allow_root_set,
        deny_root_prefixes=deny_root_set,
        pin_schemas=parsed.pin_schemas,
        allow_new_tools=parsed.allow_new_tools,
        allow_resource_prefixes=allow_res_set,
        deny_resource_prefixes=deny_res_set,
        token_budget=parsed.token_budget,
        window_decay=parsed.window_decay,
        otel_jsonl_path=parsed.otel_jsonl,
    )
    return asyncio.run(proxy.run_stdio(cmd))

  parser.print_help()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
