"""Standalone MCP JSON-RPC 2.0 stdio server exposing mcp-slim schema & security tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .guard import SecretGuard, StreamSanitizer
from .slimmer import (
    OutputSlimmer,
    SchemaSlimmer,
    SlimStats,
    TokenWindowGovernor,
    audit_elicitation_request,
    audit_resources_manifest,
    audit_tools_manifest,
    diff_tools_manifests,
)
from .telemetry import StatsLedger

SERVER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "compact_mcp_tools",
        "description": (
            "Compact an array of MCP tool definitions by stripping JSON Schema noise, "
            "inlining $defs/$ref definitions, and flattening root object unions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "description": "Array of raw MCP tool objects containing name and inputSchema.",
                    "items": {"type": "object"},
                },
                "max_desc_chars": {
                    "type": "integer",
                    "description": "Max characters per tool description (default 120).",
                },
                "preserve_root_unions": {
                    "type": "boolean",
                    "description": "Keep root anyOf/oneOf schemas intact instead of flattening.",
                },
            },
            "required": ["tools"],
        },
    },
    {
        "name": "audit_mcp_schema",
        "description": (
            "Audit token footprint, $defs/$ref chain usage, and lazy meta-tool savings "
            "across an MCP tools list."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "description": "Array of MCP tool objects to profile.",
                    "items": {"type": "object"},
                }
            },
            "required": ["tools"],
        },
    },
    {
        "name": "diff_mcp_tools",
        "description": (
            "Compare two MCP tools/list snapshots for rug-pull description edits, "
            "inputSchema tampering, and ToolAnnotations privilege escalations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "before_tools": {
                    "type": "array",
                    "description": "Baseline array of MCP tool objects.",
                    "items": {"type": "object"},
                },
                "after_tools": {
                    "type": "array",
                    "description": "Updated array of MCP tool objects to compare against baseline.",
                    "items": {"type": "object"},
                },
                "allow_new_tools": {
                    "type": "boolean",
                    "description": "Treat newly added tools as non-blocking notes.",
                },
            },
            "required": ["before_tools", "after_tools"],
        },
    },
    {
        "name": "audit_mcp_resources",
        "description": (
            "Audit MCP resources or resource read contents for URI prefix policy violations, "
            "path traversal, leaked secrets (including base64 blobs), and indirect prompt injection."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "resources": {
                    "type": "array",
                    "description": "Array of MCP resource or resource content objects.",
                    "items": {"type": "object"},
                },
                "allow_prefixes": {
                    "type": "array",
                    "description": "Optional allowlist of URI prefixes or globs.",
                    "items": {"type": "string"},
                },
                "deny_prefixes": {
                    "type": "array",
                    "description": "Optional denylist of URI prefixes or globs.",
                    "items": {"type": "string"},
                },
            },
            "required": ["resources"],
        },
    },
    {
        "name": "redact_secrets",
        "description": (
            "Scan a text or JSON payload with SecretGuard and redact leaked API keys, "
            "cloud tokens, private keys, JWTs, and prompt injections."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text or serialized JSON string to scan and sanitize.",
                },
                "vault_handles": {
                    "type": "boolean",
                    "description": "Replace secrets with session-scoped [SECRET_REF:...] handles.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "compact_mcp_output",
        "description": (
            "Compact an MCP tools/call result object using OutputSlimmer, SecretGuard, "
            "optional columnar tables, and sliding exponential token-window budget throttling."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "MCP tools/call result object containing content or structuredContent.",
                },
                "tabular_arrays": {
                    "type": "boolean",
                    "description": "Convert homogeneous arrays of flat JSON objects into columnar tables.",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Optional sliding window token budget (>0 enables adaptive throttling).",
                },
                "prior_window_tokens": {
                    "type": "number",
                    "description": "Current accumulated sliding-window tokens prior to this turn.",
                },
                "window_decay": {
                    "type": "number",
                    "description": "Exponential per-turn decay factor [0.0..1.0] (default 0.85).",
                },
            },
            "required": ["result"],
        },
    },
    {
        "name": "sanitize_stream_chunks",
        "description": (
            "Sanitize an ordered sequence of streaming text or tool-call JSON delta chunks "
            "using stateful holdback so secrets and prompt injections split across chunks are scrubbed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "description": "Ordered array of text or JSON delta chunks from a streaming call.",
                    "items": {"type": "string"},
                },
                "vault_handles": {
                    "type": "boolean",
                    "description": "Tokenize intercepted secrets into session-scoped [SECRET_REF:...] handles.",
                },
            },
            "required": ["chunks"],
        },
    },
    {
        "name": "audit_mcp_elicitation",
        "description": (
            "Audit a server-initiated MCP elicitation/create payload for credential "
            "phishing schemas, auth URL traps, and Unicode smuggling (MS007)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "MCP elicitation/create params containing message and requestedSchema.",
                }
            },
            "required": ["params"],
        },
    },
    {
        "name": "get_slim_stats",
        "description": (
            "Read cumulative token, wire-byte, and estimated LLM API cost savings "
            "recorded by mcp-slim proxy sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stats_file": {
                    "type": "string",
                    "description": "Optional custom path to stats.json ledger.",
                }
            },
        },
    },
]


def _text_result(payload: Any, is_error: bool = False) -> dict[str, Any]:
  text = (
      payload
      if isinstance(payload, str)
      else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  )
  return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_server_rpc(msg: dict[str, Any]) -> dict[str, Any] | None:
  """Processes a single JSON-RPC 2.0 request for standalone `mcp-slim serve` mode."""
  msg_id = msg.get("id")
  method = msg.get("method")
  params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

  if msg_id is None:
    return None

  if method == "initialize":
    client_proto = params.get("protocolVersion")
    proto = client_proto if isinstance(client_proto, str) and client_proto else "2024-11-05"
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": proto,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mcp-slim", "version": __version__},
        },
    }

  if method == "ping":
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

  if method == "tools/list":
    return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": SERVER_TOOLS}}

  if method in ("resources/list",):
    return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}

  if method in ("prompts/list",):
    return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}

  if method == "tools/call":
    name = params.get("name")
    args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}

    if name == "compact_mcp_tools":
      tools = args.get("tools")
      if not isinstance(tools, list):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result("Parameter 'tools' must be a JSON array.", is_error=True),
        }
      slimmer = SchemaSlimmer(
          max_desc_chars=int(args.get("max_desc_chars", 120)),
          preserve_root_unions=bool(args.get("preserve_root_unions", False)),
      )
      stats = SlimStats()
      compacted = slimmer.slim_tools_list(tools, stats=stats)
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result({
              "tools_before": stats.tools_total,
              "tools_after": stats.tools_kept,
              "tokens_before": stats.schema_tokens_before,
              "tokens_after": stats.schema_tokens_after,
              "reduction_pct": stats.reduction_pct,
              "refs_inlined": stats.refs_inlined,
              "refs_deduped": stats.refs_deduped,
              "cycles_broken": stats.cycles_broken,
              "tools": compacted,
          }),
      }

    if name == "audit_mcp_schema":
      tools = args.get("tools")
      if not isinstance(tools, list):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result("Parameter 'tools' must be a JSON array.", is_error=True),
        }
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result(audit_tools_manifest(tools)),
      }

    if name == "diff_mcp_tools":
      before_tools = args.get("before_tools")
      after_tools = args.get("after_tools")
      if not isinstance(before_tools, list) or not isinstance(after_tools, list):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result(
                "Parameters 'before_tools' and 'after_tools' must be JSON arrays.",
                is_error=True,
            ),
        }
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result(
              diff_tools_manifests(
                  before_tools,
                  after_tools,
                  allow_new_tools=bool(args.get("allow_new_tools", False)),
              )
          ),
      }

    if name == "audit_mcp_resources":
      resources = args.get("resources")
      if not isinstance(resources, list):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result(
                "Parameter 'resources' must be a JSON array.",
                is_error=True,
            ),
        }
      raw_allow = args.get("allow_prefixes")
      raw_deny = args.get("deny_prefixes")
      allow_list = (
          [str(x) for x in raw_allow if isinstance(x, str) and x.strip()]
          if isinstance(raw_allow, list)
          else None
      )
      deny_list = (
          [str(x) for x in raw_deny if isinstance(x, str) and x.strip()]
          if isinstance(raw_deny, list)
          else None
      )
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result(
              audit_resources_manifest(
                  resources,
                  allow_prefixes=allow_list,
                  deny_prefixes=deny_list,
              )
          ),
      }

    if name == "redact_secrets":
      text = str(args.get("text", ""))
      guard = SecretGuard(
          enabled=True, vault_handles=bool(args.get("vault_handles", False))
      )
      sanitized = guard.redact_text(text)
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result({
              "sanitized": sanitized,
              "total_redactions": guard.report.total_redactions,
              "by_category": guard.report.by_category,
          }),
      }

    if name == "compact_mcp_output":
      raw_result = args.get("result")
      if not isinstance(raw_result, dict):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result("Parameter 'result' must be a JSON object.", is_error=True),
        }
      budget = int(args.get("token_budget", 0) or 0)
      prior_win = float(args.get("prior_window_tokens", 0.0) or 0.0)
      decay = float(args.get("window_decay", 0.85) if args.get("window_decay") is not None else 0.85)
      gov = TokenWindowGovernor(
          budget_tokens=budget,
          decay=decay,
          initial_window_tokens=prior_win,
      )
      out_slimmer = OutputSlimmer(
          tabular_arrays=bool(args.get("tabular_arrays", False)),
          window_governor=gov,
      )
      guard = SecretGuard(enabled=True)
      stats = SlimStats()
      compacted_res = out_slimmer.slim_call_result(
          raw_result, stats=stats, guard=guard, governor=gov
      )
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result({
              "tokens_before": stats.output_tokens_before,
              "tokens_after": stats.output_tokens_after,
              "reduction_pct": stats.reduction_pct,
              "tables_compacted": stats.tables_compacted,
              "secrets_redacted": guard.report.total_redactions,
              "window": gov.snapshot(),
              "result": compacted_res,
          }),
      }

    if name == "sanitize_stream_chunks":
      chunks = args.get("chunks")
      if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result("Parameter 'chunks' must be an array of strings.", is_error=True),
        }
      guard = SecretGuard(
          enabled=True, vault_handles=bool(args.get("vault_handles", False))
      )
      sanitizer = StreamSanitizer(guard)
      emitted_chunks: list[str] = [sanitizer.push(c) for c in chunks]
      tail = sanitizer.flush()
      if tail:
        emitted_chunks.append(tail)
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result({
              "sanitized_chunks": [c for c in emitted_chunks if c],
              "sanitized_text": "".join(emitted_chunks),
              "chunks_processed": sanitizer.chunks_processed,
              "total_redactions": guard.report.total_redactions,
              "by_category": guard.report.by_category,
          }),
      }

    if name == "audit_mcp_elicitation":
      el_params = args.get("params")
      if not isinstance(el_params, dict):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": _text_result("Parameter 'params' must be a JSON object.", is_error=True),
        }
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result(audit_elicitation_request(el_params)),
      }

    if name == "get_slim_stats":
      custom_path = args.get("stats_file")
      ledger = StatsLedger(
          path=Path(custom_path) if isinstance(custom_path, str) and custom_path else None
      )
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result(ledger.summary()),
      }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": _text_result(f"Unknown tool: {name}", is_error=True),
    }

  return {
      "jsonrpc": "2.0",
      "id": msg_id,
      "error": {"code": -32601, "message": f"Method not found: {method}"},
  }


def run_serve() -> int:
  """Runs the standalone mcp-slim MCP stdio server loop."""
  for raw_line in sys.stdin:
    stripped = raw_line.strip()
    if not stripped:
      continue
    try:
      msg = json.loads(stripped)
    except json.JSONDecodeError:
      continue
    if not isinstance(msg, dict):
      continue
    resp = handle_server_rpc(msg)
    if resp is not None:
      sys.stdout.write(json.dumps(resp, ensure_ascii=False, separators=(",", ":")) + "\n")
      sys.stdout.flush()
  return 0
