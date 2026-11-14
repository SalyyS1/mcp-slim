"""Standalone MCP JSON-RPC 2.0 stdio server exposing mcp-slim schema & security tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .guard import SecretGuard
from .slimmer import LazyToolBroker, SchemaSlimmer, SlimStats, estimate_tokens
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
      slimmer = SchemaSlimmer()
      std_stats = SlimStats()
      compacted = slimmer.slim_tools_list(tools, stats=std_stats)
      broker = LazyToolBroker(schema_slimmer=slimmer)
      lazy_stats = SlimStats()
      broker.ingest_tools(tools, stats=lazy_stats)
      per_tool = []
      slim_map = {t.get("name", ""): t for t in compacted}
      for raw_t in tools:
        if not isinstance(raw_t, dict):
          continue
        t_name = str(raw_t.get("name", ""))
        raw_tok = estimate_tokens(raw_t)
        slim_tok = estimate_tokens(slim_map.get(t_name, raw_t))
        raw_dump = json.dumps(raw_t)
        per_tool.append({
            "name": t_name,
            "raw_tokens": raw_tok,
            "slim_tokens": slim_tok,
            "has_refs": "$defs" in raw_dump or "$ref" in raw_dump,
            "has_root_union": isinstance(raw_t.get("inputSchema"), dict)
            and any(u in raw_t["inputSchema"] for u in ("anyOf", "oneOf")),
        })
      return {
          "jsonrpc": "2.0",
          "id": msg_id,
          "result": _text_result({
              "tool_count": len(tools),
              "tokens_raw": std_stats.schema_tokens_before,
              "tokens_slim": std_stats.schema_tokens_after,
              "standard_reduction_pct": std_stats.reduction_pct,
              "tokens_lazy": lazy_stats.schema_tokens_after,
              "lazy_reduction_pct": lazy_stats.reduction_pct,
              "refs_inlined": std_stats.refs_inlined,
              "tools": per_tool,
          }),
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
