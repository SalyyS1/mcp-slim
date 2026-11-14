"""Async JSON-RPC 2.0 stdio proxy for Model Context Protocol (MCP) servers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Awaitable

from .guard import SecretGuard
from .slimmer import (
    LazyToolBroker,
    OutputSlimmer,
    SchemaSlimmer,
    SlimStats,
    estimate_tokens,
)
from .telemetry import OTelSpanExporter, StatsLedger, derive_server_label


class MCPProxy:
  """Intercepts and optimizes JSON-RPC 2.0 messages between an MCP client and server."""

  def __init__(
      self,
      schema_slimmer: SchemaSlimmer | None = None,
      output_slimmer: OutputSlimmer | None = None,
      secret_guard: SecretGuard | None = None,
      lazy: bool = False,
      cache_ttl: float = 0.0,
      quiet: bool = False,
      cli_only: bool = False,
      bridge_socket: str | None = None,
      no_stats: bool = False,
      stats_ledger: StatsLedger | None = None,
      block_sampling: bool = False,
      otel_exporter: OTelSpanExporter | None = None,
      otel_jsonl_path: Path | str | None = None,
  ) -> None:
    self.schema_slimmer = schema_slimmer or SchemaSlimmer()
    self.output_slimmer = output_slimmer or OutputSlimmer()
    self.secret_guard = secret_guard or SecretGuard()
    self.lazy = lazy
    self.lazy_broker = (
        LazyToolBroker(schema_slimmer=self.schema_slimmer) if lazy else None
    )
    self.cache_ttl = cache_ttl
    self.stats = SlimStats()
    self.quiet = quiet
    self.cli_only = cli_only
    self.bridge_socket = bridge_socket
    self.no_stats = no_stats
    self.block_sampling = block_sampling
    self.stats_ledger = (
        stats_ledger
        if stats_ledger is not None
        else StatsLedger(enabled=not no_stats)
    )
    if otel_exporter is not None:
      self.otel_exporter: OTelSpanExporter | None = otel_exporter
    else:
      candidate_otel = OTelSpanExporter(path=otel_jsonl_path)
      self.otel_exporter = candidate_otel if candidate_otel.enabled else None
    self._pending_methods: dict[str | int, str] = {}
    self._pending_tool_names: dict[str | int, str] = {}
    self._pending_server_methods: dict[str | int, str] = {}
    self._pending_cache_keys: dict[str | int, str] = {}
    self._read_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    self._local_responses: list[str] = []
    self._local_server_responses: list[str] = []
    self._inflight_spans: dict[tuple[str, str | int], dict[str, Any]] = {}
    self._cached_tools: dict[str, dict[str, Any]] = {}
    self._bridge_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
    self._bridge_seq: int = 0

  def pop_local_responses(self) -> list[str]:
    """Returns any locally synthesized JSON-RPC responses destined for the MCP client."""
    items = list(self._local_responses)
    self._local_responses.clear()
    return items

  def pop_local_server_responses(self) -> list[str]:
    """Returns any locally synthesized JSON-RPC responses destined for the upstream MCP server."""
    items = list(self._local_server_responses)
    self._local_server_responses.clear()
    return items

  def _emit_span(
      self,
      *,
      name: str,
      kind: str = "CLIENT",
      start_ns: int | None = None,
      end_ns: int | None = None,
      is_error: bool = False,
      error_message: str | None = None,
      attributes: dict[str, Any] | None = None,
  ) -> None:
    if self.otel_exporter is None or not self.otel_exporter.enabled:
      return
    self.otel_exporter.emit_span(
        name=name,
        kind=kind,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        is_error=is_error,
        error_message=error_message,
        attributes=attributes,
    )

  def process_client_message(self, raw_line: str) -> str:
    """Processes a client -> server JSON-RPC line.

    Returns an empty string when the request was handled locally by `LazyToolBroker`.
    """
    try:
      msg = json.loads(raw_line)
    except json.JSONDecodeError:
      return raw_line
    if not isinstance(msg, dict):
      return raw_line

    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params")
    start_wall_ns = time.time_ns()
    start_mono_ns = time.perf_counter_ns()

    # Client -> server JSON-RPC response (e.g. replying to server-initiated sampling/createMessage)
    if method is None and ("result" in msg or "error" in msg):
      srv_method = (
          self._pending_server_methods.pop(msg_id, None)
          if msg_id is not None
          else None
      )
      inflight_srv = (
          self._inflight_spans.pop(("server", msg_id), None)
          if msg_id is not None
          else None
      )
      red_before = self.secret_guard.report.total_redactions
      wire_in = len(raw_line.encode("utf-8"))

      if "result" in msg:
        if srv_method is None and isinstance(msg.get("result"), dict) and "role" in msg["result"] and "content" in msg["result"]:
          self.stats.sampling_guarded += 1
        msg["result"] = self.secret_guard.redact_value(
            msg["result"], force_redact=True
        )
        serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        serialized = self.secret_guard.redact_text(serialized, force_redact=True)
        out_line = serialized + "\n"
        wire_out = len(out_line.encode("utf-8"))
        red_delta = self.secret_guard.report.total_redactions - red_before
        if inflight_srv is not None:
          elapsed_ns = max(0, time.perf_counter_ns() - inflight_srv["start_mono_ns"])
          tok_b = inflight_srv.get("tok_before", 0)
          tok_a = inflight_srv.get("tok_after", 0)
          self._emit_span(
              name=f"mcp.{inflight_srv['method']}",
              kind="SERVER",
              start_ns=inflight_srv["start_wall_ns"],
              end_ns=inflight_srv["start_wall_ns"] + elapsed_ns,
              is_error=False,
              attributes={
                  "mcp.method.name": inflight_srv["method"],
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.tokens.before": tok_b,
                  "mcp_slim.tokens.after": tok_a,
                  "mcp_slim.tokens.saved": max(0, tok_b - tok_a),
                  "mcp_slim.wire_bytes.before": inflight_srv.get("wire_in", 0) + wire_in,
                  "mcp_slim.wire_bytes.after": inflight_srv.get("wire_out", 0) + wire_out,
                  "mcp_slim.secrets_redacted": inflight_srv.get("redactions_on_req", 0) + red_delta,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": False,
              },
          )
        return out_line

      err_obj = msg.get("error")
      err_msg = (
          str(err_obj.get("message", "RPC error"))
          if isinstance(err_obj, dict)
          else "RPC error"
      )
      if inflight_srv is not None:
        elapsed_ns = max(0, time.perf_counter_ns() - inflight_srv["start_mono_ns"])
        self._emit_span(
            name=f"mcp.{inflight_srv['method']}",
            kind="SERVER",
            start_ns=inflight_srv["start_wall_ns"],
            end_ns=inflight_srv["start_wall_ns"] + elapsed_ns,
            is_error=True,
            error_message=err_msg,
            attributes={
                "mcp.method.name": inflight_srv["method"],
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.secrets_redacted": inflight_srv.get("redactions_on_req", 0),
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": False,
            },
        )
      return raw_line

    resolved_tool_name: str | None = None
    if method == "tools/call" and isinstance(params, dict):
      tool_name = params.get("name")
      arguments = params.get("arguments") or {}

      if self.lazy_broker is not None and tool_name == LazyToolBroker.SEARCH_TOOL_NAME:
        query = str(arguments.get("query", "")) if isinstance(arguments, dict) else ""
        limit = int(arguments.get("limit", 5)) if isinstance(arguments, dict) else 5
        search_res = self.lazy_broker.search(query=query, limit=limit)
        payload = {
            "jsonrpc": msg.get("jsonrpc", "2.0"),
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            search_res, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                ],
                "isError": False,
            },
        }
        reply_line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._local_responses.append(reply_line)
        elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
        self._emit_span(
            name=f"mcp.tools/call {LazyToolBroker.SEARCH_TOOL_NAME}",
            kind="INTERNAL",
            start_ns=start_wall_ns,
            end_ns=start_wall_ns + elapsed_ns,
            attributes={
                "mcp.method.name": "tools/call",
                "mcp.tool.name": LazyToolBroker.SEARCH_TOOL_NAME,
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": False,
            },
        )
        return ""

      if self.lazy_broker is not None and tool_name == LazyToolBroker.CALL_TOOL_NAME:
        if isinstance(arguments, dict) and isinstance(arguments.get("tool_name"), str):
          params["name"] = arguments["tool_name"]
          params["arguments"] = (
              arguments.get("arguments")
              if isinstance(arguments.get("arguments"), dict)
              else {}
          )
          tool_name = params["name"]
          arguments = params["arguments"]

      if isinstance(tool_name, str):
        resolved_tool_name = tool_name

      if isinstance(tool_name, str) and not self.schema_slimmer.is_tool_allowed(tool_name):
        self.stats.blocked_calls += 1
        reason_suffix = (
            " (read-only mode active)"
            if self.schema_slimmer.read_only_only
            and not self.schema_slimmer.is_read_only_tool(tool_name)
            else ""
        )
        block_msg = (
            f"[mcp-slim] Tool '{tool_name}' is blocked by proxy "
            f"allow/deny policy{reason_suffix}."
        )
        blocked_payload = {
            "jsonrpc": msg.get("jsonrpc", "2.0"),
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": block_msg,
                    }
                ],
                "isError": True,
            },
        }
        self._local_responses.append(
            json.dumps(blocked_payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
        self._emit_span(
            name=f"mcp.tools/call {tool_name}",
            kind="CLIENT",
            start_ns=start_wall_ns,
            end_ns=start_wall_ns + elapsed_ns,
            is_error=True,
            error_message=block_msg,
            attributes={
                "mcp.method.name": "tools/call",
                "mcp.tool.name": tool_name,
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": True,
            },
        )
        return ""

      if isinstance(tool_name, str) and isinstance(arguments, dict):
        arguments = self.schema_slimmer.coerce_arguments(
            tool_name, arguments, stats=self.stats
        )
        params["arguments"] = arguments
        union_err = self.schema_slimmer.validate_union_arguments(
            tool_name, arguments
        )
        if union_err is not None:
          self.stats.blocked_calls += 1
          invalid_payload = {
              "jsonrpc": msg.get("jsonrpc", "2.0"),
              "id": msg_id,
              "result": {
                  "content": [{"type": "text", "text": union_err}],
                  "isError": True,
              },
          }
          self._local_responses.append(
              json.dumps(invalid_payload, ensure_ascii=False, separators=(",", ":"))
              + "\n"
          )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.tools/call {tool_name}",
              kind="CLIENT",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=union_err,
              attributes={
                  "mcp.method.name": "tools/call",
                  "mcp.tool.name": tool_name,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
              },
          )
          return ""

      if self.cache_ttl > 0.0 and isinstance(tool_name, str):
        if self.schema_slimmer.is_read_only_tool(tool_name):
          canonical_args = json.dumps(
              arguments if isinstance(arguments, dict) else {},
              sort_keys=True,
              separators=(",", ":"),
          )
          cache_key = f"{tool_name}:{canonical_args}"
          cached = self._read_cache.get(cache_key)
          now = time.monotonic()
          if cached is not None and now < cached[0]:
            self.stats.cache_hits += 1
            hit_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": cached[1],
            }
            self._local_responses.append(
                json.dumps(hit_payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
            self._emit_span(
                name=f"mcp.tools/call {tool_name}",
                kind="CLIENT",
                start_ns=start_wall_ns,
                end_ns=start_wall_ns + elapsed_ns,
                attributes={
                    "mcp.method.name": "tools/call",
                    "mcp.tool.name": tool_name,
                    "mcp.request.id": str(msg_id) if msg_id is not None else None,
                    "mcp_slim.cache_hit": True,
                    "mcp_slim.blocked": False,
                },
            )
            return ""
          if msg_id is not None:
            self._pending_cache_keys[msg_id] = cache_key
        else:
          self._read_cache.clear()

    if msg_id is not None and isinstance(method, str):
      self._pending_methods[msg_id] = method
      if resolved_tool_name is not None:
        self._pending_tool_names[msg_id] = resolved_tool_name

    red_req_before = self.secret_guard.report.total_redactions
    if method == "tools/call" and isinstance(params, dict):
      if "arguments" in params:
        redacted_args = self.secret_guard.redact_value(params["arguments"])
        params["arguments"] = self.secret_guard.resolve_handles_in_value(
            redacted_args
        )
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      if not self.secret_guard.vault_handles:
        serialized = self.secret_guard.redact_text(serialized)
      red_req_delta = self.secret_guard.report.total_redactions - red_req_before
      if self.otel_exporter is not None and msg_id is not None:
        self._inflight_spans[("client", msg_id)] = {
            "start_wall_ns": start_wall_ns,
            "start_mono_ns": start_mono_ns,
            "method": method,
            "tool_name": resolved_tool_name,
            "redactions_on_req": red_req_delta,
        }
      return serialized + "\n"

    if self.otel_exporter is not None and msg_id is not None and isinstance(method, str):
      self._inflight_spans[("client", msg_id)] = {
          "start_wall_ns": start_wall_ns,
          "start_mono_ns": start_mono_ns,
          "method": method,
          "tool_name": resolved_tool_name,
          "redactions_on_req": 0,
      }
    return raw_line

  def process_server_message(self, raw_line: str) -> str:
    """Processes a server -> client JSON-RPC line."""
    try:
      msg = json.loads(raw_line)
    except json.JSONDecodeError:
      return raw_line
    if not isinstance(msg, dict):
      return raw_line

    msg_id = msg.get("id")
    srv_method = msg.get("method")

    # Server-initiated requests / notifications (e.g. sampling/createMessage, elicitation/create)
    if isinstance(srv_method, str) and "result" not in msg and "error" not in msg:
      srv_params = msg.get("params")
      start_wall_ns = time.time_ns()
      start_mono_ns = time.perf_counter_ns()
      if srv_method == "sampling/createMessage":
        if self.block_sampling:
          self.stats.blocked_calls += 1
          err_text = (
              "[mcp-slim] Server-initiated sampling/createMessage is blocked by "
              "proxy policy (--block-sampling)."
          )
          if msg_id is not None:
            err_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "error": {"code": -32600, "message": err_text},
            }
            self._local_server_responses.append(
                json.dumps(err_payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name="mcp.sampling/createMessage",
              kind="SERVER",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=err_text,
              attributes={
                  "mcp.method.name": "sampling/createMessage",
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
              },
          )
          return ""

        self.stats.sampling_guarded += 1
        red_before = self.secret_guard.report.total_redactions
        tok_b_0 = self.stats.output_tokens_before
        tok_a_0 = self.stats.output_tokens_after
        if isinstance(srv_params, dict):
          new_params = dict(srv_params)
          if isinstance(new_params.get("messages"), list):
            slim_msgs = self.output_slimmer.slim_prompt_result(
                {"messages": new_params["messages"]}, stats=self.stats
            )
            new_params["messages"] = slim_msgs["messages"]
          if isinstance(new_params.get("systemPrompt"), str) and new_params["systemPrompt"]:
            sp_before = estimate_tokens(new_params["systemPrompt"])
            slim_sp = self.output_slimmer.slim_text(new_params["systemPrompt"])
            sp_after = estimate_tokens(slim_sp)
            self.stats.output_tokens_before += sp_before
            self.stats.output_tokens_after += sp_after
            new_params["systemPrompt"] = slim_sp
          msg["params"] = self.secret_guard.redact_value(
              new_params, force_redact=True
          )
        serialized_srv = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        serialized_srv = self.secret_guard.redact_text(
            serialized_srv, force_redact=True
        )
        out_srv_line = serialized_srv + "\n"
        wire_in = len(raw_line.encode("utf-8"))
        wire_out = len(out_srv_line.encode("utf-8"))
        self.stats.wire_bytes_before += wire_in
        self.stats.wire_bytes_after += wire_out
        if msg_id is not None:
          self._pending_server_methods[msg_id] = srv_method
          if self.otel_exporter is not None:
            self._inflight_spans[("server", msg_id)] = {
                "start_wall_ns": start_wall_ns,
                "start_mono_ns": start_mono_ns,
                "method": srv_method,
                "wire_in": wire_in,
                "wire_out": wire_out,
                "tok_before": self.stats.output_tokens_before - tok_b_0,
                "tok_after": self.stats.output_tokens_after - tok_a_0,
                "redactions_on_req": self.secret_guard.report.total_redactions - red_before,
            }
        return out_srv_line

      if msg_id is not None:
        self._pending_server_methods[msg_id] = srv_method
      if isinstance(srv_params, dict) and self.secret_guard.enabled:
        red_before = self.secret_guard.report.total_redactions
        clean_params = self.secret_guard.redact_value(
            srv_params, force_redact=True
        )
        if clean_params != srv_params or self.secret_guard.report.total_redactions > red_before:
          msg["params"] = clean_params
          return json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"
      return raw_line

    method = (
        self._pending_methods.pop(msg_id, None) if msg_id is not None else None
    )
    tool_name = (
        self._pending_tool_names.pop(msg_id, None)
        if msg_id is not None
        else None
    )
    inflight_cli = (
        self._inflight_spans.pop(("client", msg_id), None)
        if msg_id is not None
        else None
    )
    pending_cache_key = (
        self._pending_cache_keys.pop(msg_id, None)
        if msg_id is not None
        else None
    )
    result = msg.get("result")

    if isinstance(result, dict):
      tok_before_0 = self.stats.total_tokens_before
      tok_after_0 = self.stats.total_tokens_after
      refs_0 = self.stats.refs_inlined
      red_0 = self.secret_guard.report.total_redactions

      if method == "tools/list" or isinstance(result.get("tools"), list):
        raw_tools = result.get("tools")
        if isinstance(raw_tools, list):
          orig_tok = estimate_tokens(raw_tools)
          tools = self.secret_guard.redact_value(raw_tools)
          sanitized_tok = estimate_tokens(tools)
          slimmed_list = self.schema_slimmer.slim_tools_list(
              tools, stats=None if self.lazy_broker is not None else self.stats
          )
          if not self.quiet and self.schema_slimmer.policy_warnings:
            for warn_msg in self.schema_slimmer.policy_warnings:
              sys.stderr.write(f"[mcp-slim] Warning: {warn_msg}\n")
          for t in slimmed_list:
            if isinstance(t.get("name"), str):
              self._cached_tools[t["name"]] = t
          if self.cli_only:
            result["tools"] = []
          elif self.lazy_broker is not None:
            result["tools"] = self.lazy_broker.ingest_tools(
                tools, stats=self.stats
            )
          else:
            result["tools"] = slimmed_list
          self.stats.schema_tokens_before += orig_tok - sanitized_tok
      elif method == "tools/call" or isinstance(result.get("content"), list):
        result = self.output_slimmer.slim_call_result(result, stats=self.stats)
      elif method == "resources/read" or isinstance(result.get("contents"), list):
        result = self.output_slimmer.slim_resource_result(result, stats=self.stats)
      elif method == "prompts/get" or isinstance(result.get("messages"), list):
        result = self.output_slimmer.slim_prompt_result(result, stats=self.stats)
      result = self.secret_guard.redact_value(result)
      msg["result"] = result
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      redacted_str = (
          serialized
          if self.secret_guard.vault_handles
          else self.secret_guard.redact_text(serialized)
      )
      out_line = redacted_str + "\n"
      wire_in = len(raw_line.encode("utf-8"))
      wire_out = len(out_line.encode("utf-8"))

      if inflight_cli is not None:
        elapsed_ns = max(0, time.perf_counter_ns() - inflight_cli["start_mono_ns"])
        eff_method = method or inflight_cli["method"]
        eff_tool = tool_name or inflight_cli.get("tool_name")
        span_name = (
            f"mcp.{eff_method} {eff_tool}" if eff_tool else f"mcp.{eff_method}"
        )
        tok_b_delta = self.stats.total_tokens_before - tok_before_0
        tok_a_delta = self.stats.total_tokens_after - tok_after_0
        is_err = bool(result.get("isError"))
        self._emit_span(
            name=span_name,
            kind="CLIENT",
            start_ns=inflight_cli["start_wall_ns"],
            end_ns=inflight_cli["start_wall_ns"] + elapsed_ns,
            is_error=is_err,
            attributes={
                "mcp.method.name": eff_method,
                "mcp.tool.name": eff_tool,
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.tokens.before": tok_b_delta,
                "mcp_slim.tokens.after": tok_a_delta,
                "mcp_slim.tokens.saved": max(0, tok_b_delta - tok_a_delta),
                "mcp_slim.wire_bytes.before": wire_in,
                "mcp_slim.wire_bytes.after": wire_out,
                "mcp_slim.refs_inlined": self.stats.refs_inlined - refs_0,
                "mcp_slim.secrets_redacted": (
                    inflight_cli.get("redactions_on_req", 0)
                    + (self.secret_guard.report.total_redactions - red_0)
                ),
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": False,
            },
        )

      if isinstance(msg_id, str) and msg_id in self._bridge_waiters:
        waiter = self._bridge_waiters.pop(msg_id)
        if not waiter.done():
          try:
            waiter.set_result(json.loads(redacted_str))
          except json.JSONDecodeError:
            waiter.set_result(msg)
        return ""
      if (
          pending_cache_key is not None
          and self.cache_ttl > 0.0
          and not result.get("isError")
      ):
        try:
          redacted_msg = json.loads(redacted_str)
          if isinstance(redacted_msg.get("result"), dict):
            self._read_cache[pending_cache_key] = (
                time.monotonic() + self.cache_ttl,
                redacted_msg["result"],
            )
        except json.JSONDecodeError:
          pass
      self.stats.wire_bytes_before += wire_in
      self.stats.wire_bytes_after += wire_out
      return out_line

    if "error" in msg and inflight_cli is not None:
      elapsed_ns = max(0, time.perf_counter_ns() - inflight_cli["start_mono_ns"])
      eff_method = method or inflight_cli["method"]
      eff_tool = tool_name or inflight_cli.get("tool_name")
      span_name = (
          f"mcp.{eff_method} {eff_tool}" if eff_tool else f"mcp.{eff_method}"
      )
      err_obj = msg.get("error")
      err_msg = (
          str(err_obj.get("message", "RPC error"))
          if isinstance(err_obj, dict)
          else "RPC error"
      )
      self._emit_span(
          name=span_name,
          kind="CLIENT",
          start_ns=inflight_cli["start_wall_ns"],
          end_ns=inflight_cli["start_wall_ns"] + elapsed_ns,
          is_error=True,
          error_message=err_msg,
          attributes={
              "mcp.method.name": eff_method,
              "mcp.tool.name": eff_tool,
              "mcp.request.id": str(msg_id) if msg_id is not None else None,
              "mcp_slim.secrets_redacted": inflight_cli.get("redactions_on_req", 0),
              "mcp_slim.cache_hit": False,
              "mcp_slim.blocked": False,
          },
      )

    if isinstance(msg_id, str) and msg_id in self._bridge_waiters:
      waiter = self._bridge_waiters.pop(msg_id)
      if not waiter.done():
        waiter.set_result(msg)
      return ""

    return raw_line

  async def handle_bridge_request(
      self,
      req: dict[str, Any],
      send_upstream: Callable[[str], Awaitable[None]],
      timeout: float = 30.0,
  ) -> dict[str, Any]:
    """Handles a local CLI bridge request ('list', 'schema', or 'call') multiplexed over the live MCP session."""
    action = req.get("action", "list")
    loop = asyncio.get_running_loop()

    async def _ensure_tools_cached() -> None:
      if self._cached_tools:
        return
      self._bridge_seq += 1
      req_id = f"__mcp_slim_bridge_{self._bridge_seq}__"
      fut: asyncio.Future[dict[str, Any]] = loop.create_future()
      self._bridge_waiters[req_id] = fut
      self._pending_methods[req_id] = "tools/list"
      list_msg = {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}
      await send_upstream(json.dumps(list_msg, separators=(",", ":")) + "\n")
      await asyncio.wait_for(fut, timeout=timeout)

    if action == "list":
      await _ensure_tools_cached()
      items = [
          {"name": name, "description": t.get("description", "")}
          for name, t in self._cached_tools.items()
      ]
      return {"ok": True, "tools": items}

    if action == "schema":
      await _ensure_tools_cached()
      tool_name = str(req.get("tool", ""))
      tool_def = self._cached_tools.get(tool_name)
      if tool_def is None:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}
      return {"ok": True, "tool": tool_def}

    if action == "call":
      tool_name = str(req.get("tool", ""))
      arguments = req.get("arguments")
      if not isinstance(arguments, dict):
        arguments = {}
      self._bridge_seq += 1
      req_id = f"__mcp_slim_bridge_{self._bridge_seq}__"
      fut = loop.create_future()
      self._bridge_waiters[req_id] = fut
      call_msg = {
          "jsonrpc": "2.0",
          "id": req_id,
          "method": "tools/call",
          "params": {"name": tool_name, "arguments": arguments},
      }
      processed = self.process_client_message(
          json.dumps(call_msg, ensure_ascii=False, separators=(",", ":")) + "\n"
      )
      local_replies = self.pop_local_responses()
      if local_replies:
        self._bridge_waiters.pop(req_id, None)
        return {"ok": True, "response": json.loads(local_replies[0])}
      if processed:
        await send_upstream(processed)
      resp = await asyncio.wait_for(fut, timeout=timeout)
      return {"ok": True, "response": resp}

    return {"ok": False, "error": f"Unsupported bridge action: {action}"}

  def record_telemetry(
      self,
      command: list[str] | None = None,
      server_label: str | None = None,
  ) -> dict[str, Any] | None:
    """Persists cumulative proxy metrics when telemetry is enabled."""
    if self.no_stats or self.stats_ledger is None:
      return None
    return self.stats_ledger.record_session(
        self.stats,
        command=command,
        server_label=server_label,
        secrets_redacted=self.secret_guard.report.total_redactions,
    )

  async def run_stdio(self, command: list[str]) -> int:
    """Spawns the upstream MCP server and proxies stdio streams."""
    if self.otel_exporter is not None and self.otel_exporter.server_label == "unknown":
      self.otel_exporter.server_label = derive_server_label(command)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    stdin_lock = asyncio.Lock()

    async def _send_upstream(line_str: str) -> None:
      assert proc.stdin is not None
      async with stdin_lock:
        proc.stdin.write(line_str.encode("utf-8"))
        await proc.stdin.drain()

    bridge_server = None
    if self.bridge_socket:
      try:
        if os.path.exists(self.bridge_socket):
          os.unlink(self.bridge_socket)
      except OSError:
        pass

      async def _on_bridge_client(
          c_reader: asyncio.StreamReader, c_writer: asyncio.StreamWriter
      ) -> None:
        try:
          raw = await c_reader.readline()
          if raw:
            req_obj = json.loads(raw.decode("utf-8", errors="replace"))
            res_obj = await self.handle_bridge_request(req_obj, _send_upstream)
            c_writer.write(
                (json.dumps(res_obj, ensure_ascii=False) + "\n").encode("utf-8")
            )
            await c_writer.drain()
        except Exception as exc:
          err_line = json.dumps({"ok": False, "error": str(exc)}) + "\n"
          c_writer.write(err_line.encode("utf-8"))
          await c_writer.drain()
        finally:
          c_writer.close()

      bridge_server = await asyncio.start_unix_server(
          _on_bridge_client, path=self.bridge_socket
      )

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async def forward_client_to_server() -> None:
      try:
        while True:
          line = await reader.readline()
          if not line:
            break
          processed = self.process_client_message(
              line.decode("utf-8", errors="replace")
          )
          for local_reply in self.pop_local_responses():
            sys.stdout.write(local_reply)
            sys.stdout.flush()
          if processed:
            await _send_upstream(processed)
      except (BrokenPipeError, ConnectionResetError):
        pass
      finally:
        try:
          proc.stdin.close()
        except Exception:
          pass

    async def forward_server_to_client() -> None:
      try:
        while True:
          line = await proc.stdout.readline()
          if not line:
            break
          processed = self.process_server_message(
              line.decode("utf-8", errors="replace")
          )
          for srv_reply in self.pop_local_server_responses():
            await _send_upstream(srv_reply)
          if processed:
            sys.stdout.write(processed)
            sys.stdout.flush()
      except (BrokenPipeError, ConnectionResetError):
        pass

    rc = 0
    try:
      try:
        await asyncio.gather(forward_client_to_server(), forward_server_to_client())
      finally:
        if bridge_server is not None:
          bridge_server.close()
          await bridge_server.wait_closed()
          try:
            if self.bridge_socket and os.path.exists(self.bridge_socket):
              os.unlink(self.bridge_socket)
          except OSError:
            pass
      rc = await proc.wait()
    finally:
      self.record_telemetry(command=command)
    if not self.quiet and self.stats.total_tokens_before > 0:
      sys.stderr.write(
          f"[mcp-slim] Session complete: {self.stats.total_tokens_before} -> "
          f"{self.stats.total_tokens_after} tokens "
          f"(-{self.stats.reduction_pct}%, saved {self.stats.tokens_saved} tokens, "
          f"redacted {self.secret_guard.report.total_redactions} secrets)\n"
      )
    return rc
