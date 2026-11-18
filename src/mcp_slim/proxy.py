"""Async JSON-RPC 2.0 stdio proxy for Model Context Protocol (MCP) servers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Awaitable

from .guard import SecretGuard, StreamSanitizer, ToolCallStreamSanitizer
from .slimmer import (
    ElicitationGuard,
    JsonRpcResponseFirewall,
    LazyToolBroker,
    OutputSlimmer,
    ResourceUriGuard,
    RootsPrivacyGuard,
    SchemaSlimmer,
    SessionPolicy,
    SessionPolicyEngine,
    SlimStats,
    TemplateArgumentShield,
    ToolCallReplayGuard,
    ToolCatalogPruner,
    TokenWindowGovernor,
    ToolPinGuard,
    scan_tool_result_for_injection,
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
      block_elicitation: bool = False,
      allow_credential_elicitation: bool = False,
      elicitation_guard: ElicitationGuard | None = None,
      block_roots: bool = False,
      allow_root_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      deny_root_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      roots_guard: RootsPrivacyGuard | None = None,
      pin_schemas: bool = False,
      allow_new_tools: bool = False,
      pin_guard: ToolPinGuard | None = None,
      allow_resource_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      deny_resource_prefixes: set[str] | list[str] | tuple[str, ...] | None = None,
      resource_guard: ResourceUriGuard | None = None,
      token_budget: int = 0,
      window_decay: float = 0.85,
      window_governor: TokenWindowGovernor | None = None,
      otel_exporter: OTelSpanExporter | None = None,
      otel_jsonl_path: Path | str | None = None,
      block_result_injection: bool = False,
      detect_replay: bool = False,
      replay_guard: ToolCallReplayGuard | None = None,
      repair_truncated_json: bool = False,
      prune_tools_intent: str | None = None,
      prune_top_k: int = 12,
      tool_pruner: ToolCatalogPruner | None = None,
      strict_rpc_ids: bool = False,
      max_batch_size: int = 32,
      rpc_firewall: JsonRpcResponseFirewall | None = None,
      enforce_session_policy: bool = False,
      session_policy: SessionPolicy | dict[str, Any] | None = None,
      session_policy_engine: SessionPolicyEngine | None = None,
      shield_templates: bool = False,
      max_template_depth: int = 1,
      sanitize_templates_in_place: bool = False,
      template_shield: TemplateArgumentShield | None = None,
      server_id: str = "",
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
    self.block_elicitation = block_elicitation
    self.elicitation_guard = (
        elicitation_guard
        if elicitation_guard is not None
        else ElicitationGuard(
            block_all=block_elicitation,
            block_credential_phishing=(
                self.secret_guard.enabled and not allow_credential_elicitation
            ),
        )
    )
    self.roots_guard = (
        roots_guard
        if roots_guard is not None
        else RootsPrivacyGuard(
            block_all=block_roots,
            allow_prefixes=allow_root_prefixes,
            deny_prefixes=deny_root_prefixes,
        )
    )
    self.pin_schemas = pin_schemas
    self.pin_guard = (
        pin_guard
        if pin_guard is not None
        else ToolPinGuard(enabled=pin_schemas, allow_new_tools=allow_new_tools)
    )
    self.resource_guard = (
        resource_guard
        if resource_guard is not None
        else ResourceUriGuard(
            allow_prefixes=allow_resource_prefixes,
            deny_prefixes=deny_resource_prefixes,
            block_traversal=True,
        )
    )
    self.window_governor = (
        window_governor
        if window_governor is not None
        else (
            self.output_slimmer.window_governor
            if self.output_slimmer.window_governor is not None
            else TokenWindowGovernor(budget_tokens=token_budget, decay=window_decay)
        )
    )
    self.repair_truncated_json = repair_truncated_json
    self._server_streams: dict[str, StreamSanitizer] = {}
    self._client_tool_streams = ToolCallStreamSanitizer(
        guard=self.secret_guard,
        repair_truncated_json=repair_truncated_json,
    )
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
    self.block_result_injection = block_result_injection
    self.replay_guard = (
        replay_guard
        if replay_guard is not None
        else ToolCallReplayGuard(enabled=detect_replay)
    )
    self.tool_pruner = (
        tool_pruner
        if tool_pruner is not None
        else ToolCatalogPruner(
            intent=prune_tools_intent,
            top_k=max(1, int(prune_top_k)),
            fallback_to_all=True,
        )
    )
    self.rpc_firewall = (
        rpc_firewall
        if rpc_firewall is not None
        else JsonRpcResponseFirewall(
            enabled=strict_rpc_ids,
            max_batch_size=max(1, int(max_batch_size)),
        )
    )
    self.server_id = server_id
    self.session_policy_engine = (
        session_policy_engine
        if session_policy_engine is not None
        else SessionPolicyEngine(
            policy=session_policy,
            enabled=bool(enforce_session_policy or session_policy is not None),
        )
    )
    self.sanitize_templates_in_place = sanitize_templates_in_place
    self.template_shield = (
        template_shield
        if template_shield is not None
        else TemplateArgumentShield(
            enabled=bool(shield_templates),
            max_depth=max(0, int(max_template_depth)),
            block_exfiltration=not sanitize_templates_in_place,
        )
    )
    self._pending_methods: dict[str | int, str] = {}
    self._pending_tool_names: dict[str | int, str] = {}
    self._pending_args: dict[str | int, Any] = {}
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
      if self.rpc_firewall.enabled:
        self.rpc_firewall.note_unparseable_client_frame()
      return raw_line
    if isinstance(msg, list):
      if self.rpc_firewall.enabled:
        c_viols = self.rpc_firewall.register_client_payload(msg)
        blocking_c = [
            v
            for v in c_viols
            if v.kind in ("empty_batch", "batch_disabled", "oversized_batch")
        ]
        if blocking_c:
          self.stats.blocked_calls += 1
          self.stats.rpc_spoofs_blocked += 1
          err_reply = {
              "jsonrpc": "2.0",
              "id": None,
              "error": {"code": -32600, "message": f"[mcp-slim] {blocking_c[0].message}"},
          }
          self._local_responses.append(
              json.dumps(err_reply, ensure_ascii=False, separators=(",", ":")) + "\n"
          )
          return ""
      for item in msg:
        if isinstance(item, dict) and item.get("id") is not None and isinstance(item.get("method"), str):
          self._pending_methods[item["id"]] = item["method"]
      if self.secret_guard.enabled:
        clean_batch = self.secret_guard.resolve_handles_in_value(
            self.secret_guard.redact_value(msg)
        )
        return json.dumps(clean_batch, ensure_ascii=False, separators=(",", ":")) + "\n"
      return raw_line
    if not isinstance(msg, dict):
      if self.rpc_firewall.enabled:
        self.rpc_firewall.note_unparseable_client_frame()
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
        res_obj = msg.get("result")
        if srv_method is None and isinstance(res_obj, dict) and "role" in res_obj and "content" in res_obj:
          self.stats.sampling_guarded += 1
        if (
            isinstance(res_obj, dict)
            and isinstance(res_obj.get("roots"), list)
            and (srv_method == "roots/list" or srv_method is None)
        ):
          res_obj = {
              **res_obj,
              "roots": self.roots_guard.filter_roots(
                  res_obj["roots"], stats=self.stats
              ),
          }
          msg["result"] = res_obj
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

    if (
        self.session_policy_engine.enabled
        and isinstance(method, str)
        and not (
            method == "tools/call"
            and isinstance(params, dict)
            and self.lazy_broker is not None
            and params.get("name") in (LazyToolBroker.SEARCH_TOOL_NAME, LazyToolBroker.CALL_TOOL_NAME)
        )
    ):
      pol_dec = self.session_policy_engine.evaluate_client_request(
          msg, server_id=self.server_id
      )
      if not pol_dec.allowed:
        self.stats.blocked_calls += 1
        self.stats.policy_violations_blocked += 1
        pol_msg = f"[mcp-slim] Security Alert MS019: {pol_dec.violations[0].message}"
        if not self.quiet:
          sys.stderr.write(f"{pol_msg}\n")
        if msg_id is not None:
          if method == "tools/call":
            pol_reply: dict[str, Any] = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": pol_msg}],
                    "isError": True,
                },
            }
          else:
            pol_reply = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "error": {"code": -32600, "message": pol_msg},
            }
          self._local_responses.append(
              json.dumps(pol_reply, ensure_ascii=False, separators=(",", ":")) + "\n"
          )
        elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
        self._emit_span(
            name=f"mcp.{method}",
            kind="CLIENT",
            start_ns=start_wall_ns,
            end_ns=start_wall_ns + elapsed_ns,
            is_error=True,
            error_message=pol_msg,
            attributes={
                "mcp.method.name": method,
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": True,
                "mcp_slim.policy_violation_blocked": True,
                "mcp_slim.violation_code": pol_dec.violations[0].violation_code,
            },
        )
        return ""

    if self.template_shield.enabled and isinstance(method, str) and isinstance(params, dict):
      tpl_dec = None
      if method == "prompts/get" and isinstance(params.get("arguments"), dict):
        tpl_dec = self.template_shield.audit_prompt_arguments(
            str(params.get("name") or ""),
            params["arguments"],
            block_exfiltration=not self.sanitize_templates_in_place,
        )
      elif method in ("resources/read", "resources/subscribe"):
        res_args: dict[str, Any] = {}
        if isinstance(params.get("uri"), str):
          res_args["uri"] = params["uri"]
        if isinstance(params.get("arguments"), dict):
          res_args.update(params["arguments"])
        if res_args:
          tpl_dec = self.template_shield.audit_arguments(
              res_args,
              template_type="resource",
              uri_template=str(params.get("uriTemplate") or ""),
              block_exfiltration=not self.sanitize_templates_in_place,
          )
      elif method == "completion/complete" and isinstance(params.get("argument"), dict):
        arg_obj = params["argument"]
        arg_k = str(arg_obj.get("name") or "value")
        tpl_dec = self.template_shield.audit_arguments(
            {arg_k: arg_obj.get("value", "")},
            template_type="completion",
            block_exfiltration=not self.sanitize_templates_in_place,
        )

      if tpl_dec is not None and tpl_dec.violations:
        if not tpl_dec.allowed:
          self.stats.blocked_calls += 1
          self.stats.template_exfiltrations_blocked += 1
          tpl_msg = f"[mcp-slim] Security Alert MS020: {tpl_dec.violations[0].message}"
          if not self.quiet:
            sys.stderr.write(f"{tpl_msg}\n")
          if msg_id is not None:
            tpl_err = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "error": {"code": -32602, "message": tpl_msg},
            }
            self._local_responses.append(
                json.dumps(tpl_err, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.{method}",
              kind="CLIENT",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=tpl_msg,
              attributes={
                  "mcp.method.name": method,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.template_exfiltration_blocked": True,
                  "mcp_slim.violation_code": tpl_dec.violations[0].violation_code,
              },
          )
          return ""
        self.stats.template_exfiltrations_blocked += len(tpl_dec.violations)
        if method == "prompts/get" and isinstance(params.get("arguments"), dict):
          params["arguments"] = tpl_dec.sanitized_arguments
        elif method in ("resources/read", "resources/subscribe") and "uri" in tpl_dec.sanitized_arguments:
          params["uri"] = tpl_dec.sanitized_arguments["uri"]
        elif method == "completion/complete" and isinstance(params.get("argument"), dict):
          arg_k = str(params["argument"].get("name") or "value")
          params["argument"]["value"] = tpl_dec.sanitized_arguments.get(arg_k, "")

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
          if self.session_policy_engine.enabled:
            lazy_pol = self.session_policy_engine.evaluate_client_request(
                msg, server_id=self.server_id
            )
            if not lazy_pol.allowed:
              self.stats.blocked_calls += 1
              self.stats.policy_violations_blocked += 1
              pol_msg = f"[mcp-slim] Security Alert MS019: {lazy_pol.violations[0].message}"
              if not self.quiet:
                sys.stderr.write(f"{pol_msg}\n")
              if msg_id is not None:
                pol_reply = {
                    "jsonrpc": msg.get("jsonrpc", "2.0"),
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": pol_msg}],
                        "isError": True,
                    },
                }
                self._local_responses.append(
                    json.dumps(pol_reply, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
              return ""

      if isinstance(tool_name, str):
        resolved_tool_name = tool_name
        quarantine_mut = self.pin_guard.get_quarantine(tool_name)
        if quarantine_mut is not None:
          self.stats.blocked_calls += 1
          self.stats.rug_pulls_blocked += 1
          pin_msg = (
              f"[mcp-slim] Tool '{tool_name}' is quarantined by --pin-schemas: "
              f"{quarantine_mut.message}"
          )
          pin_payload = {
              "jsonrpc": msg.get("jsonrpc", "2.0"),
              "id": msg_id,
              "result": {
                  "content": [{"type": "text", "text": pin_msg}],
                  "isError": True,
              },
          }
          self._local_responses.append(
              json.dumps(pin_payload, ensure_ascii=False, separators=(",", ":"))
              + "\n"
          )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.tools/call {tool_name}",
              kind="CLIENT",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=pin_msg,
              attributes={
                  "mcp.method.name": "tools/call",
                  "mcp.tool.name": tool_name,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.rug_pull_quarantined": True,
              },
          )
          return ""

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

      if self.repair_truncated_json and isinstance(arguments, str):
        from ._ms015_ms016 import ToolCallJsonRepairer  # noqa: PLC0415

        rep = ToolCallJsonRepairer(require_object_root=True).repair(arguments)
        if rep.is_valid and isinstance(rep.value, dict):
          arguments = rep.value
          params["arguments"] = arguments
          if rep.repaired:
            self.stats.json_repairs += 1

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
            # Track arguments for MS010 replay detection.
            if msg_id is not None and "arguments" in params:
              self._pending_args[msg_id] = params.get("arguments")
        else:
          self._read_cache.clear()

    if method in ("resources/read", "resources/subscribe") and isinstance(params, dict):
      req_uri = params.get("uri")
      uri_err = self.resource_guard.validate_uri(req_uri)
      if uri_err is not None:
        self.stats.blocked_calls += 1
        self.stats.resources_blocked += 1
        if msg_id is not None:
          err_payload = {
              "jsonrpc": msg.get("jsonrpc", "2.0"),
              "id": msg_id,
              "error": {"code": -32602, "message": uri_err},
          }
          self._local_responses.append(
              json.dumps(err_payload, ensure_ascii=False, separators=(",", ":"))
              + "\n"
          )
        elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
        self._emit_span(
            name=f"mcp.{method}",
            kind="CLIENT",
            start_ns=start_wall_ns,
            end_ns=start_wall_ns + elapsed_ns,
            is_error=True,
            error_message=uri_err,
            attributes={
                "mcp.method.name": method,
                "mcp.resource.uri": str(req_uri) if req_uri is not None else "",
                "mcp.request.id": str(msg_id) if msg_id is not None else None,
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": True,
                "mcp_slim.resource_blocked": True,
            },
        )
        return ""

    if "id" in msg and isinstance(method, str) and self.rpc_firewall.enabled:
      self.rpc_firewall.register_client_payload(msg)
    if msg_id is not None and isinstance(method, str):
      self._pending_methods[msg_id] = method
      if resolved_tool_name is not None:
        self._pending_tool_names[msg_id] = resolved_tool_name

    red_req_before = self.secret_guard.report.total_redactions
    if (
        isinstance(method, str)
        and method in ("notifications/tools/call_delta", "notifications/tools/input_delta", "tools/callStream")
        and isinstance(params, dict)
        and (isinstance(params.get("delta"), str) or isinstance(params.get("argumentsDelta"), str))
    ):
      delta_key = "delta" if isinstance(params.get("delta"), str) else "argumentsDelta"
      call_id = params.get("callId") or params.get("streamId") or msg_id or "default"
      t_name = params.get("name") if isinstance(params.get("name"), str) else None
      eff_t_name = t_name or self._client_tool_streams.tool_name_for(call_id)
      if eff_t_name:
        quarantine_mut = self.pin_guard.get_quarantine(eff_t_name)
        if quarantine_mut is not None:
          self.stats.blocked_calls += 1
          self.stats.rug_pulls_blocked += 1
          self._client_tool_streams.discard_call(call_id)
          pin_msg = (
              f"[mcp-slim] Tool '{eff_t_name}' is quarantined by --pin-schemas: "
              f"{quarantine_mut.message}"
          )
          if msg_id is not None:
            pin_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": pin_msg}],
                    "isError": True,
                },
            }
            self._local_responses.append(json.dumps(pin_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.{method} {eff_t_name}",
              kind="CLIENT",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=pin_msg,
              attributes={
                  "mcp.method.name": method,
                  "mcp.tool.name": eff_t_name,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.rug_pull_quarantined": True,
              },
          )
          return ""
        if not self.schema_slimmer.is_tool_allowed(eff_t_name):
          self.stats.blocked_calls += 1
          self._client_tool_streams.discard_call(call_id)
          reason_suffix = (
              " (read-only mode active)"
              if self.schema_slimmer.read_only_only
              and not self.schema_slimmer.is_read_only_tool(eff_t_name)
              else ""
          )
          block_msg = f"[mcp-slim] Tool '{eff_t_name}' is blocked by proxy allow/deny policy{reason_suffix}."
          if msg_id is not None:
            block_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": block_msg}],
                    "isError": True,
                },
            }
            self._local_responses.append(json.dumps(block_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.{method} {eff_t_name}",
              kind="CLIENT",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=block_msg,
              attributes={
                  "mcp.method.name": method,
                  "mcp.tool.name": eff_t_name,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
              },
          )
          return ""

      safe_delta = self._client_tool_streams.push_delta(
          call_id, params[delta_key], tool_name=t_name
      )
      self.stats.stream_chunks_sanitized += 1
      if params.get("done") is True:
        fin = self._client_tool_streams.finish_call(
            call_id, schema_slimmer=self.schema_slimmer, stats=self.stats
        )
        final_tool_name = str(fin.get("tool_name") or eff_t_name or "")
        if fin["validation_error"] is not None:
          self.stats.blocked_calls += 1
          if msg_id is not None:
            err_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": fin["validation_error"]}],
                    "isError": True,
                },
            }
            self._local_responses.append(json.dumps(err_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
          return ""
        if (
            self.cache_ttl > 0.0
            and final_tool_name
            and not self.schema_slimmer.is_read_only_tool(final_tool_name)
        ):
          self._read_cache.clear()
        params["arguments"] = fin["arguments"]
        params.pop(delta_key, None)
      else:
        params[delta_key] = safe_delta
      return json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"

    if method in ("tools/call", "prompts/get") and isinstance(params, dict):
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

    if msg_id is None and isinstance(params, dict) and self.secret_guard.enabled:
      clean_cli_params = self.secret_guard.resolve_handles_in_value(
          self.secret_guard.redact_value(params)
      )
      if clean_cli_params != params:
        msg["params"] = clean_cli_params
        return json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"

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
      if self.rpc_firewall.enabled:
        self.stats.blocked_calls += 1
        self.stats.rpc_spoofs_blocked += 1
        if not self.quiet:
          sys.stderr.write("[mcp-slim] Security Alert MS018: Dropped non-JSON server frame.\n")
        return ""
      return raw_line

    if self.rpc_firewall.enabled:
      fw_ts = time.time_ns()
      fw_dec = self.rpc_firewall.inspect_server_payload(msg)
      if fw_dec.violations:
        dropped = (
            max(1, fw_dec.dropped_count)
            if (not fw_dec.allowed or fw_dec.dropped_count > 0)
            else 0
        )
        self.stats.blocked_calls += dropped
        self.stats.rpc_spoofs_blocked += dropped
        if not self.quiet:
          for viol in fw_dec.violations:
            sys.stderr.write(f"[mcp-slim] Security Alert MS018: {viol.message}\n")
        self._emit_span(
            name="mcp.rpc_firewall",
            kind="CLIENT",
            start_ns=fw_ts,
            end_ns=fw_ts,
            is_error=True,
            error_message=fw_dec.violations[0].message,
            attributes={
                "mcp.method.name": "jsonrpc.firewall",
                "mcp_slim.cache_hit": False,
                "mcp_slim.blocked": True,
                "mcp_slim.rpc_spoof_blocked": True,
                "mcp_slim.violation_kind": fw_dec.violations[0].kind,
            },
        )
      if not fw_dec.allowed or fw_dec.sanitized_payload is None:
        return ""
      msg = fw_dec.sanitized_payload

    if isinstance(msg, list):
      for elem in msg:
        if isinstance(elem, dict) and elem.get("id") is not None:
          self._pending_methods.pop(elem["id"], None)
          self._pending_tool_names.pop(elem["id"], None)
      redacted_batch = self.secret_guard.redact_value(msg)
      serialized_batch = json.dumps(redacted_batch, ensure_ascii=False, separators=(",", ":"))
      if not self.secret_guard.vault_handles:
        serialized_batch = self.secret_guard.redact_text(serialized_batch)
      out_batch = serialized_batch + "\n"
      self.stats.wire_bytes_before += len(raw_line.encode("utf-8"))
      self.stats.wire_bytes_after += len(out_batch.encode("utf-8"))
      return out_batch

    if not isinstance(msg, dict):
      return raw_line

    msg_id = msg.get("id")
    srv_method = msg.get("method")

    # Server-initiated requests / notifications (e.g. sampling/createMessage, elicitation/create)
    if isinstance(srv_method, str) and "result" not in msg and "error" not in msg:
      srv_params = msg.get("params")
      start_wall_ns = time.time_ns()
      start_mono_ns = time.perf_counter_ns()
      if self.session_policy_engine.enabled:
        srv_pol = self.session_policy_engine.evaluate_server_response(
            msg, server_id=self.server_id
        )
        if not srv_pol.allowed:
          self.stats.blocked_calls += 1
          self.stats.policy_violations_blocked += 1
          srv_err = f"[mcp-slim] Security Alert MS019: {srv_pol.violations[0].message}"
          if not self.quiet:
            sys.stderr.write(f"{srv_err}\n")
          if msg_id is not None:
            err_payload = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "error": {"code": -32600, "message": srv_err},
            }
            self._local_server_responses.append(
                json.dumps(err_payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name=f"mcp.{srv_method}",
              kind="SERVER",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=srv_err,
              attributes={
                  "mcp.method.name": srv_method,
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.policy_violation_blocked": True,
                  "mcp_slim.violation_code": srv_pol.violations[0].violation_code,
              },
          )
          return ""
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
                {"messages": new_params["messages"]},
                stats=self.stats,
                guard=self.secret_guard,
                force_redact=True,
            )
            new_params["messages"] = slim_msgs["messages"]
          if isinstance(new_params.get("systemPrompt"), str) and new_params["systemPrompt"]:
            sp_before = estimate_tokens(new_params["systemPrompt"])
            slim_sp = self.output_slimmer.slim_text(
                new_params["systemPrompt"],
                guard=self.secret_guard,
                force_redact=True,
            )
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

      if srv_method == "elicitation/create":
        decision = self.elicitation_guard.evaluate(srv_params)
        self.stats.elicitations_guarded += 1
        if not decision["allowed"]:
          self.stats.blocked_calls += 1
          reason_text = str(decision["reason"] or "[mcp-slim] Elicitation blocked.")
          if not self.quiet and decision["mode"] == "decline":
            sys.stderr.write(f"{reason_text}\n")
          if msg_id is not None:
            if decision["mode"] == "policy_error":
              reply_obj: dict[str, Any] = {
                  "jsonrpc": msg.get("jsonrpc", "2.0"),
                  "id": msg_id,
                  "error": {"code": -32600, "message": reason_text},
              }
            else:
              reply_obj = {
                  "jsonrpc": msg.get("jsonrpc", "2.0"),
                  "id": msg_id,
                  "result": {"action": "decline"},
              }
            self._local_server_responses.append(
                json.dumps(reply_obj, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name="mcp.elicitation/create",
              kind="SERVER",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=True,
              error_message=reason_text,
              attributes={
                  "mcp.method.name": "elicitation/create",
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.elicitation_phishing_blocked": decision["mode"] == "decline",
              },
          )
          return ""

        red_el_0 = self.secret_guard.report.total_redactions
        sch_b_0 = self.stats.schema_tokens_before
        sch_a_0 = self.stats.schema_tokens_after
        if isinstance(srv_params, dict):
          new_el = dict(srv_params)
          if isinstance(new_el.get("message"), str) and new_el["message"]:
            new_el["message"] = self.secret_guard.redact_text(
                new_el["message"], force_redact=True
            )
          if isinstance(new_el.get("requestedSchema"), dict):
            raw_sch = new_el["requestedSchema"]
            sch_tok_b = estimate_tokens(raw_sch)
            slim_sch = self.schema_slimmer.slim_schema_node(
                raw_sch, is_root=True
            )
            sch_tok_a = estimate_tokens(slim_sch)
            self.stats.schema_tokens_before += sch_tok_b
            self.stats.schema_tokens_after += sch_tok_a
            new_el["requestedSchema"] = slim_sch
          msg["params"] = self.secret_guard.redact_value(new_el, force_redact=True)
        serialized_el = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        serialized_el = self.secret_guard.redact_text(serialized_el, force_redact=True)
        out_el_line = serialized_el + "\n"
        wire_in = len(raw_line.encode("utf-8"))
        wire_out = len(out_el_line.encode("utf-8"))
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
                "tok_before": self.stats.schema_tokens_before - sch_b_0,
                "tok_after": self.stats.schema_tokens_after - sch_a_0,
                "redactions_on_req": self.secret_guard.report.total_redactions - red_el_0,
            }
        return out_el_line

      if srv_method == "roots/list":
        if self.roots_guard.block_all:
          self.stats.blocked_calls += 1
          self.stats.roots_filtered += 1
          if msg_id is not None:
            roots_reply = {
                "jsonrpc": msg.get("jsonrpc", "2.0"),
                "id": msg_id,
                "result": {"roots": []},
            }
            self._local_server_responses.append(
                json.dumps(roots_reply, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
          elapsed_ns = max(0, time.perf_counter_ns() - start_mono_ns)
          self._emit_span(
              name="mcp.roots/list",
              kind="SERVER",
              start_ns=start_wall_ns,
              end_ns=start_wall_ns + elapsed_ns,
              is_error=False,
              attributes={
                  "mcp.method.name": "roots/list",
                  "mcp.request.id": str(msg_id) if msg_id is not None else None,
                  "mcp_slim.cache_hit": False,
                  "mcp_slim.blocked": True,
                  "mcp_slim.roots_masked": True,
              },
          )
          return ""
        if msg_id is not None:
          self._pending_server_methods[msg_id] = srv_method
          if self.otel_exporter is not None:
            self._inflight_spans[("server", msg_id)] = {
                "start_wall_ns": start_wall_ns,
                "start_mono_ns": start_mono_ns,
                "method": srv_method,
                "wire_in": len(raw_line.encode("utf-8")),
                "wire_out": len(raw_line.encode("utf-8")),
                "tok_before": 0,
                "tok_after": 0,
                "redactions_on_req": 0,
            }
        return raw_line

      if msg_id is not None:
        self._pending_server_methods[msg_id] = srv_method
      if isinstance(srv_params, dict) and self.secret_guard.enabled:
        stream_field: str | None = None
        for cand in ("delta", "chunk"):
          if isinstance(srv_params.get(cand), str):
            stream_field = cand
            break
        has_bounded_progress = (
            isinstance(srv_params.get("progress"), (int, float))
            and isinstance(srv_params.get("total"), (int, float))
            and float(srv_params["total"]) > 0
        )
        if (
            stream_field is None
            and isinstance(srv_params.get("message"), str)
            and ("done" in srv_params or "final" in srv_params or "streamId" in srv_params or has_bounded_progress)
        ):
          stream_field = "message"

        if stream_field is not None:
          stream_key = str(
              srv_params.get("streamId")
              or srv_params.get("progressToken")
              or srv_method
          )
          sanitizer = self._server_streams.get(stream_key)
          if sanitizer is None:
            while len(self._server_streams) >= 256:
              oldest_key = next(iter(self._server_streams))
              self._server_streams.pop(oldest_key, None)
            sanitizer = StreamSanitizer(
                self.secret_guard,
                force_redact=not self.secret_guard.vault_handles,
            )
            self._server_streams[stream_key] = sanitizer
          is_done = bool(
              srv_params.get("done") is True
              or srv_params.get("final") is True
              or (
                  has_bounded_progress
                  and float(srv_params["progress"]) >= float(srv_params["total"])
              )
          )
          emitted_chunk = sanitizer.push(srv_params[stream_field])
          if is_done:
            emitted_chunk += sanitizer.flush()
            self._server_streams.pop(stream_key, None)
          self.stats.stream_chunks_sanitized += 1
          other_params = {
              k: (
                  emitted_chunk
                  if k == stream_field
                  else self.secret_guard.redact_value(
                      v, force_redact=not self.secret_guard.vault_handles
                  )
              )
              for k, v in srv_params.items()
          }
          msg["params"] = other_params
          return json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"

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

      governed_turn = False
      if method == "tools/list" or isinstance(result.get("tools"), list):
        raw_tools = result.get("tools")
        if isinstance(raw_tools, list):
          orig_tok = estimate_tokens(raw_tools)
          verified_tools, pin_mutations = self.pin_guard.verify_or_pin(
              raw_tools, stats=self.stats
          )
          if not self.quiet and pin_mutations:
            for pm in pin_mutations:
              sys.stderr.write(f"[mcp-slim] Security Pin Alert: {pm.message}\n")
          tools = self.secret_guard.redact_value(verified_tools)
          if self.session_policy_engine.enabled:
            tools = self.session_policy_engine.filter_tools_for_server(
                self.server_id, tools
            )
          sanitized_tok = estimate_tokens(tools)
          if (
              self.tool_pruner is not None
              and self.tool_pruner.enabled
              and self.lazy_broker is None
              and not self.cli_only
          ):
            prune_res = self.tool_pruner.prune(tools)
            self.stats.tools_pruned += prune_res.tools_pruned
            slimmed_list = self.schema_slimmer.slim_tools_list(
                prune_res.tools, stats=None
            )
            self.stats.tools_total += len(tools)
            self.stats.tools_kept += len(slimmed_list)
            self.stats.schema_tokens_before += orig_tok
            self.stats.schema_tokens_after += estimate_tokens(slimmed_list)
            if not self.quiet and self.schema_slimmer.policy_warnings:
              for warn_msg in self.schema_slimmer.policy_warnings:
                sys.stderr.write(f"[mcp-slim] Warning: {warn_msg}\n")
            for t in slimmed_list:
              if isinstance(t.get("name"), str):
                self._cached_tools[t["name"]] = t
            result["tools"] = slimmed_list
          else:
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
        governed_turn = self.window_governor.enabled
        result = self.output_slimmer.slim_call_result(
            result,
            stats=self.stats,
            guard=self.secret_guard,
            governor=self.window_governor,
        )
        # MS009: scan tool result content for prompt-injection directives.
        if isinstance(result, dict):
          injection_findings = scan_tool_result_for_injection(result, self.secret_guard)
          for finding in injection_findings:
            if not self.quiet:
              sys.stderr.write(
                  f"[mcp-slim] Security Alert MS009: {finding['message']}\n"
              )
            if self.block_result_injection:
              # Replace poisoned content item with a redaction notice.
              idx = finding.get("item_index")
              content = result.get("content")
              if isinstance(content, list) and isinstance(idx, int) and idx < len(content):
                content = list(content)
                content[idx] = {
                    "type": "text",
                    "text": "[mcp-slim] Tool result blocked: contains prompt-injection directives (MS009).",
                }
                result = {**result, "content": content}
        # MS010: detect server-side result substitution (replay guard).
        if isinstance(result, dict):
          _tool_name_ms010 = self._pending_tool_names.get(msg_id, "") if msg_id is not None else ""
          _args_ms010 = self._pending_args.get(msg_id) if msg_id is not None else None
          replay_finding = self.replay_guard.check(_tool_name_ms010, _args_ms010, result)
          if replay_finding is not None and not self.quiet:
            sys.stderr.write(
                f"[mcp-slim] Security Alert MS010: {replay_finding['message']}\n"
            )
      elif method == "resources/read" or isinstance(result.get("contents"), list):
        governed_turn = self.window_governor.enabled
        result = self.output_slimmer.slim_resource_result(
            result,
            stats=self.stats,
            guard=self.secret_guard,
            governor=self.window_governor,
        )
      elif method == "prompts/get" or isinstance(result.get("messages"), list):
        governed_turn = self.window_governor.enabled
        result = self.output_slimmer.slim_prompt_result(
            result,
            stats=self.stats,
            guard=self.secret_guard,
            governor=self.window_governor,
        )
      elif method == "resources/list" or isinstance(result.get("resources"), list):
        raw_res = result.get("resources")
        if isinstance(raw_res, list):
          res_before_tok = estimate_tokens(raw_res)
          sanitized_res = self.secret_guard.redact_value(raw_res)
          filtered_res = self.resource_guard.filter_resources_list(
              sanitized_res, stats=self.stats
          )
          result = {**result, "resources": filtered_res}
          self.stats.schema_tokens_before += res_before_tok
          self.stats.schema_tokens_after += estimate_tokens(filtered_res)
      elif method == "resources/templates/list" or isinstance(
          result.get("resourceTemplates"), list
      ):
        raw_tmpls = result.get("resourceTemplates")
        if isinstance(raw_tmpls, list):
          tmpl_before_tok = estimate_tokens(raw_tmpls)
          sanitized_tmpls = self.secret_guard.redact_value(raw_tmpls)
          filtered_tmpls = self.resource_guard.filter_resource_templates(
              sanitized_tmpls, stats=self.stats
          )
          result = {**result, "resourceTemplates": filtered_tmpls}
          self.stats.schema_tokens_before += tmpl_before_tok
          self.stats.schema_tokens_after += estimate_tokens(filtered_tmpls)
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
        span_attrs: dict[str, Any] = {
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
        }
        if governed_turn and self.window_governor.last_decision is not None:
          dec = self.window_governor.last_decision
          span_attrs["mcp_slim.window_tokens"] = round(
              self.window_governor.window_tokens, 2
          )
          span_attrs["mcp_slim.budget_pressure"] = dec.pressure
          span_attrs["mcp_slim.budget_throttled"] = dec.throttled
        self._emit_span(
            name=span_name,
            kind="CLIENT",
            start_ns=inflight_cli["start_wall_ns"],
            end_ns=inflight_cli["start_wall_ns"] + elapsed_ns,
            is_error=is_err,
            attributes=span_attrs,
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
      if self.rpc_firewall.enabled:
        self.rpc_firewall.register_client_payload(list_msg)
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
