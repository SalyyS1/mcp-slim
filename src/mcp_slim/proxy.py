"""Async JSON-RPC 2.0 stdio proxy for Model Context Protocol (MCP) servers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Callable, Awaitable

from .guard import SecretGuard
from .slimmer import LazyToolBroker, OutputSlimmer, SchemaSlimmer, SlimStats
from .telemetry import StatsLedger

_READ_ONLY_PREFIXES = (
    "get_",
    "read_",
    "list_",
    "search_",
    "fetch_",
    "describe_",
    "inspect_",
    "find_",
    "query_",
)


def _is_read_only_tool(name: str) -> bool:
  lower = name.lower()
  return lower.startswith(_READ_ONLY_PREFIXES)


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
    self.stats_ledger = (
        stats_ledger
        if stats_ledger is not None
        else StatsLedger(enabled=not no_stats)
    )
    self._pending_methods: dict[str | int, str] = {}
    self._pending_cache_keys: dict[str | int, str] = {}
    self._read_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    self._local_responses: list[str] = []
    self._cached_tools: dict[str, dict[str, Any]] = {}
    self._bridge_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
    self._bridge_seq: int = 0

  def pop_local_responses(self) -> list[str]:
    """Returns any locally synthesized JSON-RPC responses (e.g. from mcp_search_tools)."""
    items = list(self._local_responses)
    self._local_responses.clear()
    return items

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
        self._local_responses.append(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
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

      if isinstance(tool_name, str) and not self.schema_slimmer.is_tool_allowed(tool_name):
        self.stats.blocked_calls += 1
        blocked_payload = {
            "jsonrpc": msg.get("jsonrpc", "2.0"),
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"[mcp-slim] Tool '{tool_name}' is blocked by proxy "
                            "allow/deny policy."
                        ),
                    }
                ],
                "isError": True,
            },
        }
        self._local_responses.append(
            json.dumps(blocked_payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
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
          return ""

      if self.cache_ttl > 0.0 and isinstance(tool_name, str):
        if _is_read_only_tool(tool_name):
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
            return ""
          if msg_id is not None:
            self._pending_cache_keys[msg_id] = cache_key
        else:
          self._read_cache.clear()

    if msg_id is not None and isinstance(method, str):
      self._pending_methods[msg_id] = method

    if method == "tools/call" and isinstance(params, dict):
      if "arguments" in params:
        redacted_args = self.secret_guard.redact_value(params["arguments"])
        params["arguments"] = self.secret_guard.resolve_handles_in_value(
            redacted_args
        )
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      if not self.secret_guard.vault_handles:
        serialized = self.secret_guard.redact_text(serialized)
      return serialized + "\n"

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
    method = (
        self._pending_methods.pop(msg_id, None) if msg_id is not None else None
    )
    pending_cache_key = (
        self._pending_cache_keys.pop(msg_id, None)
        if msg_id is not None
        else None
    )
    result = msg.get("result")

    if isinstance(result, dict):
      if method == "tools/list" or isinstance(result.get("tools"), list):
        tools = result.get("tools")
        if isinstance(tools, list):
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
      elif method == "tools/call" or isinstance(result.get("content"), list):
        result = self.output_slimmer.slim_call_result(result, stats=self.stats)
      result = self.secret_guard.redact_value(result)
      msg["result"] = result
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      redacted_str = (
          serialized
          if self.secret_guard.vault_handles
          else self.secret_guard.redact_text(serialized)
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
      out_line = redacted_str + "\n"
      self.stats.wire_bytes_before += len(raw_line.encode("utf-8"))
      self.stats.wire_bytes_after += len(out_line.encode("utf-8"))
      return out_line

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
