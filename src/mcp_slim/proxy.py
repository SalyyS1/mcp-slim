"""Async JSON-RPC 2.0 stdio proxy for Model Context Protocol (MCP) servers."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from .guard import SecretGuard
from .slimmer import LazyToolBroker, OutputSlimmer, SchemaSlimmer, SlimStats

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
    self._pending_methods: dict[str | int, str] = {}
    self._pending_cache_keys: dict[str | int, str] = {}
    self._read_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    self._local_responses: list[str] = []

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
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      return self.secret_guard.redact_text(serialized) + "\n"

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
          if self.lazy_broker is not None:
            result["tools"] = self.lazy_broker.ingest_tools(
                tools, stats=self.stats
            )
          else:
            result["tools"] = self.schema_slimmer.slim_tools_list(
                tools, stats=self.stats
            )
      elif method == "tools/call" or isinstance(result.get("content"), list):
        result = self.output_slimmer.slim_call_result(result, stats=self.stats)
      msg["result"] = result
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      redacted_str = self.secret_guard.redact_text(serialized)
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
      return redacted_str + "\n"

    return raw_line

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
            proc.stdin.write(processed.encode("utf-8"))
            await proc.stdin.drain()
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
          sys.stdout.write(processed)
          sys.stdout.flush()
      except (BrokenPipeError, ConnectionResetError):
        pass

    await asyncio.gather(forward_client_to_server(), forward_server_to_client())
    rc = await proc.wait()
    if not self.quiet and self.stats.total_tokens_before > 0:
      sys.stderr.write(
          f"[mcp-slim] Session complete: {self.stats.total_tokens_before} -> "
          f"{self.stats.total_tokens_after} tokens "
          f"(-{self.stats.reduction_pct}%, saved {self.stats.tokens_saved} tokens, "
          f"redacted {self.secret_guard.report.total_redactions} secrets)\n"
      )
    return rc
