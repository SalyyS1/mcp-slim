"""Async JSON-RPC 2.0 stdio proxy for Model Context Protocol (MCP) servers."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .guard import SecretGuard
from .slimmer import OutputSlimmer, SchemaSlimmer, SlimStats


class MCPProxy:
  """Intercepts and optimizes JSON-RPC 2.0 messages between an MCP client and server."""

  def __init__(
      self,
      schema_slimmer: SchemaSlimmer | None = None,
      output_slimmer: OutputSlimmer | None = None,
      secret_guard: SecretGuard | None = None,
      quiet: bool = False,
  ) -> None:
    self.schema_slimmer = schema_slimmer or SchemaSlimmer()
    self.output_slimmer = output_slimmer or OutputSlimmer()
    self.secret_guard = secret_guard or SecretGuard()
    self.stats = SlimStats()
    self.quiet = quiet
    self._pending_methods: dict[str | int, str] = {}

  def process_client_message(self, raw_line: str) -> str:
    """Processes a client -> server JSON-RPC line."""
    try:
      msg = json.loads(raw_line)
    except json.JSONDecodeError:
      return raw_line
    if not isinstance(msg, dict):
      return raw_line

    msg_id = msg.get("id")
    method = msg.get("method")
    if msg_id is not None and isinstance(method, str):
      self._pending_methods[msg_id] = method

    if method == "tools/call" and isinstance(msg.get("params"), dict):
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
    method = self._pending_methods.pop(msg_id, None) if msg_id is not None else None
    result = msg.get("result")

    if isinstance(result, dict):
      if method == "tools/list" or isinstance(result.get("tools"), list):
        tools = result.get("tools")
        if isinstance(tools, list):
          result["tools"] = self.schema_slimmer.slim_tools_list(
              tools, stats=self.stats
          )
      elif method == "tools/call" or isinstance(result.get("content"), list):
        result = self.output_slimmer.slim_call_result(result, stats=self.stats)
      msg["result"] = result
      serialized = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
      return self.secret_guard.redact_text(serialized) + "\n"

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
          processed = self.process_client_message(line.decode("utf-8", errors="replace"))
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
          processed = self.process_server_message(line.decode("utf-8", errors="replace"))
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
