"""Persistent zero-dependency token & API cost telemetry ledger for mcp-slim."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import tempfile
from typing import Any

from .slimmer import SlimStats

COUNTER_KEYS: tuple[str, ...] = (
    "sessions",
    "schema_tokens_before",
    "schema_tokens_after",
    "output_tokens_before",
    "output_tokens_after",
    "wire_bytes_before",
    "wire_bytes_after",
    "refs_inlined",
    "coerced_arguments",
    "cache_hits",
    "blocked_calls",
    "secrets_redacted",
)

# Standard public API input token pricing ($ USD per 1M input tokens).
MODEL_INPUT_RATES: tuple[tuple[str, str, float], ...] = (
    ("claude_opus_4", "Claude Opus 4", 15.00),
    ("claude_sonnet_4", "Claude Sonnet 4", 3.00),
    ("gpt_4o", "GPT-4o", 2.50),
    ("gemini_2_5_pro", "Gemini 2.5 Pro", 1.25),
)

_DISABLED_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

_NPX_VALUE_FLAGS = frozenset({"-p", "--package", "-c", "--call", "--cache", "--userconfig"})
_UV_VALUE_FLAGS = frozenset({
    "--from",
    "--with",
    "--with-editable",
    "--with-requirements",
    "--python",
    "-p",
    "--directory",
    "--index-url",
    "--extra-index-url",
})
_DOCKER_VALUE_FLAGS = frozenset({
    "-e",
    "--env",
    "--env-file",
    "-v",
    "--volume",
    "--mount",
    "-p",
    "--publish",
    "-w",
    "--workdir",
    "--network",
    "--name",
    "--entrypoint",
    "-u",
    "--user",
    "--platform",
    "--cpus",
    "--memory",
    "-m",
})
_GENERIC_SCRIPT_NAMES = frozenset({
    "index.js",
    "index.mjs",
    "index.cjs",
    "cli.js",
    "main.js",
    "server.js",
    "index.py",
    "main.py",
    "server.py",
    "__main__.py",
})
_BUILD_DIR_NAMES = frozenset({"dist", "build", "bin", "lib", "out", "src"})


def _utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_stats_disabled_by_env() -> bool:
  raw = os.environ.get("MCP_SLIM_NO_STATS", "").strip().lower()
  return raw in _DISABLED_ENV_VALUES


def default_stats_path() -> Path:
  override = os.environ.get("MCP_SLIM_STATS_PATH", "").strip()
  if override:
    return Path(override).expanduser()
  return Path.home() / ".mcp-slim" / "stats.json"


def _strip_pkg_version(spec: str) -> str:
  """Strips trailing npm/pip version tags while preserving `@scope/pkg` names."""
  cleaned = spec.strip()
  if cleaned.startswith("git+"):
    tail = cleaned.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
      tail = tail[:-4]
    return tail or cleaned
  if cleaned.startswith("@"):
    at_idx = cleaned.find("@", 1)
    if at_idx != -1:
      return cleaned[:at_idx]
    return cleaned
  if "@" in cleaned and "://" not in cleaned:
    return cleaned.split("@", 1)[0]
  return cleaned


def _script_label(path_str: str) -> str:
  p = Path(path_str)
  name = p.name or path_str
  if name in _GENERIC_SCRIPT_NAMES and p.parent.name not in ("", ".", "/"):
    parent = p.parent
    if parent.name in _BUILD_DIR_NAMES and parent.parent.name not in ("", ".", "/"):
      return parent.parent.name
    return parent.name
  return name


def derive_server_label(command: Sequence[str] | str | None) -> str:
  """Derives a clean, human-readable server identifier from an upstream stdio command."""
  if command is None:
    return "unknown"
  if isinstance(command, str):
    try:
      tokens = shlex.split(command)
    except ValueError:
      tokens = command.split()
  else:
    tokens = [str(t) for t in command if str(t).strip()]

  while tokens and tokens[0] == "--":
    tokens = tokens[1:]
  if not tokens:
    return "unknown"

  exe = Path(tokens[0]).name.lower()
  if exe.endswith(".exe") or exe.endswith(".cmd"):
    exe = exe[:-4]

  # npx / bunx / pnpm dlx / yarn dlx
  if exe in ("npx", "bunx", "pnpm", "yarn"):
    i = 1
    if exe in ("pnpm", "yarn") and i < len(tokens) and tokens[i] in ("dlx", "exec"):
      i += 1
    pkg_from_flag: str | None = None
    while i < len(tokens):
      tok = tokens[i]
      if tok == "--":
        i += 1
        break
      if tok in _NPX_VALUE_FLAGS:
        if tok in ("-p", "--package") and i + 1 < len(tokens):
          pkg_from_flag = tokens[i + 1]
        i += 2
        continue
      if tok.startswith("--package="):
        pkg_from_flag = tok.split("=", 1)[1]
        i += 1
        continue
      if tok.startswith("-"):
        i += 1
        continue
      return _strip_pkg_version(tok)
    if pkg_from_flag:
      return _strip_pkg_version(pkg_from_flag)
    if i < len(tokens):
      return _strip_pkg_version(tokens[i])
    return exe

  # uvx / uv / pipx
  if exe in ("uvx", "uv", "pipx"):
    i = 1
    while i < len(tokens) and tokens[i] in ("tool", "run"):
      i += 1
    from_pkg: str | None = None
    while i < len(tokens):
      tok = tokens[i]
      if tok == "--":
        i += 1
        break
      if tok in _UV_VALUE_FLAGS:
        if tok == "--from" and i + 1 < len(tokens):
          from_pkg = tokens[i + 1]
        i += 2
        continue
      if tok.startswith("--from="):
        from_pkg = tok.split("=", 1)[1]
        i += 1
        continue
      if tok.startswith("-"):
        i += 1
        continue
      return _strip_pkg_version(tok)
    if from_pkg:
      return _strip_pkg_version(from_pkg)
    return exe

  # python / python3 / py
  if exe == "py" or exe.startswith("python"):
    i = 1
    while i < len(tokens):
      tok = tokens[i]
      if tok == "-m" and i + 1 < len(tokens):
        return tokens[i + 1]
      if tok == "-c":
        return f"{Path(tokens[0]).name} -c"
      if tok in ("-W", "-X") and i + 1 < len(tokens):
        i += 2
        continue
      if tok.startswith("-"):
        i += 1
        continue
      return _script_label(tok)
    return Path(tokens[0]).name

  # node / deno / bun
  if exe in ("node", "nodejs", "deno", "bun"):
    i = 1
    if i < len(tokens) and tokens[i] == "run":
      i += 1
    while i < len(tokens):
      tok = tokens[i]
      if tok in ("-e", "--eval"):
        return f"{exe} -e"
      if tok.startswith("-"):
        i += 1
        continue
      return _script_label(tok)
    return exe

  # docker / podman
  if exe in ("docker", "podman"):
    if len(tokens) >= 2 and tokens[1] == "run":
      i = 2
      while i < len(tokens):
        tok = tokens[i]
        if tok in _DOCKER_VALUE_FLAGS:
          i += 2
          continue
        if tok.startswith("-"):
          i += 1
          continue
        image = tok
        if ":" in image and not image.rsplit(":", 1)[-1].isdigit():
          base, tag = image.rsplit(":", 1)
          if "/" not in tag and tag == "latest":
            image = base
        return image
    return exe

  return _strip_pkg_version(Path(tokens[0]).name or tokens[0])


def _empty_bucket(updated_at: str = "") -> dict[str, Any]:
  bucket: dict[str, Any] = {k: 0 for k in COUNTER_KEYS}
  bucket["updated_at"] = updated_at
  return bucket


def _empty_ledger(updated_at: str = "") -> dict[str, Any]:
  data = _empty_bucket(updated_at=updated_at)
  data["servers"] = {}
  return data


def _safe_int(val: Any) -> int:
  if isinstance(val, bool):
    return 0
  if isinstance(val, (int, float)):
    return max(0, int(val))
  if isinstance(val, str):
    try:
      return max(0, int(val.strip()))
    except ValueError:
      return 0
  return 0


def _normalize_bucket(raw: Any) -> dict[str, Any]:
  if not isinstance(raw, dict):
    return _empty_bucket()
  out: dict[str, Any] = {k: _safe_int(raw.get(k, 0)) for k in COUNTER_KEYS}
  updated = raw.get("updated_at")
  out["updated_at"] = updated if isinstance(updated, str) else ""
  return out


def _normalize_ledger(raw: Any) -> dict[str, Any]:
  if not isinstance(raw, dict):
    return _empty_ledger()
  out = _normalize_bucket(raw)
  raw_servers = raw.get("servers")
  servers: dict[str, dict[str, Any]] = {}
  if isinstance(raw_servers, dict):
    for label, srv_bucket in raw_servers.items():
      if isinstance(label, str) and label.strip() and isinstance(srv_bucket, dict):
        servers[label] = _normalize_bucket(srv_bucket)
  out["servers"] = servers
  return out


def format_wire_bytes(num_bytes: int | float) -> str:
  """Formats wire byte counts into crisp KiB / MiB units."""
  val = max(0.0, float(num_bytes))
  if val >= 1024.0 * 1024.0:
    return f"{val / (1024.0 * 1024.0):.2f} MiB"
  return f"{val / 1024.0:.2f} KiB"


def _format_usd(amount: float) -> str:
  if amount <= 0.0:
    return "$0.0000"
  if amount >= 1.0:
    return f"${amount:,.2f}"
  if amount >= 0.01:
    return f"${amount:.3f}"
  return f"${amount:.4f}"


class StatsLedger:
  """Persistent atomic JSON ledger for cumulative MCP token, byte, and cost savings."""

  def __init__(
      self,
      path: Path | str | None = None,
      enabled: bool | None = None,
  ) -> None:
    self.path = (
        Path(path).expanduser() if path is not None else default_stats_path()
    )
    if enabled is False or is_stats_disabled_by_env():
      self.enabled = False
    else:
      self.enabled = True

  def load(self) -> dict[str, Any]:
    """Loads the cumulative ledger from disk, falling back to an empty ledger on missing/corrupt JSON."""
    try:
      if not self.path.exists():
        return _empty_ledger()
      raw_text = self.path.read_text(encoding="utf-8")
      parsed = json.loads(raw_text)
      return _normalize_ledger(parsed)
    except Exception:
      return _empty_ledger()

  def _atomic_write(self, data: dict[str, Any]) -> bool:
    """Atomically writes ledger JSON via a sibling temp file + `os.replace`."""
    tmp_path: str | None = None
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
      fd, tmp_path = tempfile.mkstemp(
          prefix=".stats.",
          suffix=".tmp",
          dir=str(self.path.parent),
      )
      try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
          fh.write(payload)
          fh.flush()
          try:
            os.fsync(fh.fileno())
          except OSError:
            pass
      except Exception:
        try:
          os.close(fd)
        except OSError:
          pass
        raise
      os.replace(tmp_path, self.path)
      tmp_path = None
      return True
    except Exception:
      if tmp_path is not None:
        try:
          os.unlink(tmp_path)
        except OSError:
          pass
      return False

  def reset(self) -> dict[str, Any]:
    """Resets cumulative statistics to zero atomically."""
    fresh = _empty_ledger(updated_at=_utc_now_iso())
    self._atomic_write(fresh)
    return fresh

  def record_session(
      self,
      stats: SlimStats | None = None,
      *,
      command: Sequence[str] | str | None = None,
      server_label: str | None = None,
      secrets_redacted: int = 0,
      schema_tokens_before: int = 0,
      schema_tokens_after: int = 0,
      output_tokens_before: int = 0,
      output_tokens_after: int = 0,
      wire_bytes_before: int = 0,
      wire_bytes_after: int = 0,
      refs_inlined: int = 0,
      coerced_arguments: int = 0,
      cache_hits: int = 0,
      blocked_calls: int = 0,
  ) -> dict[str, Any]:
    """Records one proxy session into the cumulative ledger without ever raising."""
    if not self.enabled or is_stats_disabled_by_env():
      return self.load()

    try:
      delta = {
          "sessions": 1,
          "schema_tokens_before": (
              (stats.schema_tokens_before if stats is not None else 0)
              + schema_tokens_before
          ),
          "schema_tokens_after": (
              (stats.schema_tokens_after if stats is not None else 0)
              + schema_tokens_after
          ),
          "output_tokens_before": (
              (stats.output_tokens_before if stats is not None else 0)
              + output_tokens_before
          ),
          "output_tokens_after": (
              (stats.output_tokens_after if stats is not None else 0)
              + output_tokens_after
          ),
          "wire_bytes_before": (
              (stats.wire_bytes_before if stats is not None else 0)
              + wire_bytes_before
          ),
          "wire_bytes_after": (
              (stats.wire_bytes_after if stats is not None else 0)
              + wire_bytes_after
          ),
          "refs_inlined": (
              (stats.refs_inlined if stats is not None else 0) + refs_inlined
          ),
          "coerced_arguments": (
              (stats.args_coerced if stats is not None else 0)
              + coerced_arguments
          ),
          "cache_hits": (
              (stats.cache_hits if stats is not None else 0) + cache_hits
          ),
          "blocked_calls": (
              (stats.blocked_calls if stats is not None else 0) + blocked_calls
          ),
          "secrets_redacted": max(0, int(secrets_redacted)),
      }

      label = (
          server_label.strip()
          if isinstance(server_label, str) and server_label.strip()
          else derive_server_label(command)
      )
      now_iso = _utc_now_iso()
      ledger = self.load()
      for k in COUNTER_KEYS:
        ledger[k] = _safe_int(ledger.get(k, 0)) + max(0, int(delta[k]))
      ledger["updated_at"] = now_iso

      servers = ledger.setdefault("servers", {})
      srv_entry = _normalize_bucket(servers.get(label))
      for k in COUNTER_KEYS:
        srv_entry[k] = _safe_int(srv_entry.get(k, 0)) + max(0, int(delta[k]))
      srv_entry["updated_at"] = now_iso
      servers[label] = srv_entry

      self._atomic_write(ledger)
      return ledger
    except Exception:
      return _empty_ledger()

  def summary(self) -> dict[str, Any]:
    """Returns ledger data enriched with derived savings and model input-cost estimates."""
    data = self.load()
    tokens_before = data["schema_tokens_before"] + data["output_tokens_before"]
    tokens_after = data["schema_tokens_after"] + data["output_tokens_after"]
    tokens_saved = max(0, tokens_before - tokens_after)
    wire_saved = max(0, data["wire_bytes_before"] - data["wire_bytes_after"])
    reduction_pct = (
        round((tokens_saved / tokens_before) * 100.0, 1)
        if tokens_before > 0
        else 0.0
    )

    cost_saved = {
        slug: round((tokens_saved / 1_000_000.0) * rate, 6)
        for slug, _, rate in MODEL_INPUT_RATES
    }
    sessions = data["sessions"]
    projected_30d_tokens = (
        int(round((tokens_saved / sessions) * 100 * 30))
        if sessions > 0 and tokens_saved > 0
        else 0
    )
    projected_30d_usd = {
        slug: round((projected_30d_tokens / 1_000_000.0) * rate, 4)
        for slug, _, rate in MODEL_INPUT_RATES
    }

    return {
        **data,
        "stats_file": str(self.path),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "wire_bytes_saved": wire_saved,
        "reduction_pct": reduction_pct,
        "estimated_input_cost_saved_usd": cost_saved,
        "projected_30d_100_turns_per_day_usd": projected_30d_usd,
    }

  def format_dashboard(self) -> str:
    """Renders a crisp ASCII terminal dashboard for `mcp-slim stats`."""
    info = self.summary()
    sessions = info["sessions"]
    sch_before = info["schema_tokens_before"]
    sch_after = info["schema_tokens_after"]
    sch_saved = max(0, sch_before - sch_after)
    sch_pct = round((sch_saved / sch_before) * 100.0, 1) if sch_before > 0 else 0.0

    out_before = info["output_tokens_before"]
    out_after = info["output_tokens_after"]
    out_saved = max(0, out_before - out_after)
    out_pct = round((out_saved / out_before) * 100.0, 1) if out_before > 0 else 0.0

    tok_before = info["tokens_before"]
    tok_after = info["tokens_after"]
    tok_saved = info["tokens_saved"]
    tok_pct = info["reduction_pct"]

    wb_before = info["wire_bytes_before"]
    wb_after = info["wire_bytes_after"]
    wb_saved = info["wire_bytes_saved"]
    wb_pct = round((wb_saved / wb_before) * 100.0, 1) if wb_before > 0 else 0.0

    updated_label = info["updated_at"] or "never"
    show_projection = 1 <= sessions <= 2 and tok_saved > 0

    lines: list[str] = [
        "mcp-slim Cumulative Token & Cost Savings Dashboard",
        "=" * 78,
        f"Ledger File : {self.path}",
        f"Sessions    : {sessions:,}   |   Last Updated : {updated_label}",
        "-" * 78,
        f"{'TRAFFIC STAGE':<28} | {'BEFORE':>12} | {'AFTER':>12} | {'SAVED':>18}",
        "-" * 78,
        (
            f"{'tools/list Schemas':<28} | {sch_before:>12,} | {sch_after:>12,} |"
            f" {f'{sch_saved:,} (-{sch_pct}%)':>18}"
        ),
        (
            f"{'tools/call Outputs':<28} | {out_before:>12,} | {out_after:>12,} |"
            f" {f'{out_saved:,} (-{out_pct}%)':>18}"
        ),
        (
            f"{'Total Context Tokens':<28} | {tok_before:>12,} | {tok_after:>12,} |"
            f" {f'{tok_saved:,} (-{tok_pct}%)':>18}"
        ),
        (
            f"{'JSON-RPC Wire Payload':<28} | {format_wire_bytes(wb_before):>12} |"
            f" {format_wire_bytes(wb_after):>12} |"
            f" {f'{format_wire_bytes(wb_saved)} (-{wb_pct}%)':>18}"
        ),
        "=" * 78,
        "Estimated API Input Cost Saved (Input-Rate Estimates)",
    ]
    if show_projection:
      lines.append(
          f"{'MODEL (INPUT RATE)':<36} | {'CUMULATIVE SAVED':>17} |"
          f" {'30D @ 100 TURNS/DAY':>19}"
      )
      lines.append("-" * 78)
      avg_saved_per_session = tok_saved / max(1, sessions)
      proj_tokens = avg_saved_per_session * 100.0 * 30.0
      for _, display_name, rate in MODEL_INPUT_RATES:
        cum_usd = (tok_saved / 1_000_000.0) * rate
        proj_usd = (proj_tokens / 1_000_000.0) * rate
        model_col = f"{display_name} (${rate:.2f}/1M)"
        lines.append(
            f"{model_col:<36} | {_format_usd(cum_usd):>17} |"
            f" {f'~${proj_usd:,.2f}/mo':>19}"
        )
    else:
      lines.append(f"{'MODEL (INPUT RATE)':<46} | {'ESTIMATED SAVED':>29}")
      lines.append("-" * 78)
      for _, display_name, rate in MODEL_INPUT_RATES:
        cum_usd = (tok_saved / 1_000_000.0) * rate
        model_col = f"{display_name} (${rate:.2f}/1M input tokens)"
        lines.append(f"{model_col:<46} | {_format_usd(cum_usd):>29}")

    lines.extend([
        "=" * 78,
        "Reliability & Security Shield Counters",
        "-" * 78,
        (
            f"  $defs/$ref Pointers Inlined : {info['refs_inlined']:<10,}"
            f"  Read-Cache Hits     : {info['cache_hits']:,}"
        ),
        (
            f"  Stringified Args Coerced    : {info['coerced_arguments']:<10,}"
            f"  Secrets Redacted    : {info['secrets_redacted']:,}"
        ),
        f"  Policy-Blocked Tool Calls   : {info['blocked_calls']:,}",
        "=" * 78,
        (
            f"{'SERVER':<30} | {'SESSIONS':>8} | {'RAW TOK':>9} |"
            f" {'SLIM TOK':>9} | {'SAVED':>8} | {'SECRETS':>7}"
        ),
        "-" * 78,
    ])

    servers: dict[str, dict[str, Any]] = info.get("servers", {})
    if not servers:
      lines.append(
          "  (No proxy sessions recorded yet — run `mcp-slim run -- <server>`)"
      )
    else:
      sorted_servers = sorted(
          servers.items(),
          key=lambda kv: (
              -(
                  (kv[1]["schema_tokens_before"] + kv[1]["output_tokens_before"])
                  - (kv[1]["schema_tokens_after"] + kv[1]["output_tokens_after"])
              ),
              kv[0],
          ),
      )
      for srv_name, srv in sorted_servers:
        s_raw = srv["schema_tokens_before"] + srv["output_tokens_before"]
        s_slim = srv["schema_tokens_after"] + srv["output_tokens_after"]
        s_saved = max(0, s_raw - s_slim)
        s_pct = round((s_saved / s_raw) * 100.0, 1) if s_raw > 0 else 0.0
        lines.append(
            f"{srv_name:<30.30} | {srv['sessions']:>8,} | {s_raw:>9,} |"
            f" {s_slim:>9,} | {f'-{s_pct}%':>8} | {srv['secrets_redacted']:>7,}"
        )

    lines.append("=" * 78)
    return "\n".join(lines)
