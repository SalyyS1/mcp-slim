"""Bidirectional secret redaction, indirect prompt-injection shield, and opaque handle vault for MCP traffic."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import re
import secrets
from typing import Any

# High-precision patterns covering modern LLM, cloud, and developer secrets plus
# indirect prompt-injection directives and Markdown exfiltration beacons.
# Every rule includes C-level substring triggers (checked against original text or
# `text.lower()` when `case_insensitive=True`) so CPython's `sre` engine skips
# clean payloads in sub-millisecond time.
_SECRET_RULES: tuple[tuple[str, tuple[str, ...], bool, re.Pattern[str], str], ...] = (
    (
        "anthropic_api_key",
        ("sk-ant-",),
        False,
        re.compile(r"sk-ant-(?:api03|admin01)-[A-Za-z0-9_-]{20,}"),
        "[REDACTED:ANTHROPIC_KEY]",
    ),
    (
        "openai_api_key",
        ("sk-",),
        False,
        re.compile(
            r"sk-(?:(?:proj-|svcacct-|None-)?[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{10,}|proj-[A-Za-z0-9_-]{32,})"
        ),
        "[REDACTED:OPENAI_KEY]",
    ),
    (
        "github_token",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        False,
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        "[REDACTED:GITHUB_TOKEN]",
    ),
    (
        "github_token",
        ("github_pat_",),
        False,
        re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
        "[REDACTED:GITHUB_TOKEN]",
    ),
    (
        "aws_access_key",
        ("AKIA", "ASIA", "ABIA", "ACCA"),
        False,
        re.compile(r"A(?:KIA|SIA|BIA|CCA)[0-9A-Z]{16}\b"),
        "[REDACTED:AWS_ACCESS_KEY]",
    ),
    (
        "google_api_key",
        ("AIza",),
        False,
        re.compile(r"AIza[0-9A-Za-z_-]{35}\b"),
        "[REDACTED:GCP_API_KEY]",
    ),
    (
        "slack_token",
        ("xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"),
        False,
        re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24,}"),
        "[REDACTED:SLACK_TOKEN]",
    ),
    (
        "stripe_key",
        ("sk_live_", "sk_test_", "rk_live_", "rk_test_"),
        False,
        re.compile(r"[sr]k_(?:live|test)_[0-9a-zA-Z]{24,}"),
        "[REDACTED:STRIPE_KEY]",
    ),
    (
        "huggingface_token",
        ("hf_",),
        False,
        re.compile(r"hf_[A-Za-z0-9]{30,}"),
        "[REDACTED:HF_TOKEN]",
    ),
    (
        "private_key_block",
        ("PRIVATE KEY-----",),
        False,
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
        "[REDACTED:PRIVATE_KEY_BLOCK]",
    ),
    (
        "jwt_token",
        ("eyJ",),
        False,
        re.compile(
            r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{16,}"
        ),
        "[REDACTED:JWT_TOKEN]",
    ),
    (
        "url_userinfo_secret",
        ("://",),
        False,
        re.compile(r"(://[^/\s:@\"']+:(?!\[(?:REDACTED|SECRET_REF):))([^/\s\"']+)(@)"),
        r"\1[REDACTED:URL_SECRET]\3",
    ),
    (
        "url_query_secret",
        (
            "?token=",
            "&token=",
            "?private_token=",
            "&private_token=",
            "?oauth_token=",
            "&oauth_token=",
            "?access_token=",
            "&access_token=",
            "?auth_token=",
            "&auth_token=",
            "?api_key=",
            "&api_key=",
            "?apikey=",
            "&apikey=",
            "?sig=",
            "&sig=",
            "?signature=",
            "&signature=",
            "x-amz-signature=",
            "x-amz-security-token=",
        ),
        True,
        re.compile(
            r"(?i)([?&](?:token|private_token|oauth_token|access_token|auth_token|api_key|apikey|sig|signature|x-amz-signature|x-amz-security-token)=(?!\[(?:REDACTED|SECRET_REF):))([^&\s\"'#]{6,})"
        ),
        r"\1[REDACTED:URL_QUERY_SECRET]",
    ),
    (
        "env_assignment_secret",
        (
            "api_key",
            "secret_key",
            "access_token",
            "private_key",
            "auth_token",
            "client_secret",
            "db_password",
            "database_password",
        ),
        True,
        re.compile(
            r"(?i)((?<!\[SECRET_REF:)(?<!\[REDACTED:)\b[A-Za-z0-9_]*(?:API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|CLIENT_SECRET|DB_PASSWORD|DATABASE_PASSWORD)\s*[:=]\s*['\"]?(?!\[(?:REDACTED|SECRET_REF):))([^\s'\",;}{\]]{8,})(['\"]?)"
        ),
        r"\1[REDACTED:ENV_SECRET]\3",
    ),
    (
        "markdown_exfil_beacon",
        ("![",),
        False,
        re.compile(
            r"!\[[^\]]*\]\(https?://[^)\s]+?(?:[?&](?:exfil|leak|steal|secret|token|api_key|apikey|password|passwd|env|credentials|prompt|context|chat_history)=[^)\s]*|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\{\{[^{}]+\}\})[^)\s]*\)",
            re.IGNORECASE,
        ),
        "[REDACTED:EXFIL_BEACON]",
    ),
    (
        "prompt_injection",
        (
            "<important",
            "<system",
            "<instruction",
            "<tool_call",
            "<function_call",
            "<|im_start|>",
            "<|system|>",
            "<|start_header_id|>",
            "<<sys>>",
            "[inst]",
            "previous instructions",
            "prior instructions",
            "above instructions",
            "previous prompts",
            "prior prompts",
            "previous rules",
            "prior rules",
            "system instructions",
            "system prompt",
            "developer mode",
        ),
        True,
        re.compile(
            r"(?is)(?:"
            r"<(?:IMPORTANT|SYSTEM|SYSTEM-REMINDER|INSTRUCTION|TOOL_CALL|FUNCTION_CALL)\b[^>]*>[\s\S]*?</(?:IMPORTANT|SYSTEM|SYSTEM-REMINDER|INSTRUCTION|TOOL_CALL|FUNCTION_CALL)>"
            r"|<(?:IMPORTANT|SYSTEM|SYSTEM-REMINDER|INSTRUCTION|TOOL_CALL|FUNCTION_CALL)\b[^>]*>[\s\S]*$"
            r"|<\|im_start\|>\s*(?:system|developer)\b[\s\S]*?(?:<\|im_end\|>|$)"
            r"|<\|system\|>[\s\S]*?(?:<\|end\|>|<\|endoftext\|>|$)"
            r"|<\|start_header_id\|>\s*system\s*<\|end_header_id\|>[\s\S]*?(?:<\|eot_id\|>|$)"
            r"|\[INST\]\s*<<SYS>>[\s\S]*?(?:\[/INST\]|<</SYS>>|$)"
            r"|<<SYS>>[\s\S]*?(?:<</SYS>>|$)"
            r"|(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|directives|rules)[^\n.\"]*"
            r"|(?:new|override|updated)\s+system\s+(?:instructions|prompt)\s*:[^\n\"]*"
            r"|you\s+are\s+now\s+in\s+developer\s+mode[^\n.\"]*"
            r")"
        ),
        "[REDACTED:PROMPT_INJECTION]",
    ),
)

# Trojan Source BiDi marks/overrides/isolates, zero-width spaces, word joiners,
# and Plane-14 invisible tag characters.
_UNICODE_SMUGGLING_RE = re.compile(
    r"[\u061c\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\U000e0000-\U000e007f]+"
)

# Zero-width joiners/non-joiners/BOM sandwiched between ASCII word/tag delimiters
# (defeats `<IMP\u200dORTANT>` and `ig\u200cnore` keyword splitting while preserving emoji ZWJ).
_ASCII_ZW_SPLIT_RE = re.compile(
    r"(?<=[A-Za-z0-9<|/_-])[\u200c\u200d\ufeff]+(?=[A-Za-z0-9>|/_-])"
)

_NON_VAULT_CATEGORIES = frozenset({
    "prompt_injection",
    "markdown_exfil_beacon",
})

_FRAMED_SECRET_CATEGORIES = frozenset({
    "url_userinfo_secret",
    "url_query_secret",
    "env_assignment_secret",
})

_BINARY_MIME_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "font/",
    "application/octet-stream",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/wasm",
)

_MAX_INSPECT_BLOB_CHARS = 2_800_000

_SECRET_HANDLE_PATTERN = re.compile(r"\[SECRET_REF:[a-z0-9_]+:[0-9a-f]{10}\]")

_STREAM_OPEN_XML_TAG_RE = re.compile(
    r"(?i)<(important|system-reminder|system|instruction|tool_call|function_call)\b[^>]*>"
)
_STREAM_CHATML_OPEN_RE = re.compile(r"(?i)<\|im_start\|>\s*(?:system|developer)\b")
_STREAM_SYS_PIPE_OPEN_RE = re.compile(r"(?i)<\|system\|>")
_STREAM_LLAMA3_SYS_OPEN_RE = re.compile(
    r"(?i)<\|start_header_id\|>\s*system\s*<\|end_header_id\|>"
)
_STREAM_LLAMA2_SYS_OPEN_RE = re.compile(r"(?i)<<sys>>")
_STREAM_PEM_END_RE = re.compile(
    r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)
_STREAM_LINE_DIRECTIVE_DOT_RE = re.compile(
    r"(?i)(?:(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|system)\s+"
    r"(?:instructions|prompts|directives|rules)|you\s+are\s+now\s+in\s+developer\s+mode)"
)
_STREAM_LINE_DIRECTIVE_COLON_RE = re.compile(
    r"(?i)(?:new|override|updated)\s+system\s+(?:instructions|prompt)\s*:"
)
_STREAM_ENV_PREFIX_RE = re.compile(
    r"(?i)(?:API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|CLIENT_SECRET|"
    r"DB_PASSWORD|DATABASE_PASSWORD)\s*[:=]\s*['\"]?$"
)


@dataclass
class RedactionReport:
  """Tracks how many secrets and injection payloads were intercepted and redacted."""

  total_redactions: int = 0
  handles_resolved: int = 0
  by_category: dict[str, int] = field(default_factory=dict)

  def record(self, category: str, count: int = 1) -> None:
    if count <= 0:
      return
    self.total_redactions += count
    self.by_category[category] = self.by_category.get(category, 0) + count


class SecretGuard:
  """Scans, redacts, and optionally tokenizes secrets into opaque session handles."""

  def __init__(self, enabled: bool = True, vault_handles: bool = False) -> None:
    self.enabled = enabled
    self.vault_handles = vault_handles
    self.report = RedactionReport()
    self._vault_key = secrets.token_bytes(16)
    self._handle_to_secret: dict[str, str] = {}

  def _mint_handle(self, category: str, raw_secret: str) -> str:
    digest = hmac.new(
        self._vault_key, raw_secret.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:10]
    handle = f"[SECRET_REF:{category}:{digest}]"
    self._handle_to_secret[handle] = raw_secret
    return handle

  def redact_text(self, text: str, *, force_redact: bool = False) -> str:
    if not self.enabled or not text:
      return text
    current = text
    if not current.isascii():
      current, smug_count = _UNICODE_SMUGGLING_RE.subn("", current)
      current, zw_count = _ASCII_ZW_SPLIT_RE.subn("", current)
      total_smug = smug_count + zw_count
      if total_smug:
        self.report.record("unicode_smuggling", total_smug)
    if force_redact and "[SECRET_REF:" in current:
      current, handle_count = _SECRET_HANDLE_PATTERN.subn(
          "[REDACTED:SECRET_HANDLE]", current
      )
      if handle_count:
        self.report.record("secret_handle", handle_count)
    use_vault = self.vault_handles and not force_redact
    current_lower = current.lower()
    for name, triggers, case_insensitive, pattern, replacement in _SECRET_RULES:
      haystack = current_lower if case_insensitive else current
      if not any(t in haystack for t in triggers):
        continue
      if use_vault and name not in _NON_VAULT_CATEGORIES:
        if name in _FRAMED_SECRET_CATEGORIES:

          def _replace_framed(match: re.Match[str], cat: str = name) -> str:
            secret_val = match.group(2)
            if secret_val.startswith("[SECRET_REF:") or secret_val.startswith("[REDACTED:"):
              return match.group(0)
            prefix = match.group(1)
            suffix = match.group(3) if (match.lastindex or 0) >= 3 else ""
            return f"{prefix}{self._mint_handle(cat, secret_val)}{suffix}"

          updated, count = pattern.subn(_replace_framed, current)
        else:

          def _replace_direct(match: re.Match[str], cat: str = name) -> str:
            return self._mint_handle(cat, match.group(0))

          updated, count = pattern.subn(_replace_direct, current)
        if updated != current and count:
          current = updated
          current_lower = current.lower()
          self.report.record(name, count)
      else:
        updated, count = pattern.subn(replacement, current)
        if updated != current and count:
          current = updated
          current_lower = current.lower()
          self.report.record(name, count)
    return current

  def redact_base64_blob(
      self,
      blob_b64: str,
      *,
      mime_type: str | None = None,
      force_redact: bool = False,
  ) -> str:
    """Decodes textual UTF-8 base64 MCP resource blobs, scrubs secrets/injections, and re-encodes."""
    if not self.enabled or not blob_b64 or len(blob_b64) > _MAX_INSPECT_BLOB_CHARS:
      return blob_b64
    if isinstance(mime_type, str) and mime_type:
      norm_mime = mime_type.strip().lower()
      if norm_mime != "image/svg+xml" and norm_mime.startswith(_BINARY_MIME_PREFIXES):
        return blob_b64
    compact_b64 = "".join(blob_b64.split())
    if not compact_b64:
      return blob_b64
    try:
      raw_bytes = base64.b64decode(compact_b64, validate=True)
    except (binascii.Error, ValueError):
      return blob_b64
    if b"\x00" in raw_bytes:
      return blob_b64
    try:
      decoded_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
      return blob_b64
    sanitized_text = self.redact_text(decoded_text, force_redact=force_redact)
    if sanitized_text == decoded_text:
      return blob_b64
    return base64.b64encode(sanitized_text.encode("utf-8")).decode("ascii")

  def redact_value(self, value: Any, *, force_redact: bool = False) -> Any:
    if not self.enabled:
      return value
    if isinstance(value, str):
      return self.redact_text(value, force_redact=force_redact)
    if isinstance(value, list):
      return [self.redact_value(item, force_redact=force_redact) for item in value]
    if isinstance(value, dict):
      is_resource_blob = isinstance(value.get("blob"), str) and (
          isinstance(value.get("uri"), str) or isinstance(value.get("mimeType"), str)
      )
      mime_hint = value.get("mimeType") if isinstance(value.get("mimeType"), str) else None
      out: dict[str, Any] = {}
      for k, v in value.items():
        if is_resource_blob and k == "blob" and isinstance(v, str):
          out[k] = self.redact_base64_blob(
              v, mime_type=mime_hint, force_redact=force_redact
          )
        else:
          out[k] = self.redact_value(v, force_redact=force_redact)
      return out
    return value

  def resolve_handles_in_text(self, text: str) -> str:
    """Replaces session-scoped `[SECRET_REF:...]` handles with raw secrets on outbound calls."""
    if not self.enabled or not self.vault_handles or "[SECRET_REF:" not in text:
      return text

    def _sub_handle(match: re.Match[str]) -> str:
      handle = match.group(0)
      raw = self._handle_to_secret.get(handle)
      if raw is not None:
        self.report.handles_resolved += 1
        return raw
      return handle

    return _SECRET_HANDLE_PATTERN.sub(_sub_handle, text)

  def resolve_handles_in_value(self, value: Any) -> Any:
    """Recursively resolves opaque `[SECRET_REF:...]` handles across structured tool arguments."""
    if not self.enabled or not self.vault_handles:
      return value
    if isinstance(value, str):
      return self.resolve_handles_in_text(value)
    if isinstance(value, list):
      return [self.resolve_handles_in_value(item) for item in value]
    if isinstance(value, dict):
      return {k: self.resolve_handles_in_value(v) for k, v in value.items()}
    return value


class StreamSanitizer:
  """Stateful O(N) chunk-boundary sanitizer for streaming MCP tool calls and notifications.

  Holds back trailing open tokens, incomplete `[SECRET_REF:...]` handles, unclosed PEM
  blocks, and multi-chunk prompt-injection tags up to `max_buffer_chars` so secrets or
  injections split across arbitrary byte/chunk boundaries never leak.
  """

  def __init__(
      self,
      guard: SecretGuard | None = None,
      *,
      overlap_chars: int = 160,
      max_buffer_chars: int = 8192,
      force_redact: bool = False,
      resolve_handles: bool = False,
  ) -> None:
    self.guard = guard if guard is not None else SecretGuard()
    self.overlap_chars = max(32, int(overlap_chars))
    self.max_buffer_chars = max(self.overlap_chars * 2, int(max_buffer_chars))
    self.force_redact = force_redact
    self.resolve_handles = resolve_handles
    self._carry: str = ""
    self._overflow_close_re: re.Pattern[str] | None = None
    self.chunks_processed: int = 0

  @property
  def buffered_chars(self) -> int:
    return len(self._carry)

  def reset(self) -> None:
    self._carry = ""
    self._overflow_close_re = None

  def _post_process(self, text: str) -> str:
    if not text:
      return ""
    out = self.guard.redact_text(text, force_redact=self.force_redact)
    if self.resolve_handles and not self.force_redact:
      out = self.guard.resolve_handles_in_text(out)
    return out

  @staticmethod
  def _find_open_construct(
      text: str,
  ) -> tuple[int | None, re.Pattern[str] | None]:
    """Finds the earliest unclosed multi-token secret, handle, or injection construct."""
    earliest: int | None = None
    close_re: re.Pattern[str] | None = None

    def _consider(pos: int, closer: re.Pattern[str] | None = None) -> None:
      nonlocal earliest, close_re
      if earliest is None or pos < earliest:
        earliest = pos
        close_re = closer

    # 1. XML-style prompt-injection blocks (<SYSTEM>, <IMPORTANT>, <TOOL_CALL>, ...)
    for m in _STREAM_OPEN_XML_TAG_RE.finditer(text):
      tag_name = m.group(1)
      closer = re.compile(rf"(?i)</{re.escape(tag_name)}\s*>")
      if closer.search(text, m.end()) is None:
        _consider(m.start(), closer)
        break

    # 2. ChatML / Llama / pipe system role headers
    for open_re, closer in (
        (_STREAM_CHATML_OPEN_RE, re.compile(r"(?i)<\|im_end\|>")),
        (_STREAM_SYS_PIPE_OPEN_RE, re.compile(r"(?i)<\|(?:end|endoftext)\|>")),
        (_STREAM_LLAMA3_SYS_OPEN_RE, re.compile(r"(?i)<\|eot_id\|>")),
        (_STREAM_LLAMA2_SYS_OPEN_RE, re.compile(r"(?i)<</sys>>")),
    ):
      for m in open_re.finditer(text):
        if closer.search(text, m.end()) is None:
          _consider(m.start(), closer)
          break

    # 3. Incomplete opening angle-bracket tag near end of buffer (e.g. `<SYST` or `<|im_`)
    last_lt = text.rfind("<")
    if last_lt != -1 and ">" not in text[last_lt:] and "\n" not in text[last_lt:]:
      _consider(last_lt, None)

    # 4. Multi-line PEM private key block
    last_begin = text.rfind("-----BEGIN ")
    if last_begin != -1 and _STREAM_PEM_END_RE.search(text, last_begin + 11) is None:
      _consider(last_begin, None)

    # 5. Single-line prompt-injection directives not yet terminated by newline/period/quote
    last_nl = text.rfind("\n")
    line_start = last_nl + 1
    last_line = text[line_start:]
    if last_line:
      norm_line = _ASCII_ZW_SPLIT_RE.sub("", _UNICODE_SMUGGLING_RE.sub("", last_line))
      for m in _STREAM_LINE_DIRECTIVE_DOT_RE.finditer(norm_line):
        tail = norm_line[m.end() :]
        if "\n" not in tail and "." not in tail and '"' not in tail:
          _consider(line_start, re.compile(r'[\n."]'))
          break
      for m in _STREAM_LINE_DIRECTIVE_COLON_RE.finditer(norm_line):
        tail = norm_line[m.end() :]
        if "\n" not in tail and '"' not in tail:
          _consider(line_start, re.compile(r'[\n"]'))
          break

    # 6. Unclosed Markdown image beacon `![alt](https://...` on current line
    last_img = text.rfind("![")
    if last_img != -1 and ")" not in text[last_img:] and "\n" not in text[last_img:]:
      _consider(last_img, None)

    # 7. Unclosed `[SECRET_REF:...]` or `[INST]` bracket token on current line
    last_brk = text.rfind("[")
    if last_brk != -1 and "]" not in text[last_brk:] and "\n" not in text[last_brk:]:
      _consider(last_brk, None)

    return earliest, close_re

  @staticmethod
  def _snap_to_safe_boundary(text: str, limit: int, overlap_chars: int) -> int:
    """Snaps `limit` backwards to a clean newline (or token delimiter on long lines)."""
    if limit <= 0:
      return 0
    nl_pos = text.rfind("\n", 0, limit)
    if nl_pos != -1:
      return nl_pos + 1
    if len(text) <= max(512, overlap_chars * 3):
      return 0

    idx = limit
    while idx > 0:
      ch = text[idx - 1]
      if ch in (" ", "\t", "\n", "\r", ","):
        prefix = text[max(0, idx - 40) : idx]
        if _STREAM_ENV_PREFIX_RE.search(prefix) is not None:
          idx -= 1
          continue
        return idx
      # Do not split across ASCII + zero-width joiner boundaries
      idx -= 1
    return 0

  def _avoid_splitting_complete_match(self, text: str, cut: int) -> int:
    """If any secret, handle, or injection match straddles `cut`, pulls `cut` back to `m.start()`."""
    if cut <= 0 or cut >= len(text):
      return cut
    adjusted = cut
    text_lower = text.lower()
    for _, triggers, case_insensitive, pattern, _ in _SECRET_RULES:
      haystack = text_lower if case_insensitive else text
      if not any(t in haystack for t in triggers):
        continue
      for m in pattern.finditer(text):
        if m.start() < adjusted < m.end():
          adjusted = m.start()
          break
    if "[SECRET_REF:" in text:
      for m in _SECRET_HANDLE_PATTERN.finditer(text):
        if m.start() < adjusted < m.end():
          adjusted = m.start()
          break
    return adjusted

  def push(self, chunk: str) -> str:
    """Ingests one streaming text chunk and returns the prefix guaranteed safe to emit."""
    if not chunk:
      return ""
    self.chunks_processed += 1
    if not self.guard.enabled:
      return chunk

    combined = self._carry + chunk

    # If currently discarding an oversized unclosed prompt-injection block, look for its closer
    if self._overflow_close_re is not None:
      m_close = self._overflow_close_re.search(combined)
      if m_close is None:
        self._carry = combined[-64:] if len(combined) > 64 else combined
        return ""
      combined = combined[m_close.end() :]
      self._overflow_close_re = None
      self._carry = ""
      if not combined:
        return ""

    open_idx, closer_re = self._find_open_construct(combined)
    if open_idx is not None:
      span_len = len(combined) - open_idx
      if span_len <= self.max_buffer_chars:
        if open_idx == 0:
          self._carry = combined
          return ""
        safe_cut = self._snap_to_safe_boundary(
            combined, min(open_idx, max(0, len(combined) - self.overlap_chars)), self.overlap_chars
        )
        safe_cut = self._avoid_splitting_complete_match(combined, safe_cut)
        if safe_cut <= 0:
          self._carry = combined
          return ""
        emitted = self._post_process(combined[:safe_cut])
        self._carry = combined[safe_cut:]
        return emitted

      # Unclosed injection construct exceeded max_buffer_chars: redact immediately & stay in shield mode
      if closer_re is not None:
        prefix_out = self._post_process(combined[:open_idx]) if open_idx > 0 else ""
        # Record smuggling inside the overflowed injection chunk before entering shield mode
        in_block = combined[open_idx:]
        if not in_block.isascii():
          _, smug_c = _UNICODE_SMUGGLING_RE.subn("", in_block)
          _, zw_c = _ASCII_ZW_SPLIT_RE.subn("", in_block)
          if smug_c + zw_c:
            self.guard.report.record("unicode_smuggling", smug_c + zw_c)
        self.guard.report.record("prompt_injection", 1)
        self._overflow_close_re = closer_re
        self._carry = combined[-64:] if len(combined) > 64 else combined
        return f"{prefix_out}[REDACTED:PROMPT_INJECTION]"

    if len(combined) <= self.overlap_chars:
      self._carry = combined
      return ""

    raw_limit = len(combined) - self.overlap_chars
    safe_cut = self._snap_to_safe_boundary(combined, raw_limit, self.overlap_chars)
    safe_cut = self._avoid_splitting_complete_match(combined, safe_cut)
    if safe_cut <= 0:
      if len(combined) <= self.max_buffer_chars:
        self._carry = combined
        return ""
      safe_cut = raw_limit

    emitted = self._post_process(combined[:safe_cut])
    self._carry = combined[safe_cut:]
    return emitted

  def flush(self) -> str:
    """Flushes any remaining buffered characters at the end of a stream."""
    if self._overflow_close_re is not None:
      if not self._carry.isascii():
        _, smug_c = _UNICODE_SMUGGLING_RE.subn("", self._carry)
        _, zw_c = _ASCII_ZW_SPLIT_RE.subn("", self._carry)
        if smug_c + zw_c:
          self.guard.report.record("unicode_smuggling", smug_c + zw_c)
      self._carry = ""
      self._overflow_close_re = None
      return ""
    rem = self._carry
    self._carry = ""
    if not rem:
      return ""
    if not self.guard.enabled:
      return rem
    return self._post_process(rem)


class ToolCallStreamSanitizer:
  """Sanitizes incremental LLM/MCP streaming tool-call JSON argument deltas (`tools/call`).

  Reassembles fragmented JSON argument chunks per `call_id`, scrubs cross-chunk secrets
  and prompt injections, resolves split `[SECRET_REF:...]` vault handles, and validates
  decoded JSON arguments (including `\\uXXXX`-escaped sequences) upon completion.
  """

  def __init__(
      self,
      guard: SecretGuard | None = None,
      *,
      overlap_chars: int = 160,
      max_call_chars: int = 262_144,
      max_active_calls: int = 256,
  ) -> None:
    self.guard = guard if guard is not None else SecretGuard()
    self.overlap_chars = overlap_chars
    self.max_call_chars = max(1024, int(max_call_chars))
    self.max_active_calls = max(8, int(max_active_calls))
    self._calls: dict[str | int, dict[str, Any]] = {}
    self.deltas_processed: int = 0
    self.calls_completed: int = 0

  @property
  def active_calls(self) -> int:
    return len(self._calls)

  def tool_name_for(self, call_id: str | int) -> str:
    state = self._calls.get(call_id)
    if state is None:
      return ""
    return str(state.get("tool_name") or "")

  def discard_call(self, call_id: str | int) -> None:
    self._calls.pop(call_id, None)

  def start_call(self, call_id: str | int, tool_name: str = "") -> None:
    if call_id not in self._calls:
      while len(self._calls) >= self.max_active_calls:
        oldest_id = next(iter(self._calls))
        self._calls.pop(oldest_id, None)
    self._calls[call_id] = {
        "tool_name": tool_name,
        "sanitizer": StreamSanitizer(
            self.guard,
            overlap_chars=self.overlap_chars,
            resolve_handles=self.guard.vault_handles,
        ),
        "parts": [],
        "raw_chars": 0,
        "overflowed": False,
    }

  def push_delta(
      self,
      call_id: str | int,
      arguments_delta: str,
      *,
      tool_name: str | None = None,
  ) -> str:
    """Feeds an argument JSON delta chunk for `call_id` and returns any safe sanitized delta."""
    state = self._calls.get(call_id)
    if state is None:
      self.start_call(call_id, tool_name=tool_name or "")
      state = self._calls[call_id]
    elif tool_name and not state["tool_name"]:
      state["tool_name"] = tool_name

    if not arguments_delta or state["overflowed"]:
      return ""

    self.deltas_processed += 1
    if state["raw_chars"] + len(arguments_delta) > self.max_call_chars:
      state["overflowed"] = True
      state["parts"].clear()
      state["sanitizer"].reset()
      return ""

    state["raw_chars"] += len(arguments_delta)
    safe_delta = state["sanitizer"].push(arguments_delta)
    if safe_delta:
      state["parts"].append(safe_delta)
    return safe_delta

  def finish_call(
      self,
      call_id: str | int,
      *,
      schema_slimmer: Any | None = None,
      stats: Any | None = None,
  ) -> dict[str, Any]:
    """Flushes `call_id`, decodes JSON, scrubs decoded values/escapes, and coerces/validates."""
    state = self._calls.pop(call_id, None)
    if state is None:
      return {
          "call_id": call_id,
          "tool_name": "",
          "arguments": {},
          "sanitized_json": "{}",
          "is_valid_json": True,
          "validation_error": None,
      }

    tool_name = str(state["tool_name"] or "")
    if state["overflowed"]:
      return {
          "call_id": call_id,
          "tool_name": tool_name,
          "arguments": None,
          "sanitized_json": "",
          "is_valid_json": False,
          "validation_error": (
              f"[mcp-slim] Streaming tool-call arguments for '{tool_name or call_id}' "
              f"exceeded max_call_chars ({self.max_call_chars})."
          ),
      }

    tail = state["sanitizer"].flush()
    if tail:
      state["parts"].append(tail)
    assembled = "".join(state["parts"]).strip()
    self.calls_completed += 1

    if not assembled:
      parsed: Any = {}
    else:
      try:
        parsed = json.loads(assembled)
      except json.JSONDecodeError as exc:
        return {
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": None,
            "sanitized_json": assembled,
            "is_valid_json": False,
            "validation_error": f"Malformed tool-call arguments JSON: {exc.msg}",
        }

    if not isinstance(parsed, dict):
      return {
          "call_id": call_id,
          "tool_name": tool_name,
          "arguments": None,
          "sanitized_json": assembled,
          "is_valid_json": False,
          "validation_error": "Tool-call arguments must decode to a JSON object.",
      }

    # Second pass over decoded Python strings catches JSON `\uXXXX`-escaped tags or handles
    sanitized_args = self.guard.redact_value(parsed)
    sanitized_args = self.guard.resolve_handles_in_value(sanitized_args)

    validation_err: str | None = None
    if schema_slimmer is not None and tool_name:
      sanitized_args = schema_slimmer.coerce_arguments(
          tool_name, sanitized_args, stats=stats
      )
      validation_err = schema_slimmer.validate_union_arguments(
          tool_name, sanitized_args
      )

    canonical_json = json.dumps(
        sanitized_args, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": sanitized_args,
        "sanitized_json": canonical_json,
        "is_valid_json": True,
        "validation_error": validation_err,
    }
