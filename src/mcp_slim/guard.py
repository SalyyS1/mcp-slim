"""Bidirectional secret redaction, indirect prompt-injection shield, and opaque handle vault for MCP traffic."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import hmac
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
            r"(?i)((?:API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|CLIENT_SECRET|DB_PASSWORD|DATABASE_PASSWORD)\s*[:=]\s*['\"]?(?!\[(?:REDACTED|SECRET_REF):))([^\s'\",;}{]{8,})(['\"]?)"
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
