"""Bidirectional secret redaction and session-scoped opaque handle vault for MCP traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import re
import secrets
from typing import Any

# High-precision patterns covering modern LLM, cloud, and developer secrets.
# Every rule starts with a literal prefix (no leading `\b` or unanchored alternation)
# and includes C-level substring triggers so CPython's `sre` engine uses Boyer-Moore
# `fastsearch` instead of scanning character-by-character.
_SECRET_RULES: tuple[tuple[str, tuple[str, ...], re.Pattern[str], str], ...] = (
    (
        "anthropic_api_key",
        ("sk-ant-",),
        re.compile(r"sk-ant-(?:api03|admin01)-[A-Za-z0-9_-]{20,}"),
        "[REDACTED:ANTHROPIC_KEY]",
    ),
    (
        "openai_api_key",
        ("sk-",),
        re.compile(
            r"sk-(?:(?:proj-|svcacct-|None-)?[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{10,}|proj-[A-Za-z0-9_-]{32,})"
        ),
        "[REDACTED:OPENAI_KEY]",
    ),
    (
        "github_token",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        "[REDACTED:GITHUB_TOKEN]",
    ),
    (
        "github_token",
        ("github_pat_",),
        re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
        "[REDACTED:GITHUB_TOKEN]",
    ),
    (
        "aws_access_key",
        ("AKIA", "ASIA", "ABIA", "ACCA"),
        re.compile(r"A(?:KIA|SIA|BIA|CCA)[0-9A-Z]{16}\b"),
        "[REDACTED:AWS_ACCESS_KEY]",
    ),
    (
        "google_api_key",
        ("AIza",),
        re.compile(r"AIza[0-9A-Za-z_-]{35}\b"),
        "[REDACTED:GCP_API_KEY]",
    ),
    (
        "slack_token",
        ("xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"),
        re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24,}"),
        "[REDACTED:SLACK_TOKEN]",
    ),
    (
        "stripe_key",
        ("sk_live_", "sk_test_", "rk_live_", "rk_test_"),
        re.compile(r"[sr]k_(?:live|test)_[0-9a-zA-Z]{24,}"),
        "[REDACTED:STRIPE_KEY]",
    ),
    (
        "huggingface_token",
        ("hf_",),
        re.compile(r"hf_[A-Za-z0-9]{30,}"),
        "[REDACTED:HF_TOKEN]",
    ),
    (
        "private_key_block",
        ("PRIVATE KEY-----",),
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
        "[REDACTED:PRIVATE_KEY_BLOCK]",
    ),
    (
        "jwt_token",
        ("eyJ",),
        re.compile(
            r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{16,}"
        ),
        "[REDACTED:JWT_TOKEN]",
    ),
    (
        "url_userinfo_secret",
        ("://",),
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
            "X-Amz-Signature=",
            "x-amz-signature=",
        ),
        re.compile(
            r"(?i)([?&](?:token|private_token|oauth_token|access_token|auth_token|api_key|apikey|sig|signature|x-amz-signature|x-amz-security-token)=(?!\[(?:REDACTED|SECRET_REF):))([^&\s\"'#]{6,})"
        ),
        r"\1[REDACTED:URL_QUERY_SECRET]",
    ),
    (
        "env_assignment_secret",
        (
            "API_KEY",
            "SECRET_KEY",
            "ACCESS_TOKEN",
            "PRIVATE_KEY",
            "AUTH_TOKEN",
            "CLIENT_SECRET",
            "DB_PASSWORD",
            "DATABASE_PASSWORD",
            "api_key",
            "secret_key",
            "access_token",
            "private_key",
            "auth_token",
            "client_secret",
            "db_password",
            "database_password",
        ),
        re.compile(
            r"(?i)((?:API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|CLIENT_SECRET|DB_PASSWORD|DATABASE_PASSWORD)\s*[:=]\s*['\"]?(?!\[(?:REDACTED|SECRET_REF):))([^\s'\",;}{]{8,})(['\"]?)"
        ),
        r"\1[REDACTED:ENV_SECRET]\3",
    ),
    (
        "prompt_injection",
        (
            "<IMPORTANT",
            "<important",
            "<SYSTEM",
            "<system",
            "<INSTRUCTION",
            "<instruction",
            "ignore previous instructions",
            "Ignore previous instructions",
            "ignore all previous instructions",
            "Ignore all previous instructions",
            "disregard previous instructions",
            "Disregard previous instructions",
        ),
        re.compile(
            r"(?is)(?:<(?:IMPORTANT|SYSTEM|INSTRUCTION)\b[^>]*>[\s\S]*?</(?:IMPORTANT|SYSTEM|INSTRUCTION)>|<(?:IMPORTANT|SYSTEM|INSTRUCTION)\b[^>]*>[\s\S]*$|(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions[^\n.\"]*)"
        ),
        "[REDACTED:PROMPT_INJECTION]",
    ),
)

_UNICODE_SMUGGLING_RE = re.compile(
    r"[\u202a-\u202e\u2066-\u2069\U000e0000-\U000e007f]+"
)

_FRAMED_SECRET_CATEGORIES = frozenset({
    "url_userinfo_secret",
    "url_query_secret",
    "env_assignment_secret",
})

_SECRET_HANDLE_PATTERN = re.compile(r"\[SECRET_REF:[a-z0-9_]+:[0-9a-f]{10}\]")


@dataclass
class RedactionReport:
  """Tracks how many secrets were intercepted and redacted."""

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

  def redact_text(self, text: str) -> str:
    if not self.enabled or not text:
      return text
    current = text
    if not current.isascii():
      current, smug_count = _UNICODE_SMUGGLING_RE.subn("", current)
      if smug_count:
        self.report.record("unicode_smuggling", smug_count)
    for name, triggers, pattern, replacement in _SECRET_RULES:
      if not any(t in current for t in triggers):
        continue
      if self.vault_handles and name != "prompt_injection":
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
          self.report.record(name, count)
      else:
        current, count = pattern.subn(replacement, current)
        if count:
          self.report.record(name, count)
    return current

  def redact_value(self, value: Any) -> Any:
    if not self.enabled:
      return value
    if isinstance(value, str):
      return self.redact_text(value)
    if isinstance(value, list):
      return [self.redact_value(item) for item in value]
    if isinstance(value, dict):
      return {k: self.redact_value(v) for k, v in value.items()}
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
