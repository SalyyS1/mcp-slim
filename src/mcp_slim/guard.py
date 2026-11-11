"""Bidirectional secret and credential redaction engine for MCP JSON-RPC traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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
        re.compile(r"(://[^/\s:@\"']+:)([^@\s/\"']+)(@)"),
        r"\1[REDACTED:URL_SECRET]\3",
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
            r"(?i)((?:API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|CLIENT_SECRET|DB_PASSWORD|DATABASE_PASSWORD)\s*[:=]\s*['\"]?)([^\s'\",;}{]{8,})(['\"]?)"
        ),
        r"\1[REDACTED:ENV_SECRET]\3",
    ),
)


@dataclass
class RedactionReport:
  """Tracks how many secrets were intercepted and redacted."""

  total_redactions: int = 0
  by_category: dict[str, int] = field(default_factory=dict)

  def record(self, category: str, count: int = 1) -> None:
    if count <= 0:
      return
    self.total_redactions += count
    self.by_category[category] = self.by_category.get(category, 0) + count


class SecretGuard:
  """Scans and redacts secrets from strings and nested JSON structures."""

  def __init__(self, enabled: bool = True) -> None:
    self.enabled = enabled
    self.report = RedactionReport()

  def redact_text(self, text: str) -> str:
    if not self.enabled or not text:
      return text
    current = text
    for name, triggers, pattern, replacement in _SECRET_RULES:
      if not any(t in current for t in triggers):
        continue
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
