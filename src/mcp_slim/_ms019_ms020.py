"""MS019: Capability-scoped multi-upstream MCP session policy engine and cross-server escalation guard.
MS020: Recursive prompt and resource template argument exfiltration shield.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import json
import posixpath
import re
import unicodedata
import urllib.parse
from typing import Any

_SESSION_POLICY_RULE_ID = "MS019"
_TEMPLATE_SHIELD_RULE_ID = "MS020"

_SUBWORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+")

_MUTATING_VERBS = frozenset({
    "add",
    "append",
    "apply",
    "assign",
    "attach",
    "cancel",
    "clear",
    "close",
    "commit",
    "create",
    "delete",
    "deploy",
    "destroy",
    "detach",
    "disable",
    "drop",
    "edit",
    "enable",
    "eval",
    "exec",
    "execute",
    "flush",
    "fork",
    "grant",
    "import",
    "insert",
    "install",
    "invoke",
    "kill",
    "merge",
    "migrate",
    "modify",
    "move",
    "mutate",
    "overwrite",
    "patch",
    "post",
    "publish",
    "purge",
    "push",
    "put",
    "reboot",
    "remove",
    "rename",
    "replace",
    "reset",
    "restart",
    "restore",
    "revoke",
    "rm",
    "rollback",
    "rotate",
    "run",
    "save",
    "send",
    "set",
    "spawn",
    "start",
    "stop",
    "submit",
    "terminate",
    "transfer",
    "trigger",
    "truncate",
    "uninstall",
    "unlink",
    "unset",
    "update",
    "upload",
    "upsert",
    "wipe",
    "write",
})

_DESTRUCTIVE_VERBS = frozenset({
    "delete",
    "destroy",
    "drop",
    "erase",
    " format",
    "kill",
    "overwrite",
    "purge",
    "remove",
    "reset",
    "revoke",
    "rm",
    "terminate",
    "truncate",
    "uninstall",
    "unlink",
    "wipe",
})

_READ_VERBS = frozenset({
    "analyze",
    "audit",
    "check",
    "count",
    "describe",
    "diagnose",
    "diff",
    "discover",
    "dump",
    "estimate",
    "explain",
    "export",
    "fetch",
    "filter",
    "find",
    "get",
    "grep",
    "head",
    "help",
    "info",
    "inspect",
    "list",
    "load",
    "lookup",
    "ls",
    "parse",
    "ping",
    "preview",
    "profile",
    "query",
    "read",
    "render",
    "resolve",
    "scan",
    "search",
    "select",
    "show",
    "stat",
    "stats",
    "status",
    "summarize",
    "tail",
    "test",
    "validate",
    "verify",
    "view",
    "whoami",
})

# Innermost template expansion markers across Mustache/Handlebars/Jinja2, JS/Shell, Ruby, ERB/EJS
_INNER_EXPANSION_RE = re.compile(
    r"(?:\{\{[^{}]*\}\}|\$\{[^{}]*\}|#\{[^{}]*\}|<%\s*=?\s*[^%]*?%>|\{%[^{}%]*%\})"
)
_NESTED_EXPANSION_RE = re.compile(
    r"(?:\{\{[\s\S]*\{\{[\s\S]*\}\}[\s\S]*\}\}"
    r"|\$\{[\s\S]*\$\{[\s\S]*\}[\s\S]*\}"
    r"|#\{[\s\S]*#\{[\s\S]*\}[\s\S]*\}"
    r"|\{\{[\s\S]*\$\{[\s\S]*\}[\s\S]*\}\}"
    r"|\$\{[\s\S]*\{\{[\s\S]*\}\}[\s\S]*\})"
)

# Variable identifier extraction inside {{var}}, ${var}, #{var}, or {var}
_ARG_REF_TOKEN_RE = re.compile(
    r"(?:\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}"
    r"|\$\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}"
    r"|#\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}"
    r"|\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\})"
)

# RFC 6570 URI template expression: {op varlist}
_URI_TEMPLATE_EXPR_RE = re.compile(r"\{([+#./;?&]?)([^{}]+)\}")

# Sensitive environment and credential keywords
_ENV_SECRET_KEYWORDS = frozenset({
    "aws_secret_access_key",
    "aws_access_key_id",
    "aws_session_token",
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "google_api_key",
    "github_token",
    "gh_token",
    "gitlab_token",
    "slack_token",
    "discord_token",
    "stripe_secret_key",
    "hf_token",
    "bearer",
    "password",
    "passwd",
    "secret",
    "private_key",
    "api_key",
    "apikey",
    "client_secret",
    "database_url",
    "connection_string",
    "id_rsa",
    "id_ed25519",
    "jwt_secret",
    "auth_token",
    "access_token",
    "refresh_token",
})

# Path traversal and sensitive host file targets
_PATH_TRAVERSAL_RE = re.compile(
    r"(?:(?:\.\.[/\\])"
    r"|(?:%2e%2e(?:%2f|%5c|/|\\))"
    r"|(?:^|[/\\])(?:etc/(?:passwd|shadow|hosts|sudoers|kubernetes)|proc/(?:self|1|environ|cmdline|maps)|windows/(?:system32|win\.ini)|root/\.ssh|home/[^/]+/\.ssh|\.aws/credentials|\.kube/config))",
    re.IGNORECASE,
)

# Cloud metadata SSRF targets (AWS IMDSv1/v2 IPv4 + IPv6, GCP, Alibaba, Azure/OpenStack)
_SSRF_METADATA_RE = re.compile(
    r"(?:169\.254\.169\.254|fd00:ec2::254|metadata\.google\.internal|100\.100\.100\.200|169\.254\.170\.2|instance-data)",
    re.IGNORECASE,
)

_DANGEROUS_SCHEMES_RE = re.compile(
    r"(?:^|[\s'\"=(])((?:file|gopher|dict|ldap|ldaps|tftp|jar|netdoc|php)://)",
    re.IGNORECASE,
)

_SSTI_EXPRESSIONS_RE = re.compile(
    r"(?:__class__|__mro__|__subclasses__|__globals__|__builtins__|__init__|__import__"
    r"|constructor\.prototype|process\.mainModule|Runtime\.getRuntime"
    r"|config\.items\s*\(|self\._TemplateReference"
    r"|\{\{\s*\d+\s*\*\s*\d+\s*\}\}|\$\{\s*\d+\s*\*\s*\d+\s*\})",
    re.IGNORECASE,
)

_EXFIL_BEACON_URL_RE = re.compile(
    r"https?://[^\s'\"<>]+(?:\{\{[^{}]+\}\}|\$\{[^{}]+\}|#\{[^{}]+\})",
    re.IGNORECASE,
)


def _split_subwords(name: str) -> list[str]:
  return [tok.lower() for tok in _SUBWORD_RE.findall(name)]


def _normalize_decoded_text(text: str) -> str:
  """Applies NFKC normalization and up to 2 passes of URL percent-decoding for evasion-resistant scanning."""
  curr = unicodedata.normalize("NFKC", text)
  for _ in range(2):
    decoded = urllib.parse.unquote(curr)
    decoded_nfkc = unicodedata.normalize("NFKC", decoded)
    if decoded_nfkc == curr:
      break
    curr = decoded_nfkc
  return curr


def _canonicalize_uri(uri: str) -> tuple[str, str]:
  """Returns (scheme, canonical_uri) with percent-encoding decoded and dot-segments collapsed."""
  decoded = _normalize_decoded_text(uri.strip()).replace("\\", "/")
  if "://" not in decoded:
    norm_path = posixpath.normpath("/" + decoded.lstrip("/"))
    return ("custom", norm_path)
  scheme_raw, rest = decoded.split("://", 1)
  scheme = scheme_raw.lower()
  if scheme == "file":
    if rest.startswith("/"):
      norm_p = posixpath.normpath(rest)
      return (scheme, f"file://{norm_p}")
    host, _, path_part = rest.partition("/")
    norm_p = posixpath.normpath("/" + path_part)
    return (scheme, f"file://{host}{norm_p}")
  host, sep, path_part = rest.partition("/")
  if not sep:
    return (scheme, f"{scheme}://{host}")
  path_only, q_sep, query_part = path_part.partition("?")
  norm_p = posixpath.normpath("/" + path_only)
  suffix = f"?{query_part}" if q_sep else ""
  return (scheme, f"{scheme}://{host}{norm_p}{suffix}")


@dataclass(slots=True)
class CapabilityScope:
  """Defines the authorized capabilities for an upstream server or session."""

  tools_enabled: bool = True
  tools_list: bool = True
  tools_call: bool = True
  allowed_tools: tuple[str, ...] | None = None
  denied_tools: tuple[str, ...] = ()
  resources_enabled: bool = True
  resources_list: bool = True
  resources_read: bool = True
  resources_subscribe: bool = True
  allowed_schemes: tuple[str, ...] = ("file", "http", "https", "custom")
  denied_resource_prefixes: tuple[str, ...] = ()
  prompts_enabled: bool = True
  prompts_list: bool = True
  prompts_get: bool = True
  allowed_prompts: tuple[str, ...] | None = None
  sampling_enabled: bool = False
  roots_enabled: bool = False
  elicitation_enabled: bool = False
  completions_enabled: bool = True
  logging_enabled: bool = True
  read_only: bool = False
  allow_destructive: bool = True
  max_tool_calls: int | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> CapabilityScope:
    if not isinstance(data, dict):
      return cls()
    allowed_t = data.get("allowed_tools")
    if allowed_t is not None and not isinstance(allowed_t, tuple):
      allowed_t = tuple(str(x) for x in allowed_t)
    denied_t = tuple(str(x) for x in (data.get("denied_tools") or ()))
    allowed_p = data.get("allowed_prompts")
    if allowed_p is not None and not isinstance(allowed_p, tuple):
      allowed_p = tuple(str(x) for x in allowed_p)
    raw_schemes = data.get("allowed_schemes")
    allowed_s = (
        tuple(str(x).lower() for x in raw_schemes)
        if raw_schemes is not None
        else ("file", "http", "https", "custom")
    )
    denied_rp = tuple(str(x) for x in (data.get("denied_resource_prefixes") or ()))
    raw_max_calls = data.get("max_tool_calls")
    max_calls = int(raw_max_calls) if raw_max_calls is not None else None
    return cls(
        tools_enabled=bool(data.get("tools_enabled", data.get("tools", True))),
        tools_list=bool(data.get("tools_list", True)),
        tools_call=bool(data.get("tools_call", True)),
        allowed_tools=allowed_t,
        denied_tools=denied_t,
        resources_enabled=bool(data.get("resources_enabled", data.get("resources", True))),
        resources_list=bool(data.get("resources_list", True)),
        resources_read=bool(data.get("resources_read", True)),
        resources_subscribe=bool(data.get("resources_subscribe", True)),
        allowed_schemes=allowed_s,
        denied_resource_prefixes=denied_rp,
        prompts_enabled=bool(data.get("prompts_enabled", data.get("prompts", True))),
        prompts_list=bool(data.get("prompts_list", True)),
        prompts_get=bool(data.get("prompts_get", True)),
        allowed_prompts=allowed_p,
        sampling_enabled=bool(data.get("sampling_enabled", data.get("sampling", False))),
        roots_enabled=bool(data.get("roots_enabled", data.get("roots", False))),
        elicitation_enabled=bool(
            data.get("elicitation_enabled", data.get("elicitation", False))
        ),
        completions_enabled=bool(
            data.get("completions_enabled", data.get("completions", True))
        ),
        logging_enabled=bool(data.get("logging_enabled", data.get("logging", True))),
        read_only=bool(data.get("read_only", False)),
        allow_destructive=bool(data.get("allow_destructive", True)),
        max_tool_calls=max_calls,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "tools_enabled": self.tools_enabled,
        "tools_list": self.tools_list,
        "tools_call": self.tools_call,
        "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
        "denied_tools": list(self.denied_tools),
        "resources_enabled": self.resources_enabled,
        "resources_list": self.resources_list,
        "resources_read": self.resources_read,
        "resources_subscribe": self.resources_subscribe,
        "allowed_schemes": list(self.allowed_schemes),
        "denied_resource_prefixes": list(self.denied_resource_prefixes),
        "prompts_enabled": self.prompts_enabled,
        "prompts_list": self.prompts_list,
        "prompts_get": self.prompts_get,
        "allowed_prompts": list(self.allowed_prompts) if self.allowed_prompts is not None else None,
        "sampling_enabled": self.sampling_enabled,
        "roots_enabled": self.roots_enabled,
        "elicitation_enabled": self.elicitation_enabled,
        "completions_enabled": self.completions_enabled,
        "logging_enabled": self.logging_enabled,
        "read_only": self.read_only,
        "allow_destructive": self.allow_destructive,
        "max_tool_calls": self.max_tool_calls,
    }


@dataclass(slots=True)
class SessionPolicy:
  """Defines the policy boundaries for an MCP session spanning multiple upstreams."""

  session_id: str = "default"
  server_scopes: dict[str, CapabilityScope] = field(default_factory=dict)
  default_scope: CapabilityScope | None = None
  read_only: bool = False
  allow_destructive: bool = True
  allow_cross_server_delegation: bool = False
  allowed_upstream_ids: tuple[str, ...] | None = None
  max_request_payload_bytes: int = 10 * 1024 * 1024
  namespace_separator: str = "__"

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> SessionPolicy:
    if not isinstance(data, dict):
      return cls()
    sid = str(data.get("session_id", "default"))
    ro = bool(data.get("read_only", False))
    allow_dest = bool(data.get("allow_destructive", True))
    delegation = bool(data.get("allow_cross_server_delegation", False))
    max_bytes = int(data.get("max_request_payload_bytes", 10 * 1024 * 1024))
    ns_sep = str(data.get("namespace_separator", "__") or "__")
    allowed_up = data.get("allowed_upstream_ids")
    if allowed_up is not None and not isinstance(allowed_up, tuple):
      allowed_up = tuple(str(x) for x in allowed_up)

    raw_scopes = data.get("server_scopes", {})
    parsed_scopes: dict[str, CapabilityScope] = {}
    if isinstance(raw_scopes, dict):
      for k, v in raw_scopes.items():
        if isinstance(v, dict):
          parsed_scopes[str(k)] = CapabilityScope.from_dict(v)
        elif isinstance(v, CapabilityScope):
          parsed_scopes[str(k)] = v

    def_scope = data.get("default_scope")
    if isinstance(def_scope, dict):
      def_scope_obj: CapabilityScope | None = CapabilityScope.from_dict(def_scope)
    elif isinstance(def_scope, CapabilityScope):
      def_scope_obj = def_scope
    else:
      def_scope_obj = None

    return cls(
        session_id=sid,
        server_scopes=parsed_scopes,
        default_scope=def_scope_obj,
        read_only=ro,
        allow_destructive=allow_dest,
        allow_cross_server_delegation=delegation,
        allowed_upstream_ids=allowed_up,
        max_request_payload_bytes=max_bytes,
        namespace_separator=ns_sep,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "session_id": self.session_id,
        "read_only": self.read_only,
        "allow_destructive": self.allow_destructive,
        "allow_cross_server_delegation": self.allow_cross_server_delegation,
        "allowed_upstream_ids": (
            list(self.allowed_upstream_ids) if self.allowed_upstream_ids is not None else None
        ),
        "max_request_payload_bytes": self.max_request_payload_bytes,
        "namespace_separator": self.namespace_separator,
        "server_scopes": {k: v.to_dict() for k, v in self.server_scopes.items()},
        "default_scope": self.default_scope.to_dict() if self.default_scope is not None else None,
    }


@dataclass(slots=True)
class PolicyViolation:
  """A single security or policy breach identified by SessionPolicyEngine."""

  rule_id: str = _SESSION_POLICY_RULE_ID
  violation_code: str = "POLICY_VIOLATION"
  message: str = ""
  level: str = "error"
  server_id: str = ""
  target_server_id: str | None = None
  method: str | None = None
  tool_name: str | None = None
  details: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    res: dict[str, Any] = {
        "rule_id": self.rule_id,
        "violation_code": self.violation_code,
        "message": self.message,
        "level": self.level,
        "server_id": self.server_id,
    }
    if self.target_server_id:
      res["target_server_id"] = self.target_server_id
    if self.method:
      res["method"] = self.method
    if self.tool_name:
      res["tool_name"] = self.tool_name
    if self.details:
      res["details"] = self.details
    return res

  def to_sarif_finding(self) -> dict[str, Any]:
    loc = self.tool_name or self.method or self.server_id or "session"
    return {
        "rule_id": self.rule_id,
        "level": self.level,
        "kind": self.violation_code.lower(),
        "tool": loc,
        "message": self.message,
        "violating_server": self.server_id,
        "attempted_capability": self.method or self.violation_code,
        "target_server": self.target_server_id or "",
    }


@dataclass(slots=True)
class PolicyDecision:
  """The outcome of evaluating an MCP message frame against SessionPolicy."""

  allowed: bool
  rule_id: str = _SESSION_POLICY_RULE_ID
  violations: list[PolicyViolation] = field(default_factory=list)
  frame: dict[str, Any] = field(default_factory=dict)
  resolved_server_id: str = ""

  def to_dict(self) -> dict[str, Any]:
    return {
        "allowed": self.allowed,
        "rule_id": self.rule_id,
        "resolved_server_id": self.resolved_server_id,
        "violation_count": len(self.violations),
        "violations": [v.to_dict() for v in self.violations],
        "findings": self.to_sarif_findings(),
    }

  def to_sarif_findings(self, source_uri: str = "session_policy.json") -> list[dict[str, Any]]:
    del source_uri
    return [v.to_sarif_finding() for v in self.violations]


class SessionPolicyEngine:
  """Enforces capability scopes and cross-server isolation across multi-upstream MCP sessions."""

  def __init__(
      self,
      policy: SessionPolicy | dict[str, Any] | None = None,
      enabled: bool = True,
  ) -> None:
    self.enabled = enabled
    if policy is None:
      self.policy = SessionPolicy()
    elif isinstance(policy, dict):
      self.policy = SessionPolicy.from_dict(policy)
    else:
      self.policy = policy
    self._tool_annotations: dict[tuple[str, str], dict[str, Any]] = {}
    self._call_counts: dict[str, int] = {}

  def reset_counters(self) -> None:
    self._call_counts.clear()

  def register_tools_catalog(self, server_id: str, tools: list[dict[str, Any]]) -> None:
    """Indexes MCP ToolAnnotations (readOnlyHint, destructiveHint) for accurate enforcement."""
    sid = server_id or self.policy.session_id or "default"
    sep = self.policy.namespace_separator
    for tool in tools:
      if not isinstance(tool, dict):
        continue
      name = tool.get("name")
      ann = tool.get("annotations")
      if isinstance(name, str) and name and isinstance(ann, dict):
        self._tool_annotations[(sid, name)] = dict(ann)
        if sep and sep in name:
          pfx, bare = name.split(sep, 1)
          if pfx and bare:
            self._tool_annotations[(pfx, bare)] = dict(ann)
            self._tool_annotations[(pfx, name)] = dict(ann)

  def get_scope_for_server(self, server_id: str) -> CapabilityScope:
    if server_id in self.policy.server_scopes:
      return self.policy.server_scopes[server_id]
    if self.policy.default_scope is not None:
      return self.policy.default_scope
    return CapabilityScope(
        read_only=self.policy.read_only,
        allow_destructive=self.policy.allow_destructive,
    )

  def _lookup_annotations(self, server_id: str, tool_name: str) -> dict[str, Any] | None:
    ann = self._tool_annotations.get((server_id, tool_name))
    if ann is not None:
      return ann
    sep = self.policy.namespace_separator
    if sep and sep in tool_name:
      pfx, bare = tool_name.split(sep, 1)
      ann = self._tool_annotations.get((pfx, bare)) or self._tool_annotations.get((server_id, bare))
      if ann is not None:
        return ann
    return None

  def _is_read_only_tool_name(self, name: str, server_id: str = "") -> bool:
    ann = self._lookup_annotations(server_id, name)
    if ann is not None:
      if ann.get("destructiveHint") is True:
        return False
      if isinstance(ann.get("readOnlyHint"), bool):
        return bool(ann["readOnlyHint"])

    sep = self.policy.namespace_separator
    bare = name.split(sep, 1)[1] if (sep and sep in name) else name
    tokens = _split_subwords(bare)
    if not tokens:
      return False
    # Fail-closed: any mutating verb anywhere in the subword sequence marks the tool non-read-only.
    if any(tok in _MUTATING_VERBS for tok in tokens):
      return False
    if any(tok in _READ_VERBS for tok in tokens):
      return True
    return False

  def _is_destructive_tool_name(self, name: str, server_id: str = "") -> bool:
    ann = self._lookup_annotations(server_id, name)
    if ann is not None and isinstance(ann.get("destructiveHint"), bool):
      return bool(ann["destructiveHint"])
    sep = self.policy.namespace_separator
    bare = name.split(sep, 1)[1] if (sep and sep in name) else name
    tokens = _split_subwords(bare)
    return any(tok in _DESTRUCTIVE_VERBS for tok in tokens)

  def _matches_tool_patterns(self, tool_name: str, bare_name: str, patterns: tuple[str, ...]) -> bool:
    return any(
        fnmatch.fnmatch(tool_name, pat) or (bare_name != tool_name and fnmatch.fnmatch(bare_name, pat))
        for pat in patterns
    )

  def _resolve_target(
      self,
      frame: dict[str, Any],
      server_id: str,
      target_server_id: str | None,
  ) -> tuple[str, str, str]:
    """Resolves (origin_server_id, target_server_id, bare_tool_name)."""
    params = frame.get("params") if isinstance(frame.get("params"), dict) else {}
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    explicit_origin = (
        server_id
        or str(frame.get("_source_server") or meta.get("source_server") or "")
    )
    explicit_target = (
        target_server_id
        or str(frame.get("_target_server") or meta.get("target_server") or "")
        or None
    )

    method = str(frame.get("method") or "")
    tool_name = str(params.get("name") or "") if method in ("tools/call", "prompts/get") else ""
    bare_name = tool_name
    inferred_target: str | None = None
    sep = self.policy.namespace_separator
    if method == "tools/call" and sep and sep in tool_name:
      cand_srv, cand_bare = tool_name.split(sep, 1)
      if cand_srv and cand_bare:
        bare_name = cand_bare
        if (
            cand_srv in self.policy.server_scopes
            or (
                self.policy.allowed_upstream_ids is not None
                and cand_srv in self.policy.allowed_upstream_ids
            )
            or explicit_origin
        ):
          inferred_target = cand_srv

    eff_target = explicit_target or inferred_target or explicit_origin or self.policy.session_id or "default"
    eff_origin = explicit_origin or eff_target
    return (eff_origin, eff_target, bare_name)

  def filter_tools_for_server(
      self,
      server_id: str,
      tools: list[dict[str, Any]],
  ) -> list[dict[str, Any]]:
    """Filters an upstream tools/list catalog to only tools permitted by the server's CapabilityScope."""
    if not self.enabled:
      return list(tools)
    self.register_tools_catalog(server_id, tools)
    sid = server_id or self.policy.session_id or "default"
    if self.policy.allowed_upstream_ids is not None and sid not in self.policy.allowed_upstream_ids:
      return []
    scope = self.get_scope_for_server(sid)
    if not scope.tools_enabled or not scope.tools_list:
      return []
    sep = self.policy.namespace_separator
    kept: list[dict[str, Any]] = []
    for tool in tools:
      if not isinstance(tool, dict):
        continue
      t_name = str(tool.get("name") or "")
      if not t_name:
        continue
      t_srv = sid
      bare = t_name
      if sep and sep in t_name:
        pfx, cand_bare = t_name.split(sep, 1)
        if pfx in self.policy.server_scopes:
          t_srv = pfx
          bare = cand_bare
      t_scope = self.get_scope_for_server(t_srv)
      if not t_scope.tools_enabled or not t_scope.tools_list:
        continue
      if t_scope.denied_tools and self._matches_tool_patterns(t_name, bare, t_scope.denied_tools):
        continue
      if t_scope.allowed_tools is not None and not self._matches_tool_patterns(
          t_name, bare, t_scope.allowed_tools
      ):
        continue
      if (self.policy.read_only or t_scope.read_only) and not self._is_read_only_tool_name(
          t_name, t_srv
      ):
        continue
      if (
          not self.policy.allow_destructive or not t_scope.allow_destructive
      ) and self._is_destructive_tool_name(t_name, t_srv):
        continue
      kept.append(tool)
    return kept

  def evaluate_client_request(
      self,
      frame: dict[str, Any],
      server_id: str = "",
      target_server_id: str | None = None,
  ) -> PolicyDecision:
    if not self.enabled:
      return PolicyDecision(allowed=True, frame=frame)

    violations: list[PolicyViolation] = []
    sid, target_sid, bare_name = self._resolve_target(frame, server_id, target_server_id)

    # Check request payload byte ceiling
    if self.policy.max_request_payload_bytes > 0:
      raw_bytes = len(json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
      if raw_bytes > self.policy.max_request_payload_bytes:
        violations.append(
            PolicyViolation(
                violation_code="PAYLOAD_SIZE_EXCEEDED",
                message=(
                    f"Request frame size ({raw_bytes} bytes) exceeds session policy limit "
                    f"({self.policy.max_request_payload_bytes} bytes)."
                ),
                server_id=sid,
                target_server_id=target_sid,
                method=str(frame.get("method") or ""),
                details={
                    "payload_bytes": raw_bytes,
                    "max_request_payload_bytes": self.policy.max_request_payload_bytes,
                },
            )
        )

    # Check allowed upstreams
    if self.policy.allowed_upstream_ids is not None:
      if target_sid not in self.policy.allowed_upstream_ids:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_UPSTREAM",
                message=f"Upstream server '{target_sid}' is not permitted by session policy.",
                server_id=sid,
                target_server_id=target_sid,
                method=str(frame.get("method") or "") or None,
            )
        )

    # Check cross-server delegation boundary
    if sid and target_sid and sid != target_sid:
      if not self.policy.allow_cross_server_delegation:
        violations.append(
            PolicyViolation(
                violation_code="CROSS_SERVER_ESCALATION",
                message=(
                    f"Cross-server invocation from '{sid}' to '{target_sid}' is denied "
                    "by session policy isolation boundary."
                ),
                server_id=sid,
                target_server_id=target_sid,
                method=str(frame.get("method") or "") or None,
                tool_name=(
                    str(frame.get("params", {}).get("name") or "")
                    if isinstance(frame.get("params"), dict)
                    else None
                )
                or None,
            )
        )

    method = str(frame.get("method") or "")
    params = frame.get("params") if isinstance(frame.get("params"), dict) else {}
    scope = self.get_scope_for_server(target_sid)

    if method == "tools/list":
      if not scope.tools_enabled or not scope.tools_list:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires tools capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "tools/call":
      if not scope.tools_enabled or not scope.tools_call:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires tools:call capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
      tool_name = str(params.get("name") or "")
      if tool_name:
        if scope.denied_tools and self._matches_tool_patterns(
            tool_name, bare_name, scope.denied_tools
        ):
          violations.append(
              PolicyViolation(
                  violation_code="FORBIDDEN_TOOL_ACCESS",
                  message=f"Invocation of tool '{tool_name}' is explicitly denied for server '{target_sid}'.",
                  server_id=target_sid,
                  method=method,
                  tool_name=tool_name,
              )
          )
        elif scope.allowed_tools is not None and not self._matches_tool_patterns(
            tool_name, bare_name, scope.allowed_tools
        ):
          violations.append(
              PolicyViolation(
                  violation_code="FORBIDDEN_TOOL_ACCESS",
                  message=f"Invocation of tool '{tool_name}' is not in allowed tools list for server '{target_sid}'.",
                  server_id=target_sid,
                  method=method,
                  tool_name=tool_name,
              )
          )
        if (self.policy.read_only or scope.read_only) and not self._is_read_only_tool_name(
            tool_name, target_sid
        ):
          violations.append(
              PolicyViolation(
                  violation_code="READ_ONLY_VIOLATION",
                  message=f"Mutating tool '{tool_name}' invoked in read-only session scope.",
                  server_id=target_sid,
                  method=method,
                  tool_name=tool_name,
              )
          )
        if (
            not self.policy.allow_destructive or not scope.allow_destructive
        ) and self._is_destructive_tool_name(tool_name, target_sid):
          violations.append(
              PolicyViolation(
                  violation_code="DESTRUCTIVE_TOOL_VIOLATION",
                  message=f"Destructive tool '{tool_name}' is blocked by session policy.",
                  server_id=target_sid,
                  method=method,
                  tool_name=tool_name,
              )
          )
      if scope.max_tool_calls is not None:
        next_count = self._call_counts.get(target_sid, 0) + 1
        if next_count > scope.max_tool_calls:
          violations.append(
              PolicyViolation(
                  violation_code="CALL_QUOTA_EXCEEDED",
                  message=(
                      f"Tool call count ({next_count}) exceeds max_tool_calls quota "
                      f"({scope.max_tool_calls}) for server '{target_sid}'."
                  ),
                  server_id=target_sid,
                  method=method,
                  tool_name=tool_name or None,
                  details={"call_count": next_count, "max_tool_calls": scope.max_tool_calls},
              )
          )
        elif not violations:
          self._call_counts[target_sid] = next_count

    elif method in ("resources/list", "resources/templates/list"):
      if not scope.resources_enabled or not scope.resources_list:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires resources capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method in ("resources/read", "resources/subscribe"):
      if (
          not scope.resources_enabled
          or (method == "resources/read" and not scope.resources_read)
          or (method == "resources/subscribe" and not scope.resources_subscribe)
      ):
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires resources capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
      uri = str(params.get("uri") or "")
      if uri:
        scheme, canonical_uri = _canonicalize_uri(uri)
        if scope.allowed_schemes and scheme not in scope.allowed_schemes:
          violations.append(
              PolicyViolation(
                  violation_code="UNAUTHORIZED_CAPABILITY",
                  message=f"Resource URI scheme '{scheme}' is not permitted by policy for server '{target_sid}'.",
                  server_id=target_sid,
                  method=method,
                  details={"uri": uri, "canonical_uri": canonical_uri, "scheme": scheme},
              )
          )
        if scope.denied_resource_prefixes and any(
            uri.startswith(pfx)
            or canonical_uri.startswith(pfx)
            or fnmatch.fnmatch(canonical_uri, pfx)
            for pfx in scope.denied_resource_prefixes
        ):
          violations.append(
              PolicyViolation(
                  violation_code="FORBIDDEN_RESOURCE_ACCESS",
                  message=f"Resource URI '{uri}' matches denied prefix for server '{target_sid}'.",
                  server_id=target_sid,
                  method=method,
                  details={"uri": uri, "canonical_uri": canonical_uri},
              )
          )

    elif method == "prompts/list":
      if not scope.prompts_enabled or not scope.prompts_list:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires prompts capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "prompts/get":
      if not scope.prompts_enabled or not scope.prompts_get:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires prompts:get capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
      prompt_name = str(params.get("name") or "")
      if prompt_name and scope.allowed_prompts is not None:
        if not any(fnmatch.fnmatch(prompt_name, pat) for pat in scope.allowed_prompts):
          violations.append(
              PolicyViolation(
                  violation_code="FORBIDDEN_PROMPT_ACCESS",
                  message=f"Prompt '{prompt_name}' is not in allowed prompts list for server '{target_sid}'.",
                  server_id=target_sid,
                  method=method,
                  details={"prompt_name": prompt_name},
              )
          )

    elif method == "completion/complete":
      if not scope.completions_enabled:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires completions capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "logging/setLevel":
      if not scope.logging_enabled:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires logging capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "sampling/createMessage":
      if not scope.sampling_enabled:
        violations.append(
            PolicyViolation(
                violation_code="UNAUTHORIZED_CAPABILITY",
                message=f"Method '{method}' requires sampling capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "roots/list":
      if not scope.roots_enabled:
        violations.append(
            PolicyViolation(
                violation_code="ROOTS_ESCALATION",
                message=f"Method '{method}' requires roots capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )
    elif method == "elicitation/create":
      if not scope.elicitation_enabled:
        violations.append(
            PolicyViolation(
                violation_code="ELICITATION_ESCALATION",
                message=f"Method '{method}' requires elicitation capability disabled for server '{target_sid}'.",
                server_id=target_sid,
                method=method,
            )
        )

    return PolicyDecision(
        allowed=len(violations) == 0,
        violations=violations,
        frame=frame,
        resolved_server_id=target_sid,
    )

  def evaluate_server_response(
      self,
      frame: dict[str, Any],
      server_id: str = "",
  ) -> PolicyDecision:
    if not self.enabled:
      return PolicyDecision(allowed=True, frame=frame)

    violations: list[PolicyViolation] = []
    sid = server_id or str(frame.get("_source_server") or "") or self.policy.session_id or "default"
    scope = self.get_scope_for_server(sid)

    if self.policy.allowed_upstream_ids is not None and sid not in self.policy.allowed_upstream_ids:
      violations.append(
          PolicyViolation(
              violation_code="UNAUTHORIZED_UPSTREAM",
              message=f"Server frame from unauthorized upstream '{sid}' is rejected.",
              server_id=sid,
              method=str(frame.get("method") or "") or None,
          )
      )

    method = frame.get("method")
    if method == "sampling/createMessage" and not scope.sampling_enabled:
      violations.append(
          PolicyViolation(
              violation_code="UNAUTHORIZED_CAPABILITY",
              message=f"Server '{sid}' initiated unauthorized sampling/createMessage request.",
              server_id=sid,
              method="sampling/createMessage",
          )
      )
    elif method == "roots/list" and not scope.roots_enabled:
      violations.append(
          PolicyViolation(
              violation_code="ROOTS_ESCALATION",
              message=f"Server '{sid}' initiated unauthorized roots/list enumeration request.",
              server_id=sid,
              method="roots/list",
          )
      )
    elif method == "elicitation/create" and not scope.elicitation_enabled:
      violations.append(
          PolicyViolation(
              violation_code="ELICITATION_ESCALATION",
              message=f"Server '{sid}' initiated unauthorized elicitation/create prompt request.",
              server_id=sid,
              method="elicitation/create",
          )
      )

    return PolicyDecision(
        allowed=len(violations) == 0,
        violations=violations,
        frame=frame,
        resolved_server_id=sid,
    )

  def audit_session_transcript(self, events: list[dict[str, Any]]) -> dict[str, Any]:
    all_violations: list[dict[str, Any]] = []
    sarif_findings: list[dict[str, Any]] = []
    for idx, ev in enumerate(events):
      if not isinstance(ev, dict):
        continue
      direction = str(ev.get("direction", "client_to_server")).lower()
      server_id = str(ev.get("server_id") or "")
      target_sid = ev.get("target_server_id")
      frame = ev.get("frame") or ev.get("payload") or ev.get("message") or {}
      if not isinstance(frame, dict):
        continue
      if direction in ("client_to_server", "client", "outbound", "request"):
        dec = self.evaluate_client_request(
            frame,
            server_id=server_id,
            target_server_id=str(target_sid) if target_sid is not None else None,
        )
      else:
        dec = self.evaluate_server_response(frame, server_id=server_id or "default")
      for v in dec.violations:
        vd = v.to_dict()
        vd["event_index"] = idx
        all_violations.append(vd)
        sf = v.to_sarif_finding()
        sf["event_index"] = idx
        sarif_findings.append(sf)

    return {
        "valid": len(all_violations) == 0,
        "session_id": self.policy.session_id,
        "total_events": len(events),
        "violation_count": len(all_violations),
        "security_findings": len(sarif_findings),
        "violations": all_violations,
        "findings": sarif_findings,
    }


def audit_session_policy(
    frame: dict[str, Any] | list[dict[str, Any]],
    policy: dict[str, Any] | SessionPolicy,
    server_id: str = "",
    target_server_id: str | None = None,
) -> PolicyDecision:
  """Validates an MCP message frame against a capability session policy."""
  engine = SessionPolicyEngine(policy=policy, enabled=True)
  if isinstance(frame, list):
    rep = engine.audit_session_transcript(frame)
    viols = [
        PolicyViolation(
            violation_code=str(v.get("violation_code", "POLICY_VIOLATION")),
            message=str(v.get("message", "")),
            level=str(v.get("level", "error")),
            server_id=str(v.get("server_id", "")),
            target_server_id=v.get("target_server_id"),
            method=v.get("method"),
            tool_name=v.get("tool_name"),
            details=dict(v.get("details") or {}),
        )
        for v in rep["violations"]
    ]
    return PolicyDecision(allowed=len(viols) == 0, violations=viols, frame={})
  direction = str(frame.get("_direction", "client")).lower()
  if direction in ("server", "server_to_client"):
    return engine.evaluate_server_response(frame, server_id=server_id)
  return engine.evaluate_client_request(
      frame, server_id=server_id, target_server_id=target_server_id
  )


# ---------------------------------------------------------------------------
# MS020: Template Argument Exfiltration Shield
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemplateShieldViolation:
  """Security violation detected within a prompt or resource template argument."""

  rule_id: str = _TEMPLATE_SHIELD_RULE_ID
  violation_code: str = "TEMPLATE_ARGUMENT_VIOLATION"
  argument_key: str = ""
  raw_value: str = ""
  sanitized_value: str = ""
  message: str = ""
  level: str = "error"
  depth: int = 0

  def to_dict(self) -> dict[str, Any]:
    return {
        "rule_id": self.rule_id,
        "violation_code": self.violation_code,
        "argument_key": self.argument_key,
        "raw_value": self.raw_value,
        "sanitized_value": self.sanitized_value,
        "message": self.message,
        "level": self.level,
        "depth": self.depth,
    }

  def to_sarif_finding(self, template_type: str = "prompt") -> dict[str, Any]:
    loc = f"{template_type}:{self.argument_key}" if self.argument_key else template_type
    return {
        "rule_id": self.rule_id,
        "level": self.level,
        "kind": self.violation_code.lower(),
        "tool": loc,
        "message": self.message,
        "template_type": template_type,
        "argument_key": self.argument_key,
        "expansion_depth": self.depth,
    }


@dataclass(slots=True)
class TemplateArgumentDecision:
  """The outcome of evaluating template arguments against injection and exfiltration attacks."""

  allowed: bool
  rule_id: str = _TEMPLATE_SHIELD_RULE_ID
  template_type: str = "prompt"
  uri_template: str = ""
  violations: list[TemplateShieldViolation] = field(default_factory=list)
  arguments_before: dict[str, Any] = field(default_factory=dict)
  sanitized_arguments: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "allowed": self.allowed,
        "rule_id": self.rule_id,
        "template_type": self.template_type,
        "uri_template": self.uri_template,
        "violation_count": len(self.violations),
        "security_findings": len(self.violations),
        "violations": [v.to_dict() for v in self.violations],
        "findings": self.to_sarif_findings(),
        "arguments_before": self.arguments_before,
        "sanitized_arguments": self.sanitized_arguments,
    }

  def to_sarif_findings(self, source_uri: str = "template_arguments.json") -> list[dict[str, Any]]:
    del source_uri
    return [v.to_sarif_finding(self.template_type) for v in self.violations]


class TemplateArgumentShield:
  """Audits and neutralizes recursive template expansion, URI breakout, path traversal, and secret exfiltration."""

  def __init__(
      self,
      enabled: bool = True,
      max_depth: int = 1,
      block_exfiltration: bool = True,
  ) -> None:
    self.enabled = enabled
    self.max_depth = max(0, int(max_depth))
    self.block_exfiltration = block_exfiltration

  def calculate_expansion_depth(self, text: str) -> int:
    """Computes exact stack-matched maximum nesting depth across template syntaxes."""
    stack: list[str] = []
    max_d = 0
    i = 0
    n = len(text)
    while i < n:
      two = text[i : i + 2]
      if two == "{{":
        stack.append("}}")
        if len(stack) > max_d:
          max_d = len(stack)
        i += 2
        continue
      if two in ("${", "#{"):
        stack.append("}")
        if len(stack) > max_d:
          max_d = len(stack)
        i += 2
        continue
      if two == "<%":
        stack.append("%>")
        if len(stack) > max_d:
          max_d = len(stack)
        i += 2
        continue
      if two == "{%":
        stack.append("%}")
        if len(stack) > max_d:
          max_d = len(stack)
        i += 2
        continue
      if stack:
        expected = stack[-1]
        if len(expected) == 2 and two == expected:
          stack.pop()
          i += 2
          continue
        if len(expected) == 1 and text[i] == expected:
          stack.pop()
          i += 1
          continue
      i += 1
    return max_d

  def _sanitize_string(self, text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\x00", "").replace("%00", "").replace("%2500", "")
    # Bounded fixed-point neutralization of arbitrarily nested template delimiters
    for _ in range(16):
      nxt = _INNER_EXPANSION_RE.sub("[TEMPLATE_TOKEN_STRIPPED]", cleaned)
      if nxt == cleaned:
        break
      cleaned = nxt
    cleaned = _PATH_TRAVERSAL_RE.sub("[PATH_TRAVERSAL_STRIPPED]", cleaned)
    # Also strip URL-encoded traversal sequences that survived raw regex
    cleaned = re.sub(
        r"(?:%252e%252e(?:%252f|%255c)|%2e%2e(?:%2f|%5c))",
        "[PATH_TRAVERSAL_STRIPPED]",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _SSRF_METADATA_RE.sub("[METADATA_STRIPPED]", cleaned)
    cleaned = _DANGEROUS_SCHEMES_RE.sub(
        lambda m: m.group(0)[: len(m.group(0)) - len(m.group(1))] + "[SCHEME_STRIPPED]://",
        cleaned,
    )
    return cleaned

  def sanitize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
    return self._sanitize_node(arguments)

  def _sanitize_node(self, node: Any) -> Any:
    if isinstance(node, str):
      return self._sanitize_string(node)
    if isinstance(node, dict):
      return {str(k): self._sanitize_node(v) for k, v in node.items()}
    if isinstance(node, list):
      return [self._sanitize_node(x) for x in node]
    if isinstance(node, tuple):
      return [self._sanitize_node(x) for x in node]
    return node

  def _flatten_string_leaves(
      self,
      node: Any,
      prefix: str = "",
      out: list[tuple[str, str]] | None = None,
  ) -> list[tuple[str, str]]:
    if out is None:
      out = []
    if isinstance(node, str):
      out.append((prefix or "value", node))
    elif isinstance(node, dict):
      for k, v in node.items():
        sub_key = f"{prefix}.{k}" if prefix else str(k)
        self._flatten_string_leaves(v, sub_key, out)
    elif isinstance(node, (list, tuple)):
      for idx, item in enumerate(node):
        sub_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
        self._flatten_string_leaves(item, sub_key, out)
    return out

  def detect_violations(
      self,
      key: str,
      value: Any,
      template_type: str = "prompt",
      *,
      max_depth: int | None = None,
  ) -> list[TemplateShieldViolation]:
    del template_type
    if not isinstance(value, str):
      return []

    eff_max_depth = self.max_depth if max_depth is None else max(0, int(max_depth))
    violations: list[TemplateShieldViolation] = []
    text = value
    norm = _normalize_decoded_text(text)
    sanitized = self._sanitize_string(text)

    # 1. Null-byte truncation / C-string smuggling
    if "\x00" in text or "\x00" in norm or "%00" in text.lower() or "%2500" in text.lower():
      violations.append(
          TemplateShieldViolation(
              violation_code="NULL_BYTE_INJECTION",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' contains null-byte injection sequence.",
              level="error",
              depth=0,
          )
      )

    # 2. Recursive template expansion depth
    depth = max(self.calculate_expansion_depth(text), self.calculate_expansion_depth(norm))
    if depth > eff_max_depth or _NESTED_EXPANSION_RE.search(text) or _NESTED_EXPANSION_RE.search(norm):
      violations.append(
          TemplateShieldViolation(
              violation_code="RECURSIVE_EXPANSION_DEPTH",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=(
                  f"Argument '{key}' contains template expansion depth {depth} "
                  f"exceeding maximum allowed depth {eff_max_depth}."
              ),
              level="error",
              depth=depth,
          )
      )

    # 3. Server-Side Template Injection (SSTI) gadgets
    if _SSTI_EXPRESSIONS_RE.search(text) or _SSTI_EXPRESSIONS_RE.search(norm):
      violations.append(
          TemplateShieldViolation(
              violation_code="SSTI_EXPRESSION",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' contains server-side template injection (SSTI) expression.",
              level="error",
              depth=depth,
          )
      )

    # 4. Path traversal & sensitive host targets (including double-encoded and fullwidth NFKC)
    if _PATH_TRAVERSAL_RE.search(text) or _PATH_TRAVERSAL_RE.search(norm):
      violations.append(
          TemplateShieldViolation(
              violation_code="PATH_TRAVERSAL_IN_TEMPLATE",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' contains path traversal sequence targeting sensitive files.",
              level="error",
              depth=depth,
          )
      )

    # 5. Cloud metadata SSRF hosts
    if _SSRF_METADATA_RE.search(text) or _SSRF_METADATA_RE.search(norm):
      violations.append(
          TemplateShieldViolation(
              violation_code="SSRF_METADATA_IN_TEMPLATE",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' contains cloud metadata IP address or SSRF endpoint.",
              level="error",
              depth=depth,
          )
      )

    # 6. Unauthorized / dangerous URI schemes
    if _DANGEROUS_SCHEMES_RE.search(text) or _DANGEROUS_SCHEMES_RE.search(norm):
      violations.append(
          TemplateShieldViolation(
              violation_code="UNAUTHORIZED_SCHEME_IN_TEMPLATE",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' injects an unauthorized URI scheme into template parameters.",
              level="error",
              depth=depth,
          )
      )

    # 7. Environment variable & secret exfiltration directives
    lower_norm = norm.lower()
    exfil_matched = False
    for kw in _ENV_SECRET_KEYWORDS:
      if kw in lower_norm:
        if any(
            marker in norm
            for marker in (
                "${",
                "{{",
                "#{",
                "$",
                "env.",
                "env:",
                "env[",
                "environ",
                "process.env",
            )
        ):
          violations.append(
              TemplateShieldViolation(
                  violation_code="SECRET_EXFILTRATION_PAYLOAD",
                  argument_key=key,
                  raw_value=text,
                  sanitized_value=sanitized,
                  message=f"Argument '{key}' contains credential exfiltration directive targeting '{kw}'.",
                  level="error",
                  depth=depth,
              )
          )
          exfil_matched = True
          break
    if not exfil_matched and (_EXFIL_BEACON_URL_RE.search(text) or _EXFIL_BEACON_URL_RE.search(norm)):
      violations.append(
          TemplateShieldViolation(
              violation_code="SECRET_EXFILTRATION_PAYLOAD",
              argument_key=key,
              raw_value=text,
              sanitized_value=sanitized,
              message=f"Argument '{key}' embeds template interpolation inside an external exfiltration URL.",
              level="error",
              depth=depth,
          )
      )

    return violations

  def _analyze_argument_graph(
      self,
      flat_leaves: list[tuple[str, str]],
      eff_max_depth: int,
  ) -> list[TemplateShieldViolation]:
    """Detects cross-argument expansion cycles (a -> b -> a) and multi-hop transitive expansion chains."""
    top_values: dict[str, str] = {}
    for k, v in flat_leaves:
      if "." not in k and "[" not in k:
        top_values[k] = v
    if len(top_values) < 2:
      return []

    adj: dict[str, list[str]] = {}
    for k, v in top_values.items():
      refs: list[str] = []
      for m in _ARG_REF_TOKEN_RE.finditer(v):
        ref_name = next((g for g in m.groups() if g), "")
        root_ref = ref_name.split(".", 1)[0]
        if root_ref in top_values and root_ref not in refs:
          refs.append(root_ref)
      adj[k] = refs

    violations: list[TemplateShieldViolation] = []
    # 3-color DFS: 0 = unvisited, 1 = visiting (on stack), 2 = visited
    color: dict[str, int] = {k: 0 for k in top_values}
    memo_hops: dict[str, int] = {}
    reported_cycles: set[str] = set()

    def dfs(node: str, stack_path: list[str]) -> int:
      color[node] = 1
      max_child_hops = 0
      for nxt in adj.get(node, []):
        if color[nxt] == 1:
          cycle_chain = " -> ".join([*stack_path[stack_path.index(nxt) :], nxt])
          if node not in reported_cycles:
            reported_cycles.add(node)
            violations.append(
                TemplateShieldViolation(
                    violation_code="CROSS_ARG_EXPANSION_CYCLE",
                    argument_key=node,
                    raw_value=top_values[node],
                    sanitized_value=self._sanitize_string(top_values[node]),
                    message=(
                        f"Cross-argument recursive template cycle detected ({cycle_chain}), "
                        "enabling exponential prompt amplification."
                    ),
                    level="error",
                    depth=len(stack_path) + 1,
                )
            )
          continue
        child_hops = memo_hops[nxt] if color[nxt] == 2 else dfs(nxt, [*stack_path, nxt])
        if 1 + child_hops > max_child_hops:
          max_child_hops = 1 + child_hops
      color[node] = 2
      memo_hops[node] = max_child_hops
      return max_child_hops

    for k in top_values:
      if color[k] == 0:
        dfs(k, [k])

    for k, hops in memo_hops.items():
      if k in reported_cycles:
        continue
      # hops >= 2 means k -> b -> c (at least 2 cross-argument dereferences)
      if hops > max(1, eff_max_depth):
        violations.append(
            TemplateShieldViolation(
                violation_code="TRANSITIVE_EXPANSION_DEPTH",
                argument_key=k,
                raw_value=top_values[k],
                sanitized_value=self._sanitize_string(top_values[k]),
                message=(
                    f"Argument '{k}' triggers a {hops}-hop cross-argument expansion chain "
                    f"exceeding max allowed depth {eff_max_depth}."
                ),
                level="error",
                depth=hops,
            )
        )
    return violations

  def _validate_uri_template_binding(
      self,
      uri_template: str,
      arguments: dict[str, Any],
  ) -> list[TemplateShieldViolation]:
    """Validates RFC 6570 URI template placeholders against segment breakout and query smuggling."""
    if not uri_template or "{" not in uri_template:
      return []
    violations: list[TemplateShieldViolation] = []
    for m in _URI_TEMPLATE_EXPR_RE.finditer(uri_template):
      op = m.group(1)
      raw_vars = m.group(2)
      # Reserved (+) and fragment (#) operators intentionally allow slashes; single-segment operators do not.
      if op in ("+", "#"):
        continue
      for spec in raw_vars.split(","):
        var_name = spec.strip().rstrip("*").split(":", 1)[0].strip()
        if not var_name or var_name not in arguments:
          continue
        val = arguments[var_name]
        if not isinstance(val, str):
          continue
        norm = _normalize_decoded_text(val)
        # Single-segment placeholders ({var}, {/var}, {.var}, {;var}, {?var}, {&var}) must not smuggle /, \, ?, #, &, or ..
        forbidden = ("/", "\\", "?", "#", "&") if op in ("", "/", ".") else ("/", "\\", "#")
        if any(ch in val or ch in norm for ch in forbidden) or ".." in norm:
          violations.append(
              TemplateShieldViolation(
                  violation_code="URI_TEMPLATE_SEGMENT_BREAKOUT",
                  argument_key=var_name,
                  raw_value=val,
                  sanitized_value=urllib.parse.quote(self._sanitize_string(val), safe=""),
                  message=(
                      f"Parameter '{var_name}' breaks out of RFC 6570 URI template placeholder "
                      f"'{{{op}{var_name}}}' via unencoded delimiter or dot-segment smuggling."
                  ),
                  level="error",
                  depth=0,
              )
          )
    return violations

  def audit_arguments(
      self,
      arguments: dict[str, Any],
      template_type: str = "prompt",
      uri_template: str = "",
      max_depth: int | None = None,
      block_exfiltration: bool | None = None,
  ) -> TemplateArgumentDecision:
    if not isinstance(arguments, dict):
      arguments = {}
    if not self.enabled:
      return TemplateArgumentDecision(
          allowed=True,
          template_type=template_type,
          uri_template=uri_template,
          arguments_before=arguments,
          sanitized_arguments=dict(arguments),
      )

    eff_max_depth = self.max_depth if max_depth is None else max(0, int(max_depth))
    eff_block = self.block_exfiltration if block_exfiltration is None else bool(block_exfiltration)

    flat_leaves = self._flatten_string_leaves(arguments)
    violations: list[TemplateShieldViolation] = []
    for path_key, raw_str in flat_leaves:
      violations.extend(
          self.detect_violations(
              path_key,
              raw_str,
              template_type,
              max_depth=eff_max_depth,
          )
      )

    violations.extend(self._analyze_argument_graph(flat_leaves, eff_max_depth))
    if uri_template:
      violations.extend(self._validate_uri_template_binding(uri_template, arguments))

    sanitized = self.sanitize_arguments(arguments)
    allowed = (len(violations) == 0) if eff_block else True
    return TemplateArgumentDecision(
        allowed=allowed,
        template_type=template_type,
        uri_template=uri_template,
        violations=violations,
        arguments_before=arguments,
        sanitized_arguments=sanitized,
    )

  def audit_prompt_arguments(
      self,
      prompt_name: str,
      arguments: dict[str, Any],
      max_depth: int | None = None,
      block_exfiltration: bool | None = None,
  ) -> TemplateArgumentDecision:
    return self.audit_arguments(
        arguments=arguments,
        template_type="prompt",
        uri_template=prompt_name,
        max_depth=max_depth,
        block_exfiltration=block_exfiltration,
    )

  def audit_resource_template(
      self,
      uri_template: str,
      parameters: dict[str, Any],
      max_depth: int | None = None,
      block_exfiltration: bool | None = None,
  ) -> TemplateArgumentDecision:
    return self.audit_arguments(
        arguments=parameters,
        template_type="resource",
        uri_template=uri_template,
        max_depth=max_depth,
        block_exfiltration=block_exfiltration,
    )


def shield_template_arguments(
    template_type: str,
    arguments: dict[str, Any],
    uri_template: str = "",
    max_depth: int = 1,
    block_exfiltration: bool = True,
) -> TemplateArgumentDecision:
  """Audits and neutralizes recursive template expansion, URI breakout, path traversal, and secret leaks."""
  shield = TemplateArgumentShield(
      enabled=True,
      max_depth=max_depth,
      block_exfiltration=block_exfiltration,
  )
  return shield.audit_arguments(
      arguments=arguments,
      template_type=template_type,
      uri_template=uri_template,
      max_depth=max_depth,
      block_exfiltration=block_exfiltration,
  )
