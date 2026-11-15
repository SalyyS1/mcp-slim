"""mcp-slim: Zero-dependency MCP JSON-RPC 2.0 proxy for token compression, lazy discovery, and secret redaction."""

__version__ = "1.3.0"

from .guard import (
    RedactionReport,
    SecretGuard,
    StreamSanitizer,
    ToolCallStreamSanitizer,
)
from .proxy import MCPProxy
from .server import handle_server_rpc, run_serve
from .slimmer import (
    BranchContract,
    ElicitationGuard,
    LazyToolBroker,
    OutputSlimmer,
    ResourceUriGuard,
    RootsPrivacyGuard,
    SchemaSlimmer,
    SlimStats,
    TokenWindowGovernor,
    ToolMutation,
    ToolPinGuard,
    UnionContract,
    WindowBudgetDecision,
    audit_elicitation_request,
    audit_resources_manifest,
    audit_tools_manifest,
    build_audit_sarif,
    canonicalize_resource_uri,
    compute_tool_fingerprint,
    detect_tool_shadowing,
    diff_tools_manifests,
    estimate_tokens,
)
from .telemetry import OTelSpanExporter, StatsLedger, derive_server_label

__all__ = [
    "BranchContract",
    "ElicitationGuard",
    "LazyToolBroker",
    "MCPProxy",
    "OTelSpanExporter",
    "OutputSlimmer",
    "RedactionReport",
    "ResourceUriGuard",
    "RootsPrivacyGuard",
    "SchemaSlimmer",
    "SecretGuard",
    "SlimStats",
    "StatsLedger",
    "StreamSanitizer",
    "TokenWindowGovernor",
    "ToolCallStreamSanitizer",
    "ToolMutation",
    "ToolPinGuard",
    "UnionContract",
    "WindowBudgetDecision",
    "audit_elicitation_request",
    "audit_resources_manifest",
    "audit_tools_manifest",
    "build_audit_sarif",
    "canonicalize_resource_uri",
    "compute_tool_fingerprint",
    "derive_server_label",
    "detect_tool_shadowing",
    "diff_tools_manifests",
    "estimate_tokens",
    "handle_server_rpc",
    "run_serve",
]
