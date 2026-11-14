"""mcp-slim: Zero-dependency MCP JSON-RPC 2.0 proxy for token compression, lazy discovery, and secret redaction."""

__version__ = "1.0.0"

from .guard import RedactionReport, SecretGuard
from .proxy import MCPProxy
from .server import handle_server_rpc, run_serve
from .slimmer import (
    BranchContract,
    LazyToolBroker,
    OutputSlimmer,
    SchemaSlimmer,
    SlimStats,
    ToolMutation,
    ToolPinGuard,
    UnionContract,
    audit_tools_manifest,
    build_audit_sarif,
    compute_tool_fingerprint,
    diff_tools_manifests,
    estimate_tokens,
)
from .telemetry import OTelSpanExporter, StatsLedger, derive_server_label

__all__ = [
    "BranchContract",
    "LazyToolBroker",
    "MCPProxy",
    "OTelSpanExporter",
    "OutputSlimmer",
    "RedactionReport",
    "SchemaSlimmer",
    "SecretGuard",
    "SlimStats",
    "StatsLedger",
    "ToolMutation",
    "ToolPinGuard",
    "UnionContract",
    "audit_tools_manifest",
    "build_audit_sarif",
    "compute_tool_fingerprint",
    "derive_server_label",
    "diff_tools_manifests",
    "estimate_tokens",
    "handle_server_rpc",
    "run_serve",
]
