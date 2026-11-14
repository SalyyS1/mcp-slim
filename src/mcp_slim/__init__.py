"""mcp-slim: Zero-dependency MCP JSON-RPC 2.0 proxy for token compression, lazy discovery, and secret redaction."""

__version__ = "0.9.0"

from .guard import RedactionReport, SecretGuard
from .proxy import MCPProxy
from .server import handle_server_rpc, run_serve
from .slimmer import (
    BranchContract,
    LazyToolBroker,
    OutputSlimmer,
    SchemaSlimmer,
    SlimStats,
    UnionContract,
    audit_tools_manifest,
    build_audit_sarif,
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
    "UnionContract",
    "audit_tools_manifest",
    "build_audit_sarif",
    "derive_server_label",
    "estimate_tokens",
    "handle_server_rpc",
    "run_serve",
]
