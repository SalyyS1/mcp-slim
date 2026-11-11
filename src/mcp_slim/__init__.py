"""mcp-slim: Zero-dependency MCP JSON-RPC 2.0 proxy for token compression, lazy discovery, and secret redaction."""

from .guard import RedactionReport, SecretGuard
from .proxy import MCPProxy
from .slimmer import LazyToolBroker, OutputSlimmer, SchemaSlimmer, SlimStats, estimate_tokens

__all__ = [
    "LazyToolBroker",
    "MCPProxy",
    "OutputSlimmer",
    "RedactionReport",
    "SchemaSlimmer",
    "SecretGuard",
    "SlimStats",
    "estimate_tokens",
]

__version__ = "0.3.0"
