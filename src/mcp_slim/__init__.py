"""mcp-slim: Zero-config MCP proxy that cuts token usage by 60-80% and redacts leaked secrets."""

__version__ = "0.1.0"

from .guard import SecretGuard, RedactionReport
from .slimmer import SchemaSlimmer, OutputSlimmer, SlimStats

__all__ = [
    "__version__",
    "SecretGuard",
    "RedactionReport",
    "SchemaSlimmer",
    "OutputSlimmer",
    "SlimStats",
]
