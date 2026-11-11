<div align="center">

# ⚡ `mcp-slim`

**Zero-Config MCP Context Optimizer & Real-Time Secret Firewall**  
*Cut Model Context Protocol (MCP) tool schema & output tokens by **50–80%** and block leaked secrets before they reach your LLM context window.*

[![CI](https://github.com/SalyyS1/mcp-slim/actions/workflows/ci.yml/badge.svg)](https://github.com/SalyyS1/mcp-slim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/SalyyS1/mcp-slim)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-success)](https://github.com/SalyyS1/mcp-slim/blob/main/pyproject.toml)
[![MCP Protocol](https://img.shields.io/badge/MCP-JSON--RPC%202.0-8A2BE2)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SalyyS1/mcp-slim/blob/main/LICENSE)

</div>

---

## 🔥 Why `mcp-slim`?

When connecting **Claude Code**, **Cursor**, **Windsurf**, or **Claude Desktop** to 3–5 MCP servers (GitHub, Postgres, Slack, Filesystem, Jira), two critical problems happen on **every single turn**:

1. **Massive Token Bloat (25,000–60,000+ tokens/turn):** Auto-generated Pydantic/Zod `tools/list` JSON schemas include `$schema` declarations, `title` echoes, verbose multi-paragraph descriptions, `anyOf: [{type: "string"}, {type: "null"}]` wrappers, and uncompacted `tools/call` JSON payloads filled with `null`/`[]`/`{}` fields.
2. **Silent Credential Leaks into Cloud LLMs:** MCP tools reading logs, `.env` files, database rows, or git remotes frequently return raw API keys (`sk-ant-...`, `ghp_...`, `AKIA...`, `xoxb-...`, JWTs, or database URLs with passwords) straight into the LLM context window.

**`mcp-slim` is a zero-dependency stdio JSON-RPC 2.0 proxy** that sits transparently between your MCP client and any MCP server (`npx`, `uvx`, `python`, `docker`) with **<1 ms overhead**.

```mermaid
flowchart LR
    Client["Claude Code / Cursor / Windsurf"]
    Proxy["⚡ mcp-slim Proxy\n• Schema Slimmer (-52% tokens)\n• Output Compactor (-43% tokens)\n• Bidirectional Secret Guard"]
    Server["Upstream MCP Server\n(GitHub / Postgres / Filesystem / Slack)"]

    Client <-->|"Compact & Redacted JSON-RPC"| Proxy
    Proxy <-->|"Raw Verbose JSON-RPC"| Server
```

---

## 📊 Real Benchmark (`mcp-slim benchmark`)

Run the built-in reproducible benchmark anytime:

```bash
$ mcp-slim benchmark
mcp-slim Benchmark Report
================================================================
Stage                    |     Before |      After |    Reduction
----------------------------------------------------------------
tools/list Schema        |        572 |        277 |       -51.6%
tools/call JSON Output   |        207 |        118 |       -43.0%
----------------------------------------------------------------
Total (1 Turn)           |        779 |        395 |       -49.3%
================================================================
Secrets intercepted & redacted: 1
```

---

## 🚀 Quick Start (30 Seconds)

### 1. Install (Zero third-party dependencies!)

```bash
pip install git+https://github.com/SalyyS1/mcp-slim.git
# or with uv:
uv tool install git+https://github.com/SalyyS1/mcp-slim.git
```

### 2. Wrap Your Existing MCP Config in 1 Command

Automatically wrap all servers in your Claude Desktop, Claude Code, Cursor, or Windsurf config file:

```bash
# Preview wrapped config:
mcp-slim wrap ~/.cursor/mcp.json

# Apply in-place:
mcp-slim wrap -i ~/.cursor/mcp.json
```

**Before (`mcp.json`):**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

**After (`mcp-slim wrap -i mcp.json`):**
```json
{
  "mcpServers": {
    "github": {
      "command": "mcp-slim",
      "args": ["run", "--", "npx", "-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

---

## 🛠️ What `mcp-slim` Optimizes

### 1. `SchemaSlimmer` (`tools/list`)
- **Strips JSON Schema boilerplate:** Removes `$schema`, `$id`, `title`, `additionalProperties`, `examples`, and `default` metadata that LLMs do not need to invoke tools.
- **Simplifies nullable unions:** Rewrites `{"anyOf": [{"type": "string"}, {"type": "null"}]}` into `{"type": "string"}` while preserving `required` arrays.
- **Prunes self-repeating descriptions:** Drops redundant parameter descriptions that merely repeat the parameter name (e.g. `repo`: `"The repo"`, `branch`: `"The branch"`).
- **Condenses multi-sentence essays:** Keeps the high-signal first sentence of long tool and parameter descriptions (configurable via `--max-desc` and `--max-param-desc`).
- **Selective Tool Filtering:** Expose only the tools you actually use with `--allow-tools` or hide noisy tools with `--deny-tools`.

```bash
mcp-slim run --allow-tools search_repositories,get_file_contents -- npx -y @modelcontextprotocol/server-github
```

### 2. `OutputSlimmer` (`tools/call`)
- **JSON Compaction & Null Pruning:** Parses JSON strings returned inside `content[].text`, removes keys with `null`, `""`, `[]`, or `{}` values, and re-serializes without indentation whitespace.
- **ANSI Escape Stripping:** Removes terminal color codes from CLI tool outputs.
- **Runaway Array/String Truncation:** Caps oversized arrays and multi-megabyte log blobs before they overflow the context window.

### 3. `SecretGuard` (Bidirectional Firewall)
Scans both outgoing `tools/call` arguments and incoming `tools/call` results in real time:

| Provider / Pattern | Redaction Tag |
| :--- | :--- |
| Anthropic API Keys (`sk-ant-api03-...`) | `[REDACTED:ANTHROPIC_KEY]` |
| OpenAI Project & Service Keys (`sk-proj-...`) | `[REDACTED:OPENAI_KEY]` |
| GitHub PATs & Fine-Grained Tokens (`ghp_...`, `github_pat_...`) | `[REDACTED:GITHUB_TOKEN]` |
| AWS Access Key IDs (`AKIA...`, `ASIA...`) | `[REDACTED:AWS_ACCESS_KEY]` |
| Google Cloud API Keys (`AIza...`) | `[REDACTED:GCP_API_KEY]` |
| Slack Bot & User Tokens (`xoxb-...`, `xoxp-...`) | `[REDACTED:SLACK_TOKEN]` |
| Stripe Live/Test Keys (`sk_live_...`, `rk_live_...`) | `[REDACTED:STRIPE_KEY]` |
| HuggingFace Tokens (`hf_...`) | `[REDACTED:HF_TOKEN]` |
| PEM / OpenSSH Private Keys (`-----BEGIN ... PRIVATE KEY-----`) | `[REDACTED:PRIVATE_KEY_BLOCK]` |
| JSON Web Tokens (`eyJ...`) | `[REDACTED:JWT_TOKEN]` |
| Database & HTTP URL Passwords (`postgres://user:pass@host`) | `[REDACTED:URL_SECRET]` |
| `.env` Secret Assignments (`API_KEY=...`, `DB_PASSWORD=...`) | `[REDACTED:ENV_SECRET]` |

---

## 🧪 Running Tests

`mcp-slim` uses pure Python 3.10+ standard library and runs with `unittest` or `pytest`:

```bash
git clone https://github.com/SalyyS1/mcp-slim.git
cd mcp-slim
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## 📄 License

Released under the [MIT License](./LICENSE). Contributions, issues, and feature requests are welcome!
