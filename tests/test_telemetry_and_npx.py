"""Unit and integration tests for v0.6.0 telemetry ledger, stats CLI, Node/NPX launcher, and CI action."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import mcp_slim
from mcp_slim.cli import SAMPLE_BENCHMARK_OUTPUT, SAMPLE_BENCHMARK_TOOLS, main, wrap_config
from mcp_slim.proxy import MCPProxy
from mcp_slim.slimmer import SlimStats
from mcp_slim.telemetry import StatsLedger, derive_server_label

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestServerLabelDerivation(unittest.TestCase):
  def test_derives_clean_labels_across_launchers(self) -> None:
    self.assertEqual(
        derive_server_label(["npx", "-y", "@modelcontextprotocol/server-github"]),
        "@modelcontextprotocol/server-github",
    )
    self.assertEqual(
        derive_server_label(["npx", "--yes", "@playwright/mcp@latest", "--headless"]),
        "@playwright/mcp",
    )
    self.assertEqual(
        derive_server_label(["uvx", "mcp-server-git", "--repository", "."]),
        "mcp-server-git",
    )
    self.assertEqual(
        derive_server_label([
            "uvx",
            "--from",
            "git+https://github.com/org/mcp-postgres.git",
            "mcp-postgres",
        ]),
        "mcp-postgres",
    )
    self.assertEqual(
        derive_server_label(["python3", "-m", "mcp_server_sqlite", "--db", "app.db"]),
        "mcp_server_sqlite",
    )
    self.assertEqual(
        derive_server_label(["python3", "-c", "print(1)"]),
        "python3 -c",
    )
    self.assertEqual(
        derive_server_label(["node", "/opt/mcp-github/dist/index.js"]),
        "mcp-github",
    )
    self.assertEqual(
        derive_server_label([
            "docker",
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server:latest",
        ]),
        "ghcr.io/github/github-mcp-server",
    )
    self.assertEqual(
        derive_server_label(["/usr/local/bin/mcp-server-git"]),
        "mcp-server-git",
    )


class TestStatsLedgerAndCLI(unittest.TestCase):
  def test_version_bumped_to_0_6_0(self) -> None:
    self.assertEqual(mcp_slim.__version__, "0.6.0")

  def test_ledger_accumulation_and_per_server_breakdown(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      stats_path = Path(tmpdir) / "stats.json"
      ledger = StatsLedger(path=stats_path)

      s1 = SlimStats(
          schema_tokens_before=4000,
          schema_tokens_after=2000,
          output_tokens_before=1000,
          output_tokens_after=550,
          wire_bytes_before=20000,
          wire_bytes_after=10200,
          refs_inlined=3,
          args_coerced=2,
          cache_hits=4,
          blocked_calls=1,
      )
      ledger.record_session(
          s1,
          command=["npx", "-y", "@modelcontextprotocol/server-github"],
          secrets_redacted=2,
      )

      s2 = SlimStats(
          schema_tokens_before=1800,
          schema_tokens_after=920,
          output_tokens_before=600,
          output_tokens_after=300,
          wire_bytes_before=9600,
          wire_bytes_after=4800,
          refs_inlined=1,
          args_coerced=1,
          cache_hits=2,
          blocked_calls=0,
      )
      ledger.record_session(
          s2,
          command=["uvx", "mcp-server-git", "--repository", "."],
          secrets_redacted=1,
      )

      data = ledger.load()
      self.assertEqual(data["sessions"], 2)
      self.assertEqual(data["schema_tokens_before"], 5800)
      self.assertEqual(data["schema_tokens_after"], 2920)
      self.assertEqual(data["output_tokens_before"], 1600)
      self.assertEqual(data["output_tokens_after"], 850)
      self.assertEqual(data["wire_bytes_before"], 29600)
      self.assertEqual(data["wire_bytes_after"], 15000)
      self.assertEqual(data["refs_inlined"], 4)
      self.assertEqual(data["coerced_arguments"], 3)
      self.assertEqual(data["cache_hits"], 6)
      self.assertEqual(data["blocked_calls"], 1)
      self.assertEqual(data["secrets_redacted"], 3)
      self.assertTrue(data["updated_at"].endswith("Z"))

      servers = data["servers"]
      self.assertIn("@modelcontextprotocol/server-github", servers)
      self.assertIn("mcp-server-git", servers)
      self.assertEqual(servers["@modelcontextprotocol/server-github"]["sessions"], 1)
      self.assertEqual(servers["@modelcontextprotocol/server-github"]["secrets_redacted"], 2)
      self.assertEqual(servers["mcp-server-git"]["refs_inlined"], 1)

      dashboard = ledger.format_dashboard()
      self.assertIn("Claude Opus 4 ($15.00/1M)", dashboard)
      self.assertIn("Claude Sonnet 4 ($3.00/1M)", dashboard)
      self.assertIn("GPT-4o ($2.50/1M)", dashboard)
      self.assertIn("Gemini 2.5 Pro ($1.25/1M)", dashboard)
      # 2 sessions recorded -> should display 30-day projection at 100 turns/day
      self.assertIn("30D @ 100 TURNS/DAY", dashboard)
      self.assertIn("KiB", dashboard)
      self.assertIn("SERVER", dashboard)
      self.assertIn("mcp-server-git", dashboard)

  def test_corruption_recovery_and_non_crashing_writes(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      stats_path = Path(tmpdir) / "corrupt_stats.json"
      stats_path.write_text("{corrupted_json: [1, 2, ", encoding="utf-8")

      ledger = StatsLedger(path=stats_path)
      recovered = ledger.load()
      self.assertEqual(recovered["sessions"], 0)
      self.assertEqual(recovered["servers"], {})

      # Recording after corruption overwrites atomically with clean JSON
      ledger.record_session(
          SlimStats(schema_tokens_before=500, schema_tokens_after=250),
          command=["mcp-server-git"],
      )
      after = json.loads(stats_path.read_text(encoding="utf-8"))
      self.assertEqual(after["sessions"], 1)
      self.assertEqual(after["schema_tokens_before"], 500)

      # Non-dict JSON root also falls back cleanly
      stats_path.write_text('["unexpected", "list"]', encoding="utf-8")
      self.assertEqual(ledger.load()["sessions"], 0)

      # Unwritable path (parent is a regular file) never raises an exception
      blocker_file = Path(tmpdir) / "blocker"
      blocker_file.write_text("not a dir", encoding="utf-8")
      bad_ledger = StatsLedger(path=blocker_file / "stats.json")
      res = bad_ledger.record_session(
          SlimStats(schema_tokens_before=100, schema_tokens_after=50)
      )
      self.assertIsInstance(res, dict)

  def test_no_stats_flag_and_env_var_disable_recording(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      env_path = Path(tmpdir) / "env_disabled.json"
      with mock.patch.dict(
          os.environ,
          {"MCP_SLIM_STATS_PATH": str(env_path), "MCP_SLIM_NO_STATS": "1"},
      ):
        ledger = StatsLedger()
        self.assertFalse(ledger.enabled)
        ledger.record_session(
            SlimStats(schema_tokens_before=400, schema_tokens_after=200)
        )
        self.assertFalse(env_path.exists())

      proxy_disabled = MCPProxy(
          no_stats=True,
          stats_ledger=StatsLedger(path=env_path, enabled=False),
      )
      self.assertIsNone(proxy_disabled.record_telemetry(command=["mcp-server-git"]))
      self.assertFalse(env_path.exists())

  def test_stats_cli_json_and_reset_and_live_stdio_session(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      stats_path = Path(tmpdir) / "live_stats.json"
      mock_server = (
          "import sys, json\n"
          "tools = " + repr(SAMPLE_BENCHMARK_TOOLS) + "\n"
          "out = " + repr(SAMPLE_BENCHMARK_OUTPUT) + "\n"
          "for line in sys.stdin:\n"
          "  msg = json.loads(line)\n"
          "  m = msg.get('method')\n"
          "  if m == 'tools/list':\n"
          "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'tools':tools}}), flush=True)\n"
          "  elif m == 'tools/call':\n"
          "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':out}), flush=True)\n"
          "    break\n"
      )
      req_list = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
      req_call = json.dumps({
          "jsonrpc": "2.0",
          "id": 2,
          "method": "tools/call",
          "params": {"name": "search_repositories", "arguments": {"query": "mcp", "page": "2"}},
      }) + "\n"

      def _run_interactive_session(extra_flags: list[str]) -> str:
        with subprocess.Popen(
            [sys.executable, "-m", "mcp_slim.cli", "run", "--quiet", *extra_flags, "--", sys.executable, "-c", mock_server],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        ) as proc:
          assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
          proc.stdin.write(req_list)
          proc.stdin.flush()
          list_line = proc.stdout.readline()
          proc.stdin.write(req_call)
          proc.stdin.flush()
          call_line = proc.stdout.readline()
          proc.stdin.close()
          rc = proc.wait(timeout=10)
          proc.stdout.close()
          proc.stderr.close()
          self.assertEqual(rc, 0)
          return list_line + call_line

      env = dict(os.environ)
      env["PYTHONPATH"] = str(REPO_ROOT / "src")
      env["MCP_SLIM_STATS_PATH"] = str(stats_path)
      env.pop("MCP_SLIM_NO_STATS", None)

      # 1. Run with --no-stats first: stats file must NOT be created
      out_nostats = _run_interactive_session(["--no-stats"])
      self.assertIn("search_repositories", out_nostats)
      self.assertFalse(stats_path.exists())

      # 2. Run normal proxy session: stats file IS created and populated
      out_live = _run_interactive_session([])
      self.assertIn("[REDACTED:GITHUB_TOKEN]", out_live)
      self.assertTrue(stats_path.exists())

      # 3. Query via `mcp-slim stats --json --stats-file ...`
      buf_json = io.StringIO()
      with contextlib.redirect_stdout(buf_json):
        rc_json = main(["stats", "--json", "--stats-file", str(stats_path)])
      self.assertEqual(rc_json, 0)
      parsed_summary = json.loads(buf_json.getvalue())
      self.assertEqual(parsed_summary["sessions"], 1)
      self.assertGreater(parsed_summary["schema_tokens_before"], parsed_summary["schema_tokens_after"])
      self.assertGreater(parsed_summary["output_tokens_before"], parsed_summary["output_tokens_after"])
      self.assertGreater(parsed_summary["wire_bytes_before"], parsed_summary["wire_bytes_after"])
      self.assertGreaterEqual(parsed_summary["refs_inlined"], 1)
      self.assertGreaterEqual(parsed_summary["coerced_arguments"], 1)
      self.assertGreaterEqual(parsed_summary["secrets_redacted"], 1)
      self.assertGreater(parsed_summary["estimated_input_cost_saved_usd"]["claude_opus_4"], 0.0)

      # 4. Render ASCII dashboard via `mcp-slim stats --stats-file ...`
      buf_ascii = io.StringIO()
      with contextlib.redirect_stdout(buf_ascii):
        rc_ascii = main(["stats", "--stats-file", str(stats_path)])
      self.assertEqual(rc_ascii, 0)
      self.assertIn("mcp-slim Cumulative Token & Cost Savings Dashboard", buf_ascii.getvalue())
      self.assertIn("30D @ 100 TURNS/DAY", buf_ascii.getvalue())

      # 5. Reset via `mcp-slim stats --reset --json --stats-file ...`
      buf_reset = io.StringIO()
      with contextlib.redirect_stdout(buf_reset):
        rc_reset = main(["stats", "--reset", "--json", "--stats-file", str(stats_path)])
      self.assertEqual(rc_reset, 0)
      reset_summary = json.loads(buf_reset.getvalue())
      self.assertEqual(reset_summary["sessions"], 0)
      self.assertEqual(reset_summary["tokens_saved"], 0)
      self.assertEqual(reset_summary["servers"], {})


class TestNodeLauncherActionAndPlayground(unittest.TestCase):
  def test_package_json_and_node_bin_launcher_benchmark(self) -> None:
    pkg_path = REPO_ROOT / "package.json"
    bin_path = REPO_ROOT / "bin" / "mcp-slim.js"
    self.assertTrue(pkg_path.exists())
    self.assertTrue(bin_path.exists())

    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    self.assertEqual(pkg["name"], "mcp-slim")
    self.assertEqual(pkg["version"], "0.6.0")
    self.assertEqual(pkg["bin"]["mcp-slim"], "bin/mcp-slim.js")
    self.assertFalse(pkg.get("dependencies"))

    node_bin = shutil.which("node")
    self.assertIsNotNone(node_bin, "node must be installed in test environment")

    res = subprocess.run(
        [str(node_bin), str(bin_path), "--benchmark"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    self.assertEqual(res.returncode, 0)
    self.assertIn("mcp-slim Benchmark Report", res.stdout)
    self.assertIn("Secrets intercepted & redacted: 1", res.stdout)

    ver_res = subprocess.run(
        [str(node_bin), str(bin_path), "--version"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    self.assertIn("mcp-slim 0.6.0", ver_res.stdout)

  def test_wrap_config_runners_npx_and_uvx_idempotent(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = Path(tmpdir) / "mcp.json"
      raw_cfg = {
          "mcpServers": {
              "github": {
                  "command": "npx",
                  "args": ["-y", "@modelcontextprotocol/server-github"],
              }
          }
      }
      cfg.write_text(json.dumps(raw_cfg), encoding="utf-8")
      self.assertEqual(wrap_config(cfg, in_place=True, lazy=True, runner="npx"), 0)
      wrapped_npx = json.loads(cfg.read_text(encoding="utf-8"))
      self.assertEqual(wrapped_npx["mcpServers"]["github"]["command"], "npx")
      self.assertEqual(
          wrapped_npx["mcpServers"]["github"]["args"],
          [
              "-y",
              "github:SalyyS1/mcp-slim",
              "run",
              "--lazy",
              "--",
              "npx",
              "-y",
              "@modelcontextprotocol/server-github",
          ],
      )
      # Second wrap is idempotent (0 newly wrapped)
      self.assertEqual(wrap_config(cfg, in_place=True, lazy=True, runner="npx"), 0)
      self.assertEqual(
          json.loads(cfg.read_text(encoding="utf-8")),
          wrapped_npx,
      )

  def test_composite_github_action_and_playground_assets(self) -> None:
    action_path = REPO_ROOT / "action.yml"
    self.assertTrue(action_path.exists())
    action_text = action_path.read_text(encoding="utf-8")
    self.assertIn("MCP Slim Schema & Token Audit", action_text)
    self.assertIn("using: \"composite\"", action_text)
    self.assertIn("GITHUB_STEP_SUMMARY", action_text)
    self.assertIn("mcp_slim.cli audit", action_text)

    playground = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    self.assertNotIn("<script src=", playground)
    self.assertIn("Multi-Server Workspace", playground)
    self.assertIn("FastMCP Nested $defs/$ref", playground)
    self.assertIn("Lazy Discovery Mode (--lazy)", playground)
    self.assertIn("assets/benchmark-tokens.svg", playground)
    self.assertIn("assets/benchmark-latency.svg", playground)


if __name__ == "__main__":
  unittest.main()
