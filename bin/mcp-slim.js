#!/usr/bin/env node
"use strict";

const { spawn, spawnSync } = require("child_process");
const path = require("path");

function findPython() {
  const candidates = [];
  if (process.env.MCP_SLIM_PYTHON && process.env.MCP_SLIM_PYTHON.trim()) {
    candidates.push([process.env.MCP_SLIM_PYTHON.trim(), []]);
  }
  candidates.push(["python3", []]);
  candidates.push(["python", []]);
  if (process.platform === "win32") {
    candidates.push(["py", ["-3"]]);
  } else {
    candidates.push(["py", ["-3"]]);
  }

  const versionProbe =
    "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)";

  for (const [cmd, prefixArgs] of candidates) {
    const res = spawnSync(cmd, [...prefixArgs, "-c", versionProbe], {
      stdio: "ignore",
      windowsHide: true,
    });
    if (!res.error && res.status === 0) {
      return { cmd, prefixArgs };
    }
  }
  return null;
}

function main() {
  const found = findPython();
  if (!found) {
    process.stderr.write(
      "[mcp-slim] Error: Python 3.10+ is required on PATH (tried python3, python, py -3).\n" +
        "Install Python 3.10+ from https://www.python.org/downloads/ or set MCP_SLIM_PYTHON.\n"
    );
    process.exit(1);
  }

  const srcDir = path.resolve(__dirname, "..", "src");
  const existingPythonPath = process.env.PYTHONPATH || "";
  const pythonPath = existingPythonPath
    ? `${srcDir}${path.delimiter}${existingPythonPath}`
    : srcDir;

  const env = Object.assign({}, process.env, { PYTHONPATH: pythonPath });
  const cliArgs = [
    ...found.prefixArgs,
    "-m",
    "mcp_slim.cli",
    ...process.argv.slice(2),
  ];

  const child = spawn(found.cmd, cliArgs, {
    stdio: "inherit",
    env,
    windowsHide: true,
  });

  const forwardSignal = (sig) => {
    if (!child.killed) {
      try {
        child.kill(sig);
      } catch (_) {
        // Ignore if child already exited.
      }
    }
  };
  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));

  child.on("error", (err) => {
    process.stderr.write(
      `[mcp-slim] Failed to launch ${found.cmd}: ${err.message}\n`
    );
    process.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (typeof code === "number") {
      process.exit(code);
    }
    if (signal) {
      process.exit(1);
    }
    process.exit(0);
  });
}

main();
