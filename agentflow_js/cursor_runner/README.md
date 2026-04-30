# AgentFlow Cursor SDK Bridge

This directory contains the local Node bridge used by AgentFlow's Python `cursor_sdk` runner.

## Purpose

- Accept one JSON request over `stdin`
- Execute Cursor SDK locally (`local.cwd`)
- Emit NDJSON events to `stdout`
- Emit diagnostics to `stderr`

Every `stdout` line is machine-readable JSON.

## Install

```bash
cd agentflow_js/cursor_runner
npm install
```

## Manual smoke test

```bash
node src/cursor_runner.mjs <<'JSON'
{"apiKey":"$CURSOR_API_KEY","cwd":"/path/to/repo","prompt":"hello","model":"auto"}
JSON
```

## Notes

- Requires Node.js.
- Requires `CURSOR_API_KEY` passed from AgentFlow (never printed by the bridge).
- `runtime: cloud` is not implemented in this phase.
