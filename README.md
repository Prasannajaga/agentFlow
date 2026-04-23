# AgentFlow

Phase 1 adds a local validation CLI for agent YAML files.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Validate an agent config

```bash
.venv/bin/createAgent validate examples/text-agent.yaml
```

The CLI prints a short success message followed by normalized JSON.

## Validate the invalid example

```bash
.venv/bin/createAgent validate examples/invalid-agent.yaml
```

The command exits non-zero and prints field-level validation errors.

## Aliases

The same validator is also available through the `agentflow` entrypoint:

```bash
.venv/bin/agentflow validate examples/text-agent.yaml
```

Phase 1 also accepts the prompt's alternate command shape and treats it as local validation only:

```bash
.venv/bin/agentflow create examples/text-agent.yaml
```
