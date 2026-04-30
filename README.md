# AgentFlow

Phase 5 adds background run execution on top of the Phase 4 persistence model. Runs are created in Postgres, claimed directly from Postgres by a worker process, and executed with the same deterministic fake provider.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Configure Postgres

```bash
.venv/bin/agentflow db setup --local
```

That writes a `.env` file that the CLI now loads automatically.

If you already have a full Postgres URL, you can store that instead:

```bash
.venv/bin/agentflow db setup --url postgresql+psycopg://postgres:postgres@localhost:5432/flow_agent
```

To inspect the resolved configuration:

```bash
.venv/bin/agentflow db show
```

## Apply the schema

```bash
.venv/bin/agentflow db migrate
```

Checked-in schema files live under `agentflow/sql/migrations/`.

The migration approach is intentionally simple:

- Schema changes are authored as versioned `.sql` files in the repo.
- `agentflow db migrate` applies unapplied files in filename order.
- Applied versions are recorded in the `schema_migrations` table inside Postgres.
- The SQL files remain the source of truth for the database schema.

## Validate an agent config

```bash
.venv/bin/createAgent validate examples/text-agent.yaml
```

The CLI prints a short success message followed by normalized JSON.

## Register an agent config

```bash
.venv/bin/agentflow register examples/text-agent.yaml
```

The command validates the YAML, inserts one logical agent row plus one immutable version row, and prints the generated IDs and config hash.

## List registered agents

```bash
.venv/bin/agentflow list
```

The command prints one row per logical agent, ordered newest first, including the latest version number when available.

## Show one agent

```bash
.venv/bin/agentflow show <agent_id>
```

The command prints agent metadata plus the latest registered version summary.

## Show versions for one agent

```bash
.venv/bin/agentflow versions <agent_id>
```

The command prints all known versions for that agent, ordered by highest version number first.

## Create one pending agent run

```bash
.venv/bin/agentflow run <agent_id>
```

The command loads the latest agent version, creates an `agent_runs` row with `pending`, and returns immediately.

## Start the worker

```bash
.venv/bin/agentflow worker
```

The worker polls Postgres for pending runs, claims one safely, and moves it through `running` and `completed` or `failed`.

## External CLI runner

Agent configs can define an external command runner:

```yaml
runner:
  type: external_cli
  command: python
  args:
    - "-c"
    - "from pathlib import Path; Path('agentflow_demo_output.txt').write_text('hello from external runner\\n', encoding='utf-8')"
  cwd: "."
  timeout_seconds: 30
```

Behavior in this phase:

- `runner.type: external_cli` is supported.
- `command`/`args` run in `cwd` (relative path only).
- After execution, AgentFlow records base/result commit SHA and changed files.
- If git changes exist, AgentFlow commits them with message `agentflow run <run_id>`.
- Run detail dashboard shows a simple code-changes summary.

Not included yet:

- Resume/continue sessions
- Diff/patch viewer
- Streaming runner events

## List runs

```bash
.venv/bin/agentflow runs
```

The command prints persisted runs newest first.

## Show one run

```bash
.venv/bin/agentflow run-show <run_id>
```

The command prints the stored run metadata and pretty-prints `output_json` for completed runs.

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

## Verify inserted rows

```sql
SELECT id, name, created_at FROM agent_definitions;
SELECT id, agent_id, version_number, config_hash FROM agent_versions;
SELECT id, agent_id, version_id, status, created_at, started_at, ended_at FROM agent_runs ORDER BY created_at DESC;
SELECT version, name, applied_at FROM schema_migrations ORDER BY version;
```
