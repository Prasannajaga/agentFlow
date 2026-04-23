# AgentFlow

Phase 2 adds Postgres-backed agent registration on top of the Phase 1 YAML validation CLI, using strict checked-in SQL migration files with lightweight tracking in `schema_migrations`.

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
SELECT version, name, applied_at FROM schema_migrations ORDER BY version;
```
