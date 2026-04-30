# Testing AgentFlow

This project uses `pytest` for tests.

## Install test dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
```

## Create a dedicated Postgres test database

```bash
createdb agentflow_test
```

Set `TEST_DATABASE_URL` to that dedicated database:

```bash
export TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/agentflow_test
```

Safety guard:

- DB integration tests refuse to run if the database name in `TEST_DATABASE_URL` does not include `test`.
- Override intentionally only when needed:

```bash
export AGENTFLOW_ALLOW_NON_TEST_DB=1
```

## Run tests

```bash
pytest
```

Run coverage:

```bash
pytest --cov=agentflow
```

## Notes

- DB integration tests are marked with `@pytest.mark.db`.
- If `TEST_DATABASE_URL` is not set, DB tests are skipped with a clear message.
- Tests apply SQL migrations from `agentflow/sql/migrations` directly.
- Tests use temporary artifact directories via `AGENTFLOW_ARTIFACT_STORAGE_DIR`.
- Tests use the fake provider only and do not require real API keys.
- Tests do not call external LLM providers or network APIs.
