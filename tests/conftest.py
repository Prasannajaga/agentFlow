from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from agentflow.config import ARTIFACT_STORAGE_DIR_ENV, clear_settings_cache
from agentflow.db.session import create_session_factory, get_engine
from agentflow.services.db_migrations import apply_sql_migrations

TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"
ALLOW_NON_TEST_DB_ENV = "AGENTFLOW_ALLOW_NON_TEST_DB"


def make_agent_yaml(
    name: str = "test-agent",
    version: int = 1,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "provider": {
            "type": "fake",
            "model": "stub-model",
        },
        "system_prompt": "You are a test agent.",
        "runtime": {
            "max_steps": 10,
            "timeout_seconds": 30,
            "retry": {
                "max_attempts": 1,
                "backoff_seconds": 0,
            },
        },
        "tools": [],
        "task": {
            "type": "test",
            "input_schema": {
                "prompt": "string",
            },
        },
    }
    if extra:
        payload = _deep_merge(payload, extra)

    return yaml.safe_dump(payload, sort_keys=False)


def write_agent_yaml(tmp_path: Path, content: str | Mapping[str, Any]) -> Path:
    path = tmp_path / "agent.yaml"
    if isinstance(content, Mapping):
        serialized = yaml.safe_dump(dict(content), sort_keys=False)
    else:
        serialized = content
    path.write_text(serialized, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def artifact_storage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    storage_dir = tmp_path / "artifacts"
    storage_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ARTIFACT_STORAGE_DIR_ENV, str(storage_dir))
    return storage_dir


@pytest.fixture(scope="session")
def db_session_factory() -> sessionmaker[Session]:
    test_database_url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if not test_database_url:
        pytest.skip(
            f"DB integration tests require {TEST_DATABASE_URL_ENV}. "
            "Example: postgresql+psycopg://postgres:postgres@localhost:5432/agentflow_test"
        )

    _assert_safe_test_database_url(test_database_url)

    previous_database_url = os.environ.get("DATABASE_URL")
    try:
        os.environ["DATABASE_URL"] = test_database_url
        clear_settings_cache()
        get_engine.cache_clear()
        apply_sql_migrations()
        yield create_session_factory(database_url=test_database_url)
    finally:
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url
        clear_settings_cache()
        get_engine.cache_clear()


@pytest.fixture(autouse=True)
def db_test_isolation(request: pytest.FixtureRequest) -> None:
    if "db" not in request.keywords:
        yield
        return

    session_factory = request.getfixturevalue("db_session_factory")
    _truncate_all_data(session_factory)
    yield
    _truncate_all_data(session_factory)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    test_database_url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if test_database_url:
        return

    skip_db = pytest.mark.skip(
        reason=(
            f"Skipping DB integration tests because {TEST_DATABASE_URL_ENV} is not set. "
            "Set it to a dedicated PostgreSQL test database URL to run marked db tests."
        )
    )
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)


def _assert_safe_test_database_url(database_url: str) -> None:
    try:
        parsed = make_url(database_url)
    except Exception as exc:
        raise pytest.UsageError(f"Invalid {TEST_DATABASE_URL_ENV}: {exc}") from exc

    database_name = (parsed.database or "").strip()
    if not database_name:
        raise pytest.UsageError(
            f"{TEST_DATABASE_URL_ENV} must include a database name."
        )

    allow_non_test_db = os.environ.get(ALLOW_NON_TEST_DB_ENV, "").strip() == "1"
    if "test" not in database_name.lower() and not allow_non_test_db:
        raise pytest.UsageError(
            "Refusing to run DB tests against a database whose name does not include 'test'. "
            f"DB name was '{database_name}'. "
            f"Set {ALLOW_NON_TEST_DB_ENV}=1 to override intentionally."
        )


def _truncate_all_data(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        table_names = session.execute(
            text(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename <> 'schema_migrations'
                ORDER BY tablename
                """
            )
        ).scalars().all()

        if not table_names:
            session.commit()
            return

        qualified = ", ".join(_quote_identifier(name) for name in table_names)
        session.execute(text(f"TRUNCATE TABLE {qualified} CASCADE"))
        session.commit()


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
