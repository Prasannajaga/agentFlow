from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentflow.config import get_settings
from agentflow.db.session import get_engine

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "sql" / "migrations"


@dataclass(frozen=True)
class DbMigrationResult:
    applied_migrations: tuple[str, ...]
    skipped_migrations: tuple[str, ...]


def list_migration_files(migrations_dir: Path | None = None) -> tuple[Path, ...]:
    resolved_dir = migrations_dir or MIGRATIONS_DIR
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Migration directory not found: {resolved_dir}")

    return tuple(
        sorted(
            path
            for path in resolved_dir.iterdir()
            if path.is_file() and path.suffix == ".sql"
        )
    )


def apply_sql_migrations(migrations_dir: Path | None = None) -> DbMigrationResult:
    migration_files = list_migration_files(migrations_dir)
    engine = get_engine(get_settings().database_url)

    applied: list[str] = []
    skipped: list[str] = []

    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            _ensure_schema_migrations_table(cursor)
            raw_connection.commit()

            applied_versions = _fetch_applied_versions(cursor)

            for migration_file in migration_files:
                version = migration_file.name.split("_", 1)[0]
                if version in applied_versions:
                    skipped.append(migration_file.name)
                    continue

                sql_text = migration_file.read_text(encoding="utf-8").strip()
                if not sql_text:
                    skipped.append(migration_file.name)
                    continue

                try:
                    cursor.execute(sql_text)
                    cursor.execute(
                        """
                        INSERT INTO schema_migrations (version, name)
                        VALUES (%s, %s)
                        """,
                        (version, migration_file.name),
                    )
                except Exception:
                    raw_connection.rollback()
                    raise

                raw_connection.commit()
                applied_versions.add(version)
                applied.append(migration_file.name)
    finally:
        raw_connection.close()

    return DbMigrationResult(
        applied_migrations=tuple(applied),
        skipped_migrations=tuple(skipped),
    )


def _ensure_schema_migrations_table(cursor: object) -> None:
    # We keep migration tracking in Postgres itself while leaving the actual schema
    # definition in checked-in .sql files under agentflow/sql/migrations/.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _fetch_applied_versions(cursor: object) -> set[str]:
    cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
    return {str(row[0]) for row in cursor.fetchall()}
