from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from sqlalchemy.engine import URL, make_url

DATABASE_URL_ENV = "DATABASE_URL"
DATABASE_DRIVER_ENV = "DATABASE_DRIVER"
DATABASE_HOST_ENV = "DATABASE_HOST"
DATABASE_PORT_ENV = "DATABASE_PORT"
DATABASE_NAME_ENV = "DATABASE_NAME"
DATABASE_USER_ENV = "DATABASE_USER"
DATABASE_PASSWORD_ENV = "DATABASE_PASSWORD"
ARTIFACT_STORAGE_DIR_ENV = "AGENTFLOW_ARTIFACT_STORAGE_DIR"
WORKER_HEARTBEAT_INTERVAL_SECONDS_ENV = "WORKER_HEARTBEAT_INTERVAL_SECONDS"
WORKER_STALE_THRESHOLD_SECONDS_ENV = "WORKER_STALE_THRESHOLD_SECONDS"

DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DATABASE_DRIVER = "postgresql+psycopg"
DEFAULT_DATABASE_HOST = "localhost"
DEFAULT_DATABASE_PORT = 5432
DEFAULT_WORKER_HEARTBEAT_INTERVAL_SECONDS = 5
DEFAULT_WORKER_STALE_THRESHOLD_SECONDS = 600

LOCAL_DATABASE_ENV_VARS = (
    DATABASE_DRIVER_ENV,
    DATABASE_HOST_ENV,
    DATABASE_PORT_ENV,
    DATABASE_NAME_ENV,
    DATABASE_USER_ENV,
    DATABASE_PASSWORD_ENV,
)


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    database_url: str
    database_url_redacted: str
    database_source: str
    database_driver: str | None
    database_host: str | None
    database_port: int | None
    database_name: str | None
    database_user: str | None
    database_password_configured: bool


def _strip_env_value(env: Mapping[str, str], key: str) -> str:
    return env.get(key, "").strip()


def _env_value_is_set(env: Mapping[str, str], key: str) -> bool:
    return key in env and env.get(key) is not None


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None

    if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]

    return key, value


@lru_cache(maxsize=1)
def load_env_file(path: str | None = None) -> Path | None:
    env_path = Path(path) if path is not None else DEFAULT_ENV_FILE
    if not env_path.is_file():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        assignment = _parse_env_assignment(line)
        if assignment is None:
            continue

        key, value = assignment
        os.environ.setdefault(key, value)

    return env_path


def build_database_url(
    *,
    driver: str = DEFAULT_DATABASE_DRIVER,
    host: str = DEFAULT_DATABASE_HOST,
    port: int = DEFAULT_DATABASE_PORT,
    name: str,
    user: str,
    password: str | None = None,
) -> str:
    return URL.create(
        drivername=driver,
        username=user,
        password=password or None,
        host=host,
        port=port,
        database=name,
    ).render_as_string(hide_password=False)


def redact_database_url(database_url: str) -> str:
    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except Exception:
        return database_url


def _settings_from_url(database_url: str, source: str) -> Settings:
    try:
        parsed = make_url(database_url)
    except Exception as exc:
        raise ConfigurationError(f"Invalid {source}: {exc}") from exc

    return Settings(
        database_url=database_url,
        database_url_redacted=parsed.render_as_string(hide_password=True),
        database_source=source,
        database_driver=parsed.drivername,
        database_host=parsed.host,
        database_port=parsed.port,
        database_name=parsed.database,
        database_user=parsed.username,
        database_password_configured=parsed.password is not None,
    )


def resolve_settings(env: Mapping[str, str] | None = None) -> Settings:
    if env is None:
        load_env_file()
        env = os.environ

    database_url = _strip_env_value(env, DATABASE_URL_ENV)
    if database_url:
        return _settings_from_url(database_url, DATABASE_URL_ENV)

    local_mode_enabled = any(_env_value_is_set(env, key) for key in LOCAL_DATABASE_ENV_VARS)
    if not local_mode_enabled:
        raise ConfigurationError(
            "DATABASE_URL is not set. "
            "Set DATABASE_URL directly or use DATABASE_NAME and DATABASE_USER "
            "with optional DATABASE_HOST, DATABASE_PORT, DATABASE_PASSWORD, and DATABASE_DRIVER."
        )

    driver = _strip_env_value(env, DATABASE_DRIVER_ENV) or DEFAULT_DATABASE_DRIVER
    host = _strip_env_value(env, DATABASE_HOST_ENV) or DEFAULT_DATABASE_HOST
    port_text = _strip_env_value(env, DATABASE_PORT_ENV) or str(DEFAULT_DATABASE_PORT)
    name = _strip_env_value(env, DATABASE_NAME_ENV)
    user = _strip_env_value(env, DATABASE_USER_ENV)
    password = env.get(DATABASE_PASSWORD_ENV)

    missing_vars = [key for key, value in ((DATABASE_NAME_ENV, name), (DATABASE_USER_ENV, user)) if not value]
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise ConfigurationError(
            f"Missing required local database settings: {missing}. "
            "Set DATABASE_URL instead or provide the missing DATABASE_* values."
        )

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ConfigurationError(f"{DATABASE_PORT_ENV} must be an integer.") from exc

    if port <= 0:
        raise ConfigurationError(f"{DATABASE_PORT_ENV} must be greater than zero.")

    database_url = build_database_url(
        driver=driver,
        host=host,
        port=port,
        name=name,
        user=user,
        password=password,
    )
    return _settings_from_url(database_url, "DATABASE_*")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return resolve_settings()


def get_database_url() -> str:
    return get_settings().database_url


def clear_settings_cache() -> None:
    load_env_file.cache_clear()
    get_settings.cache_clear()


def get_artifact_storage_dir() -> Path:
    load_env_file()
    configured = os.environ.get(ARTIFACT_STORAGE_DIR_ENV, "").strip()
    return Path(configured) if configured else Path("data") / "artifacts"


def get_worker_heartbeat_interval_seconds() -> int:
    return _resolve_positive_int_env(
        WORKER_HEARTBEAT_INTERVAL_SECONDS_ENV,
        default=DEFAULT_WORKER_HEARTBEAT_INTERVAL_SECONDS,
    )


def get_worker_stale_threshold_seconds() -> int:
    return _resolve_positive_int_env(
        WORKER_STALE_THRESHOLD_SECONDS_ENV,
        default=DEFAULT_WORKER_STALE_THRESHOLD_SECONDS,
    )


def _resolve_positive_int_env(name: str, *, default: int) -> int:
    load_env_file()
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")

    return parsed
