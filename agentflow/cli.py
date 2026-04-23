from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError

from agentflow.config import (
    ConfigurationError,
    DATABASE_DRIVER_ENV,
    DATABASE_HOST_ENV,
    DATABASE_NAME_ENV,
    DATABASE_PASSWORD_ENV,
    DATABASE_PORT_ENV,
    DATABASE_URL_ENV,
    DATABASE_USER_ENV,
    DEFAULT_DATABASE_DRIVER,
    DEFAULT_DATABASE_HOST,
    DEFAULT_DATABASE_PORT,
    build_database_url,
    clear_settings_cache,
    get_settings,
    redact_database_url,
)
from agentflow.services.agent_registry import register_agent
from agentflow.services.yaml_loader import AgentYamlError, load_agent_document, normalize_agent_config

DEFAULT_LOCAL_DATABASE_NAME = "flow_agent"
DEFAULT_LOCAL_DATABASE_USER = "postgres"
DEFAULT_LOCAL_DATABASE_PASSWORD = "postgres"


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def _add_agent_file_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command_name: str,
    help_text: str,
) -> None:
    command_parser = subparsers.add_parser(command_name, help=help_text)
    command_parser.add_argument("path", type=Path, help="Path to an agent YAML file.")


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate agent YAML files, register them in Postgres, and manage DB config.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command_name, help_text in (
        ("validate", "Validate an agent YAML file."),
        ("create", "Phase 1 compatibility alias that validates an agent YAML file."),
        ("register", "Validate an agent YAML file and register it in Postgres."),
    ):
        _add_agent_file_command(subparsers, command_name=command_name, help_text=help_text)

    db_parser = subparsers.add_parser("db", help="Database configuration helpers.")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)

    db_show_parser = db_subparsers.add_parser("show", help="Show the resolved database settings.")
    db_show_parser.add_argument("--json", action="store_true", help="Print the resolved settings as JSON.")

    db_setup_parser = db_subparsers.add_parser("setup", help="Write database settings to an env file.")
    db_setup_mode = db_setup_parser.add_mutually_exclusive_group(required=True)
    db_setup_mode.add_argument("--url", help="Full SQLAlchemy database URL to store.")
    db_setup_mode.add_argument(
        "--local",
        action="store_true",
        help="Write localhost-friendly DATABASE_* settings instead of DATABASE_URL.",
    )
    db_setup_parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Env file to update. Defaults to .env in the current directory.",
    )
    db_setup_parser.add_argument(
        "--host",
        default=DEFAULT_DATABASE_HOST,
        help=f"Database host for --local. Defaults to {DEFAULT_DATABASE_HOST}.",
    )
    db_setup_parser.add_argument(
        "--port",
        type=parse_positive_int,
        default=DEFAULT_DATABASE_PORT,
        help=f"Database port for --local. Defaults to {DEFAULT_DATABASE_PORT}.",
    )
    db_setup_parser.add_argument(
        "--name",
        default=DEFAULT_LOCAL_DATABASE_NAME,
        help=f"Database name for --local. Defaults to {DEFAULT_LOCAL_DATABASE_NAME}.",
    )
    db_setup_parser.add_argument(
        "--user",
        default=DEFAULT_LOCAL_DATABASE_USER,
        help=f"Database user for --local. Defaults to {DEFAULT_LOCAL_DATABASE_USER}.",
    )
    db_setup_parser.add_argument(
        "--password",
        default=DEFAULT_LOCAL_DATABASE_PASSWORD,
        help="Database password for --local. Ignored when --no-password is used.",
    )
    db_setup_parser.add_argument(
        "--no-password",
        action="store_true",
        help="Omit DATABASE_PASSWORD when writing local settings.",
    )
    db_setup_parser.add_argument(
        "--driver",
        default=DEFAULT_DATABASE_DRIVER,
        help=f"SQLAlchemy driver for --local. Defaults to {DEFAULT_DATABASE_DRIVER}.",
    )
    db_setup_parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the env entries instead of writing the env file.",
    )

    return parser


def format_validation_error(error: dict[str, object]) -> str:
    location = format_location(error.get("loc", ()))
    message = str(error.get("msg", "Invalid value"))
    return f"- {location}: {message}"


def format_location(location: object) -> str:
    parts: list[str] = []

    for part in location if isinstance(location, tuple) else ():
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
            continue

        parts.append(str(part))

    return ".".join(parts) or "<root>"


def extract_env_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    if "=" not in stripped:
        return None

    key, _ = stripped.split("=", 1)
    key = key.strip()
    return key or None


def serialize_env_value(value: str) -> str:
    if value == "":
        return '""'

    if any(character.isspace() for character in value) or any(character in value for character in '#"\''):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    return value


def update_env_file(path: Path, values: dict[str, str | None]) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in existing_lines:
        key = extract_env_key(line)
        if key is None or key not in values:
            updated_lines.append(line)
            continue

        seen_keys.add(key)
        value = values[key]
        if value is None:
            continue

        updated_lines.append(f"{key}={serialize_env_value(value)}")

    for key, value in values.items():
        if key in seen_keys or value is None:
            continue

        updated_lines.append(f"{key}={serialize_env_value(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(updated_lines).rstrip("\n") + "\n", encoding="utf-8")


def build_env_preview(values: dict[str, str | None]) -> str:
    preview_lines: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        preview_lines.append(f"{key}={serialize_env_value(value)}")
    return "\n".join(preview_lines)


def apply_env_values(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
            continue

        os.environ[key] = value


def run_db_show(as_json: bool) -> int:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print("Database configuration is incomplete.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1

    payload = {
        "source": settings.database_source,
        "database_url": settings.database_url_redacted,
        "driver": settings.database_driver,
        "host": settings.database_host,
        "port": settings.database_port,
        "database": settings.database_name,
        "user": settings.database_user,
        "password_configured": settings.database_password_configured,
    }

    if as_json:
        print(json.dumps(payload, indent=2))
        return 0

    print("Database configuration resolved")
    print(f"source: {payload['source']}")
    print(f"database_url: {payload['database_url']}")
    print(f"driver: {payload['driver']}")
    print(f"host: {payload['host']}")
    print(f"port: {payload['port']}")
    print(f"database: {payload['database']}")
    print(f"user: {payload['user']}")
    print(f"password_configured: {payload['password_configured']}")
    return 0


def run_db_setup(args: argparse.Namespace) -> int:
    if args.url is not None:
        database_url = args.url.strip()
        if not database_url:
            print("Database setup failed: --url cannot be empty.", file=sys.stderr)
            return 1

        env_values = {
            DATABASE_URL_ENV: database_url,
            DATABASE_DRIVER_ENV: None,
            DATABASE_HOST_ENV: None,
            DATABASE_PORT_ENV: None,
            DATABASE_NAME_ENV: None,
            DATABASE_USER_ENV: None,
            DATABASE_PASSWORD_ENV: None,
        }
        setup_source = DATABASE_URL_ENV
    else:
        password = None if args.no_password else args.password
        database_url = build_database_url(
            driver=args.driver,
            host=args.host,
            port=args.port,
            name=args.name,
            user=args.user,
            password=password,
        )
        env_values = {
            DATABASE_URL_ENV: None,
            DATABASE_DRIVER_ENV: args.driver,
            DATABASE_HOST_ENV: args.host,
            DATABASE_PORT_ENV: str(args.port),
            DATABASE_NAME_ENV: args.name,
            DATABASE_USER_ENV: args.user,
            DATABASE_PASSWORD_ENV: password,
        }
        setup_source = "DATABASE_*"

    if args.print_only:
        print(build_env_preview(env_values))
        return 0

    update_env_file(args.env_file, env_values)
    apply_env_values(env_values)
    clear_settings_cache()

    print(f"Database configuration written to {args.env_file.resolve()}")
    print(f"source: {setup_source}")
    print(f"database_url: {redact_database_url(database_url)}")
    if args.url is None:
        print("local_settings: ready")
    return 0


def run_validate(path: Path) -> int:
    try:
        document = load_agent_document(path)
    except FileNotFoundError:
        print(f"Validation failed: file not found: {path}", file=sys.stderr)
        return 1
    except IsADirectoryError:
        print(f"Validation failed: expected a file but got a directory: {path}", file=sys.stderr)
        return 1
    except AgentYamlError as exc:
        print(f"Validation failed for {path}", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Validation failed for {path}", file=sys.stderr)
        for error in exc.errors():
            print(format_validation_error(error), file=sys.stderr)
        return 1

    normalized = normalize_agent_config(document.config)
    print(f"Validation succeeded for {path}")
    print(json.dumps(normalized, indent=2))
    return 0


def run_register(path: Path) -> int:
    try:
        result = register_agent(path)
    except FileNotFoundError:
        print(f"Registration failed: file not found: {path}", file=sys.stderr)
        return 1
    except IsADirectoryError:
        print(f"Registration failed: expected a file but got a directory: {path}", file=sys.stderr)
        return 1
    except AgentYamlError as exc:
        print(f"Registration failed for {path}", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Registration failed for {path}", file=sys.stderr)
        for error in exc.errors():
            print(format_validation_error(error), file=sys.stderr)
        return 1
    except ConfigurationError as exc:
        print("Registration failed: missing configuration.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1
    except SQLAlchemyError as exc:
        print("Registration failed due to a database error.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)

        if isinstance(exc, OperationalError):
            print(
                "- Verify DATABASE_URL or your DATABASE_HOST/DATABASE_PORT settings point to a reachable Postgres instance.",
                file=sys.stderr,
            )
        elif isinstance(exc, ProgrammingError):
            print("- Verify the schema exists by running `alembic upgrade head`.", file=sys.stderr)

        return 1

    print("Registration successful")
    print(f"agent_id: {result.agent_id}")
    print(f"version_id: {result.version_id}")
    print(f"version_number: {result.version_number}")
    print(f"config_hash: {result.config_hash}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(prog=Path(sys.argv[0]).name)
    args = parser.parse_args(args_list)

    if args.command in {"validate", "create"}:
        return run_validate(args.path)
    if args.command == "register":
        return run_register(args.path)
    if args.command == "db":
        if args.db_command == "show":
            return run_db_show(args.json)
        if args.db_command == "setup":
            return run_db_setup(args)

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
