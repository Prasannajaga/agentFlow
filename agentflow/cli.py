from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Sequence

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
from agentflow.services.db_migrations import apply_sql_migrations
from agentflow.services.agent_registry import register_agent
from agentflow.services.agent_runner import (
    AgentNotFoundError,
    AgentRunExecutionFailedError,
    AgentVersionNotFoundError,
    run_registered_agent,
)
from agentflow.services.agent_queries import (
    AgentDetail,
    AgentSummary,
    AgentVersionSummary,
    get_registered_agent,
    list_agent_versions,
    list_registered_agents,
)
from agentflow.services.run_queries import AgentRunDetail, AgentRunSummary, get_agent_run, list_agent_runs
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
        description="Validate agent YAML files, register and inspect agents in Postgres, and manage DB config.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command_name, help_text in (
        ("validate", "Validate an agent YAML file."),
        ("create", "Phase 1 compatibility alias that validates an agent YAML file."),
        ("register", "Validate an agent YAML file and register it in Postgres."),
    ):
        _add_agent_file_command(subparsers, command_name=command_name, help_text=help_text)

    subparsers.add_parser("list", help="List registered agents.")

    show_parser = subparsers.add_parser("show", help="Show one registered agent.")
    show_parser.add_argument("agent_id", help="UUID of the agent to display.")

    versions_parser = subparsers.add_parser("versions", help="List versions for one registered agent.")
    versions_parser.add_argument("agent_id", help="UUID of the agent whose versions should be displayed.")

    run_parser = subparsers.add_parser("run", help="Run one registered agent synchronously.")
    run_parser.add_argument("agent_id", help="UUID of the agent to run.")

    subparsers.add_parser("runs", help="List agent runs.")

    run_show_parser = subparsers.add_parser("run-show", help="Show one persisted agent run.")
    run_show_parser.add_argument("run_id", help="UUID of the run to display.")

    db_parser = subparsers.add_parser("db", help="Database configuration helpers.")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)

    db_show_parser = db_subparsers.add_parser("show", help="Show the resolved database settings.")
    db_show_parser.add_argument("--json", action="store_true", help="Print the resolved settings as JSON.")

    db_subparsers.add_parser(
        "migrate",
        help="Apply versioned SQL schema files tracked in schema_migrations.",
    )

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


def parse_agent_id(value: str) -> uuid.UUID:
    return parse_uuid_value(value, label="agent_id")


def parse_run_id(value: str) -> uuid.UUID:
    return parse_uuid_value(value, label="run_id")


def parse_uuid_value(value: str, *, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {label} '{value}'. Expected a UUID.") from exc


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def format_optional_timestamp(value: datetime | None) -> str:
    if value is None:
        return "-"

    return format_timestamp(value)


def summarize_text(value: str | None, *, fallback: str = "-", width: int | None = None) -> str:
    if value is None:
        return fallback

    collapsed = " ".join(value.split())
    if not collapsed:
        return fallback

    if width is not None and len(collapsed) > width:
        return f"{collapsed[: max(width - 3, 0)].rstrip()}..."

    return collapsed


def print_key_value(key: str, value: object, *, file: IO[str] = sys.stdout) -> None:
    print(f"{key}: {value}", file=file)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "  ".join("-" * widths[index] for index in range(len(headers)))
    body_lines = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in string_rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def print_agent_table(agents: Sequence[AgentSummary]) -> None:
    rows = [
        (
            agent.agent_id,
            agent.name,
            summarize_text(agent.description, width=50),
            format_timestamp(agent.created_at),
            agent.latest_version.version_number if agent.latest_version is not None else "-",
        )
        for agent in agents
    ]
    print(
        render_table(
            ("agent_id", "name", "description", "created_at", "latest_version"),
            rows,
        )
    )


def print_versions_table(versions: Sequence[AgentVersionSummary]) -> None:
    rows = [
        (
            version.version_id,
            version.version_number,
            version.config_hash,
            format_timestamp(version.created_at),
        )
        for version in versions
    ]
    print(render_table(("version_id", "version_number", "config_hash", "created_at"), rows))


def print_runs_table(runs: Sequence[AgentRunSummary]) -> None:
    rows = [
        (
            run.run_id,
            run.agent_id,
            run.status,
            format_timestamp(run.created_at),
            format_optional_timestamp(run.started_at),
            format_optional_timestamp(run.ended_at),
        )
        for run in runs
    ]
    print(render_table(("run_id", "agent_id", "status", "created_at", "started_at", "ended_at"), rows))


def print_agent_detail(agent: AgentDetail) -> None:
    print_key_value("agent_id", agent.agent_id)
    print_key_value("name", agent.name)
    print_key_value("description", summarize_text(agent.description))
    print_key_value("created_at", format_timestamp(agent.created_at))
    print_key_value("updated_at", format_timestamp(agent.updated_at))

    print()
    print("latest_version:")
    if agent.latest_version is None:
        print("  none")
        return

    print(f"  version_id: {agent.latest_version.version_id}")
    print(f"  version_number: {agent.latest_version.version_number}")
    print(f"  config_hash: {agent.latest_version.config_hash}")
    print(f"  created_at: {format_timestamp(agent.latest_version.created_at)}")


def print_run_summary(run: AgentRunDetail, *, file: IO[str] = sys.stdout) -> None:
    print_key_value("run_id", run.run_id, file=file)
    print_key_value("agent_id", run.agent_id, file=file)
    print_key_value("version_id", run.version_id, file=file)
    print_key_value("status", run.status, file=file)
    print_key_value("created_at", format_timestamp(run.created_at), file=file)
    print_key_value("started_at", format_optional_timestamp(run.started_at), file=file)
    print_key_value("ended_at", format_optional_timestamp(run.ended_at), file=file)
    print_key_value("output_summary", summarize_run_output(run.output_json), file=file)


def print_run_detail(run: AgentRunDetail, *, file: IO[str] = sys.stdout) -> None:
    print_run_summary(run, file=file)

    if run.error_message:
        print_key_value("error_message", run.error_message, file=file)

    if run.output_json is not None:
        print(file=file)
        print("output_json:", file=file)
        print(json.dumps(run.output_json, indent=2), file=file)


def summarize_run_output(output_json: dict[str, object] | None) -> str:
    if output_json is None:
        return "-"

    parts: list[str] = []
    provider = output_json.get("provider")
    model = output_json.get("model")
    message = output_json.get("message")

    if provider:
        parts.append(str(provider))
    if model:
        parts.append(str(model))
    if message:
        parts.append(summarize_text(str(message), width=60))

    return " | ".join(parts) or "stored"


def format_database_error_message(exc: SQLAlchemyError) -> str:
    original = getattr(exc, "orig", None)
    if original is not None:
        return str(original)

    return str(exc)


def handle_cli_query_error(exc: Exception, *, action: str) -> int:
    if isinstance(exc, ValueError):
        print(str(exc), file=sys.stderr)
        return 1

    if isinstance(exc, ConfigurationError):
        print(f"{action} failed: missing configuration.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1

    if isinstance(exc, SQLAlchemyError):
        print(f"{action} failed due to a database error.", file=sys.stderr)
        print(f"- {format_database_error_message(exc)}", file=sys.stderr)

        if isinstance(exc, OperationalError):
            print(
                "- Verify DATABASE_URL or your DATABASE_HOST/DATABASE_PORT settings point to a reachable Postgres instance.",
                file=sys.stderr,
            )
        elif isinstance(exc, ProgrammingError):
            print("- Verify the schema exists by running `agentflow db migrate`.", file=sys.stderr)

        return 1

    print(f"{action} failed.", file=sys.stderr)
    print(f"- {exc}", file=sys.stderr)
    return 1


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


def run_db_migrate() -> int:
    try:
        result = apply_sql_migrations()
    except ConfigurationError as exc:
        print("Database migration failed: missing configuration.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print("Database migration failed: migration files not found.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1
    except SQLAlchemyError as exc:
        print("Database migration failed due to a database error.", file=sys.stderr)
        print(f"- {format_database_error_message(exc)}", file=sys.stderr)

        if isinstance(exc, OperationalError):
            print(
                "- Verify DATABASE_URL or your DATABASE_HOST/DATABASE_PORT settings point to a reachable Postgres instance.",
                file=sys.stderr,
            )

        return 1
    except Exception as exc:
        print("Database migration failed.", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1

    if result.applied_migrations:
        print("Database migrations applied and recorded in schema_migrations")
        for migration_name in result.applied_migrations:
            print(f"- {migration_name}")
        return 0

    print("Database schema is up to date")
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
        print(f"- {format_database_error_message(exc)}", file=sys.stderr)

        if isinstance(exc, OperationalError):
            print(
                "- Verify DATABASE_URL or your DATABASE_HOST/DATABASE_PORT settings point to a reachable Postgres instance.",
                file=sys.stderr,
            )
        elif isinstance(exc, ProgrammingError):
            print("- Verify the schema exists by running `agentflow db migrate`.", file=sys.stderr)

        return 1

    print("Registration successful")
    print(f"agent_id: {result.agent_id}")
    print(f"version_id: {result.version_id}")
    print(f"version_number: {result.version_number}")
    print(f"config_hash: {result.config_hash}")
    return 0


def run_list_agents() -> int:
    try:
        agents = list_registered_agents()
    except Exception as exc:
        return handle_cli_query_error(exc, action="List agents")

    if not agents:
        print("No registered agents found.")
        return 0

    print_agent_table(agents)
    return 0


def run_show_agent(agent_id_text: str) -> int:
    try:
        agent_id = parse_agent_id(agent_id_text)
        agent = get_registered_agent(agent_id)
    except Exception as exc:
        return handle_cli_query_error(exc, action="Show agent")

    if agent is None:
        print(f"Agent not found: {agent_id}", file=sys.stderr)
        return 1

    print_agent_detail(agent)
    return 0


def run_list_agent_versions(agent_id_text: str) -> int:
    try:
        agent_id = parse_agent_id(agent_id_text)
        versions = list_agent_versions(agent_id)
    except Exception as exc:
        return handle_cli_query_error(exc, action="List agent versions")

    if versions is None:
        print(f"Agent not found: {agent_id}", file=sys.stderr)
        return 1

    print_key_value("agent_id", agent_id)
    print()

    if not versions:
        print("No versions found.")
        return 0

    print_versions_table(versions)
    return 0


def run_run_agent(agent_id_text: str) -> int:
    try:
        agent_id = parse_agent_id(agent_id_text)
        run = run_registered_agent(agent_id)
    except AgentNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except AgentVersionNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except AgentRunExecutionFailedError as exc:
        print("Run failed", file=sys.stderr)
        print_run_detail(exc.run, file=sys.stderr)
        return 1
    except Exception as exc:
        return handle_cli_query_error(exc, action="Run agent")

    print("Run completed")
    print_run_summary(run)
    return 0


def run_list_runs() -> int:
    try:
        runs = list_agent_runs()
    except Exception as exc:
        return handle_cli_query_error(exc, action="List runs")

    if not runs:
        print("No runs found.")
        return 0

    print_runs_table(runs)
    return 0


def run_show_run(run_id_text: str) -> int:
    try:
        run_id = parse_run_id(run_id_text)
        run = get_agent_run(run_id)
    except Exception as exc:
        return handle_cli_query_error(exc, action="Show run")

    if run is None:
        print(f"Run not found: {run_id}", file=sys.stderr)
        return 1

    print_run_detail(run)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(prog=Path(sys.argv[0]).name)
    args = parser.parse_args(args_list)

    if args.command in {"validate", "create"}:
        return run_validate(args.path)
    if args.command == "register":
        return run_register(args.path)
    if args.command == "list":
        return run_list_agents()
    if args.command == "show":
        return run_show_agent(args.agent_id)
    if args.command == "versions":
        return run_list_agent_versions(args.agent_id)
    if args.command == "run":
        return run_run_agent(args.agent_id)
    if args.command == "runs":
        return run_list_runs()
    if args.command == "run-show":
        return run_show_run(args.run_id)
    if args.command == "db":
        if args.db_command == "show":
            return run_db_show(args.json)
        if args.db_command == "setup":
            return run_db_setup(args)
        if args.db_command == "migrate":
            return run_db_migrate()

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
