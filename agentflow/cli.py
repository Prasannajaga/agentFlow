from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from agentflow.services.yaml_loader import AgentYamlError, load_agent_config


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate an agent YAML file and print normalized JSON.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command_name, help_text in (
        ("validate", "Validate an agent YAML file."),
        ("create", "Phase 1 compatibility alias that validates an agent YAML file."),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("path", type=Path, help="Path to an agent YAML file.")

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


def run_validate(path: Path) -> int:
    try:
        config = load_agent_config(path)
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

    normalized = config.model_dump(mode="json", exclude_none=True)
    print(f"Validation succeeded for {path}")
    print(json.dumps(normalized, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(prog=Path(sys.argv[0]).name)
    args = parser.parse_args(args_list)

    if args.command in {"validate", "create"}:
        return run_validate(args.path)

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
