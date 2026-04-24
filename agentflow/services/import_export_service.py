from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentInputPreset, AgentVersion, VersionLabel, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.yaml_loader import AgentYamlError, parse_yaml_text

EXPORT_FORMAT_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
AGENT_DEFINITION_FILENAME = "agent_definition.json"
VERSION_LABELS_FILENAME = "version_labels.json"
PRESETS_FILENAME = "presets.json"
VERSIONS_DIRNAME = "versions"


class AgentPackageError(RuntimeError):
    """Base error for agent package export/import failures."""


class AgentExportNotFoundError(AgentPackageError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"Agent not found: {agent_id}")
        self.agent_id = agent_id


class AgentPackageValidationError(AgentPackageError):
    pass


@dataclass(frozen=True)
class ExportedAgentDefinition:
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ExportedVersion:
    version_number: int
    raw_yaml: str
    normalized_config_json: dict[str, Any]
    config_hash: str
    created_at: datetime


@dataclass(frozen=True)
class ExportedPreset:
    name: str
    description: str | None
    input_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ExportedVersionLabel:
    version_number: int
    label: str
    created_at: datetime


@dataclass(frozen=True)
class AgentExportData:
    source_agent_id: uuid.UUID
    agent_definition: ExportedAgentDefinition
    versions: tuple[ExportedVersion, ...]
    presets: tuple[ExportedPreset, ...]
    version_labels: tuple[ExportedVersionLabel, ...]
    exported_at: datetime


@dataclass(frozen=True)
class AgentExportResult:
    source_agent_id: uuid.UUID
    version_count: int
    preset_count: int
    version_label_count: int
    output_path: Path


@dataclass(frozen=True)
class AgentImportResult:
    new_agent_id: uuid.UUID
    version_count: int
    preset_count: int
    version_label_count: int


def export_agent_package(
    agent_id: uuid.UUID,
    output_path: Path,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentExportResult:
    export_data = collect_agent_export_data(agent_id, session_factory=session_factory)
    write_export_package(export_data, output_path)
    return AgentExportResult(
        source_agent_id=agent_id,
        version_count=len(export_data.versions),
        preset_count=len(export_data.presets),
        version_label_count=len(export_data.version_labels),
        output_path=output_path.resolve(),
    )


def collect_agent_export_data(
    agent_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentExportData:
    resolved_session_factory = session_factory or create_session_factory()
    with resolved_session_factory() as session:
        agent = session.get(AgentDefinition, agent_id)
        if agent is None:
            raise AgentExportNotFoundError(agent_id)

        versions = session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_number.asc(), AgentVersion.created_at.asc())
        ).scalars().all()

        presets = session.execute(
            select(AgentInputPreset)
            .where(AgentInputPreset.agent_id == agent_id)
            .order_by(AgentInputPreset.name.asc(), AgentInputPreset.created_at.asc())
        ).scalars().all()

        labels = session.execute(
            select(VersionLabel, AgentVersion.version_number)
            .join(AgentVersion, AgentVersion.id == VersionLabel.version_id)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_number.asc(), VersionLabel.label.asc(), VersionLabel.created_at.asc())
        ).all()

    if not versions:
        raise AgentPackageValidationError(f"Cannot export agent {agent_id}: no versions found.")

    exported_versions = tuple(
        ExportedVersion(
            version_number=version.version_number,
            raw_yaml=version.raw_yaml,
            normalized_config_json=dict(version.normalized_config_json),
            config_hash=version.config_hash,
            created_at=version.created_at,
        )
        for version in versions
    )
    exported_presets = tuple(
        ExportedPreset(
            name=preset.name,
            description=preset.description,
            input_json=dict(preset.input_json),
            created_at=preset.created_at,
            updated_at=preset.updated_at,
        )
        for preset in presets
    )
    exported_labels = tuple(
        ExportedVersionLabel(
            version_number=version_number,
            label=version_label.label,
            created_at=version_label.created_at,
        )
        for version_label, version_number in labels
    )

    return AgentExportData(
        source_agent_id=agent.id,
        agent_definition=ExportedAgentDefinition(
            name=agent.name,
            description=agent.description,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        ),
        versions=exported_versions,
        presets=exported_presets,
        version_labels=exported_labels,
        exported_at=utc_now(),
    )


def write_export_package(export_data: AgentExportData, output_path: Path) -> None:
    if output_path.exists():
        if output_path.is_dir() and any(output_path.iterdir()):
            raise AgentPackageValidationError(
                f"Export path is not empty: {output_path}"
            )
        raise AgentPackageValidationError(
            f"Export path already exists: {output_path}"
        )

    parent = output_path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    temp_dir = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_json_file(
            temp_dir / MANIFEST_FILENAME,
            _manifest_payload(export_data),
        )
        _write_json_file(
            temp_dir / AGENT_DEFINITION_FILENAME,
            {
                "name": export_data.agent_definition.name,
                "description": export_data.agent_definition.description,
                "created_at": _to_iso8601(export_data.agent_definition.created_at),
                "updated_at": _to_iso8601(export_data.agent_definition.updated_at),
            },
        )
        _write_json_file(
            temp_dir / PRESETS_FILENAME,
            [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "input_json": preset.input_json,
                    "created_at": _to_iso8601(preset.created_at),
                    "updated_at": _to_iso8601(preset.updated_at),
                }
                for preset in export_data.presets
            ],
        )
        _write_json_file(
            temp_dir / VERSION_LABELS_FILENAME,
            [
                {
                    "version_number": label.version_number,
                    "label": label.label,
                    "created_at": _to_iso8601(label.created_at),
                }
                for label in export_data.version_labels
            ],
        )

        versions_dir = temp_dir / VERSIONS_DIRNAME
        versions_dir.mkdir(parents=True, exist_ok=False)

        width = max(3, len(str(max(version.version_number for version in export_data.versions))))
        for version in export_data.versions:
            filename = f"{version.version_number:0{width}d}"
            (versions_dir / f"{filename}.yaml").write_text(version.raw_yaml, encoding="utf-8")
            _write_json_file(
                versions_dir / f"{filename}.meta.json",
                {
                    "version_number": version.version_number,
                    "normalized_config_json": version.normalized_config_json,
                    "config_hash": version.config_hash,
                    "created_at": _to_iso8601(version.created_at),
                },
            )

        temp_dir.rename(output_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def import_agent_package(
    package_path: Path,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentImportResult:
    package_data = read_import_package(package_path)
    resolved_session_factory = session_factory or create_session_factory()

    new_agent_id = uuid.uuid4()
    version_id_by_number: dict[int, uuid.UUID] = {}

    with resolved_session_factory() as session:
        with session.begin():
            definition = AgentDefinition(
                id=new_agent_id,
                name=package_data.agent_definition.name,
                description=package_data.agent_definition.description,
                created_at=package_data.agent_definition.created_at,
                updated_at=package_data.agent_definition.updated_at,
            )
            session.add(definition)

            for version in package_data.versions:
                version_id = uuid.uuid4()
                version_id_by_number[version.version_number] = version_id
                session.add(
                    AgentVersion(
                        id=version_id,
                        agent_id=new_agent_id,
                        version_number=version.version_number,
                        raw_yaml=version.raw_yaml,
                        normalized_config_json=dict(version.normalized_config_json),
                        config_hash=version.config_hash,
                        created_at=version.created_at,
                    )
                )

            for preset in package_data.presets:
                session.add(
                    AgentInputPreset(
                        id=uuid.uuid4(),
                        agent_id=new_agent_id,
                        name=preset.name,
                        description=preset.description,
                        input_json=dict(preset.input_json),
                        created_at=preset.created_at,
                        updated_at=preset.updated_at,
                    )
                )

            for label in package_data.version_labels:
                version_id = version_id_by_number.get(label.version_number)
                if version_id is None:
                    raise AgentPackageValidationError(
                        f"Version label references unknown version_number: {label.version_number}"
                    )
                session.add(
                    VersionLabel(
                        id=uuid.uuid4(),
                        version_id=version_id,
                        label=label.label,
                        created_at=label.created_at,
                    )
                )

    return AgentImportResult(
        new_agent_id=new_agent_id,
        version_count=len(package_data.versions),
        preset_count=len(package_data.presets),
        version_label_count=len(package_data.version_labels),
    )


def read_import_package(package_path: Path) -> AgentExportData:
    if not package_path.exists():
        raise AgentPackageValidationError(f"Package path does not exist: {package_path}")
    if not package_path.is_dir():
        raise AgentPackageValidationError(f"Package path must be a directory: {package_path}")

    manifest_payload = _read_json_file(package_path / MANIFEST_FILENAME)
    if not isinstance(manifest_payload, dict):
        raise AgentPackageValidationError("manifest.json must contain a JSON object.")
    format_version = manifest_payload.get("format_version")
    if format_version != EXPORT_FORMAT_VERSION:
        raise AgentPackageValidationError(
            f"Unsupported export format_version: {format_version}"
        )

    exported_at = _parse_datetime(
        manifest_payload.get("exported_at"),
        field_name="manifest.exported_at",
    )
    source_agent_id = _parse_uuid(
        manifest_payload.get("source_agent_id"),
        field_name="manifest.source_agent_id",
    )

    definition_payload = _read_json_file(package_path / AGENT_DEFINITION_FILENAME)
    if not isinstance(definition_payload, dict):
        raise AgentPackageValidationError("agent_definition.json must contain a JSON object.")
    agent_definition = ExportedAgentDefinition(
        name=_parse_non_empty_string(definition_payload.get("name"), field_name="agent_definition.name"),
        description=_parse_optional_string(definition_payload.get("description"), field_name="agent_definition.description"),
        created_at=_parse_datetime(definition_payload.get("created_at"), field_name="agent_definition.created_at"),
        updated_at=_parse_datetime(definition_payload.get("updated_at"), field_name="agent_definition.updated_at"),
    )

    versions_dir = package_path / VERSIONS_DIRNAME
    if not versions_dir.exists() or not versions_dir.is_dir():
        raise AgentPackageValidationError("versions directory is required.")
    versions = _read_versions(versions_dir)
    _validate_version_order(versions)

    preset_payload = _read_json_file(package_path / PRESETS_FILENAME)
    if not isinstance(preset_payload, list):
        raise AgentPackageValidationError("presets.json must contain a JSON array.")
    presets = tuple(_parse_preset(item, index=index) for index, item in enumerate(preset_payload))

    labels_payload = _read_json_file(package_path / VERSION_LABELS_FILENAME)
    if not isinstance(labels_payload, list):
        raise AgentPackageValidationError("version_labels.json must contain a JSON array.")
    labels = tuple(_parse_version_label(item, index=index) for index, item in enumerate(labels_payload))

    manifest_version_count = manifest_payload.get("version_count")
    if manifest_version_count is not None and manifest_version_count != len(versions):
        raise AgentPackageValidationError(
            f"Manifest version_count mismatch: expected {manifest_version_count}, found {len(versions)}"
        )
    manifest_preset_count = manifest_payload.get("preset_count")
    if manifest_preset_count is not None and manifest_preset_count != len(presets):
        raise AgentPackageValidationError(
            f"Manifest preset_count mismatch: expected {manifest_preset_count}, found {len(presets)}"
        )

    version_numbers = {version.version_number for version in versions}
    for label in labels:
        if label.version_number not in version_numbers:
            raise AgentPackageValidationError(
                f"version_labels.json contains unknown version_number: {label.version_number}"
            )

    return AgentExportData(
        source_agent_id=source_agent_id,
        agent_definition=agent_definition,
        versions=versions,
        presets=presets,
        version_labels=labels,
        exported_at=exported_at,
    )


def _manifest_payload(export_data: AgentExportData) -> dict[str, Any]:
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "exported_at": _to_iso8601(export_data.exported_at),
        "source_agent_id": str(export_data.source_agent_id),
        "agent_name": export_data.agent_definition.name,
        "version_count": len(export_data.versions),
        "preset_count": len(export_data.presets),
        "version_label_count": len(export_data.version_labels),
        "includes": {
            "versions": True,
            "presets": True,
            "version_labels": True,
            "run_labels": False,
        },
    }


def _read_versions(versions_dir: Path) -> tuple[ExportedVersion, ...]:
    meta_paths = sorted(versions_dir.glob("*.meta.json"))
    if not meta_paths:
        raise AgentPackageValidationError("versions directory must include *.meta.json files.")

    versions: list[ExportedVersion] = []
    for meta_path in meta_paths:
        yaml_path = versions_dir / f"{meta_path.name[:-10]}.yaml"
        if not yaml_path.exists():
            raise AgentPackageValidationError(
                f"Missing YAML file for version metadata: {yaml_path.name}"
            )

        meta_payload = _read_json_file(meta_path)
        if not isinstance(meta_payload, dict):
            raise AgentPackageValidationError(f"{meta_path.name} must contain a JSON object.")

        raw_yaml = yaml_path.read_text(encoding="utf-8")
        try:
            parse_yaml_text(raw_yaml)
        except AgentYamlError as exc:
            raise AgentPackageValidationError(
                f"Invalid YAML in {yaml_path.name}: {exc}"
            ) from exc

        normalized_config_json = meta_payload.get("normalized_config_json")
        if not isinstance(normalized_config_json, dict):
            raise AgentPackageValidationError(
                f"{meta_path.name}: normalized_config_json must be a JSON object."
            )

        versions.append(
            ExportedVersion(
                version_number=_parse_positive_int(meta_payload.get("version_number"), field_name=f"{meta_path.name}.version_number"),
                raw_yaml=raw_yaml,
                normalized_config_json=normalized_config_json,
                config_hash=_parse_non_empty_string(meta_payload.get("config_hash"), field_name=f"{meta_path.name}.config_hash"),
                created_at=_parse_datetime(meta_payload.get("created_at"), field_name=f"{meta_path.name}.created_at"),
            )
        )

    versions.sort(key=lambda item: item.version_number)
    return tuple(versions)


def _validate_version_order(versions: tuple[ExportedVersion, ...]) -> None:
    version_numbers = [version.version_number for version in versions]
    expected = list(range(1, len(versions) + 1))
    if version_numbers != expected:
        raise AgentPackageValidationError(
            f"Version numbering must be contiguous starting at 1. Found: {version_numbers}"
        )


def _parse_preset(value: object, *, index: int) -> ExportedPreset:
    if not isinstance(value, dict):
        raise AgentPackageValidationError(f"presets[{index}] must be a JSON object.")
    input_json = value.get("input_json")
    if not isinstance(input_json, dict):
        raise AgentPackageValidationError(f"presets[{index}].input_json must be a JSON object.")

    return ExportedPreset(
        name=_parse_non_empty_string(value.get("name"), field_name=f"presets[{index}].name"),
        description=_parse_optional_string(value.get("description"), field_name=f"presets[{index}].description"),
        input_json=input_json,
        created_at=_parse_datetime(value.get("created_at"), field_name=f"presets[{index}].created_at"),
        updated_at=_parse_datetime(value.get("updated_at"), field_name=f"presets[{index}].updated_at"),
    )


def _parse_version_label(value: object, *, index: int) -> ExportedVersionLabel:
    if not isinstance(value, dict):
        raise AgentPackageValidationError(f"version_labels[{index}] must be a JSON object.")
    return ExportedVersionLabel(
        version_number=_parse_positive_int(
            value.get("version_number"),
            field_name=f"version_labels[{index}].version_number",
        ),
        label=_parse_non_empty_string(value.get("label"), field_name=f"version_labels[{index}].label"),
        created_at=_parse_datetime(value.get("created_at"), field_name=f"version_labels[{index}].created_at"),
    )


def _write_json_file(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_file(path: Path) -> object:
    if not path.exists():
        raise AgentPackageValidationError(f"Missing required file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AgentPackageValidationError(f"Invalid JSON in {path.name}: {exc.msg}") from exc


def _parse_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise AgentPackageValidationError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise AgentPackageValidationError(f"{field_name} must be a non-empty string.")
    return normalized


def _parse_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentPackageValidationError(f"{field_name} must be a string or null.")
    normalized = value.strip()
    return normalized or None


def _parse_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise AgentPackageValidationError(f"{field_name} must be a positive integer.")
    return value


def _parse_datetime(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise AgentPackageValidationError(f"{field_name} must be an ISO-8601 timestamp string.")

    normalized = value.strip()
    if not normalized:
        raise AgentPackageValidationError(f"{field_name} must be an ISO-8601 timestamp string.")

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AgentPackageValidationError(f"{field_name} is not a valid timestamp: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _parse_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if not isinstance(value, str):
        raise AgentPackageValidationError(f"{field_name} must be a UUID string.")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise AgentPackageValidationError(f"{field_name} must be a UUID string.") from exc


def _to_iso8601(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
