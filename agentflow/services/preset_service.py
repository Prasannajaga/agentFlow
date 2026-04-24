from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentInputPreset, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.agent_runner import PreparedAgentRun, create_run_for_agent


class PresetError(RuntimeError):
    """Base error for input preset operations."""


class PresetInvalidError(PresetError):
    pass


class PresetDuplicateError(PresetError):
    pass


class PresetNotFoundError(PresetError):
    pass


class PresetAgentNotFoundError(PresetError):
    pass


@dataclass(frozen=True)
class InputPresetRecord:
    preset_id: uuid.UUID
    agent_id: uuid.UUID
    name: str
    description: str | None
    input_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def create_input_preset(
    agent_id: uuid.UUID,
    *,
    name: str,
    input_json: dict[str, Any],
    description: str | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> InputPresetRecord:
    normalized_name = _normalize_preset_name(name)
    if not isinstance(input_json, dict):
        raise PresetInvalidError("Preset input_json must be a JSON object.")
    normalized_description = description.strip() if description and description.strip() else None
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentDefinition, agent_id) is None:
                raise PresetAgentNotFoundError(f"Agent not found: {agent_id}")
            preset = AgentInputPreset(
                agent_id=agent_id,
                name=normalized_name,
                description=normalized_description,
                input_json=dict(input_json),
                created_at=now,
                updated_at=now,
            )
            session.add(preset)
            try:
                session.flush()
            except IntegrityError as exc:
                raise PresetDuplicateError(f"Agent already has preset named: {normalized_name}") from exc
            return _build_preset_record(preset)


def list_input_presets(
    agent_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[InputPresetRecord] | None:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        if session.get(AgentDefinition, agent_id) is None:
            return None
        rows = session.execute(
            select(AgentInputPreset)
            .where(AgentInputPreset.agent_id == agent_id)
            .order_by(AgentInputPreset.name.asc(), AgentInputPreset.created_at.asc())
        ).scalars().all()
        return [_build_preset_record(preset) for preset in rows]


def get_input_preset(
    preset_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> InputPresetRecord | None:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        preset = session.get(AgentInputPreset, preset_id)
        return _build_preset_record(preset) if preset is not None else None


def run_from_preset(
    preset_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> PreparedAgentRun:
    preset = get_input_preset(preset_id, session_factory=session_factory)
    if preset is None:
        raise PresetNotFoundError(f"Preset not found: {preset_id}")
    return create_run_for_agent(
        preset.agent_id,
        input_json=dict(preset.input_json),
        session_factory=session_factory,
    )


def _normalize_preset_name(name: str) -> str:
    normalized = " ".join(name.strip().split())
    if not normalized:
        raise PresetInvalidError("Preset name cannot be empty.")
    if len(normalized) > 120:
        raise PresetInvalidError("Preset name cannot be longer than 120 characters.")
    return normalized


def _build_preset_record(preset: AgentInputPreset) -> InputPresetRecord:
    return InputPresetRecord(
        preset_id=preset.id,
        agent_id=preset.agent_id,
        name=preset.name,
        description=preset.description,
        input_json=dict(preset.input_json),
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )
