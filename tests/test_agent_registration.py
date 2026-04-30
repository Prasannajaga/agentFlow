from __future__ import annotations

from sqlalchemy import select

import pytest

from agentflow.db.models import AgentDefinition, AgentVersion
from agentflow.services.agent_registry import register_agent_from_yaml_text
from conftest import make_agent_yaml

pytestmark = pytest.mark.db


def test_register_new_agent_creates_definition_and_version(db_session_factory) -> None:
    raw_yaml = make_agent_yaml(name="registration-agent")

    result = register_agent_from_yaml_text(raw_yaml, session_factory=db_session_factory)

    with db_session_factory() as session:
        definition = session.get(AgentDefinition, result.agent_id)
        version = session.get(AgentVersion, result.version_id)

    assert definition is not None
    assert definition.name == "registration-agent"

    assert version is not None
    assert version.version_number == 1
    assert version.config_hash == result.config_hash
    assert version.raw_yaml == raw_yaml
    assert version.normalized_config_json["name"] == "registration-agent"


def test_registering_changed_yaml_creates_new_immutable_version(db_session_factory) -> None:
    raw_yaml_v1 = make_agent_yaml(name="versioned-agent", extra={"system_prompt": "v1 prompt"})
    first = register_agent_from_yaml_text(raw_yaml_v1, session_factory=db_session_factory)

    raw_yaml_v2 = make_agent_yaml(name="versioned-agent", extra={"system_prompt": "v2 prompt"})
    second = register_agent_from_yaml_text(
        raw_yaml_v2,
        agent_id=first.agent_id,
        session_factory=db_session_factory,
    )

    with db_session_factory() as session:
        versions = session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == first.agent_id)
            .order_by(AgentVersion.version_number.asc())
        ).scalars().all()

    assert first.version_number == 1
    assert second.version_number == 2
    assert len(versions) == 2
    assert versions[0].raw_yaml == raw_yaml_v1
    assert versions[1].raw_yaml == raw_yaml_v2
    assert versions[0].normalized_config_json["system_prompt"] == "v1 prompt"
    assert versions[1].normalized_config_json["system_prompt"] == "v2 prompt"


def test_registering_same_config_again_creates_next_version_row(db_session_factory) -> None:
    raw_yaml = make_agent_yaml(name="idempotency-check-agent")
    first = register_agent_from_yaml_text(raw_yaml, session_factory=db_session_factory)
    second = register_agent_from_yaml_text(
        raw_yaml,
        agent_id=first.agent_id,
        session_factory=db_session_factory,
    )

    with db_session_factory() as session:
        versions = session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == first.agent_id)
            .order_by(AgentVersion.version_number.asc())
        ).scalars().all()

    assert second.version_number == 2
    assert second.config_hash == first.config_hash
    assert [version.version_number for version in versions] == [1, 2]
