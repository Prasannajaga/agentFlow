from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentVersion
from agentflow.db.session import create_session_factory
from agentflow.services.yaml_loader import load_agent_document, normalize_agent_config
from agentflow.utils.hashing import compute_config_hash


@dataclass(frozen=True)
class AgentRegistrationResult:
    agent_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    config_hash: str


class AgentRegistrationError(RuntimeError):
    """Base error for agent registration failures."""


class AgentRegistrationAgentNotFoundError(AgentRegistrationError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"Agent not found: {agent_id}")
        self.agent_id = agent_id


def register_agent(
    path: Path,
    *,
    agent_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRegistrationResult:
    document = load_agent_document(path)
    normalized_config = normalize_agent_config(document.config)
    config_hash = compute_config_hash(normalized_config)

    now = datetime.now(timezone.utc)
    version_id = uuid.uuid4()
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        if agent_id is None:
            agent_id = uuid.uuid4()
            version_number = _register_new_agent(
                session=session,
                agent_id=agent_id,
                version_id=version_id,
                name=document.config.name,
                description=document.config.description,
                raw_yaml=document.raw_yaml,
                normalized_config=normalized_config,
                config_hash=config_hash,
                created_at=now,
            )
        else:
            version_number = _register_agent_version(
                session=session,
                agent_id=agent_id,
                version_id=version_id,
                name=document.config.name,
                description=document.config.description,
                raw_yaml=document.raw_yaml,
                normalized_config=normalized_config,
                config_hash=config_hash,
                created_at=now,
            )

    return AgentRegistrationResult(
        agent_id=agent_id,
        version_id=version_id,
        version_number=version_number,
        config_hash=config_hash,
    )


def _register_new_agent(
    session: Session,
    *,
    agent_id: uuid.UUID,
    version_id: uuid.UUID,
    name: str,
    description: str | None,
    raw_yaml: str,
    normalized_config: dict[str, object],
    config_hash: str,
    created_at: datetime,
) -> int:
    with session.begin():
        definition = AgentDefinition(
            id=agent_id,
            name=name,
            description=description,
            created_at=created_at,
            updated_at=created_at,
        )
        version = AgentVersion(
            id=version_id,
            agent_id=agent_id,
            version_number=1,
            raw_yaml=raw_yaml,
            normalized_config_json=normalized_config,
            config_hash=config_hash,
            created_at=created_at,
        )

        session.add(definition)
        session.add(version)
    return 1


def _register_agent_version(
    session: Session,
    *,
    agent_id: uuid.UUID,
    version_id: uuid.UUID,
    name: str,
    description: str | None,
    raw_yaml: str,
    normalized_config: dict[str, object],
    config_hash: str,
    created_at: datetime,
) -> int:
    with session.begin():
        definition = session.get(AgentDefinition, agent_id, with_for_update=True)
        if definition is None:
            raise AgentRegistrationAgentNotFoundError(agent_id)

        current_max = session.execute(
            select(func.max(AgentVersion.version_number))
            .where(AgentVersion.agent_id == agent_id)
        ).scalar_one()
        version_number = int(current_max or 0) + 1

        definition.name = name
        definition.description = description
        definition.updated_at = created_at

        version = AgentVersion(
            id=version_id,
            agent_id=agent_id,
            version_number=version_number,
            raw_yaml=raw_yaml,
            normalized_config_json=normalized_config,
            config_hash=config_hash,
            created_at=created_at,
        )
        session.add(version)

    return version_number
