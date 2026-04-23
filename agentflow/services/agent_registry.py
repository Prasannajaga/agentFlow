from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


def register_agent(
    path: Path,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRegistrationResult:
    document = load_agent_document(path)
    normalized_config = normalize_agent_config(document.config)
    config_hash = compute_config_hash(normalized_config)

    now = datetime.now(timezone.utc)
    agent_id = uuid.uuid4()
    version_id = uuid.uuid4()
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        _register_agent_version(
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
        version_number=1,
        config_hash=config_hash,
    )


def _register_agent_version(
    session: Session,
    agent_id: uuid.UUID,
    version_id: uuid.UUID,
    name: str,
    description: str | None,
    raw_yaml: str,
    normalized_config: dict[str, object],
    config_hash: str,
    created_at: datetime,
) -> None:
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
