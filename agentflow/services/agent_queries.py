from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentVersion
from agentflow.db.session import create_session_factory


@dataclass(frozen=True)
class AgentVersionSummary:
    version_id: uuid.UUID
    version_number: int
    config_hash: str
    created_at: datetime


@dataclass(frozen=True)
class AgentSummary:
    agent_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    latest_version: AgentVersionSummary | None


@dataclass(frozen=True)
class AgentDetail:
    agent_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    latest_version: AgentVersionSummary | None


def list_registered_agents(
    session_factory: sessionmaker[Session] | None = None,
) -> list[AgentSummary]:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        rows = session.execute(_agent_with_latest_version_query()).all()

    return [
        AgentSummary(
            agent_id=row.id,
            name=row.name,
            description=row.description,
            created_at=row.created_at,
            latest_version=_build_version_summary(
                version_id=row.latest_version_id,
                version_number=row.latest_version_number,
                config_hash=row.latest_config_hash,
                created_at=row.latest_version_created_at,
            ),
        )
        for row in rows
    ]


def get_registered_agent(
    agent_id: uuid.UUID,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentDetail | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        row = session.execute(
            _agent_with_latest_version_query().where(AgentDefinition.id == agent_id)
        ).one_or_none()

    if row is None:
        return None

    return AgentDetail(
        agent_id=row.id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
        latest_version=_build_version_summary(
            version_id=row.latest_version_id,
            version_number=row.latest_version_number,
            config_hash=row.latest_config_hash,
            created_at=row.latest_version_created_at,
        ),
    )


def list_agent_versions(
    agent_id: uuid.UUID,
    session_factory: sessionmaker[Session] | None = None,
) -> list[AgentVersionSummary] | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        exists = session.execute(
            select(AgentDefinition.id).where(AgentDefinition.id == agent_id)
        ).scalar_one_or_none()
        if exists is None:
            return None

        rows = session.execute(
            select(
                AgentVersion.id,
                AgentVersion.version_number,
                AgentVersion.config_hash,
                AgentVersion.created_at,
            )
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_number.desc(), AgentVersion.created_at.desc())
        ).all()

    return [
        AgentVersionSummary(
            version_id=row.id,
            version_number=row.version_number,
            config_hash=row.config_hash,
            created_at=row.created_at,
        )
        for row in rows
    ]


def _agent_with_latest_version_query() -> Select[tuple[object, ...]]:
    latest_version_number_subquery = (
        select(
            AgentVersion.agent_id.label("agent_id"),
            func.max(AgentVersion.version_number).label("latest_version_number"),
        )
        .group_by(AgentVersion.agent_id)
        .subquery()
    )

    latest_version_subquery = (
        select(
            AgentVersion.agent_id.label("agent_id"),
            AgentVersion.id.label("latest_version_id"),
            AgentVersion.version_number.label("latest_version_number"),
            AgentVersion.config_hash.label("latest_config_hash"),
            AgentVersion.created_at.label("latest_version_created_at"),
        )
        .join(
            latest_version_number_subquery,
            (AgentVersion.agent_id == latest_version_number_subquery.c.agent_id)
            & (AgentVersion.version_number == latest_version_number_subquery.c.latest_version_number),
        )
        .subquery()
    )

    return (
        select(
            AgentDefinition.id,
            AgentDefinition.name,
            AgentDefinition.description,
            AgentDefinition.created_at,
            AgentDefinition.updated_at,
            latest_version_subquery.c.latest_version_id,
            latest_version_subquery.c.latest_version_number,
            latest_version_subquery.c.latest_config_hash,
            latest_version_subquery.c.latest_version_created_at,
        )
        .outerjoin(
            latest_version_subquery,
            AgentDefinition.id == latest_version_subquery.c.agent_id,
        )
        .order_by(AgentDefinition.created_at.desc(), AgentDefinition.id.desc())
    )


def _build_version_summary(
    *,
    version_id: uuid.UUID | None,
    version_number: int | None,
    config_hash: str | None,
    created_at: datetime | None,
) -> AgentVersionSummary | None:
    if version_id is None or version_number is None or config_hash is None or created_at is None:
        return None

    return AgentVersionSummary(
        version_id=version_id,
        version_number=version_number,
        config_hash=config_hash,
        created_at=created_at,
    )
