from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from agentflow.config import get_settings


@lru_cache
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def create_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    resolved_database_url = database_url or get_settings().database_url
    return sessionmaker(
        bind=get_engine(resolved_database_url),
        autoflush=False,
        expire_on_commit=False,
    )
