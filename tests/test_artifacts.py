from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.agent_runner import create_run_for_agent, execute_claimed_run
from agentflow.services.artifact_service import list_run_artifacts
from agentflow.services.run_queries import claim_next_pending_run
from conftest import make_agent_yaml

pytestmark = pytest.mark.db


def test_successful_run_writes_output_artifact_to_storage_dir(artifact_storage_dir: Path, db_session_factory) -> None:
    registration = register_agent_from_yaml_text(make_agent_yaml(), session_factory=db_session_factory)
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)

    claimed = claim_next_pending_run(worker_id="artifact-worker", session_factory=db_session_factory)
    assert claimed is not None

    completed = execute_claimed_run(claimed, session_factory=db_session_factory)
    artifacts = list_run_artifacts(completed.run_id, session_factory=db_session_factory)

    assert artifacts is not None
    assert len(artifacts) >= 1

    output_artifact = next(artifact for artifact in artifacts if artifact.name == "output.json")
    artifact_path = Path(output_artifact.file_path).resolve()

    assert output_artifact.artifact_type == "json"
    assert artifact_path.is_file()
    assert artifact_path.is_relative_to(artifact_storage_dir.resolve())

    file_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert file_payload == completed.output_json
