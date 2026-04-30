from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from agentflow.db.models import AgentRun, AgentVersion, VersionLabel
from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.import_export_service import export_agent_package, import_agent_package
from agentflow.services.label_service import add_run_label, add_version_label
from agentflow.services.preset_service import create_input_preset, list_input_presets
from agentflow.services.agent_runner import create_run_for_agent
from conftest import make_agent_yaml

pytestmark = pytest.mark.db


def test_export_and_import_preserve_versions_and_metadata(tmp_path: Path, db_session_factory) -> None:
    first = register_agent_from_yaml_text(
        make_agent_yaml(name="portable-agent", extra={"system_prompt": "portable-v1"}),
        session_factory=db_session_factory,
    )
    second = register_agent_from_yaml_text(
        make_agent_yaml(name="portable-agent", extra={"system_prompt": "portable-v2"}),
        agent_id=first.agent_id,
        session_factory=db_session_factory,
    )

    add_version_label(first.version_id, "stable", session_factory=db_session_factory)
    add_version_label(second.version_id, "candidate", session_factory=db_session_factory)
    create_input_preset(
        first.agent_id,
        name="baseline",
        input_json={"topic": "tests"},
        description="baseline preset",
        session_factory=db_session_factory,
    )

    labeled_run = create_run_for_agent(first.agent_id, session_factory=db_session_factory).run
    add_run_label(labeled_run.run_id, "internal-only", session_factory=db_session_factory)

    export_dir = tmp_path / "agent-export"
    export_result = export_agent_package(first.agent_id, export_dir, session_factory=db_session_factory)

    assert export_result.version_count == 2
    assert (export_dir / "manifest.json").is_file()
    assert (export_dir / "agent_definition.json").is_file()
    assert (export_dir / "presets.json").is_file()
    assert (export_dir / "version_labels.json").is_file()
    assert (export_dir / "versions" / "001.yaml").is_file()
    assert (export_dir / "versions" / "001.meta.json").is_file()
    assert (export_dir / "versions" / "002.yaml").is_file()
    assert (export_dir / "versions" / "002.meta.json").is_file()

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["includes"]["run_labels"] is False

    import_result = import_agent_package(export_dir, session_factory=db_session_factory)

    with db_session_factory() as session:
        source_versions = session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == first.agent_id)
            .order_by(AgentVersion.version_number.asc())
        ).scalars().all()
        imported_versions = session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == import_result.new_agent_id)
            .order_by(AgentVersion.version_number.asc())
        ).scalars().all()

    assert [version.version_number for version in source_versions] == [1, 2]
    assert [version.version_number for version in imported_versions] == [1, 2]

    for source, imported in zip(source_versions, imported_versions, strict=True):
        assert imported.version_number == source.version_number
        assert imported.config_hash == source.config_hash
        assert imported.raw_yaml == source.raw_yaml
        assert imported.normalized_config_json == source.normalized_config_json

    imported_presets = list_input_presets(import_result.new_agent_id, session_factory=db_session_factory)
    assert imported_presets is not None
    assert len(imported_presets) == 1
    assert imported_presets[0].name == "baseline"
    assert imported_presets[0].input_json == {"topic": "tests"}

    with db_session_factory() as session:
        source_labels = session.execute(
            select(AgentVersion.version_number, VersionLabel.label)
            .join(VersionLabel, VersionLabel.version_id == AgentVersion.id)
            .where(AgentVersion.agent_id == first.agent_id)
            .order_by(AgentVersion.version_number.asc(), VersionLabel.label.asc())
        ).all()
        imported_labels = session.execute(
            select(AgentVersion.version_number, VersionLabel.label)
            .join(VersionLabel, VersionLabel.version_id == AgentVersion.id)
            .where(AgentVersion.agent_id == import_result.new_agent_id)
            .order_by(AgentVersion.version_number.asc(), VersionLabel.label.asc())
        ).all()
        imported_run_count = session.execute(
            select(func.count()).select_from(AgentRun).where(AgentRun.agent_id == import_result.new_agent_id)
        ).scalar_one()

    assert [(row.version_number, row.label) for row in imported_labels] == [
        (row.version_number, row.label) for row in source_labels
    ]
    assert imported_run_count == 0
