CREATE TABLE IF NOT EXISTS run_labels (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    label VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_run_labels_run_id_label UNIQUE (run_id, label)
);

CREATE INDEX IF NOT EXISTS ix_run_labels_label ON run_labels(label);

CREATE TABLE IF NOT EXISTS version_labels (
    id UUID PRIMARY KEY,
    version_id UUID NOT NULL REFERENCES agent_versions(id) ON DELETE CASCADE,
    label VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_version_labels_version_id_label UNIQUE (version_id, label)
);

CREATE INDEX IF NOT EXISTS ix_version_labels_label ON version_labels(label);

CREATE TABLE IF NOT EXISTS agent_input_presets (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_definitions(id) ON DELETE CASCADE,
    name VARCHAR(120) NOT NULL,
    description TEXT,
    input_json JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_agent_input_presets_agent_id_name UNIQUE (agent_id, name)
);

CREATE INDEX IF NOT EXISTS ix_agent_input_presets_agent_id ON agent_input_presets(agent_id);
