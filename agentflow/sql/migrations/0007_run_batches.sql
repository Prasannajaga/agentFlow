CREATE TABLE IF NOT EXISTS run_batches (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_definitions(id),
    version_id UUID NOT NULL REFERENCES agent_versions(id),
    name VARCHAR(120),
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_run_batches_agent_id_created_at
    ON run_batches(agent_id, created_at);

CREATE TABLE IF NOT EXISTS run_batch_items (
    id UUID PRIMARY KEY,
    batch_id UUID NOT NULL REFERENCES run_batches(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    preset_id UUID REFERENCES agent_input_presets(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_run_batch_items_batch_id_run_id UNIQUE (batch_id, run_id)
);

CREATE INDEX IF NOT EXISTS ix_run_batch_items_batch_id
    ON run_batch_items(batch_id);

CREATE INDEX IF NOT EXISTS ix_run_batch_items_run_id
    ON run_batch_items(run_id);
