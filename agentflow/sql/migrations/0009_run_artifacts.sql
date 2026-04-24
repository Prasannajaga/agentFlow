CREATE TABLE IF NOT EXISTS run_artifacts (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    artifact_type VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    mime_type VARCHAR(120),
    size_bytes BIGINT,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_run_artifacts_run_id_created_at
    ON run_artifacts(run_id, created_at);
