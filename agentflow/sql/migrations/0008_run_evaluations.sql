CREATE TABLE IF NOT EXISTS run_evaluations (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    evaluator_type VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    score DOUBLE PRECISION,
    passed BOOLEAN,
    summary TEXT,
    expected_json JSON,
    actual_json JSON,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_run_evaluations_run_id_created_at
    ON run_evaluations(run_id, created_at);
