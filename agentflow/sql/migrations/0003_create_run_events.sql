CREATE TABLE IF NOT EXISTS run_events (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    event_type VARCHAR(64) NOT NULL,
    message TEXT,
    payload_json JSON,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_run_events_run_id
    ON run_events (run_id);

CREATE INDEX IF NOT EXISTS ix_run_events_run_id_created_at
    ON run_events (run_id, created_at);
