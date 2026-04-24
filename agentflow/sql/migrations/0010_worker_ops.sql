CREATE TABLE IF NOT EXISTS worker_heartbeats (
    id UUID PRIMARY KEY,
    worker_name VARCHAR(255) NOT NULL,
    host VARCHAR(255),
    pid INTEGER,
    status VARCHAR(32) NOT NULL,
    last_heartbeat_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata_json JSON,
    CONSTRAINT uq_worker_heartbeats_worker_name UNIQUE (worker_name)
);

CREATE INDEX IF NOT EXISTS ix_worker_heartbeats_last_heartbeat_at
    ON worker_heartbeats(last_heartbeat_at);

ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS claimed_by_worker VARCHAR(255),
    ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_agent_runs_status_claimed_at
    ON agent_runs(status, claimed_at);

CREATE INDEX IF NOT EXISTS ix_agent_runs_claimed_by_worker_status
    ON agent_runs(claimed_by_worker, status);
