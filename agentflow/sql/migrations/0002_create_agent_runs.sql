CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_definitions(id),
    version_id UUID NOT NULL REFERENCES agent_versions(id),
    status VARCHAR(32) NOT NULL,
    input_json JSON,
    resolved_config_json JSON NOT NULL,
    output_json JSON,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
