CREATE TABLE IF NOT EXISTS agent_definitions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_versions (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_definitions(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    raw_yaml TEXT NOT NULL,
    normalized_config_json JSON NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_agent_versions_agent_id_version_number
        UNIQUE (agent_id, version_number)
);
