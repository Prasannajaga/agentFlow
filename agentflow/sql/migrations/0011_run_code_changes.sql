CREATE TABLE IF NOT EXISTS run_code_changes (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    base_commit_sha TEXT,
    result_commit_sha TEXT,
    commit_message TEXT,
    changed_files_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_run_code_changes_run_id
    ON run_code_changes(run_id);

CREATE INDEX IF NOT EXISTS ix_run_code_changes_result_commit_sha
    ON run_code_changes(result_commit_sha);
