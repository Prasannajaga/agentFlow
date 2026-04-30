ALTER TABLE run_code_changes
    ADD COLUMN IF NOT EXISTS runner_type TEXT NOT NULL DEFAULT 'cursor_sdk';

CREATE INDEX IF NOT EXISTS ix_run_code_changes_runner_type
    ON run_code_changes(runner_type);
