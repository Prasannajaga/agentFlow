ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER,
    ADD COLUMN IF NOT EXISTS max_attempts INTEGER,
    ADD COLUMN IF NOT EXISTS last_error_type VARCHAR(64),
    ADD COLUMN IF NOT EXISTS retryable BOOLEAN;

UPDATE agent_runs
SET
    attempt_count = COALESCE(attempt_count, 0),
    max_attempts = CASE
        WHEN max_attempts IS NULL OR max_attempts < 1 THEN 1
        ELSE max_attempts
    END,
    retryable = COALESCE(retryable, TRUE);

ALTER TABLE agent_runs
    ALTER COLUMN attempt_count SET DEFAULT 0,
    ALTER COLUMN attempt_count SET NOT NULL,
    ALTER COLUMN max_attempts SET DEFAULT 1,
    ALTER COLUMN max_attempts SET NOT NULL,
    ALTER COLUMN retryable SET DEFAULT TRUE,
    ALTER COLUMN retryable SET NOT NULL;
