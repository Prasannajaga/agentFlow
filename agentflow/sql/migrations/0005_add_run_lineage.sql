ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS source_run_id UUID REFERENCES agent_runs(id);
