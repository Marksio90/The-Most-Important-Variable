-- Tabela uruchomień analizy
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    dataset TEXT,
    target TEXT,
    problem_type TEXT,
    model_name TEXT,
    metrics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);
