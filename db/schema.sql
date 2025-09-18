-- schema.sql — TMIV history + metadata (SQLite)

BEGIN;

-- Główna tabela historii „uruchomień”
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL UNIQUE,       -- np. run_20250918_112233Z
    dataset           TEXT    NOT NULL,              -- nazwa zbioru (string przyjazny UI)
    target            TEXT    NOT NULL,              -- kolumna celu

    problem_type      TEXT,                          -- 'regression' | 'classification' | 'other' | ...
    engine            TEXT,                          -- 'sklearn' | 'lightgbm' | 'xgboost' | 'catboost' | 'auto' | ...

    status            TEXT    NOT NULL DEFAULT 'running',
                                                     -- 'running' | 'completed' | 'failed' | 'cancelled'
    created_at        TEXT    NOT NULL,              -- ISO-8601 UTC z 'Z' (zapisywane przez aplikację)
    updated_at        TEXT    NOT NULL,              -- jw.
    duration_seconds  REAL,                          -- czas treningu w sekundach

    metrics_json      TEXT    NOT NULL DEFAULT '{}', -- metryki (JSON)
    parameters_json   TEXT    NOT NULL DEFAULT '{}', -- parametry/trening (JSON)
    tags_json         TEXT    NOT NULL DEFAULT '[]', -- tagi (JSON array)

    notes             TEXT,                          -- notatki/komentarz
    schema_version    INTEGER NOT NULL DEFAULT 3,    -- wersja schematu używana przez aplikację

    CONSTRAINT chk_status CHECK (status IN ('running','completed','failed','cancelled'))
);

-- Indeksy pod najczęstsze filtry
CREATE INDEX IF NOT EXISTS idx_runs_dataset      ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_runs_target       ON runs(target);
CREATE INDEX IF NOT EXISTS idx_runs_engine       ON runs(engine);
CREATE INDEX IF NOT EXISTS idx_runs_status       ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at   ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_problem_type ON runs(problem_type);

-- Metadane systemowe (klucz → wartość), m.in. schema_version, domyślne modele
CREATE TABLE IF NOT EXISTS system_metadata (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL                    -- zazwyczaj datetime('now') po stronie SQLite (UTC)
);

-- Opcjonalnie: predefiniuj wersję schematu (aplikacja i tak nadpisze)
INSERT OR IGNORE INTO system_metadata (key, value, updated_at)
VALUES ('schema_version', '3', datetime('now'));

COMMIT;
