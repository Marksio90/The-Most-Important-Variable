-- ===================================
-- ULEPSZONA STRUKTURA BAZY DANYCH
-- dla trackingu eksperymentów ML
-- ===================================

-- Włącz foreign keys dla SQLite
PRAGMA foreign_keys = ON;

-- Tabela główna dla uruchomień eksperymentów
CREATE TABLE IF NOT EXISTS runs (
    -- Klucz główny
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Unikalne ID uruchomienia (UUID lub custom)
    run_id TEXT UNIQUE NOT NULL,
    
    -- Metadane czasowe
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    
    -- Podstawowe informacje o eksperymencie
    dataset TEXT NOT NULL CHECK(length(dataset) > 0),
    target TEXT NOT NULL CHECK(length(target) > 0),
    problem_type TEXT CHECK(problem_type IN (
        'regression', 'classification', 'clustering', 
        'time_series', 'nlp', 'computer_vision', 'other'
    )),
    
    -- Model i engine
    model_name TEXT,
    engine TEXT,
    model_version TEXT,
    
    -- Status uruchomienia
    status TEXT NOT NULL DEFAULT 'running' CHECK(status IN (
        'queued', 'running', 'completed', 'failed', 'cancelled', 'timeout'
    )),
    
    -- Metryki i wyniki (JSON)
    metrics_json TEXT DEFAULT '{}' CHECK(json_valid(metrics_json)),
    parameters_json TEXT DEFAULT '{}' CHECK(json_valid(parameters_json)),
    
    -- Dodatkowe metadane
    duration_seconds REAL,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    
    -- Organizacja i zarządzanie
    tags TEXT DEFAULT '[]' CHECK(json_valid(tags)),
    notes TEXT,
    user_id TEXT,
    project_name TEXT,
    experiment_name TEXT,
    
    -- Wersjonowanie i śledzenie
    git_commit TEXT,
    code_version TEXT,
    environment TEXT,
    
    -- Metadane systemu
    hostname TEXT,
    python_version TEXT,
    platform TEXT,
    
    -- Ścieżki do artefaktów
    model_path TEXT,
    artifacts_path TEXT,
    logs_path TEXT,
    
    -- Audit trail
    created_by TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    schema_version INTEGER DEFAULT 1,
    
    -- Constrainty biznesowe
    CONSTRAINT chk_duration CHECK(duration_seconds >= 0),
    CONSTRAINT chk_memory CHECK(memory_usage_mb >= 0),
    CONSTRAINT chk_cpu CHECK(cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    CONSTRAINT chk_times CHECK(
        (started_at IS NULL OR started_at >= created_at) AND
        (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
    )
);

-- ===================================
-- INDEKSY DLA OPTYMALNEJ WYDAJNOŚCI
-- ===================================

-- Podstawowe indeksy wyszukiwania
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_runs_target ON runs(target);
CREATE INDEX IF NOT EXISTS idx_runs_model_name ON runs(model_name);
CREATE INDEX IF NOT EXISTS idx_runs_engine ON runs(engine);
CREATE INDEX IF NOT EXISTS idx_runs_problem_type ON runs(problem_type);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

-- Indeksy czasowe (kluczowe dla analytics)
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at);
CREATE INDEX IF NOT EXISTS idx_runs_completed_at ON runs(completed_at);

-- Indeksy organizacyjne
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_name);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_runs_deleted ON runs(is_deleted);

-- Indeksy kompozytowe dla częstych zapytań
CREATE INDEX IF NOT EXISTS idx_runs_dataset_target ON runs(dataset, target);
CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_project_experiment ON runs(project_name, experiment_name);
CREATE INDEX IF NOT EXISTS idx_runs_user_project ON runs(user_id, project_name);

-- Indeks dla aktywnych eksperymentów
CREATE INDEX IF NOT EXISTS idx_runs_active ON runs(status, created_at) 
    WHERE is_deleted = FALSE AND status IN ('queued', 'running');

-- ===================================
-- TABELE POMOCNICZE
-- ===================================

-- Tabela dla szczegółowych metryk (opcjonalna, dla dużych eksperymentów)
CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    step INTEGER DEFAULT 0,
    epoch INTEGER,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, metric_name, step, epoch)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON run_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON run_metrics(step);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON run_metrics(timestamp);

-- Tabela dla logów eksperymentów
CREATE TABLE IF NOT EXISTS run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    log_level TEXT NOT NULL CHECK(log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source TEXT,  -- np. 'training', 'validation', 'preprocessing'
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_logs_run_id ON run_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_logs_level ON run_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON run_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_source ON run_logs(source);

-- Tabela dla artefaktów (modele, ploti, raporty)
CREATE TABLE IF NOT EXISTS run_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    artifact_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK(artifact_type IN (
        'model', 'plot', 'report', 'dataset', 'config', 'other'
    )),
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    mime_type TEXT,
    checksum TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, artifact_name)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON run_artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON run_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_name ON run_artifacts(artifact_name);

-- Tabela dla tagów (normalizowana struktura)
CREATE TABLE IF NOT EXISTS run_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, tag_name)
);

CREATE INDEX IF NOT EXISTS idx_tags_run_id ON run_tags(run_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON run_tags(tag_name);

-- Tabela metadanych systemu
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ===================================
-- TRIGGERY DLA AUTOMATYZACJI
-- ===================================

-- Trigger dla aktualizacji updated_at
CREATE TRIGGER IF NOT EXISTS update_runs_updated_at 
    AFTER UPDATE ON runs
    FOR EACH ROW
BEGIN
    UPDATE runs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger dla automatycznego obliczania duration
CREATE TRIGGER IF NOT EXISTS calculate_run_duration 
    AFTER UPDATE OF completed_at ON runs
    FOR EACH ROW 
    WHEN NEW.completed_at IS NOT NULL AND NEW.started_at IS NOT NULL
BEGIN
    UPDATE runs 
    SET duration_seconds = (
        JULIANDAY(NEW.completed_at) - JULIANDAY(NEW.started_at)
    ) * 86400
    WHERE id = NEW.id;
END;

-- Trigger dla walidacji statusów
CREATE TRIGGER IF NOT EXISTS validate_status_transitions 
    BEFORE UPDATE OF status ON runs
    FOR EACH ROW
BEGIN
    -- Nie pozwól na powrót z 'completed' do 'running'
    SELECT CASE 
        WHEN OLD.status = 'completed' AND NEW.status = 'running' THEN
            RAISE(ABORT, 'Cannot change status from completed to running')
        WHEN OLD.status = 'failed' AND NEW.status = 'running' THEN
            RAISE(ABORT, 'Cannot change status from failed to running')
    END;
END;

-- ===================================
-- VIEWS DLA CZĘSTYCH ZAPYTAŃ
-- ===================================

-- View dla aktywnych eksperymentów
CREATE VIEW IF NOT EXISTS active_runs AS
SELECT 
    run_id,
    dataset,
    target,
    model_name,
    status,
    created_at,
    duration_seconds,
    user_id,
    project_name
FROM runs 
WHERE is_deleted = FALSE 
  AND status IN ('queued', 'running')
ORDER BY created_at DESC;

-- View dla najlepszych wyników (wymaga JSON extraction)
CREATE VIEW IF NOT EXISTS best_runs AS
SELECT 
    run_id,
    dataset,
    target,
    model_name,
    problem_type,
    status,
    metrics_json,
    created_at,
    -- Przykład: wyciągnij accuracy z JSON
    CAST(json_extract(metrics_json, '$.accuracy') AS REAL) as accuracy,
    CAST(json_extract(metrics_json, '$.f1_score') AS REAL) as f1_score,
    CAST(json_extract(metrics_json, '$.rmse') AS REAL) as rmse
FROM runs 
WHERE status = 'completed' 
  AND is_deleted = FALSE
  AND metrics_json != '{}'
ORDER BY created_at DESC;

-- View dla statystyk użytkowników
CREATE VIEW IF NOT EXISTS user_stats AS
SELECT 
    user_id,
    COUNT(*) as total_runs,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
    AVG(duration_seconds) as avg_duration,
    MAX(created_at) as last_run,
    COUNT(DISTINCT dataset) as unique_datasets
FROM runs 
WHERE is_deleted = FALSE
GROUP BY user_id;

-- ===================================
-- PRZYKŁADY ZAPYTAŃ ANALITYCZNYCH
-- ===================================

-- 1. Top 10 najlepszych modeli dla danego datasetu
/*
SELECT 
    run_id,
    model_name,
    CAST(json_extract(metrics_json, '$.accuracy') AS REAL) as accuracy,
    created_at
FROM runs 
WHERE dataset = 'iris' 
  AND status = 'completed'
  AND json_extract(metrics_json, '$.accuracy') IS NOT NULL
ORDER BY accuracy DESC 
LIMIT 10;
*/

-- 2. Trend wydajności w czasie
/*
SELECT 
    DATE(created_at) as date,
    COUNT(*) as runs_count,
    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS REAL)) as avg_accuracy,
    AVG(duration_seconds) as avg_duration
FROM runs 
WHERE dataset = 'titanic' 
  AND status = 'completed'
  AND created_at >= DATE('now', '-30 days')
GROUP BY DATE(created_at)
ORDER BY date;
*/

-- 3. Porównanie engineów
/*
SELECT 
    engine,
    COUNT(*) as total_runs,
    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS REAL)) as avg_accuracy,
    AVG(duration_seconds) as avg_duration,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) * 100.0 / COUNT(*) as failure_rate
FROM runs 
WHERE problem_type = 'classification'
  AND created_at >= DATE('now', '-90 days')
GROUP BY engine
ORDER BY avg_accuracy DESC;
*/

-- ===================================
-- FUNKCJE POMOCNICZE (SQLite 3.38+)
-- ===================================

-- Inicjalizacja domyślnych metadanych
INSERT OR IGNORE INTO system_metadata (key, value, description) VALUES 
    ('schema_version', '2', 'Current database schema version'),
    ('created_at', DATETIME('now'), 'Database creation timestamp'),
    ('last_migration', DATETIME('now'), 'Last migration timestamp');

-- ===================================
-- KOMENTARZE I DOKUMENTACJA
-- ===================================

/*
DESIGN DECISIONS:

1. **Normalization vs Denormalization**: 
   - Główna tabela runs jest częściowo zdenormalizowana dla wydajności
   - Oddzielne tabele dla metryk, logów i artefaktów dla elastyczności

2. **JSON Storage**: 
   - Metryki w JSON dla elastyczności
   - SQLite ma dobre wsparcie dla JSON queries (od v3.38)
   - JSON validation constraints zapewniają integralność

3. **Indexing Strategy**:
   - Kompozytowe indeksy dla częstych query patterns
   - Partial indexes dla aktywnych eksperymentów
   - Covering indexes tam gdzie to możliwe

4. **Audit Trail**:
   - Soft delete z is_deleted flag
   - created_by i updated_at dla trackingu zmian
   - Git commit i code version dla reproducibility

5. **Constraints**:
   - Check constraints dla business logic
   - Foreign keys dla referential integrity
   - Unique constraints dla data quality

PERFORMANCE CONSIDERATIONS:
- Indeksy zaprojektowane dla typowych query patterns
- JSON queries mogą być wolniejsze - rozważ denormalizację dla krytycznych metryk
- Partycjonowanie po dacie dla bardzo dużych datasets
- Regular VACUUM dla SQLite maintenance

SCALABILITY:
- Dla >1M rekordów rozważ PostgreSQL z JSON support
- Archiwizacja starych eksperymentów
- Connection pooling dla concurrent access
- Read replicas dla analytics queries
*/