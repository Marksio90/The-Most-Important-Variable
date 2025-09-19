# db/db_utils.py — NAPRAWIONY: poprawiona obsługa timezone, lokalne czasy, rozbudowane funkcje
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.ml_integration import ModelConfig, TrainingResult
from backend.utils import format_datetime_for_display, local_now_iso


# ==========================
# Modele danych - ROZSZERZONE
# ==========================
@dataclass
class TrainingRecord:
    """Rozszerzony rekord treningu modelu."""
    dataset_name: str
    target_column: str
    engine: str
    problem_type: str
    status: str = "completed"
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_time: Optional[float] = None
    created_at: Optional[datetime] = None
    run_id: Optional[str] = None
    notes: str = ""
    
    # NOWE POLA
    model_size_mb: Optional[float] = None
    best_score: Optional[float] = None
    cv_mean_score: Optional[float] = None
    cv_std_score: Optional[float] = None
    n_features_selected: Optional[int] = None
    hyperparams_tuned: bool = False
    ensemble_used: bool = False

    def __post_init__(self):
        if self.created_at is None:
            # Używaj lokalnego czasu zamiast UTC
            self.created_at = datetime.now()
        if self.run_id is None:
            # Format z lokalną strefą czasową
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"

    def get_local_created_at_str(self) -> str:
        """Zwraca lokalny czas jako string do wyświetlenia."""
        if self.created_at is None:
            return "N/A"
        
        # Jeśli datetime jest "naive" (bez timezone), traktuj jako lokalny
        if self.created_at.tzinfo is None:
            return self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Konwertuj na lokalny czas
            local_time = self.created_at.astimezone()
            return local_time.strftime("%Y-%m-%d %H:%M:%S")


# ==========================
# Zarządca bazy - ROZBUDOWANY
# ==========================
class DatabaseManager:
    """Zarządca bazy danych SQLite dla TMIV z naprawioną obsługą czasu."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _connect(self):
        """Połączenie z bazą danych z optymalną konfiguracją."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_database(self) -> None:
        """Inicjalizuje bazę danych z rozszerzoną strukturą."""
        with self._connect() as conn:
            cur = conn.cursor()
            
            # Tabela historii treningów - ROZSZERZONA
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    metrics TEXT,   -- JSON
                    metadata TEXT,  -- JSON
                    training_time REAL,
                    notes TEXT,
                    created_at TEXT,  -- ISO8601 w lokalnej strefie czasowej
                    
                    -- NOWE KOLUMNY
                    model_size_mb REAL,
                    best_score REAL,
                    cv_mean_score REAL,
                    cv_std_score REAL,
                    n_features_selected INTEGER,
                    hyperparams_tuned BOOLEAN DEFAULT 0,
                    ensemble_used BOOLEAN DEFAULT 0,
                    config_used TEXT  -- JSON konfiguracji
                )
            """)
            
            # Tabela domyślnych modeli
            cur.execute("""
                CREATE TABLE IF NOT EXISTS default_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT,
                    UNIQUE(dataset_name, target_column)
                )
            """)
            
            # NOWA: Tabela eksportów i artefaktów
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    export_type TEXT NOT NULL, -- model, report, chart, etc.
                    file_path TEXT NOT NULL,
                    file_size_mb REAL,
                    created_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES training_history (run_id)
                )
            """)
            
            # NOWA: Tabela tagów i kategorii
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES training_history (run_id),
                    UNIQUE(run_id, tag)
                )
            """)
            
            # Indeksy przyspieszające - ROZSZERZONE
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_created ON training_history(created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_dataset ON training_history(dataset_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_target ON training_history(target_column)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_engine ON training_history(engine)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_problem_type ON training_history(problem_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_best_score ON training_history(best_score DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_default_pair ON default_models(dataset_name, target_column)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exports_run ON model_exports(run_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_run ON training_tags(run_id)")
            
            conn.commit()

    def save_training_record(self, record: TrainingRecord) -> bool:
        """Zapisuje rekord treningu z lokalnym czasem."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                
                # Przygotuj lokalny timestamp
                if record.created_at:
                    if record.created_at.tzinfo is None:
                        # Już lokalny czas
                        created_at_str = record.created_at.isoformat()
                    else:
                        # Konwertuj na lokalny
                        local_time = record.created_at.astimezone()
                        created_at_str = local_time.isoformat()
                else:
                    created_at_str = local_now_iso()
                
                # Dodatkowe metryki z metadanych
                config_json = None
                if record.metadata and 'config_used' in record.metadata:
                    config_json = json.dumps(record.metadata['config_used'], ensure_ascii=False)
                
                cur.execute(
                    """
                    INSERT OR REPLACE INTO training_history 
                    (run_id, dataset_name, target_column, engine, problem_type, 
                     status, metrics, metadata, training_time, notes, created_at,
                     model_size_mb, best_score, cv_mean_score, cv_std_score,
                     n_features_selected, hyperparams_tuned, ensemble_used, config_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.dataset_name,
                        record.target_column,
                        record.engine,
                        record.problem_type,
                        record.status,
                        json.dumps(record.metrics, ensure_ascii=False) if record.metrics else None,
                        json.dumps(record.metadata, ensure_ascii=False) if record.metadata else None,
                        record.training_time,
                        record.notes,
                        created_at_str,
                        record.model_size_mb,
                        record.best_score,
                        record.cv_mean_score,
                        record.cv_std_score,
                        record.n_features_selected,
                        int(record.hyperparams_tuned) if record.hyperparams_tuned else 0,
                        int(record.ensemble_used) if record.ensemble_used else 0,
                        config_json
                    ),
                )
                conn.commit()
                return True
                
        except Exception as e:
            print(f"[DB] Błąd zapisu do bazy danych: {e}")
            return False

    def get_training_history(self, limit: int = 50, dataset_name: Optional[str] = None, **filters) -> List[TrainingRecord]:
        """Pobiera historię treningów z filtrami."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                query = """
                    SELECT run_id, dataset_name, target_column, engine, problem_type,
                           status, metrics, metadata, training_time, notes, created_at,
                           model_size_mb, best_score, cv_mean_score, cv_std_score,
                           n_features_selected, hyperparams_tuned, ensemble_used
                    FROM training_history
                """
                params: List[Any] = []
                conditions = []
                
                # Filtry
                if dataset_name:
                    conditions.append("dataset_name = ?")
                    params.append(dataset_name)
                
                if filters.get('engine'):
                    conditions.append("engine = ?")
                    params.append(filters['engine'])
                
                if filters.get('problem_type'):
                    conditions.append("problem_type = ?")
                    params.append(filters['problem_type'])
                
                if filters.get('min_score') is not None:
                    conditions.append("best_score >= ?")
                    params.append(filters['min_score'])
                
                # Dodaj warunki WHERE
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                # Sortowanie i limit
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

            # Parsowanie wyników
            records: List[TrainingRecord] = []
            for row in rows:
                # Parsuj JSON
                metrics = {}
                metadata = {}
                try:
                    metrics = json.loads(row[6]) if row[6] else {}
                except Exception:
                    metrics = {}
                try:
                    metadata = json.loads(row[7]) if row[7] else {}
                except Exception:
                    metadata = {}

                # Parsuj czas - NAPRAWIONE
                created_at = None
                if row[10]:  # created_at
                    try:
                        # Spróbuj parsować jako ISO format
                        created_at = datetime.fromisoformat(row[10])
                    except Exception:
                        try:
                            # Fallback na pandas
                            created_at = pd.to_datetime(row[10]).to_pydatetime()
                        except Exception:
                            # Ostatni fallback
                            created_at = datetime.now()

                records.append(
                    TrainingRecord(
                        run_id=row[0],
                        dataset_name=row[1],
                        target_column=row[2],
                        engine=row[3],
                        problem_type=row[4],
                        status=row[5],
                        metrics=metrics,
                        metadata=metadata,
                        training_time=row[8],
                        notes=row[9] or "",
                        created_at=created_at,
                        model_size_mb=row[11],
                        best_score=row[12],
                        cv_mean_score=row[13],
                        cv_std_score=row[14],
                        n_features_selected=row[15],
                        hyperparams_tuned=bool(row[16]) if row[16] is not None else False,
                        ensemble_used=bool(row[17]) if row[17] is not None else False,
                    )
                )
            return records
            
        except Exception as e:
            print(f"[DB] Błąd odczytu historii: {e}")
            return []

    def get_training_record(self, run_id: str) -> Optional[TrainingRecord]:
        """Pobiera pojedynczy rekord treningu."""
        records = self.get_training_history(limit=1)
        filtered = [r for r in records if r.run_id == run_id]
        return filtered[0] if filtered else None

    def update_training_record(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Aktualizuje istniejący rekord treningu."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                
                # Buduj dynamiczne UPDATE
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    if key in ['notes', 'status', 'best_score', 'model_size_mb']:
                        set_clauses.append(f"{key} = ?")
                        params.append(value)
                
                if not set_clauses:
                    return False
                
                params.append(run_id)
                
                query = f"UPDATE training_history SET {', '.join(set_clauses)} WHERE run_id = ?"
                cur.execute(query, params)
                
                conn.commit()
                return cur.rowcount > 0
                
        except Exception as e:
            print(f"[DB] Błąd aktualizacji rekordu: {e}")
            return False

    # NOWE FUNKCJE DO ZARZĄDZANIA EKSPORTAMI
    def save_export_record(self, run_id: str, export_type: str, file_path: str, file_size_mb: float = 0) -> bool:
        """Zapisuje informację o eksporcie artefaktu."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO model_exports (run_id, export_type, file_path, file_size_mb, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, export_type, file_path, file_size_mb, local_now_iso())
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[DB] Błąd zapisu eksportu: {e}")
            return False

    def get_exports_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Pobiera listę eksportów dla danego run_id."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT export_type, file_path, file_size_mb, created_at 
                    FROM model_exports 
                    WHERE run_id = ? 
                    ORDER BY created_at DESC
                    """,
                    (run_id,)
                )
                rows = cur.fetchall()
                
                return [
                    {
                        'export_type': row[0],
                        'file_path': row[1], 
                        'file_size_mb': row[2],
                        'created_at': row[3]
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"[DB] Błąd pobierania eksportów: {e}")
            return []

    def add_training_tag(self, run_id: str, tag: str) -> bool:
        """Dodaje tag do treningu."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO training_tags (run_id, tag, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, tag.strip(), local_now_iso())
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            print(f"[DB] Błąd dodawania tagu: {e}")
            return False

    def get_tags_for_run(self, run_id: str) -> List[str]:
        """Pobiera tagi dla danego treningu."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT tag FROM training_tags WHERE run_id = ? ORDER BY created_at", (run_id,))
                rows = cur.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            print(f"[DB] Błąd pobierania tagów: {e}")
            return []

    def remove_training_tag(self, run_id: str, tag: str) -> bool:
        """Usuwa tag z treningu."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM training_tags WHERE run_id = ? AND tag = ?", (run_id, tag))
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            print(f"[DB] Błąd usuwania tagu: {e}")
            return False

    # ROZBUDOWANE STATYSTYKI I ANALIZY
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Pobiera rozbudowane statystyki z bazy danych."""
        stats: Dict[str, Any] = {}
        
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                
                # Podstawowe statystyki
                cur.execute("SELECT COUNT(*) FROM training_history")
                stats["total_runs"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT dataset_name) FROM training_history")
                stats["unique_datasets"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT target_column) FROM training_history")
                stats["unique_targets"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM training_history WHERE status = 'completed'")
                stats["completed_runs"] = cur.fetchone()[0]

                # Statystyki silników
                cur.execute("""
                    SELECT engine, COUNT(*) as count, AVG(best_score) as avg_score
                    FROM training_history 
                    WHERE best_score IS NOT NULL
                    GROUP BY engine
                    ORDER BY count DESC
                """)
                engine_stats = []
                for row in cur.fetchall():
                    engine_stats.append({
                        'engine': row[0],
                        'count': row[1],
                        'avg_score': round(row[2], 4) if row[2] else None
                    })
                stats["engine_performance"] = engine_stats

                # Statystyki typów problemów
                cur.execute("""
                    SELECT problem_type, COUNT(*) as count, 
                           AVG(training_time) as avg_training_time,
                           AVG(best_score) as avg_score
                    FROM training_history 
                    GROUP BY problem_type
                """)
                problem_stats = []
                for row in cur.fetchall():
                    problem_stats.append({
                        'problem_type': row[0],
                        'count': row[1],
                        'avg_training_time': round(row[2], 2) if row[2] else None,
                        'avg_score': round(row[3], 4) if row[3] else None
                    })
                stats["problem_type_stats"] = problem_stats

                # Najlepsze modele
                cur.execute("""
                    SELECT run_id, dataset_name, target_column, engine, best_score, created_at
                    FROM training_history 
                    WHERE best_score IS NOT NULL
                    ORDER BY best_score DESC
                    LIMIT 10
                """)
                best_models = []
                for row in cur.fetchall():
                    best_models.append({
                        'run_id': row[0],
                        'dataset_name': row[1],
                        'target_column': row[2],
                        'engine': row[3],
                        'best_score': round(row[4], 4),
                        'created_at': row[5]
                    })
                stats["best_models"] = best_models

                # Ostatnie trenringi
                cur.execute("""
                    SELECT created_at FROM training_history 
                    ORDER BY created_at DESC LIMIT 1
                """)
                last = cur.fetchone()
                stats["last_run_date"] = last[0] if last else None

                # Statystyki zaawansowanych opcji
                cur.execute("SELECT COUNT(*) FROM training_history WHERE hyperparams_tuned = 1")
                stats["hyperparams_tuned_count"] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM training_history WHERE ensemble_used = 1") 
                stats["ensemble_used_count"] = cur.fetchone()[0]
                
        except Exception as e:
            print(f"[DB] Błąd pobierania statystyk: {e}")
            
        return stats

    def get_performance_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Pobiera trendy wydajności modeli w czasie."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                
                # Trendy dla ostatnich N dni
                cur.execute("""
                    SELECT DATE(created_at) as date,
                           COUNT(*) as runs_count,
                           AVG(best_score) as avg_score,
                           AVG(training_time) as avg_training_time,
                           engine
                    FROM training_history 
                    WHERE created_at >= DATE('now', '-{} days')
                    AND best_score IS NOT NULL
                    GROUP BY DATE(created_at), engine
                    ORDER BY date DESC
                """.format(days))
                
                trends = {}
                for row in cur.fetchall():
                    date_str = row[0]
                    if date_str not in trends:
                        trends[date_str] = []
                    
                    trends[date_str].append({
                        'engine': row[4],
                        'runs_count': row[1],
                        'avg_score': round(row[2], 4) if row[2] else None,
                        'avg_training_time': round(row[3], 2) if row[3] else None
                    })
                
                return trends
                
        except Exception as e:
            print(f"[DB] Błąd pobierania trendów: {e}")
            return {}

    # POZOSTAŁE FUNKCJE BEZ ZMIAN LUB Z DROBNYMI POPRAWKAMI
    def set_default_model(self, dataset_name: str, target_column: str, run_id: str) -> bool:
        """Ustawia domyślny model dla pary dataset/target."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO default_models 
                    (dataset_name, target_column, run_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dataset_name, target_column, run_id, local_now_iso()),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[DB] Błąd ustawiania domyślnego modelu: {e}")
            return False

    def get_default_model(self, dataset_name: str, target_column: str) -> Optional[str]:
        """Pobiera run_id domyślnego modelu dla pary dataset/target."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT run_id FROM default_models 
                    WHERE dataset_name = ? AND target_column = ?
                    """,
                    (dataset_name, target_column),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            print(f"[DB] Błąd pobierania domyślnego modelu: {e}")
            return None

    def delete_training_record(self, run_id: str) -> bool:
        """Usuwa rekord treningu i powiązane dane."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # Usuń z historii
                cur.execute("DELETE FROM training_history WHERE run_id = ?", (run_id,))
                # Wyczyść z default_models
                cur.execute("DELETE FROM default_models WHERE run_id = ?", (run_id,))
                # Usuń eksporty
                cur.execute("DELETE FROM model_exports WHERE run_id = ?", (run_id,))
                # Usuń tagi
                cur.execute("DELETE FROM training_tags WHERE run_id = ?", (run_id,))
                
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            print(f"[DB] Błąd usuwania rekordu: {e}")
            return False

    def prune_history(self, keep_last: int = 1000) -> int:
        """Usuwa najstarsze wpisy, zostawiając najnowsze."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # Policz ile mamy
                cur.execute("SELECT COUNT(*) FROM training_history")
                total = cur.fetchone()[0]
                if total <= keep_last:
                    return 0
                
                # Znajdź próg daty  
                cur.execute(
                    """
                    SELECT created_at FROM training_history 
                    ORDER BY created_at DESC LIMIT 1 OFFSET ?
                    """,
                    (keep_last - 1,),
                )
                row = cur.fetchone()
                if not row:
                    return 0
                cutoff = row[0]
                
                # Pobierz run_id do usunięcia
                cur.execute(
                    "SELECT run_id FROM training_history WHERE created_at < ?", 
                    (cutoff,)
                )
                run_ids_to_delete = [row[0] for row in cur.fetchall()]
                
                # Usuń powiązane dane
                for run_id in run_ids_to_delete:
                    cur.execute("DELETE FROM model_exports WHERE run_id = ?", (run_id,))
                    cur.execute("DELETE FROM training_tags WHERE run_id = ?", (run_id,))
                    cur.execute("DELETE FROM default_models WHERE run_id = ?", (run_id,))
                
                # Usuń główne rekordy
                cur.execute("DELETE FROM training_history WHERE created_at < ?", (cutoff,))
                deleted = cur.rowcount
                
                conn.commit()
                return deleted or 0
        except Exception as e:
            print(f"[DB] Błąd pruning historii: {e}")
            return 0

    def backup_database(self, backup_path: Union[str, Path]) -> bool:
        """Tworzy kopię zapasową bazy danych."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._connect() as source:
                backup_conn = sqlite3.connect(backup_path)
                source.backup(backup_conn)
                backup_conn.close()
                
            return True
        except Exception as e:
            print(f"[DB] Błąd backup: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Zwraca informacje o bazie danych."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                
                # Rozmiar bazy
                db_size_mb = self.db_path.stat().st_size / 1024 / 1024
                
                # Liczba rekordów w tabelach
                cur.execute("SELECT COUNT(*) FROM training_history")
                history_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM model_exports") 
                exports_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM training_tags")
                tags_count = cur.fetchone()[0]
                
                # Informacje o schemacie
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cur.fetchall()]
                
                return {
                    'db_path': str(self.db_path),
                    'db_size_mb': round(db_size_mb, 2),
                    'history_records': history_count,
                    'export_records': exports_count,
                    'tag_records': tags_count,
                    'tables': tables,
                    'sqlite_version': sqlite3.sqlite_version
                }
        except Exception as e:
            print(f"[DB] Błąd pobierania info o DB: {e}")
            return {}


# ==========================
# Helpery modułowe (API) - ROZSZERZONE
# ==========================
def create_training_record(
    model_config: ModelConfig,
    result: TrainingResult,
    df: pd.DataFrame,
    dataset_name: str = "dataset"
) -> TrainingRecord:
    """Tworzy rozszerzony TrainingRecord z wyników treningu."""
    target_column = model_config.target
    engine = result.metadata.get("engine", model_config.engine) if result.metadata else model_config.engine
    problem_type = result.metadata.get("problem_type", "unknown") if result.metadata else "unknown"

    metrics = result.metrics or {}
    metadata = (result.metadata or {}).copy()
    
    # Dodatkowe metadane
    metadata.update({
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "target_column": target_column,
        "model_config": {
            "test_size": model_config.test_size,
            "cv_folds": model_config.cv_folds,
            "random_state": model_config.random_state,
            "stratify": model_config.stratify,
            "enable_probabilities": model_config.enable_probabilities,
            "feature_engineering": getattr(model_config, "feature_engineering", False),
            "feature_selection": getattr(model_config, "feature_selection", False),
            "hyperparameter_tuning": getattr(model_config, "hyperparameter_tuning", False),
            "ensemble_methods": getattr(model_config, "ensemble_methods", False),
        },
    })

    # Wyciągnij dodatkowe metryki
    best_score = None
    if problem_type == "regression":
        best_score = metrics.get("r2")
    elif problem_type == "classification":
        best_score = metrics.get("accuracy") or metrics.get("f1_macro")

    # CV scores
    cv_mean_score = None
    cv_std_score = None
    if result.cross_val_scores:
        # Znajdź główną metrykę CV
        main_metric_key = None
        if problem_type == "regression":
            main_metric_key = "r2_test"
        elif problem_type == "classification":
            main_metric_key = "accuracy_test"
        
        if main_metric_key and main_metric_key in result.cross_val_scores:
            scores = result.cross_val_scores[main_metric_key]
            cv_mean_score = np.mean(scores)
            cv_std_score = np.std(scores)

    return TrainingRecord(
        dataset_name=dataset_name,
        target_column=target_column,
        engine=engine,
        problem_type=problem_type,
        metrics=metrics,
        metadata=metadata,
        status="completed",
        best_score=best_score,
        cv_mean_score=cv_mean_score,
        cv_std_score=cv_std_score,
        n_features_selected=result.feature_importance.shape[0] if not result.feature_importance.empty else None,
        hyperparams_tuned=bool(result.best_params),
        ensemble_used=getattr(model_config, "ensemble_methods", False),
    )


def save_training_record(db_manager: DatabaseManager, record: TrainingRecord) -> bool:
    """Zapisuje rekord treningu do bazy danych."""
    return db_manager.save_training_record(record)


def get_training_history(
    db_manager: DatabaseManager,
    limit: int = 50,
    dataset_name: Optional[str] = None,
    **filters
) -> List[TrainingRecord]:
    """Pobiera historię treningów z bazy danych."""
    return db_manager.get_training_history(limit=limit, dataset_name=dataset_name, **filters)


def export_history_to_csv(db_manager: DatabaseManager, output_path: Union[str, Path]) -> bool:
    """Eksportuje historię treningów do pliku CSV (płaski format)."""
    try:
        history = db_manager.get_training_history(limit=10000)
        if not history:
            return False

        rows: List[Dict[str, Any]] = []
        for r in history:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "dataset_name": r.dataset_name,
                "target_column": r.target_column,
                "engine": r.engine,
                "problem_type": r.problem_type,
                "status": r.status,
                "training_time": r.training_time,
                "created_at": r.get_local_created_at_str(),
                "notes": r.notes,
                "best_score": r.best_score,
                "cv_mean_score": r.cv_mean_score,
                "cv_std_score": r.cv_std_score,
                "n_features_selected": r.n_features_selected,
                "hyperparams_tuned": r.hyperparams_tuned,
                "ensemble_used": r.ensemble_used,
            }
            
            # metryki podstawowe jako kolumny
            for k, v in (r.metrics or {}).items():
                if isinstance(v, (int, float)):
                    row[f"metric_{k}"] = v
                    
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"[DB] Błąd eksportu do CSV: {e}")
        return False


def import_history_from_csv(db_manager: DatabaseManager, csv_path: Union[str, Path]) -> bool:
    """Importuje historię treningów z pliku CSV."""
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # zbuduj metryki z kolumn metric_*
            metrics: Dict[str, Any] = {}
            for col in df.columns:
                if col.startswith("metric_") and pd.notna(row[col]):
                    metrics[col.replace("metric_", "")] = row[col]

            # parse created_at
            created_at: Optional[datetime] = None
            if pd.notna(row.get("created_at")):
                try:
                    # Spróbuj parsować bezpośrednio
                    created_at = datetime.fromisoformat(str(row["created_at"]))
                except Exception:
                    try:
                        # Przez pandas
                        created_at = pd.to_datetime(row["created_at"]).to_pydatetime()
                    except Exception:
                        created_at = None

            record = TrainingRecord(
                run_id=row.get("run_id", None) or None,
                dataset_name=str(row.get("dataset_name", "imported")),
                target_column=str(row.get("target_column", "unknown")),
                engine=str(row.get("engine", "unknown")),
                problem_type=str(row.get("problem_type", "unknown")),
                status=str(row.get("status", "completed")),
                metrics=metrics,
                training_time=float(row.get("training_time")) if pd.notna(row.get("training_time")) else None,
                notes=str(row.get("notes", "")),
                created_at=created_at,
                best_score=float(row.get("best_score")) if pd.notna(row.get("best_score")) else None,
                cv_mean_score=float(row.get("cv_mean_score")) if pd.notna(row.get("cv_mean_score")) else None,
                cv_std_score=float(row.get("cv_std_score")) if pd.notna(row.get("cv_std_score")) else None,
                n_features_selected=int(row.get("n_features_selected")) if pd.notna(row.get("n_features_selected")) else None,
                hyperparams_tuned=bool(row.get("hyperparams_tuned", False)),
                ensemble_used=bool(row.get("ensemble_used", False)),
            )
            
            db_manager.save_training_record(record)
            
        return True
        
    except Exception as e:
        print(f"[DB] Błąd importu z CSV: {e}")
        return False


# ==========================
# Kompatybilność (stare API) - ROZSZERZONE
# ==========================
class MLExperimentTracker:
    """Rozszerzony alias dla DatabaseManager."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_manager = DatabaseManager(db_path)

    def log_run(self, record) -> str:
        """Loguje run do bazy."""
        if isinstance(record, TrainingRecord):
            tr = record
        else:
            tr = TrainingRecord(
                dataset_name=getattr(record, "dataset", getattr(record, "dataset_name", "unknown")),
                target_column=getattr(record, "target", getattr(record, "target_column", "unknown")),
                engine=getattr(record, "engine", "unknown"),
                problem_type=getattr(record, "problem_type", "unknown"),
                run_id=getattr(record, "run_id", None),
                metrics=getattr(record, "metrics", {}),
                notes=getattr(record, "notes", ""),
                status=getattr(record, "status", "completed"),
                best_score=getattr(record, "best_score", None),
            )
        self.db_manager.save_training_record(tr)
        return tr.run_id

    def get_history(self, query_filter=None) -> pd.DataFrame:
        """Zwraca historię jako DataFrame."""
        hist = self.db_manager.get_training_history(limit=1000)
        if not hist:
            return pd.DataFrame()
            
        rows: List[Dict[str, Any]] = []
        for r in hist:
            rows.append({
                "run_id": r.run_id,
                "dataset": r.dataset_name,
                "target": r.target_column,
                "engine": r.engine,
                "status": r.status,
                "best_score": r.best_score,
                "created_at": r.get_local_created_at_str(),
            })
            
        return pd.DataFrame(rows)

    def set_default_model(self, dataset: str, target: str, run_id: str) -> bool:
        return self.db_manager.set_default_model(dataset, target, run_id)

    def get_default_model(self, dataset: str, target: str) -> Optional[str]:
        return self.db_manager.get_default_model(dataset, target)

    def delete_run(self, run_id: str) -> bool:
        return self.db_manager.delete_training_record(run_id)

    def get_stats(self) -> Dict[str, Any]:
        """Pobiera statystyki eksperymentów."""
        return self.db_manager.get_comprehensive_statistics()