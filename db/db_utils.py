# db/db_utils.py — KOMPLETNY: zarządzanie bazą danych i historią treningów (SQLite)
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.ml_integration import ModelConfig, TrainingResult


# ==========================
# Modele danych
# ==========================
@dataclass
class TrainingRecord:
    """Rekord treningu modelu do zapisania w bazie danych."""
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

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.run_id is None:
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S") if self.created_at else "unknown"
            self.run_id = f"run_{timestamp}"


# ==========================
# Zarządca bazy
# ==========================
class DatabaseManager:
    """Zarządca bazy danych SQLite dla TMIV."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    # -- Połączenie / kontekst --
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_database(self) -> None:
        """Inicjalizuje bazę danych i tworzy tabele + indeksy."""
        with self._connect() as conn:
            cur = conn.cursor()
            # Tabela historii treningów
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
                    created_at TEXT  -- ISO8601 w UTC
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
            # Indeksy przyspieszające listowania
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_created ON training_history(created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_dataset ON training_history(dataset_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_target ON training_history(target_column)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_default_pair ON default_models(dataset_name, target_column)")
            conn.commit()

    # -- CRUD dla historii --
    def save_training_record(self, record: TrainingRecord) -> bool:
        """Zapisuje rekord treningu do bazy danych."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO training_history 
                    (run_id, dataset_name, target_column, engine, problem_type, 
                     status, metrics, metadata, training_time, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        (record.created_at or datetime.now(timezone.utc)).isoformat(),
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[DB] Błąd zapisu do bazy danych: {e}")
            return False

    def get_training_history(self, limit: int = 50, dataset_name: Optional[str] = None) -> List[TrainingRecord]:
        """Pobiera historię treningów z bazy danych."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                query = """
                    SELECT run_id, dataset_name, target_column, engine, problem_type,
                           status, metrics, metadata, training_time, notes, created_at
                    FROM training_history
                """
                params: List[Any] = []
                if dataset_name:
                    query += " WHERE dataset_name = ?"
                    params.append(dataset_name)
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

            records: List[TrainingRecord] = []
            for row in rows:
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

                created_at = None
                if row[10]:
                    try:
                        created_at = datetime.fromisoformat(row[10])
                    except Exception:
                        try:
                            created_at = pd.to_datetime(row[10], utc=True).to_pydatetime()
                        except Exception:
                            created_at = datetime.now(timezone.utc)

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
                    )
                )
            return records
        except Exception as e:
            print(f"[DB] Błąd odczytu historii: {e}")
            return []

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
                    (dataset_name, target_column, run_id, datetime.now(timezone.utc).isoformat()),
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
        """Usuwa rekord treningu i wpis domyślnego modelu (jeśli wskazywał ten run)."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # Usuń z historii
                cur.execute("DELETE FROM training_history WHERE run_id = ?", (run_id,))
                # Wyczyść z tabeli default_models, jeśli wskazywała na ten run
                cur.execute("DELETE FROM default_models WHERE run_id = ?", (run_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            print(f"[DB] Błąd usuwania rekordu: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Pobiera statystyki z bazy danych."""
        stats: Dict[str, Any] = {}
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM training_history")
                stats["total_runs"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT dataset_name) FROM training_history")
                stats["unique_datasets"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT target_column) FROM training_history")
                stats["unique_targets"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM training_history WHERE status = 'completed'")
                stats["completed_runs"] = cur.fetchone()[0]

                cur.execute("SELECT created_at FROM training_history ORDER BY created_at DESC LIMIT 1")
                last = cur.fetchone()
                stats["last_run_date"] = last[0] if last else None
        except Exception as e:
            print(f"[DB] Błąd pobierania statystyk: {e}")
        return stats

    # Dodatkowe: porządki / pruning
    def prune_history(self, keep_last: int = 500) -> int:
        """
        Usuwa najstarsze wpisy, zostawiając `keep_last` najnowszych.
        Zwraca liczbę usuniętych rekordów.
        """
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # policz ile mamy
                cur.execute("SELECT COUNT(*) FROM training_history")
                total = cur.fetchone()[0]
                if total <= keep_last:
                    return 0
                # znajdź próg daty
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
                # usuń starsze
                cur.execute("DELETE FROM training_history WHERE created_at < ?", (cutoff,))
                deleted = cur.rowcount
                conn.commit()
                return deleted or 0
        except Exception as e:
            print(f"[DB] Błąd pruning historii: {e}")
            return 0


# ==========================
# Helpery modułowe (API)
# ==========================
def create_training_record(
    model_config: ModelConfig,
    result: TrainingResult,
    df: pd.DataFrame
) -> TrainingRecord:
    """Tworzy TrainingRecord z wyników treningu."""
    dataset_name = "dataset"  # app.py nadpisze prawdziwą nazwę
    target_column = model_config.target
    engine = result.metadata.get("engine", model_config.engine) if result.metadata else model_config.engine
    problem_type = result.metadata.get("problem_type", "unknown") if result.metadata else "unknown"

    metrics = result.metrics or {}
    metadata = (result.metadata or {}).copy()
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
        },
    })

    return TrainingRecord(
        dataset_name=dataset_name,
        target_column=target_column,
        engine=engine,
        problem_type=problem_type,
        metrics=metrics,
        metadata=metadata,
        status="completed",
    )


def save_training_record(db_manager: DatabaseManager, record: TrainingRecord) -> bool:
    """Zapisuje rekord treningu do bazy danych."""
    return db_manager.save_training_record(record)


def get_training_history(
    db_manager: DatabaseManager,
    limit: int = 50,
    dataset_name: Optional[str] = None
) -> List[TrainingRecord]:
    """Pobiera historię treningów z bazy danych."""
    return db_manager.get_training_history(limit=limit, dataset_name=dataset_name)


def export_history_to_csv(db_manager: DatabaseManager, output_path: Union[str, Path]) -> bool:
    """Eksportuje historię treningów do pliku CSV (płaski format)."""
    try:
        history = db_manager.get_training_history(limit=1000)
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
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "notes": r.notes,
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
                    created_at = datetime.fromisoformat(str(row["created_at"]))
                except Exception:
                    try:
                        created_at = pd.to_datetime(row["created_at"], utc=True).to_pydatetime()
                    except Exception:
                        created_at = None

            record = TrainingRecord(
                run_id=row.get("run_id", None) or None,
                dataset_name=str(row.get("dataset_name", "unknown")),
                target_column=str(row.get("target_column", "unknown")),
                engine=str(row.get("engine", "unknown")),
                problem_type=str(row.get("problem_type", "unknown")),
                status=str(row.get("status", "completed")),
                metrics=metrics,
                training_time=float(row.get("training_time")) if pd.notna(row.get("training_time")) else None,
                notes=str(row.get("notes", "")),
                created_at=created_at,
            )
            db_manager.save_training_record(record)
        return True
    except Exception as e:
        print(f"[DB] Błąd importu z CSV: {e}")
        return False


# ==========================
# Kompatybilność (stare API)
# ==========================
class MLExperimentTracker:
    """Alias dla DatabaseManager dla kompatybilności."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_manager = DatabaseManager(db_path)

    def log_run(self, record) -> str:
        """Loguje run do bazy; przyjmuje TrainingRecord lub podobny obiekt."""
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
            )
        self.db_manager.save_training_record(tr)
        return tr.run_id

    def get_history(self, query_filter=None) -> pd.DataFrame:
        """Zwraca historię jako DataFrame."""
        hist = self.db_manager.get_training_history()
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
                "created_at": r.created_at,
            })
        return pd.DataFrame(rows)

    def set_default_model(self, dataset: str, target: str, run_id: str) -> bool:
        return self.db_manager.set_default_model(dataset, target, run_id)

    def get_default_model(self, dataset: str, target: str) -> Optional[str]:
        return self.db_manager.get_default_model(dataset, target)

    def delete_run(self, run_id: str) -> bool:
        return self.db_manager.delete_training_record(run_id)


class RunRecord:
    """Alias dla TrainingRecord (zgodność ze starszą warstwą)."""
    def __init__(self, **kwargs):
        self.dataset = kwargs.get("dataset", kwargs.get("dataset_name", "unknown"))
        self.target = kwargs.get("target", kwargs.get("target_column", "unknown"))
        self.run_id = kwargs.get("run_id")
        self.problem_type = kwargs.get("problem_type", "unknown")
        self.engine = kwargs.get("engine", "unknown")
        self.status = kwargs.get("status", "completed")
        self.metrics = kwargs.get("metrics", {})
        self.notes = kwargs.get("notes", "")
        self.duration_seconds = kwargs.get("duration_seconds")
        self.tags = kwargs.get("tags", [])


class ProblemType:
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    OTHER = "other"


class RunStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class QueryFilter:
    """Nieużywany placeholder dla zgodności."""
    def __init__(self, limit: int = 50):
        self.limit = limit
