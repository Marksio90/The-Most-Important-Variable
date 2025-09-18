# db_utils.py — TMIV history tracker (UTC, default model, CSV, backups)
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime, date, timezone
from enum import Enum
import sqlite3
import json
import logging
import pandas as pd
import threading

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== ENUMS =======================
class ProblemType(Enum):
    """Typy problemów ML."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    OTHER = "other"


class RunStatus(Enum):
    """Statusy uruchomień."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ===================== MODELS ======================
@dataclass
class RunRecord:
    """Model rekordu uruchomienia."""
    dataset: str
    target: str
    run_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    problem_type: Optional[ProblemType] = None
    engine: Optional[str] = None
    status: RunStatus = RunStatus.RUNNING
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        # Zapisz czasy w UTC, zawsze tz-aware
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at

    def _to_iso_z(self, dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        # dt musi być aware; konwertuj do UTC i serwuj z "Z"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika z obsługą enum i datetime."""
        data = asdict(self)

        # Enum -> wartości
        if self.problem_type:
            data["problem_type"] = self.problem_type.value
        if self.status:
            data["status"] = self.status.value

        # Daty -> ISO-8601 Z
        if self.created_at:
            data["created_at"] = self._to_iso_z(self.created_at)
        if self.updated_at:
            data["updated_at"] = self._to_iso_z(self.updated_at)

        return data


@dataclass
class QueryFilter:
    """Filtr do zapytań o historię."""
    dataset: Optional[str] = None
    target: Optional[str] = None
    engine: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    status: Optional[RunStatus] = None
    tags: Optional[List[str]] = None
    date_from: Optional[Union[datetime, date, str]] = None
    date_to: Optional[Union[datetime, date, str]] = None
    run_ids: Optional[List[str]] = None
    limit: int = 200
    offset: int = 0
    order_by: str = "created_at"
    order_desc: bool = True


# ===================== ERRORS ======================
class DatabaseError(Exception):
    """Niestandardowy błąd bazy danych."""
    pass


# ================ MAIN TRACKER CLASS ===============
class MLExperimentTracker:
    """Tracker uruchomień ML (SQLite, thread-safe)."""

    # Podbijamy wersję schematu: 3 (dodajemy klucze modelu domyślnego)
    SCHEMA_VERSION = 3

    def __init__(
        self,
        db_path: Union[str, Path] = "tmiv_out/history.sqlite",
        auto_backup: bool = True
    ):
        self.db_path = Path(db_path)
        self.auto_backup = auto_backup
        self._lock = threading.RLock()

        self._initialize_database()

    # --------------- lifecycle ----------------------
    def _initialize_database(self) -> None:
        """Inicjalizuje bazę i migracje."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self.get_connection() as conn:
                self._configure_sqlite(conn)
                self._create_tables(conn)
                self._run_migrations(conn)
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager dla połączeń z bazą."""
        conn = None
        try:
            with self._lock:
                conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0,
                    isolation_level=None  # autocommit
                )
                self._configure_sqlite(conn)
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()

    def _configure_sqlite(self, conn: sqlite3.Connection) -> None:
        """Parametry SQLite pod wydajność/przewidywalność."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Tworzy tabele, jeśli brak."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL,
                target TEXT NOT NULL,
                problem_type TEXT,
                engine TEXT,
                status TEXT DEFAULT 'running',

                -- Metadane czasowe (ISO-8601, UTC Z)
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                duration_seconds REAL,

                -- JSON
                metrics_json TEXT DEFAULT '{}',
                parameters_json TEXT DEFAULT '{}',
                tags_json TEXT DEFAULT '[]',

                -- Dodatkowe
                notes TEXT,

                -- Metadane systemu
                schema_version INTEGER DEFAULT 3,

                CONSTRAINT chk_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
            )
        """)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset)",
            "CREATE INDEX IF NOT EXISTS idx_runs_target ON runs(target)",
            "CREATE INDEX IF NOT EXISTS idx_runs_engine ON runs(engine)",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_runs_problem_type ON runs(problem_type)"
        ]:
            conn.execute(idx_sql)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Migracje schematu."""
        current = self._get_schema_version(conn)

        # v1 -> v2 (Twoja poprzednia migracja: tags_json, schema_version)
        if current < 2:
            try:
                cursor = conn.execute("PRAGMA table_info(runs)")
                columns = [row[1] for row in cursor.fetchall()]
                if "tags_json" not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN tags_json TEXT DEFAULT '[]'")
                if "schema_version" not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN schema_version INTEGER DEFAULT 2")
                logger.info("Migrated to schema v2")
            except sqlite3.OperationalError as e:
                logger.warning(f"Migration v2 warning: {e}")

        # v2 -> v3 (nic nie zmieniamy w strukturze runs; użyjemy system_metadata)
        if current < 3:
            # tylko podbij wersję
            logger.info("Migrated to schema v3 (system_metadata used for default models)")

        self._set_schema_version(conn, self.SCHEMA_VERSION)

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute("SELECT value FROM system_metadata WHERE key='schema_version'").fetchone()
            return int(row[0]) if row else 1
        except (sqlite3.OperationalError, ValueError, TypeError):
            return 1

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO system_metadata (key, value, updated_at) VALUES "
            "('schema_version', ?, datetime('now'))",
            (str(version),)
        )

    # --------------- CRUD runs ----------------------
    def _validate_record(self, record: RunRecord) -> None:
        if not record.run_id or not record.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if not record.dataset or not record.dataset.strip():
            raise ValueError("dataset cannot be empty")
        if not record.target or not record.target.strip():
            raise ValueError("target cannot be empty")
        # JSON-owalne pola
        json.dumps(record.metrics)
        json.dumps(record.parameters)
        json.dumps(record.tags)

    def _iso_z(self, dt: Union[datetime, str, None]) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def log_run(self, record: RunRecord) -> bool:
        """Wstawia nowy run (albo aktualizuje, jeśli run_id istnieje)."""
        try:
            with self.get_connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM runs WHERE run_id = ?", (record.run_id,)
                ).fetchone()
                if existing:
                    logger.warning(f"Run {record.run_id} already exists, updating…")
                    return self.update_run(record)

                self._validate_record(record)

                conn.execute("""
                    INSERT INTO runs (
                        run_id, dataset, target, problem_type, engine, status,
                        created_at, updated_at, duration_seconds,
                        metrics_json, parameters_json, tags_json, notes, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.run_id,
                    record.dataset,
                    record.target,
                    record.problem_type.value if record.problem_type else None,
                    record.engine,
                    record.status.value,
                    self._iso_z(record.created_at) or self._iso_z(datetime.now(timezone.utc)),
                    self._iso_z(record.updated_at) or self._iso_z(datetime.now(timezone.utc)),
                    record.duration_seconds,
                    json.dumps(record.metrics, ensure_ascii=False),
                    json.dumps(record.parameters, ensure_ascii=False),
                    json.dumps(record.tags, ensure_ascii=False),
                    record.notes,
                    self.SCHEMA_VERSION
                ))
                logger.info(f"Logged run {record.run_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to log run {record.run_id}: {e}")
            return False

    def update_run(self, record: RunRecord) -> bool:
        """Aktualizuje istniejący run."""
        try:
            with self.get_connection() as conn:
                record.updated_at = datetime.now(timezone.utc)
                conn.execute("""
                    UPDATE runs SET
                        dataset = ?, target = ?, problem_type = ?, engine = ?,
                        status = ?, updated_at = ?, duration_seconds = ?,
                        metrics_json = ?, parameters_json = ?, tags_json = ?, notes = ?
                    WHERE run_id = ?
                """, (
                    record.dataset, record.target,
                    record.problem_type.value if record.problem_type else None,
                    record.engine, record.status.value,
                    self._iso_z(record.updated_at), record.duration_seconds,
                    json.dumps(record.metrics, ensure_ascii=False),
                    json.dumps(record.parameters, ensure_ascii=False),
                    json.dumps(record.tags, ensure_ascii=False),
                    record.notes,
                    record.run_id
                ))
                logger.info(f"Updated run {record.run_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to update run {record.run_id}: {e}")
            return False

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Pobiera pojedynczy run."""
        try:
            with self.get_connection() as conn:
                row = conn.execute("""
                    SELECT run_id, dataset, target, problem_type, engine, status,
                           created_at, updated_at, duration_seconds,
                           metrics_json, parameters_json, tags_json, notes
                    FROM runs WHERE run_id = ?
                """, (run_id,)).fetchone()
                return self._row_to_record(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None

    def delete_run(self, run_id: str) -> bool:
        """Usuwa run po run_id."""
        try:
            with self.get_connection() as conn:
                cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                ok = cur.rowcount > 0
                if ok:
                    logger.info(f"Deleted run {run_id}")
                else:
                    logger.warning(f"Run {run_id} not found")
                return ok
        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {e}")
            return False

    # --------------- history & query ----------------
    def _safe_json_loads(self, s: Optional[str]) -> Any:
        try:
            if not s or pd.isna(s):
                return {}
            return json.loads(s)
        except Exception:
            return {}

    def _format_date(self, val: Union[datetime, date, str]) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, (datetime, date)):
            if isinstance(val, datetime):
                return self._iso_z(val) or ""
            return datetime(val.year, val.month, val.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return str(val)

    def _build_query(self, f: QueryFilter) -> Tuple[str, Dict[str, Any]]:
        base = """
            SELECT run_id, dataset, target, problem_type, engine, status,
                   created_at, updated_at, duration_seconds,
                   metrics_json, parameters_json, tags_json, notes
            FROM runs WHERE 1=1
        """
        cond, params = [], {}

        if f.dataset:
            cond.append("dataset LIKE :dataset"); params["dataset"] = f"%{f.dataset}%"
        if f.target:
            cond.append("target LIKE :target"); params["target"] = f"%{f.target}%"
        if f.engine:
            cond.append("engine LIKE :engine"); params["engine"] = f"%{f.engine}%"
        if f.problem_type:
            cond.append("problem_type = :pt"); params["pt"] = f.problem_type.value
        if f.status:
            cond.append("status = :st"); params["st"] = f.status.value
        if f.run_ids:
            placeholders = ",".join([f":r{i}" for i in range(len(f.run_ids))])
            cond.append(f"run_id IN ({placeholders})")
            for i, rid in enumerate(f.run_ids):
                params[f"r{i}"] = rid
        if f.date_from:
            cond.append("created_at >= :df"); params["df"] = self._format_date(f.date_from)
        if f.date_to:
            cond.append("created_at <= :dt"); params["dt"] = self._format_date(f.date_to)
        if f.tags:
            for i, tag in enumerate(f.tags):
                cond.append(f"tags_json LIKE :tg{i}")
                params[f"tg{i}"] = f'%"{tag}"%'

        if cond:
            base += " AND " + " AND ".join(cond)

        valid = ["created_at", "updated_at", "dataset", "target", "engine", "status"]
        order_col = f.order_by if f.order_by in valid else "created_at"
        order_dir = "DESC" if f.order_desc else "ASC"
        base += f" ORDER BY {order_col} {order_dir}"
        base += f" LIMIT {f.limit} OFFSET {f.offset}"
        return base, params

    def _row_to_record(self, row: tuple) -> RunRecord:
        (run_id, dataset, target, problem_type, engine, status,
         created_at, updated_at, duration_seconds,
         metrics_json, parameters_json, tags_json, notes) = row

        def _parse_iso_z(s: Optional[str]) -> Optional[datetime]:
            if not s:
                return None
            try:
                # pd.to_datetime lepiej łapie 'Z'
                dt = pd.to_datetime(s, utc=True).to_pydatetime()
                return dt
            except Exception:
                try:
                    return datetime.fromisoformat(s.replace("Z", "+00:00"))
                except Exception:
                    return None

        return RunRecord(
            run_id=run_id,
            dataset=dataset,
            target=target,
            problem_type=ProblemType(problem_type) if problem_type else None,
            engine=engine,
            status=RunStatus(status) if status else RunStatus.RUNNING,
            created_at=_parse_iso_z(created_at),
            updated_at=_parse_iso_z(updated_at),
            duration_seconds=duration_seconds,
            metrics=self._safe_json_loads(metrics_json),
            parameters=self._safe_json_loads(parameters_json),
            notes=notes,
            tags=self._safe_json_loads(tags_json) if tags_json else []
        )

    def get_history(self, filter_params: Optional[QueryFilter] = None) -> pd.DataFrame:
        """Pobiera historię (UTC w kolumnach czasowych)."""
        f = filter_params or QueryFilter()
        try:
            with self.get_connection() as conn:
                sql, params = self._build_query(f)
                df = pd.read_sql_query(sql, conn, params=params)
                if df.empty:
                    return pd.DataFrame()

                # JSON -> python
                for jcol, ncol in [("metrics_json", "metrics"),
                                   ("parameters_json", "parameters"),
                                   ("tags_json", "tags")]:
                    if jcol in df.columns:
                        df[ncol] = df[jcol].apply(self._safe_json_loads)
                        df = df.drop(columns=[jcol])

                # Daty -> pandas datetime (UTC)
                for dcol in ["created_at", "updated_at"]:
                    if dcol in df.columns:
                        df[dcol] = pd.to_datetime(df[dcol], utc=True, errors="coerce")

                return df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return pd.DataFrame()

    def clear_history(self, confirm: bool = False) -> bool:
        if not confirm:
            logger.warning("clear_history requires confirm=True")
            return False
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM runs")
            logger.info("History cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

    # --------------- export / backup ----------------
    def export_to_csv(self,
                      filter_params: Optional[QueryFilter] = None,
                      filepath: Optional[Path] = None) -> Union[bytes, bool]:
        """Eksport historii do CSV (metryki spłaszczone)."""
        try:
            df = self.get_history(filter_params)
            if df.empty:
                logger.warning("No data to export")
                return b"" if filepath is None else False

            df_export = df.copy()

            # Rozwiń metryki do kolumn metric_*
            if "metrics" in df_export.columns:
                for idx, metrics in df_export["metrics"].items():
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and not pd.isna(v):
                                df_export.loc[idx, f"metric_{k}"] = v
                df_export = df_export.drop(columns=["metrics"])

            # JSONy -> string
            for col in ["tags", "parameters", "notes"]:
                if col in df_export.columns:
                    df_export[col] = df_export[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else (x if x is not None else "")
                    )

            csv_data = df_export.to_csv(index=False)
            if filepath:
                Path(filepath).write_text(csv_data, encoding="utf-8")
                logger.info(f"Exported {len(df)} records to {filepath}")
                return True
            return csv_data.encode("utf-8")
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return b"" if filepath is None else False

    def backup_database(self, backup_path: Optional[Path] = None) -> Optional[Path]:
        """VACUUM INTO (lub kopia pliku) do backupu."""
        try:
            if backup_path is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
                backup_path = self.db_path.parent / f"backup_{timestamp}.sqlite"
            try:
                with self.get_connection() as conn:
                    conn.execute(f"VACUUM INTO '{backup_path}'")
            except sqlite3.OperationalError:
                import shutil
                shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Podstawowe statystyki bazy."""
        try:
            with self.get_connection() as conn:
                total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                status_stats = {
                    s.value: conn.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (s.value,)).fetchone()[0]
                    for s in RunStatus
                }
                popular_datasets = dict(conn.execute("""
                    SELECT dataset, COUNT(*) FROM runs GROUP BY dataset ORDER BY COUNT(*) DESC LIMIT 10
                """).fetchall())
                popular_targets = dict(conn.execute("""
                    SELECT target, COUNT(*) FROM runs GROUP BY target ORDER BY COUNT(*) DESC LIMIT 10
                """).fetchall())
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0.0
                return {
                    "total_runs": total_runs,
                    "status_distribution": status_stats,
                    "popular_datasets": popular_datasets,
                    "popular_targets": popular_targets,
                    "database_size_mb": round(db_size_mb, 2),
                    "schema_version": self.SCHEMA_VERSION
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    # --------------- default model registry ----------
    @staticmethod
    def _default_key(dataset: str, target: str) -> str:
        # prosty i czytelny klucz
        return f"default_model__{dataset}__{target}"

    def set_default_model(self, dataset: str, target: str, run_id: str) -> bool:
        """Ustaw run_id jako model domyślny dla (dataset, target)."""
        try:
            with self.get_connection() as conn:
                key = self._default_key(dataset, target)
                conn.execute(
                    "INSERT OR REPLACE INTO system_metadata (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    (key, run_id)
                )
            logger.info(f"Set default model for {dataset}/{target} -> {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set default model: {e}")
            return False

    def unset_default_model(self, dataset: str, target: str) -> bool:
        """Usuwa domyślny model dla (dataset, target)."""
        try:
            with self.get_connection() as conn:
                key = self._default_key(dataset, target)
                conn.execute("DELETE FROM system_metadata WHERE key = ?", (key,))
            return True
        except Exception as e:
            logger.error(f"Failed to unset default model: {e}")
            return False

    def get_default_model(self, dataset: str, target: str) -> Optional[str]:
        """Zwraca run_id modelu domyślnego, jeśli ustawiony."""
        try:
            with self.get_connection() as conn:
                key = self._default_key(dataset, target)
                row = conn.execute("SELECT value FROM system_metadata WHERE key = ?", (key,)).fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get default model: {e}")
            return None

    def list_default_models(self) -> Dict[Tuple[str, str], str]:
        """Wyszczególnia wszystkie domyślne modele."""
        try:
            with self.get_connection() as conn:
                rows = conn.execute("SELECT key, value FROM system_metadata WHERE key LIKE 'default_model__%'").fetchall()
                out: Dict[Tuple[str, str], str] = {}
                for k, v in rows:
                    # default_model__DATASET__TARGET
                    parts = k.split("__", 2)
                    if len(parts) == 3:
                        _, ds, tg = parts
                        out[(ds, tg)] = v
                return out
        except Exception as e:
            logger.error(f"Failed to list default models: {e}")
            return {}

# ================= BACKWARD COMPAT LAYER =================
def ensure_db() -> sqlite3.Connection:
    """Zgodność wsteczna – zwraca połączenie (uwaga: zamknij po użyciu!)."""
    tracker = MLExperimentTracker()
    return tracker.get_connection().__enter__()  # zgodnie z wcześniejszym API

def migrate_runs_table(conn: sqlite3.Connection) -> None:
    """Zgodność wsteczna – migracje i tak są automatyczne."""
    pass

def log_run(conn: sqlite3.Connection, **kwargs) -> None:
    """Zgodność wsteczna – logowanie runu."""
    record = RunRecord(
        dataset=kwargs.get("dataset", ""),
        target=kwargs.get("target", ""),
        run_id=kwargs.get("run_id", f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}"),
        problem_type=ProblemType(kwargs.get("problem_type")) if kwargs.get("problem_type") else None,
        engine=kwargs.get("engine_name"),
        metrics=kwargs.get("metrics", {})
    )
    MLExperimentTracker().log_run(record)

def get_history(conn: sqlite3.Connection, limit: int = 200) -> pd.DataFrame:
    """Zgodność wsteczna – pobieranie historii."""
    return MLExperimentTracker().get_history(QueryFilter(limit=limit))

def clear_history(conn: sqlite3.Connection) -> None:
    """Zgodność wsteczna – czyszczenie."""
    MLExperimentTracker().clear_history(confirm=True)

def export_history_csv(conn: sqlite3.Connection) -> bytes:
    """Zgodność wsteczna – eksport CSV."""
    return MLExperimentTracker().export_to_csv()
