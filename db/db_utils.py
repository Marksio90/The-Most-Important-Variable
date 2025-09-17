from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime, date
from enum import Enum
import sqlite3
import json
import logging
import pandas as pd
import hashlib
import threading
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class RunRecord:
    """Model rekordu uruchomienia eksperymentu."""
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
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika z obsługą enum i datetime."""
        data = asdict(self)
        
        # Konwertuj enum do wartości
        if self.problem_type:
            data['problem_type'] = self.problem_type.value
        if self.status:
            data['status'] = self.status.value
            
        # Konwertuj datetime do ISO string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
            
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

class DatabaseError(Exception):
    """Niestandardowy błąd bazy danych."""
    pass

class MLExperimentTracker:
    """Uproszczony tracker eksperymentów ML z thread-safe SQLite backend."""
    
    # Wersja schematu bazy danych
    SCHEMA_VERSION = 2
    
    def __init__(
        self, 
        db_path: Union[str, Path] = "tmiv_out/history.sqlite",
        auto_backup: bool = True
    ):
        self.db_path = Path(db_path)
        self.auto_backup = auto_backup
        self._lock = threading.RLock()  # Prostszy lock zamiast ThreadPoolExecutor
        
        # Inicjalizuj bazę danych
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Inicjalizuje bazę danych i przeprowadza migracje."""
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
        """Uproszczony context manager dla połączeń z bazą danych."""
        conn = None
        try:
            with self._lock:  # Thread safety dla SQLite
                conn = sqlite3.connect(
                    str(self.db_path), 
                    check_same_thread=False,
                    timeout=30.0,
                    isolation_level=None  # Autocommit mode
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
        """Konfiguruje SQLite dla optymalnej wydajności."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        conn.execute("PRAGMA foreign_keys=ON")
        # Usunięto mmap_size - może powodować problemy na niektórych systemach
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Tworzy tabele w bazie danych."""
        
        # Główna tabela eksperymentów - uproszczona struktura
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL,
                target TEXT NOT NULL,
                problem_type TEXT,
                engine TEXT,
                status TEXT DEFAULT 'running',
                
                -- Metadane czasowe
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                duration_seconds REAL,
                
                -- Dane JSON - uproszczone
                metrics_json TEXT DEFAULT '{}',
                parameters_json TEXT DEFAULT '{}',
                tags_json TEXT DEFAULT '[]',
                
                -- Dodatkowe pola
                notes TEXT,
                
                -- Metadane systemu
                schema_version INTEGER DEFAULT 2,
                
                CONSTRAINT chk_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
            )
        """)
        
        # Podstawowe indeksy dla wydajności
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset)",
            "CREATE INDEX IF NOT EXISTS idx_runs_target ON runs(target)",
            "CREATE INDEX IF NOT EXISTS idx_runs_engine ON runs(engine)",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_runs_problem_type ON runs(problem_type)"
        ]
        
        for idx_sql in indexes:
            conn.execute(idx_sql)
        
        # Tabela metadanych systemu
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
    
    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Przeprowadza migracje schematu."""
        current_version = self._get_schema_version(conn)
        
        if current_version < 2:
            # Migracja v1 -> v2: dodaj nowe kolumny jeśli nie istnieją
            try:
                # Sprawdź czy kolumny istnieją
                cursor = conn.execute("PRAGMA table_info(runs)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'tags_json' not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN tags_json TEXT DEFAULT '[]'")
                if 'schema_version' not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN schema_version INTEGER DEFAULT 2")
                    
                logger.info("Migrated database schema to version 2")
            except sqlite3.OperationalError as e:
                logger.warning(f"Migration warning: {e}")
        
        # Zapisz aktualną wersję
        self._set_schema_version(conn, self.SCHEMA_VERSION)
    
    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Pobiera wersję schematu."""
        try:
            cursor = conn.execute(
                "SELECT value FROM system_metadata WHERE key = 'schema_version'"
            )
            result = cursor.fetchone()
            return int(result[0]) if result else 1
        except (sqlite3.OperationalError, ValueError, TypeError):
            return 1
    
    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        """Ustawia wersję schematu."""
        conn.execute(
            """
            INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
            VALUES ('schema_version', ?, datetime('now'))
            """,
            (str(version),)
        )
    
    def log_run(self, record: RunRecord) -> bool:
        """Loguje nowy eksperyment."""
        try:
            with self.get_connection() as conn:
                # Sprawdź czy run_id już istnieje
                existing = conn.execute(
                    "SELECT id FROM runs WHERE run_id = ?", (record.run_id,)
                ).fetchone()
                
                if existing:
                    logger.warning(f"Run {record.run_id} already exists, updating...")
                    return self.update_run(record)
                
                # Waliduj dane
                self._validate_record(record)
                
                # Wstaw nowy rekord
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
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.duration_seconds,
                    json.dumps(record.metrics, ensure_ascii=False),
                    json.dumps(record.parameters, ensure_ascii=False),
                    json.dumps(record.tags, ensure_ascii=False),
                    record.notes,
                    self.SCHEMA_VERSION
                ))
                
                logger.info(f"Successfully logged run {record.run_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log run {record.run_id}: {e}")
            return False
    
    def update_run(self, record: RunRecord) -> bool:
        """Aktualizuje istniejący eksperyment."""
        try:
            with self.get_connection() as conn:
                record.updated_at = datetime.now()
                
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
                    record.updated_at.isoformat(), record.duration_seconds,
                    json.dumps(record.metrics, ensure_ascii=False),
                    json.dumps(record.parameters, ensure_ascii=False),
                    json.dumps(record.tags, ensure_ascii=False),
                    record.notes,
                    record.run_id
                ))
                
                logger.info(f"Successfully updated run {record.run_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update run {record.run_id}: {e}")
            return False
    
    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Pobiera pojedynczy eksperyment."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT run_id, dataset, target, problem_type, engine, status,
                           created_at, updated_at, duration_seconds,
                           metrics_json, parameters_json, tags_json, notes
                    FROM runs WHERE run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return self._row_to_record(row)
                
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    def get_history(self, filter_params: Optional[QueryFilter] = None) -> pd.DataFrame:
        """Pobiera historię eksperymentów z zaawansowanym filtrowaniem."""
        if filter_params is None:
            filter_params = QueryFilter()
        
        try:
            with self.get_connection() as conn:
                query, params = self._build_query(filter_params)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return pd.DataFrame()
                
                # Parsuj JSON kolumny bezpiecznie
                for json_col, new_col in [
                    ('metrics_json', 'metrics'),
                    ('parameters_json', 'parameters'), 
                    ('tags_json', 'tags')
                ]:
                    if json_col in df.columns:
                        df[new_col] = df[json_col].apply(self._safe_json_loads)
                        df = df.drop(columns=[json_col])
                
                # Konwertuj daty
                for date_col in ['created_at', 'updated_at']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                return df.reset_index(drop=True)
                
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return pd.DataFrame()
    
    def _safe_json_loads(self, json_str: str) -> Any:
        """Bezpiecznie parsuje JSON string."""
        try:
            if not json_str or pd.isna(json_str):
                return {}
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _build_query(self, filter_params: QueryFilter) -> Tuple[str, Dict[str, Any]]:
        """Buduje zapytanie SQL z parametrami filtrowania."""
        
        base_query = """
            SELECT run_id, dataset, target, problem_type, engine, status,
                   created_at, updated_at, duration_seconds,
                   metrics_json, parameters_json, tags_json, notes
            FROM runs WHERE 1=1
        """
        
        conditions = []
        params = {}
        
        # Filtrowanie - używamy nazwanych parametrów dla bezpieczeństwa
        if filter_params.dataset:
            conditions.append("dataset LIKE :dataset")
            params['dataset'] = f"%{filter_params.dataset}%"
        
        if filter_params.target:
            conditions.append("target LIKE :target")
            params['target'] = f"%{filter_params.target}%"
        
        if filter_params.engine:
            conditions.append("engine LIKE :engine")
            params['engine'] = f"%{filter_params.engine}%"
        
        if filter_params.problem_type:
            conditions.append("problem_type = :problem_type")
            params['problem_type'] = filter_params.problem_type.value
        
        if filter_params.status:
            conditions.append("status = :status")
            params['status'] = filter_params.status.value
        
        if filter_params.run_ids:
            placeholders = ",".join([f":run_id_{i}" for i in range(len(filter_params.run_ids))])
            conditions.append(f"run_id IN ({placeholders})")
            for i, run_id in enumerate(filter_params.run_ids):
                params[f'run_id_{i}'] = run_id
        
        if filter_params.date_from:
            conditions.append("created_at >= :date_from")
            params['date_from'] = self._format_date(filter_params.date_from)
        
        if filter_params.date_to:
            conditions.append("created_at <= :date_to")
            params['date_to'] = self._format_date(filter_params.date_to)
        
        # Filtrowanie po tagach (JSON search) - uproszczone
        if filter_params.tags:
            for i, tag in enumerate(filter_params.tags):
                conditions.append(f"tags_json LIKE :tag_{i}")
                params[f'tag_{i}'] = f'%"{tag}"%'
        
        # Dodaj warunki do zapytania
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        # Sortowanie
        valid_columns = ['created_at', 'updated_at', 'dataset', 'target', 'engine', 'status']
        order_column = filter_params.order_by if filter_params.order_by in valid_columns else 'created_at'
        order_direction = "DESC" if filter_params.order_desc else "ASC"
        
        base_query += f" ORDER BY {order_column} {order_direction}"
        
        # Limit i offset
        base_query += f" LIMIT {filter_params.limit} OFFSET {filter_params.offset}"
        
        return base_query, params
    
    def _format_date(self, date_value: Union[datetime, date, str]) -> str:
        """Formatuje datę do ISO string."""
        if isinstance(date_value, str):
            return date_value
        elif isinstance(date_value, (date, datetime)):
            return date_value.isoformat()
        else:
            return str(date_value)
    
    def _row_to_record(self, row: tuple) -> RunRecord:
        """Konwertuje wiersz z bazy na RunRecord."""
        (run_id, dataset, target, problem_type, engine, status,
         created_at, updated_at, duration_seconds,
         metrics_json, parameters_json, tags_json, notes) = row
        
        return RunRecord(
            run_id=run_id,
            dataset=dataset,
            target=target,
            problem_type=ProblemType(problem_type) if problem_type else None,
            engine=engine,
            status=RunStatus(status) if status else RunStatus.RUNNING,
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            duration_seconds=duration_seconds,
            metrics=self._safe_json_loads(metrics_json),
            parameters=self._safe_json_loads(parameters_json),
            notes=notes,
            tags=self._safe_json_loads(tags_json)
        )
    
    def _validate_record(self, record: RunRecord) -> None:
        """Waliduje rekord przed zapisem."""
        if not record.run_id or not record.run_id.strip():
            raise ValueError("run_id cannot be empty")
        
        if not record.dataset or not record.dataset.strip():
            raise ValueError("dataset cannot be empty")
        
        if not record.target or not record.target.strip():
            raise ValueError("target cannot be empty")
        
        # Waliduj JSON serializable
        try:
            json.dumps(record.metrics)
            json.dumps(record.parameters)
            json.dumps(record.tags)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    def delete_run(self, run_id: str) -> bool:
        """Usuwa eksperyment."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted run {run_id}")
                else:
                    logger.warning(f"Run {run_id} not found for deletion")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {e}")
            return False
    
    def clear_history(self, confirm: bool = False) -> bool:
        """Czyści całą historię (wymaga potwierdzenia)."""
        if not confirm:
            logger.warning("clear_history requires confirm=True parameter")
            return False
        
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM runs")
                
            logger.info("Successfully cleared all history")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False
    
    def export_to_csv(self, 
                     filter_params: Optional[QueryFilter] = None, 
                     filepath: Optional[Path] = None) -> Union[bytes, bool]:
        """Eksportuje historię do CSV."""
        try:
            df = self.get_history(filter_params)
            
            if df.empty:
                logger.warning("No data to export")
                return b"" if filepath is None else False
            
            # Flatten nested JSON columns dla CSV
            df_export = df.copy()
            
            # Rozwiń metryki do osobnych kolumn (tylko numeryczne)
            if 'metrics' in df_export.columns:
                for idx, metrics in df_export['metrics'].items():
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and not pd.isna(value):
                                df_export.loc[idx, f"metric_{key}"] = value
                df_export = df_export.drop(columns=['metrics'])
            
            # Konwertuj listy na stringi
            for col in ['tags', 'parameters']:
                if col in df_export.columns:
                    df_export[col] = df_export[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
                    )
            
            csv_data = df_export.to_csv(index=False)
            
            if filepath:
                # Zapisz do pliku
                Path(filepath).write_text(csv_data, encoding='utf-8')
                logger.info(f"Exported {len(df)} records to {filepath}")
                return True
            else:
                # Zwróć bytes
                return csv_data.encode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return b"" if filepath is None else False
    
    def backup_database(self, backup_path: Optional[Path] = None) -> Optional[Path]:
        """Tworzy backup bazy danych."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.db_path.parent / f"backup_{timestamp}.sqlite"
            
            # Użyj VACUUM INTO dla atomowego backup (SQLite 3.27+)
            try:
                with self.get_connection() as conn:
                    conn.execute(f"VACUUM INTO '{backup_path}'")
            except sqlite3.OperationalError:
                # Fallback - kopiuj plik (starsze wersje SQLite)
                import shutil
                shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobiera statystyki bazy danych."""
        try:
            with self.get_connection() as conn:
                # Podstawowe statystyki
                total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                
                # Statystyki po statusie
                status_stats = {}
                for status in RunStatus:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM runs WHERE status = ?", 
                        (status.value,)
                    ).fetchone()[0]
                    status_stats[status.value] = count
                
                # Popularne datasety
                popular_datasets = conn.execute("""
                    SELECT dataset, COUNT(*) as count 
                    FROM runs 
                    GROUP BY dataset 
                    ORDER BY count DESC 
                    LIMIT 10
                """).fetchall()
                
                # Popularne targety
                popular_targets = conn.execute("""
                    SELECT target, COUNT(*) as count 
                    FROM runs 
                    GROUP BY target 
                    ORDER BY count DESC 
                    LIMIT 10
                """).fetchall()
                
                # Rozmiar bazy danych
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                
                return {
                    'total_runs': total_runs,
                    'status_distribution': status_stats,
                    'popular_datasets': dict(popular_datasets),
                    'popular_targets': dict(popular_targets),
                    'database_size_mb': round(db_size_mb, 2),
                    'schema_version': self.SCHEMA_VERSION
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

# Kompatybilność wsteczna z oryginalnym API
def ensure_db() -> sqlite3.Connection:
    """Kompatybilność wsteczna - tworzy połączenie."""
    tracker = MLExperimentTracker()
    return tracker.get_connection().__enter__()

def migrate_runs_table(conn: sqlite3.Connection) -> None:
    """Kompatybilność wsteczna - migracja już automatyczna."""
    pass

def log_run(conn: sqlite3.Connection, **kwargs) -> None:
    """Kompatybilność wsteczna - logowanie eksperymentu."""
    record = RunRecord(
        dataset=kwargs.get('dataset', ''),
        target=kwargs.get('target', ''),
        run_id=kwargs.get('run_id', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        problem_type=ProblemType(kwargs.get('problem_type')) if kwargs.get('problem_type') else None,
        engine=kwargs.get('engine_name'),
        metrics=kwargs.get('metrics', {})
    )
    
    tracker = MLExperimentTracker()
    tracker.log_run(record)

def get_history(conn: sqlite3.Connection, limit: int = 200) -> pd.DataFrame:
    """Kompatybilność wsteczna - pobieranie historii."""
    tracker = MLExperimentTracker()
    filter_params = QueryFilter(limit=limit)
    return tracker.get_history(filter_params)

def clear_history(conn: sqlite3.Connection) -> None:
    """Kompatybilność wsteczna - czyszczenie historii."""
    tracker = MLExperimentTracker()
    tracker.clear_history(confirm=True)

def export_history_csv(conn: sqlite3.Connection) -> bytes:
    """Kompatybilność wsteczna - eksport CSV."""
    tracker = MLExperimentTracker()
    return tracker.export_to_csv()