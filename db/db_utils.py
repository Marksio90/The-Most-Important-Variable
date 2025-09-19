# db/db_utils.py — KOMPLETNY: zarządzanie bazą danych i historią treningów
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from backend.ml_integration import ModelConfig, TrainingResult


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


class DatabaseManager:
    """Zarządca bazy danych SQLite dla TMIV."""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Inicjalizuje bazę danych i tworzy tabele."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabela historii treningów
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    metrics TEXT,  -- JSON
                    metadata TEXT, -- JSON
                    training_time REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela domyślnych modeli
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS default_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dataset_name, target_column)
                )
            """)
            
            conn.commit()
    
    def save_training_record(self, record: TrainingRecord) -> bool:
        """Zapisuje rekord treningu do bazy danych."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO training_history 
                    (run_id, dataset_name, target_column, engine, problem_type, 
                     status, metrics, metadata, training_time, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.run_id,
                    record.dataset_name,
                    record.target_column,
                    record.engine,
                    record.problem_type,
                    record.status,
                    json.dumps(record.metrics) if record.metrics else None,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.training_time,
                    record.notes,
                    record.created_at.isoformat() if record.created_at else None
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Błąd zapisu do bazy danych: {e}")
            return False
    
    def get_training_history(self, limit: int = 50, dataset_name: Optional[str] = None) -> List[TrainingRecord]:
        """Pobiera historię treningów z bazy danych."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT run_id, dataset_name, target_column, engine, problem_type,
                           status, metrics, metadata, training_time, notes, created_at
                    FROM training_history
                """
                params = []
                
                if dataset_name:
                    query += " WHERE dataset_name = ?"
                    params.append(dataset_name)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    try:
                        # Parse JSON fields
                        metrics = json.loads(row[6]) if row[6] else {}
                        metadata = json.loads(row[7]) if row[7] else {}
                        
                        # Parse datetime
                        created_at = None
                        if row[10]:
                            try:
                                created_at = datetime.fromisoformat(row[10])
                            except:
                                created_at = datetime.now(timezone.utc)
                        
                        record = TrainingRecord(
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
                            created_at=created_at
                        )
                        records.append(record)
                        
                    except Exception as e:
                        print(f"Błąd parsowania rekordu: {e}")
                        continue
                
                return records
                
        except Exception as e:
            print(f"Błąd odczytu z bazy danych: {e}")
            return []
    
    def set_default_model(self, dataset_name: str, target_column: str, run_id: str) -> bool:
        """Ustawia domyślny model dla pary dataset/target."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO default_models 
                    (dataset_name, target_column, run_id, created_at)
                    VALUES (?, ?, ?, ?)
                """, (dataset_name, target_column, run_id, datetime.now(timezone.utc).isoformat()))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Błąd ustawiania domyślnego modelu: {e}")
            return False
    
    def get_default_model(self, dataset_name: str, target_column: str) -> Optional[str]:
        """Pobiera run_id domyślnego modelu dla pary dataset/target."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT run_id FROM default_models 
                    WHERE dataset_name = ? AND target_column = ?
                """, (dataset_name, target_column))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            print(f"Błąd pobierania domyślnego modelu: {e}")
            return None
    
    def delete_training_record(self, run_id: str) -> bool:
        """Usuwa rekord treningu z bazy danych."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM training_history WHERE run_id = ?", (run_id,))
                
                # Usuń też z domyślnych modeli jeśli był ustawiony
                cursor.execute("DELETE FROM default_models WHERE run_id = ?", (run_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Błąd usuwania rekordu: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobiera statystyki z bazy danych."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Podstawowe statystyki
                cursor.execute("SELECT COUNT(*) FROM training_history")
                total_runs = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT dataset_name) FROM training_history")
                unique_datasets = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT target_column) FROM training_history")
                unique_targets = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM training_history WHERE status = 'completed'")
                completed_runs = cursor.fetchone()[0]
                
                # Ostatni trening
                cursor.execute("SELECT created_at FROM training_history ORDER BY created_at DESC LIMIT 1")
                last_run = cursor.fetchone()
                last_run_date = last_run[0] if last_run else None
                
                return {
                    'total_runs': total_runs,
                    'unique_datasets': unique_datasets,
                    'unique_targets': unique_targets,
                    'completed_runs': completed_runs,
                    'last_run_date': last_run_date
                }
                
        except Exception as e:
            print(f"Błąd pobierania statystyk: {e}")
            return {}


# ================== HELPER FUNCTIONS ==================
def create_training_record(
    model_config: ModelConfig, 
    result: TrainingResult, 
    df: pd.DataFrame
) -> TrainingRecord:
    """Tworzy TrainingRecord z wyników treningu."""
    
    # Podstawowe informacje
    dataset_name = "dataset"  # będzie nadpisane w app.py
    target_column = model_config.target
    engine = result.metadata.get('engine', model_config.engine) if result.metadata else model_config.engine
    problem_type = result.metadata.get('problem_type', 'unknown') if result.metadata else 'unknown'
    
    # Metryki i metadane
    metrics = result.metrics or {}
    metadata = (result.metadata or {}).copy()
    
    # Dodatkowe informacje o danych
    metadata.update({
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'target_column': target_column,
        'model_config': {
            'test_size': model_config.test_size,
            'cv_folds': model_config.cv_folds,
            'random_state': model_config.random_state,
            'stratify': model_config.stratify,
            'enable_probabilities': model_config.enable_probabilities
        }
    })
    
    return TrainingRecord(
        dataset_name=dataset_name,
        target_column=target_column,
        engine=engine,
        problem_type=problem_type,
        metrics=metrics,
        metadata=metadata,
        status="completed"
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
    """Eksportuje historię treningów do pliku CSV."""
    try:
        history = db_manager.get_training_history(limit=1000)
        
        if not history:
            return False
        
        # Konwertuj do DataFrame
        data = []
        for record in history:
            row = {
                'run_id': record.run_id,
                'dataset_name': record.dataset_name,
                'target_column': record.target_column,
                'engine': record.engine,
                'problem_type': record.problem_type,
                'status': record.status,
                'training_time': record.training_time,
                'created_at': record.created_at.isoformat() if record.created_at else None,
                'notes': record.notes
            }
            
            # Dodaj główne metryki jako kolumny
            if record.metrics:
                for metric_name, metric_value in record.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        row[f'metric_{metric_name}'] = metric_value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"Błąd eksportu do CSV: {e}")
        return False


def import_history_from_csv(db_manager: DatabaseManager, csv_path: Union[str, Path]) -> bool:
    """Importuje historię treningów z pliku CSV."""
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # Odbuduj metryki
            metrics = {}
            for col in df.columns:
                if col.startswith('metric_') and not pd.isna(row[col]):
                    metric_name = col.replace('metric_', '')
                    metrics[metric_name] = row[col]
            
            # Stwórz rekord
            record = TrainingRecord(
                run_id=row.get('run_id', f"imported_{len(metrics)}"),
                dataset_name=row.get('dataset_name', 'unknown'),
                target_column=row.get('target_column', 'unknown'),
                engine=row.get('engine', 'unknown'),
                problem_type=row.get('problem_type', 'unknown'),
                status=row.get('status', 'completed'),
                metrics=metrics,
                training_time=row.get('training_time'),
                notes=row.get('notes', ''),
                created_at=pd.to_datetime(row.get('created_at')).to_pydatetime() if row.get('created_at') else None
            )
            
            db_manager.save_training_record(record)
        
        return True
        
    except Exception as e:
        print(f"Błąd importu z CSV: {e}")
        return False


# ================== COMPATIBILITY FUNCTIONS ==================
# Dla kompatybilności ze starym kodem

class MLExperimentTracker:
    """Alias dla DatabaseManager dla kompatybilności."""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_manager = DatabaseManager(db_path)
    
    def log_run(self, record) -> str:
        """Loguje run do bazy danych."""
        # Konwertuj stary format na nowy jeśli potrzeba
        if hasattr(record, 'run_id'):
            training_record = record
        else:
            # Konwersja ze starého formatu
            training_record = TrainingRecord(
                dataset_name=getattr(record, 'dataset', 'unknown'),
                target_column=getattr(record, 'target', 'unknown'),
                engine=getattr(record, 'engine', 'unknown'),
                problem_type=getattr(record, 'problem_type', 'unknown'),
                run_id=getattr(record, 'run_id', None),
                metrics=getattr(record, 'metrics', {}),
                notes=getattr(record, 'notes', '')
            )
        
        self.db_manager.save_training_record(training_record)
        return training_record.run_id
    
    def get_history(self, query_filter=None):
        """Pobiera historię w formacie DataFrame dla kompatybilności."""
        history = self.db_manager.get_training_history()
        
        if not history:
            return pd.DataFrame()
        
        # Konwertuj do DataFrame
        data = []
        for record in history:
            data.append({
                'run_id': record.run_id,
                'dataset': record.dataset_name,
                'target': record.target_column,
                'engine': record.engine,
                'status': record.status,
                'created_at': record.created_at
            })
        
        return pd.DataFrame(data)
    
    def set_default_model(self, dataset: str, target: str, run_id: str) -> bool:
        """Ustawia domyślny model."""
        return self.db_manager.set_default_model(dataset, target, run_id)
    
    def get_default_model(self, dataset: str, target: str) -> Optional[str]:
        """Pobiera domyślny model."""
        return self.db_manager.get_default_model(dataset, target)
    
    def delete_run(self, run_id: str) -> bool:
        """Usuwa run."""
        return self.db_manager.delete_training_record(run_id)


# Pozostałe aliasy dla kompatybilności
class RunRecord:
    """Alias dla TrainingRecord."""
    def __init__(self, **kwargs):
        # Mapowanie starych nazw pól na nowe
        self.dataset = kwargs.get('dataset', kwargs.get('dataset_name', 'unknown'))
        self.target = kwargs.get('target', kwargs.get('target_column', 'unknown'))
        self.run_id = kwargs.get('run_id')
        self.problem_type = kwargs.get('problem_type', 'unknown')
        self.engine = kwargs.get('engine', 'unknown')
        self.status = kwargs.get('status', 'completed')
        self.metrics = kwargs.get('metrics', {})
        self.notes = kwargs.get('notes', '')
        self.duration_seconds = kwargs.get('duration_seconds')
        self.tags = kwargs.get('tags', [])


class ProblemType:
    """Enum-like klasa dla typu problemu."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    OTHER = "other"


class RunStatus:
    """Enum-like klasa dla statusu."""
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class QueryFilter:
    """Klasa filtra dla kompatybilności."""
    def __init__(self, limit: int = 50):
        self.limit = limit