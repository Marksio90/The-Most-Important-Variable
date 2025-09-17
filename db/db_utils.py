# db/db_utils.py
from __future__ import annotations
from pathlib import Path
import sqlite3
import json
from typing import Optional
import pandas as pd

DB_PATH = Path("tmiv_out/history.sqlite")

def ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def migrate_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dataset TEXT NOT NULL,
            target TEXT NOT NULL,
            problem_type TEXT,
            engine TEXT,
            run_id TEXT UNIQUE,
            metrics_json TEXT
        );
        """
    )
    conn.commit()

def log_run(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    target: str,
    problem_type: Optional[str],
    engine_name: Optional[str],
    metrics: dict,
    run_id: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
        (created_at, dataset, target, problem_type, engine, run_id, metrics_json)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)
        """,
        (dataset, target, problem_type or "", engine_name or "", run_id, json.dumps(metrics)),
    )
    conn.commit()

def get_history(conn: sqlite3.Connection, limit: int = 200) -> pd.DataFrame:
    cur = conn.execute(
        "SELECT created_at, dataset, target, problem_type, engine, run_id, metrics_json FROM runs ORDER BY id ASC"
    )
    rows = cur.fetchall()
    cols = ["created_at", "dataset", "target", "problem_type", "engine", "run_id", "metrics_json"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    if limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)

def clear_history(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM runs")
    conn.commit()

def export_history_csv(conn: sqlite3.Connection) -> bytes:
    df = get_history(conn, limit=0)
    return df.to_csv(index=False).encode("utf-8")
