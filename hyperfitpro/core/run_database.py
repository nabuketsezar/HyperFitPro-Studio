from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sqlite3
import json
import time

from .config import load_app_settings, DEFAULT_RUN_DB


SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    run_folder TEXT NOT NULL UNIQUE,
    model_number INTEGER,
    model_name TEXT,
    method TEXT,
    rmse REAL,
    mae REAL,
    r2 REAL,
    aic REAL,
    aicc REAL,
    bic REAL,
    objective REAL,
    elapsed_s REAL,
    n_datasets INTEGER,
    n_points INTEGER,
    summary_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_number, model_name);
'''


@dataclass
class RunRecord:
    id: int | None
    created_at: float
    run_folder: str
    model_number: int | None = None
    model_name: str = ''
    method: str = ''
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    objective: float | None = None
    elapsed_s: float | None = None
    n_datasets: int = 0
    n_points: int = 0
    summary_json: str = ''


class RunDatabase:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            try:
                path = load_app_settings().run_database_path
            except Exception:
                path = DEFAULT_RUN_DB
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        return sqlite3.connect(str(self.path))

    def initialize(self):
        with self.connect() as con:
            con.executescript(SCHEMA_SQL)

    def index_run_summary(self, run_folder: str | Path, summary: dict[str, Any]) -> int:
        run_folder = str(Path(run_folder).expanduser())
        model = summary.get('model') or {}
        fit = summary.get('fit_result') or {}
        datasets = summary.get('datasets') or []
        n_points = sum(int(d.get('n', 0) or 0) for d in datasets)
        values = {
            'created_at': time.time(),
            'run_folder': run_folder,
            'model_number': model.get('number'),
            'model_name': model.get('name', ''),
            'method': fit.get('method', ''),
            'rmse': fit.get('rmse'),
            'mae': fit.get('mae'),
            'r2': fit.get('r2'),
            'aic': fit.get('aic'),
            'aicc': fit.get('aicc'),
            'bic': fit.get('bic'),
            'objective': fit.get('objective'),
            'elapsed_s': fit.get('elapsed_s'),
            'n_datasets': len(datasets),
            'n_points': n_points,
            'summary_json': json.dumps(summary, default=str),
        }
        cols = ','.join(values.keys())
        ph = ','.join('?' for _ in values)
        updates = ','.join(f'{k}=excluded.{k}' for k in values if k != 'run_folder')
        sql = f'INSERT INTO runs ({cols}) VALUES ({ph}) ON CONFLICT(run_folder) DO UPDATE SET {updates}'
        with self.connect() as con:
            cur = con.execute(sql, list(values.values()))
            con.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = con.execute('SELECT id FROM runs WHERE run_folder=?', (run_folder,)).fetchone()
            return int(row[0]) if row else -1

    def index_existing_run_folder(self, run_folder: str | Path) -> int:
        run_folder = Path(run_folder).expanduser()
        summary_path = run_folder / 'run_summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        return self.index_run_summary(run_folder, summary)

    def list_runs(self, limit: int = 50, model_number: int | None = None) -> list[dict[str, Any]]:
        sql = 'SELECT id,created_at,run_folder,model_number,model_name,method,rmse,mae,r2,aic,aicc,bic,objective,elapsed_s,n_datasets,n_points FROM runs'
        params: list[Any] = []
        if model_number is not None:
            sql += ' WHERE model_number=?'; params.append(model_number)
        sql += ' ORDER BY created_at DESC LIMIT ?'; params.append(int(limit))
        with self.connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            con.row_factory = sqlite3.Row
            row = con.execute('SELECT * FROM runs WHERE id=?', (int(run_id),)).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d['summary'] = json.loads(d.get('summary_json') or '{}')
        except Exception:
            d['summary'] = {}
        return d

    def delete_missing_run_folders(self) -> int:
        rows = self.list_runs(limit=100000)
        removed = 0
        with self.connect() as con:
            for r in rows:
                if not Path(r['run_folder']).exists():
                    con.execute('DELETE FROM runs WHERE id=?', (r['id'],)); removed += 1
            con.commit()
        return removed


def index_run(run_folder: str | Path, summary: dict[str, Any]) -> int:
    return RunDatabase().index_run_summary(run_folder, summary)
