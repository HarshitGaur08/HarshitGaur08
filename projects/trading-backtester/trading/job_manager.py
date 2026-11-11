"""SQLite-backed job queue and bounded worker pool for backtests."""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from trading.engine import BacktestConfig, run_backtest

logger = logging.getLogger(__name__)
_STOP = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BacktestJobManager:
    """Persist submitted jobs and execute them on a fixed number of threads."""

    def __init__(
        self,
        database_path: str | Path = "./data/backtests.sqlite3",
        *,
        workers: int = 2,
        max_pending_jobs: int = 1000,
    ) -> None:
        if workers < 1 or max_pending_jobs < 1:
            raise ValueError("workers and max_pending_jobs must be positive")
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.max_pending_jobs = max_pending_jobs
        self._queue: queue.Queue[str | object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._state_lock = threading.Lock()
        self._started = False
        self._stopping = False
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._started and not self._stopping

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'succeeded', 'failed')
                    ),
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_backtest_jobs_status "
                "ON backtest_jobs(status, created_at)"
            )

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._stopping = False
            self._started = True
            with self._connect() as connection:
                connection.execute(
                    "UPDATE backtest_jobs SET status='queued', updated_at=? "
                    "WHERE status='running'",
                    (_utc_now(),),
                )
                pending = connection.execute(
                    "SELECT job_id FROM backtest_jobs WHERE status='queued' "
                    "ORDER BY created_at"
                ).fetchall()
            for row in pending:
                self._queue.put(row["job_id"])
            self._threads = [
                threading.Thread(
                    target=self._worker,
                    name=f"backtest-worker-{index + 1}",
                    daemon=True,
                )
                for index in range(self.workers)
            ]
            for thread in self._threads:
                thread.start()
            logger.info(
                "Started backtest worker pool workers=%d recovered=%d",
                self.workers,
                len(pending),
            )

    def stop(self, timeout: float = 10.0) -> None:
        with self._state_lock:
            if not self._started:
                return
            self._stopping = True
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(_STOP)
            self._started = False
        for thread in threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Worker %s did not stop before timeout", thread.name)
        self._threads.clear()

    def submit(self, payload: dict[str, Any]) -> str:
        with self._state_lock:
            if not self._started or self._stopping:
                raise RuntimeError("Job manager is not accepting work")
            job_id = str(uuid.uuid4())
            now = _utc_now()
            with self._connect() as connection:
                pending_count = connection.execute(
                    "SELECT COUNT(*) FROM backtest_jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                if pending_count >= self.max_pending_jobs:
                    raise OverflowError("Backtest queue is at capacity")
                connection.execute(
                    "INSERT INTO backtest_jobs "
                    "(job_id, status, payload, created_at, updated_at) "
                    "VALUES (?, 'queued', ?, ?, ?)",
                    (job_id, json.dumps(payload), now, now),
                )
            self._queue.put(job_id)
            return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT job_id, status, result, error, created_at, updated_at "
                "FROM backtest_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._execute(str(item))
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM backtest_jobs "
                "WHERE job_id=? AND status='queued'",
                (job_id,),
            ).fetchone()
            if row is None:
                return
            connection.execute(
                "UPDATE backtest_jobs SET status='running', updated_at=? "
                "WHERE job_id=? AND status='queued'",
                (_utc_now(), job_id),
            )
            payload = json.loads(row["payload"])

        try:
            frame = pd.DataFrame(payload["bars"])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
            config = BacktestConfig(**payload["config"])
            result = run_backtest(frame, config)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._finish(job_id, status="failed", error=str(exc))
            logger.warning("Backtest job failed job_id=%s error=%s", job_id, exc)
        except Exception as exc:
            self._finish(
                job_id,
                status="failed",
                error="Unexpected backtest execution failure",
            )
            logger.exception("Unexpected backtest failure job_id=%s", job_id)
        else:
            self._finish(
                job_id,
                status="succeeded",
                result=json.dumps(result, allow_nan=False),
            )

    def _finish(
        self,
        job_id: str,
        *,
        status: str,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE backtest_jobs SET status=?, result=?, error=?, updated_at=? "
                "WHERE job_id=?",
                (status, result, error, _utc_now(), job_id),
            )
