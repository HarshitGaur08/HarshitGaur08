"""HTTP interface for submitting and inspecting asynchronous backtest jobs."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from trading.engine import BacktestConfig
from trading.job_manager import BacktestJobManager

logger = logging.getLogger(__name__)
MAX_BARS_PER_JOB = 100_000


class MarketBar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    close: float = Field(gt=0, allow_inf_nan=False)


class BacktestRequest(BaseModel):
    bars: list[MarketBar] = Field(min_length=2, max_length=MAX_BARS_PER_JOB)
    config: BacktestConfig = Field(default_factory=BacktestConfig)


class SubmitResponse(BaseModel):
    job_id: str
    status: str


def create_app(
    manager: BacktestJobManager | None = None,
    *,
    database_path: str | Path | None = None,
) -> FastAPI:
    job_manager = manager or BacktestJobManager(
        database_path or os.getenv("BACKTEST_DB", "./data/backtests.sqlite3"),
        workers=int(os.getenv("BACKTEST_WORKERS", "2")),
        max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "1000")),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        job_manager.start()
        try:
            yield
        finally:
            job_manager.stop()

    app = FastAPI(
        title="Algorithmic Trading Backtester",
        version="1.0.0",
        description="Submit vectorized moving-average backtests as asynchronous jobs.",
        lifespan=lifespan,
    )
    app.state.job_manager = job_manager

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness() -> dict[str, str]:
        if not job_manager.is_running:
            raise HTTPException(status_code=503, detail="Worker pool is not running")
        return {"status": "ready"}

    @app.post(
        "/api/v1/backtests",
        response_model=SubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_backtest(request: BacktestRequest) -> SubmitResponse:
        timestamps = [bar.timestamp for bar in request.bars]
        if any(
            timestamp.tzinfo is None or timestamp.utcoffset() is None
            for timestamp in timestamps
        ):
            raise HTTPException(
                status_code=422,
                detail="Bar timestamps must include a timezone",
            )
        normalized_timestamps = [
            timestamp.astimezone(timezone.utc) for timestamp in timestamps
        ]
        if normalized_timestamps != sorted(normalized_timestamps) or len(
            normalized_timestamps
        ) != len(set(normalized_timestamps)):
            raise HTTPException(
                status_code=422,
                detail="Bar timestamps must be unique and chronologically sorted",
            )
        try:
            request.config.validate()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(request.bars) < request.config.slow_window + 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    "At least slow_window + 1 bars are required "
                    "for this strategy configuration"
                ),
            )
        try:
            job_id = job_manager.submit(
                {
                    "bars": [
                        {
                            "timestamp": bar.timestamp.isoformat(),
                            "close": bar.close,
                        }
                        for bar in request.bars
                    ],
                    "config": asdict(request.config),
                }
            )
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error("Backtest submission rejected: %s", exc)
            raise HTTPException(status_code=503, detail="Worker pool is unavailable") from exc
        return SubmitResponse(job_id=job_id, status="queued")

    @app.get("/api/v1/backtests/{job_id}")
    async def get_backtest(job_id: str, response: Response) -> dict[str, Any]:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Backtest job not found")
        if job["status"] in {"queued", "running"}:
            response.status_code = status.HTTP_202_ACCEPTED
        return job

    return app


app = create_app()
