"""HTTP API for grounded questions against the local vector index."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from rag_service.pipeline import QueryEngine

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question must not be blank")
        return normalized


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


def create_app(engine: QueryEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or await run_in_threadpool(
            QueryEngine.from_environment
        )
        yield

    app = FastAPI(
        title="Distributed Document Q&A",
        version="1.0.0",
        description="Answers questions using retrieved document context and citations.",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness() -> dict[str, str]:
        if not hasattr(app.state, "engine"):
            raise HTTPException(status_code=503, detail="Query engine is not ready")
        return {"status": "ready"}

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        try:
            result = await run_in_threadpool(
                app.state.engine.answer,
                request.question,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            logger.exception("Question answering failed")
            raise HTTPException(
                status_code=503,
                detail="Question answering is temporarily unavailable",
            ) from exc
        return QueryResponse.model_validate(result)

    return app


app = create_app()
