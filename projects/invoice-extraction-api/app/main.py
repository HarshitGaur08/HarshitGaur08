"""FastAPI service that extracts structured fields from invoice documents."""

from __future__ import annotations

import logging
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.ml_model import InvoiceFields, InvoiceModel, InvoiceModelError

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "8"))
MAX_IMAGE_DIMENSION = 2048
MAX_IMAGE_PIXELS = 25_000_000

if MAX_UPLOAD_BYTES <= 0 or MAX_PDF_PAGES <= 0:
    raise ValueError("MAX_UPLOAD_BYTES and MAX_PDF_PAGES must be positive integers")


class InvoiceResult(BaseModel):
    """Normalized invoice extraction response."""

    document_id: str
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None
    currency: str | None
    amount: float | None
    tax_amount: float | None
    total_amount: float | None
    page_count: int
    extraction_status: str


class ExtractionResponse(BaseModel):
    invoice: InvoiceResult


def decode_document(content: bytes) -> list[Image.Image]:
    """Decode a supported image or PDF into bounded RGB page images."""
    if content.startswith(b"%PDF-"):
        try:
            with fitz.open(stream=content, filetype="pdf") as document:
                if document.needs_pass:
                    raise HTTPException(
                        status_code=422,
                        detail="Password-protected PDFs are not supported",
                    )
                if not 1 <= document.page_count <= MAX_PDF_PAGES:
                    raise HTTPException(
                        status_code=422,
                        detail=f"PDF must contain between 1 and {MAX_PDF_PAGES} pages",
                    )

                images: list[Image.Image] = []
                for page in document:
                    longest_side = max(page.rect.width, page.rect.height)
                    if longest_side <= 0:
                        raise HTTPException(
                            status_code=422,
                            detail="PDF contains a page with invalid dimensions",
                        )
                    scale = min(2.0, MAX_IMAGE_DIMENSION / longest_side)
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(scale, scale),
                        alpha=False,
                    )
                    images.append(
                        Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    )
                return images
        except HTTPException:
            raise
        except (fitz.FileDataError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail="Invalid or unreadable PDF") from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise HTTPException(
                        status_code=422,
                        detail="Upload must be a PNG, JPEG, or PDF invoice",
                    )
                image.verify()
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=422,
                        detail="Image dimensions exceed the supported pixel limit",
                    )
                image.thumbnail(
                    (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                    Image.Resampling.LANCZOS,
                )
                return [image.convert("RGB")]
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail="Upload must be a readable PNG, JPEG, or PDF invoice",
        ) from exc


def create_app(
    extractor: InvoiceModel | None = None,
    *,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Create the API; accepting an extractor allows isolated endpoint tests."""
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be a positive integer")
    invoice_model = extractor or InvoiceModel(
        model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await run_in_threadpool(invoice_model.load)
        yield

    app = FastAPI(
        title="Enterprise Invoice Extraction API",
        version="1.0.0",
        description="Extract normalized invoice fields from a PDF or image.",
        lifespan=lifespan,
    )
    app.state.invoice_model = invoice_model

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness() -> JSONResponse:
        if not invoice_model.is_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "model_not_loaded"},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.post(
        "/api/v1/invoices/extract",
        response_model=ExtractionResponse,
        summary="Extract invoice fields",
    )
    async def extract_invoice(
        file: Annotated[UploadFile, File(description="Invoice PDF or image")],
    ) -> ExtractionResponse:
        request_id = str(uuid.uuid4())
        try:
            content = await file.read(max_upload_bytes + 1)
        finally:
            await file.close()

        if not content:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {max_upload_bytes}-byte upload limit",
            )

        pages = await run_in_threadpool(decode_document, content)
        logger.info(
            "Processing invoice request_id=%s filename=%s bytes=%d pages=%d",
            request_id,
            file.filename,
            len(content),
            len(pages),
        )

        try:
            fields = await run_in_threadpool(invoice_model.extract, pages)
        except InvoiceModelError as exc:
            logger.error("Invoice extraction failed request_id=%s: %s", request_id, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Invoice extraction failed (request_id={request_id})",
            ) from exc

        values = fields.model_dump()
        extracted_count = sum(
            values[name] is not None
            for name in ("vendor", "amount", "tax_amount", "total_amount")
        )
        status = "EXTRACTED" if extracted_count == 4 else "PARTIAL"
        return ExtractionResponse(
            invoice=InvoiceResult(
                document_id=request_id,
                vendor=fields.vendor,
                invoice_number=fields.invoice_number,
                invoice_date=fields.invoice_date,
                currency=fields.currency,
                amount=fields.amount,
                tax_amount=fields.tax_amount,
                total_amount=fields.total_amount,
                page_count=len(pages),
                extraction_status=status,
            )
        )

    return app


app = create_app()
