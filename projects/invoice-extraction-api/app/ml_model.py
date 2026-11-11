"""Hugging Face vision-language model wrapper for invoice extraction."""

from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Read this invoice page and extract the requested fields.
Return exactly one JSON object with these keys:
vendor, invoice_number, invoice_date, currency, amount, tax_amount, total_amount.
Use null for a field that is not visible. Use ISO 8601 (YYYY-MM-DD) for
invoice_date, a three-letter ISO 4217 code for currency, and JSON numbers for
amount, tax_amount, and total_amount. Do not infer values that are not present.
The amount field is the pre-tax/net amount; total_amount is the amount due."""


class InvoiceModelError(RuntimeError):
    """Raised when the model cannot return a valid invoice extraction."""


class InvoiceFields(BaseModel):
    """Validated invoice fields produced by the model."""

    model_config = ConfigDict(extra="forbid")

    vendor: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    amount: float | None = Field(default=None, ge=0)
    tax_amount: float | None = Field(default=None, ge=0)
    total_amount: float | None = Field(default=None, ge=0)


class InvoiceModel:
    """Loads a Hugging Face VLM once and extracts fields from invoice pages."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_id = model_id
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self._processor is not None and self._model is not None

    def load(self) -> None:
        """Load the processor and model; startup fails explicitly if unavailable."""
        if self.is_ready:
            return

        with self._load_lock:
            if self.is_ready:
                return

            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype="auto",
                    device_map="auto",
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.exception("Could not load invoice model %s", self.model_id)
                raise InvoiceModelError(
                    f"Could not load model {self.model_id!r}: {exc}"
                ) from exc

            self._processor = processor
            self._model = model
            self._torch = torch
            logger.info("Loaded invoice model %s", self.model_id)

    def extract(self, images: list[Image.Image]) -> InvoiceFields:
        """Extract and merge fields from one or more already-decoded pages."""
        if not self.is_ready:
            raise InvoiceModelError("Invoice model is not loaded")
        if not images:
            raise InvoiceModelError("No invoice pages were provided")

        combined: dict[str, Any] = {}
        for image in images:
            page_fields = self._extract_page(image)
            for name, value in page_fields.model_dump().items():
                if combined.get(name) is None and value is not None:
                    combined[name] = value

        result = InvoiceFields.model_validate(combined)
        if not any(
            (result.vendor, result.amount is not None, result.tax_amount is not None,
             result.total_amount is not None)
        ):
            raise InvoiceModelError("No recognizable invoice fields were found")
        return result

    def _extract_page(self, image: Image.Image) -> InvoiceFields:
        assert self._processor is not None and self._model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]

        try:
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            ).to(self._model.device)
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=256)

            input_length = inputs["input_ids"].shape[-1]
            output_tokens = generated[:, input_length:]
            response = self._processor.batch_decode(
                output_tokens,
                skip_special_tokens=True,
            )[0]
            return self._parse_response(response)
        except InvoiceModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
            logger.exception("Invoice model inference failed")
            raise InvoiceModelError(f"Invoice model inference failed: {exc}") from exc

    @staticmethod
    def _parse_response(response: str) -> InvoiceFields:
        start = response.find("{")
        if start < 0:
            raise InvoiceModelError("Model response did not contain a JSON object")

        try:
            payload, _ = json.JSONDecoder().raw_decode(response[start:])
            if not isinstance(payload, dict):
                raise InvoiceModelError("Model response JSON must be an object")
            return InvoiceFields.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Invoice model returned invalid structured data")
            raise InvoiceModelError(
                "Model response was not valid invoice JSON"
            ) from exc
