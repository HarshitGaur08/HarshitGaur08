from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app, decode_document
from app.ml_model import InvoiceFields


class FakeInvoiceModel:
    is_ready = False

    def load(self) -> None:
        self.is_ready = True

    def extract(self, images: list[Image.Image]) -> InvoiceFields:
        assert len(images) == 1
        return InvoiceFields(
            vendor="Example Supplies",
            invoice_number="INV-42",
            invoice_date="2026-10-10",
            currency="USD",
            amount=100.0,
            tax_amount=8.0,
            total_amount=108.0,
        )


def test_image_is_decoded_to_rgb() -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")

    pages = decode_document(buffer.getvalue())

    assert len(pages) == 1
    assert pages[0].mode == "RGB"


def test_extract_returns_normalized_invoice() -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    client = TestClient(create_app(FakeInvoiceModel()))

    with client:
        response = client.post(
            "/api/v1/invoices/extract",
            files={"file": ("invoice.png", buffer.getvalue(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()["invoice"]
    assert payload["vendor"] == "Example Supplies"
    assert payload["amount"] == 100.0
    assert payload["tax_amount"] == 8.0
    assert payload["extraction_status"] == "EXTRACTED"


def test_rejects_empty_file() -> None:
    client = TestClient(create_app(FakeInvoiceModel()))

    with client:
        response = client.post(
            "/api/v1/invoices/extract",
            files={"file": ("empty.png", b"", "image/png")},
        )

    assert response.status_code == 422
