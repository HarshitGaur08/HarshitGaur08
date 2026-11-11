# Enterprise Invoice Extraction API

A FastAPI service that accepts invoice images or PDFs, runs a Hugging Face
vision-language model, validates the extracted fields, and returns an
SAP OData V2-style entity. It is designed as an integration boundary for SAP
BTP and S/4HANA; it does not connect to an SAP tenant or persist invoices.

## Architecture

```mermaid
flowchart LR
    Client["SAP BTP / S/4HANA<br/>or integration client"] -->|multipart PDF/image| API["FastAPI<br/>upload validation"]
    API --> Decode["PDF/image decoder<br/>bounded page count"]
    Decode --> Model["InvoiceModel<br/>Hugging Face VLM"]
    Model --> Validate["Pydantic field validation"]
    Validate --> OData["SAP OData V2-style<br/>JSON entity"]
    OData --> Client
    API --> Health["Liveness / readiness"]
```

## Project layout

```text
invoice-extraction-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI routes, validation and OData response
│   └── ml_model.py       # Hugging Face model loading and inference
├── tests/
│   └── test_api.py
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Requirements and setup

- Python 3.11+
- The model requires a supported PyTorch environment. CPU inference works but
  can be slow; for production, use a GPU-enabled image/runtime and provision
  sufficient memory for the selected model.
- The first start downloads the model from Hugging Face. Provide network access
  or pre-populate the Hugging Face cache in your deployment environment.

```powershell
cd projects/invoice-extraction-api
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Configuration:

| Variable | Default | Description |
|---|---|---|
| `HF_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face vision-language model ID |
| `MAX_UPLOAD_BYTES` | `15728640` | Maximum upload size (15 MiB) |
| `MAX_PDF_PAGES` | `8` | Maximum pages rasterized per PDF |
| `LOG_LEVEL` | `INFO` | Python logging level |

## API

### `POST /api/v1/invoices/extract`

Send a PDF, PNG, or JPEG as the `file` multipart field. Uploads are limited to
15 MiB by default, and PDFs are limited to eight pages.

```powershell
curl.exe -X POST http://localhost:8000/api/v1/invoices/extract `
  -F "file=@invoice.pdf;type=application/pdf"
```

Successful response (fields the model cannot read or determine are `null`):

```json
{
  "d": {
    "__metadata": {
      "type": "ZINVOICE_EXTRACTION_SRV.InvoiceExtraction"
    },
    "DocumentId": "f378bf63-0135-4afe-b95e-a949ff5e3a79",
    "Vendor": "Example Supplies Ltd.",
    "InvoiceNumber": "INV-2026-1042",
    "DocumentDate": "2026-10-10",
    "Currency": "USD",
    "Amount": 100.0,
    "Tax": 8.0,
    "TotalAmount": 108.0,
    "PageCount": 1,
    "ExtractionStatus": "EXTRACTED"
  }
}
```

`Amount` is the pre-tax amount; `Tax` is the tax amount; `TotalAmount` is the
amount due. `ExtractionStatus` is `PARTIAL` when one or more of the vendor,
amount, tax, or total fields are missing. Consumers should validate extracted
values before posting them into a financial system.

Other endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Returns 200 only after the model has loaded |
| `GET` | `/docs` | Interactive OpenAPI documentation |

Common errors use standard HTTP status codes: `413` for oversized uploads,
`422` for unsupported or invalid documents, and `502` when inference or
structured model output fails. Responses include a request identifier on
inference failures; logs do not include invoice contents or extracted values.

## Docker

```powershell
docker build -t invoice-extraction-api .
docker run --rm -p 8000:8000 `
  -v invoice-hf-cache:/home/app/.cache/huggingface `
  invoice-extraction-api
```

The image runs as a non-root user. The model weights are downloaded on startup
and cached in the mounted Hugging Face cache. Use an appropriately sized host
or GPU-enabled runtime for the selected model.

## SAP integration notes

The `d` envelope and entity metadata follow the familiar SAP Gateway OData V2
JSON convention, making the response straightforward for an ABAP HTTP/OData
consumer to deserialize. For a real SAP BTP or S/4HANA deployment:

1. Place the service behind an authenticated API gateway and require TLS.
2. Configure SAP BTP destinations and OAuth 2.0 / client-certificate
   authentication according to the tenant's integration policy.
3. Map this response to the target CDS/OData entity and apply SAP-side business
   validation, authorization, duplicate detection, and idempotency controls.
4. Treat model output as untrusted extraction, not an automatically approved
   accounting entry. Add human review for low-confidence or incomplete results.

No SAP credentials, tenant-specific URLs, or external services are required to
run the local API.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
