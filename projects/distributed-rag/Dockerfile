FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    CHROMA_DIR=/data/chroma \
    OBJECT_STORE_DIR=/data/objects \
    STORAGE_BACKEND=local

WORKDIR /service

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY rag_service ./rag_service
COPY ingest.py query.py ./
RUN mkdir -p /data/chroma /data/objects && chown -R app:app /service /data /home/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
