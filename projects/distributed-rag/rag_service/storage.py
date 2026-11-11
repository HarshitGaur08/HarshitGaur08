"""Local and Amazon S3 object storage adapters for source documents."""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from typing import Protocol

logger = logging.getLogger(__name__)


class ObjectStore(Protocol):
    """Minimal object-store interface used by the ingestion pipeline."""

    def put(self, key: str, content: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str = "") -> list[str]: ...


def _validate_key(key: str) -> str:
    normalized = PurePosixPath(key)
    if not key or not normalized.parts or normalized.is_absolute() or any(
        part in {"", ".", ".."} for part in normalized.parts
    ):
        raise ValueError("Object key must be a non-empty relative path")
    return normalized.as_posix()


class LocalObjectStore:
    """Filesystem-backed S3 simulation suitable for local development."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _object_path(self, key: str) -> Path:
        object_path = (self.root / _validate_key(key)).resolve()
        if not object_path.is_relative_to(self.root):
            raise ValueError("Object key escapes the configured storage root")
        return object_path

    def put(self, key: str, content: bytes) -> None:
        destination = self._object_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(destination)
        except OSError:
            temporary.unlink(missing_ok=True)
            logger.exception("Could not write local object %s", key)
            raise

    def get(self, key: str) -> bytes:
        try:
            return self._object_path(key).read_bytes()
        except OSError:
            logger.exception("Could not read local object %s", key)
            raise

    def list_keys(self, prefix: str = "") -> list[str]:
        normalized_prefix = prefix.strip("/")
        if normalized_prefix:
            _validate_key(normalized_prefix)
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
            and path.name != ".keep"
            and (
                not normalized_prefix
                or path.relative_to(self.root).as_posix() == normalized_prefix
                or path.relative_to(self.root).as_posix().startswith(
                    normalized_prefix + "/"
                )
            )
        )


class S3ObjectStore:
    """AWS S3 adapter. Credentials are resolved by the standard boto3 chain."""

    def __init__(self, bucket: str, region: str | None = None) -> None:
        if not bucket.strip():
            raise ValueError("S3_BUCKET must be set when STORAGE_BACKEND=s3")
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise RuntimeError(
                "Install boto3 to use STORAGE_BACKEND=s3"
            ) from exc

        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)
        self._aws_errors = (BotoCoreError, ClientError)

    def put(self, key: str, content: bytes) -> None:
        normalized = _validate_key(key)
        try:
            self.client.put_object(Bucket=self.bucket, Key=normalized, Body=content)
        except self._aws_errors:
            logger.exception("Could not upload object to S3: %s", normalized)
            raise

    def get(self, key: str) -> bytes:
        normalized = _validate_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=normalized)
            return response["Body"].read()
        except self._aws_errors:
            logger.exception("Could not download object from S3: %s", normalized)
            raise

    def list_keys(self, prefix: str = "") -> list[str]:
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            return sorted(
                item["Key"]
                for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix)
                for item in page.get("Contents", [])
            )
        except self._aws_errors:
            logger.exception("Could not list S3 objects with prefix %s", prefix)
            raise


def create_object_store() -> ObjectStore:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalObjectStore(os.getenv("OBJECT_STORE_DIR", "./data/objects"))
    if backend == "s3":
        return S3ObjectStore(
            bucket=os.getenv("S3_BUCKET", ""),
            region=os.getenv("AWS_REGION"),
        )
    raise ValueError(f"Unsupported STORAGE_BACKEND: {backend!r}")
