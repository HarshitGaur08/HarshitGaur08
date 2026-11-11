"""Upload local documents to the configured object store and index their text."""

from __future__ import annotations

import argparse
import logging
import os

from rag_service.pipeline import ingest_directory, load_embeddings, load_vector_store
from rag_service.storage import create_object_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing .txt, .md, or .pdf files")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--max-file-mib", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.max_file_mib <= 0:
        parser.error("--max-file-mib must be positive")

    summary = ingest_directory(
        args.directory,
        object_store=create_object_store(),
        vector_store=load_vector_store(embeddings=load_embeddings()),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
    )
    print(
        f"Ingestion complete: files={summary.files}, chunks={summary.chunks}, "
        f"skipped={summary.skipped}"
    )
    if summary.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
