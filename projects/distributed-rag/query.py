"""Command-line interface for asking questions about indexed documents."""

from __future__ import annotations

import argparse
import json
import logging
import os

from rag_service.pipeline import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer from indexed documents")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        result = QueryEngine.from_environment().answer(args.question)
    except (OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).exception("Question answering failed")
        raise SystemExit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
