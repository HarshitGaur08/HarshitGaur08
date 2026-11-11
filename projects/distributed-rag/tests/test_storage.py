from pathlib import Path

import pytest

from rag_service.storage import LocalObjectStore


def test_local_object_store_round_trip_and_prefix_listing(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    store.put("handbook/guide.txt", b"sample contents")
    store.put("handbook/archive/old.txt", b"archived")
    store.put("other/guide.txt", b"not in prefix")

    assert store.get("handbook/guide.txt") == b"sample contents"
    assert store.list_keys("handbook") == [
        "handbook/archive/old.txt",
        "handbook/guide.txt",
    ]


@pytest.mark.parametrize("key", ["../secret", "/absolute.txt", "folder/../secret"])
def test_local_object_store_rejects_traversal(tmp_path: Path, key: str) -> None:
    store = LocalObjectStore(tmp_path / "objects")

    with pytest.raises(ValueError):
        store.put(key, b"not allowed")
