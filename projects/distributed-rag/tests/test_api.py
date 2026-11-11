from fastapi.testclient import TestClient

from rag_service.api import create_app


class FakeQueryEngine:
    def answer(self, question: str) -> dict:
        return {
            "answer": f"Answer for: {question}",
            "sources": [{"document": "guide.md", "page": 1, "relevance": 0.9}],
        }


def test_query_endpoint_returns_answer_and_sources() -> None:
    client = TestClient(create_app(FakeQueryEngine()))

    with client:
        response = client.post(
            "/api/v1/query",
            json={"question": "What does the guide say?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer for: What does the guide say?"
    assert response.json()["sources"][0]["document"] == "guide.md"


def test_query_endpoint_rejects_empty_question() -> None:
    client = TestClient(create_app(FakeQueryEngine()))

    with client:
        response = client.post("/api/v1/query", json={"question": " "})

    assert response.status_code == 422
