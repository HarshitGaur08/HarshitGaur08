from types import SimpleNamespace

from rag_service.pipeline import QueryEngine


class FakeVectorStore:
    def __init__(self, matches):
        self.matches = matches

    def similarity_search_with_relevance_scores(self, _question, *, k):
        return self.matches[:k]


class FakeLanguageModel:
    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return "The answer is in the handbook."


def test_query_returns_answer_and_source_citation() -> None:
    document = SimpleNamespace(
        page_content="The retention period is 30 days.",
        metadata={"source": "handbook.pdf", "page": 3},
    )
    model = FakeLanguageModel()
    engine = QueryEngine(
        FakeVectorStore([(document, 0.9)]),
        model,
        top_k=2,
    )

    result = engine.answer("What is the retention period?")

    assert result["answer"] == "The answer is in the handbook."
    assert result["sources"][0]["document"] == "handbook.pdf"
    assert result["sources"][0]["page"] == 3
    assert "The retention period is 30 days." in model.prompt


def test_query_does_not_call_model_without_relevant_context() -> None:
    document = SimpleNamespace(
        page_content="Unrelated content",
        metadata={"source": "other.txt", "page": 1},
    )
    model = FakeLanguageModel()
    engine = QueryEngine(
        FakeVectorStore([(document, 0.1)]),
        model,
        score_threshold=0.25,
    )

    result = engine.answer("Question without matching context")

    assert result["sources"] == []
    assert "could not find relevant" in result["answer"]
    assert model.prompt is None
