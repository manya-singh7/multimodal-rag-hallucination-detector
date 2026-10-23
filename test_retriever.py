"""Tests for retrieval filtering, bibliography skipping, and score cutoffs in retriever.py."""

import numpy as np
import pytest

import retriever


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """Stub the embedding model to ensure tests run completely offline without model downloads."""
    monkeypatch.setattr(retriever, "_get_model", lambda: None)


def test_bibliography_pages_skipped():
    """Verify that chunks on known bibliography pages are skipped."""
    assert retriever.is_bibliography_chunk("paper1.pdf", 27)
    assert retriever.is_bibliography_chunk("paper1.pdf", 33)
    assert not retriever.is_bibliography_chunk("paper1.pdf", 26)

    assert retriever.is_bibliography_chunk("paper2.pdf", 10)
    assert retriever.is_bibliography_chunk("paper2.pdf", 12)
    assert not retriever.is_bibliography_chunk("paper2.pdf", 9)

    mock_docs = [
        {
            "id": "p1_content",
            "text": "Retrieval augmented generation grounds model responses in research evidence.",
            "source": "paper1.pdf",
            "page": 5,
        },
        {
            "id": "p1_bib",
            "text": "Smith, J. Language models in practice. arXiv 2023, arXiv:2301.00001.",
            "source": "paper1.pdf",
            "page": 29,
        },
        {
            "id": "p2_bib",
            "text": "References [1] Doe, A. Grounding benchmarks for multimodal models, 2024.",
            "source": "paper2.pdf",
            "page": 11,
        },
    ]

    out = retriever.retrieve("retrieval augmented generation", documents=mock_docs)
    retrieved_ids = [r["id"] for r in out["results"]]

    assert "p1_content" in retrieved_ids
    assert "p1_bib" not in retrieved_ids
    assert "p2_bib" not in retrieved_ids


def test_low_scoring_chunk_dropped():
    """Verify that chunks scoring below MIN_SCORE are dropped when a higher-scoring chunk passes."""
    mock_docs = [
        {
            "id": "relevant_chunk",
            "text": "Quantum superposition allows qubits to represent zero and one simultaneously.",
            "source": "quantum.pdf",
            "page": 1,
        },
        {
            "id": "marginal_chunk",
            "text": "The computer laboratory was established in nineteen ninety five with standard equipment.",
            "source": "history.pdf",
            "page": 1,
        },
    ]

    out = retriever.retrieve("quantum superposition qubits", top_k=5, documents=mock_docs)
    retrieved_ids = [r["id"] for r in out["results"]]

    assert "relevant_chunk" in retrieved_ids
    assert "marginal_chunk" not in retrieved_ids
    for r in out["results"]:
        assert r["score"] >= retriever.MIN_SCORE


def test_best_result_kept_when_below_min_score():
    """Verify that when all chunks score below MIN_SCORE, the single best result is preserved."""
    mock_docs = [
        {
            "id": "weak_match_1",
            "text": "Apples and oranges are common orchard fruits harvested during autumn months.",
            "source": "orchard.pdf",
            "page": 1,
        },
        {
            "id": "weak_match_2",
            "text": "Bananas and pineapples grow primarily in tropical and subtropical climates.",
            "source": "tropics.pdf",
            "page": 1,
        },
    ]

    out = retriever.retrieve("astrophysics planetary astronomy", top_k=5, documents=mock_docs)
    assert len(out["results"]) == 1
    assert len(out["text_chunks"]) == 1
    assert out["results"][0]["score"] < retriever.MIN_SCORE
