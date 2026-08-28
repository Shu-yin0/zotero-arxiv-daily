"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import pytest
import numpy as np

from zotero_arxiv_daily.reranker.local import LocalReranker


@pytest.mark.slow
def test_local_reranker(config, monkeypatch):
    class FakeSimilarity:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return self.values

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.array([[len(text), 1.0] for text in texts], dtype=float)

        def similarity(self, first, second):
            return FakeSimilarity(first @ second.T)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
