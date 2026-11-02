"""Tests that need no database or model download."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.ingest import split
from app.llm import build_messages
from app.store import Hit, rrf


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert [i for i, _ in fused][:2] == [1, 3]
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_split_overlaps_and_hashes_are_unique():
    text = " ".join(f"sentence number {i}." for i in range(200))
    chunks = split("doc1", "Doc 1", text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 5
    assert all(len(c.content) <= 200 for c in chunks)
    assert len({c.content_hash for c in chunks}) == len(chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_same_text_in_different_docs_is_kept():
    a = split("a", "A", "same text", 100, 10)[0]
    b = split("b", "B", "same text", 100, 10)[0]
    assert a.content_hash != b.content_hash


def test_prompt_numbers_sources():
    hits = [Hit(1, "d1", "Warsaw", "Warsaw is the capital of Poland.", 0.9),
            Hit(2, "d2", "Tesla", "Tesla was born in 1856.", 0.8)]
    system, human = build_messages("When was Tesla born?", hits)
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert "[1] (Warsaw)" in human.content and "[2] (Tesla)" in human.content
    assert human.content.endswith("Question: When was Tesla born?")
