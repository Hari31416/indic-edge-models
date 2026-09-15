"""Unit tests for dataset manifest loader and synthetic slice generation."""

import tempfile
from pathlib import Path
from indic_edge_embeddings.eval.loader import (
    CorpusDoc,
    EvaluationManifest,
    Query,
    create_romanized_query_slice,
)


def test_manifest_save_and_load():
    """Verify manifest can be saved to disk and loaded without loss."""
    queries = {
        "q1": Query(query_id="q1", text="भारत", language="hi", script="native"),
    }
    corpus = {
        "d1": CorpusDoc(
            doc_id="d1", text="भारत दक्षिण एशिया का एक देश है।", title="भारत"
        ),
    }
    qrels = {"q1": {"d1": 1}}

    manifest = EvaluationManifest(
        name="test_hi",
        language="hi",
        split="dev",
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = manifest.save(tmp_dir)
        loaded = EvaluationManifest.load(save_path)

        assert loaded.name == manifest.name
        assert loaded.language == "hi"
        assert len(loaded.queries) == 1
        assert loaded.queries["q1"].text == "भारत"
        assert loaded.corpus["d1"].title == "भारत"
        assert loaded.qrels["q1"]["d1"] == 1


def test_create_romanized_query_slice():
    """Verify Romanized query slice creation preserves source IDs and qrels."""
    queries = {
        "q1": Query(query_id="q1", text="भारत", language="hi", script="native"),
    }
    corpus = {
        "d1": CorpusDoc(doc_id="d1", text="भारत दक्षिण एशिया का एक देश है।"),
    }
    qrels = {"q1": {"d1": 1}}

    manifest = EvaluationManifest(
        name="test_hi",
        language="hi",
        split="dev",
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )

    # Dummy transliteration function for test
    def dummy_transliterate(text: str) -> str:
        return "bharat" if text == "भारत" else text

    roman_manifest = create_romanized_query_slice(manifest, dummy_transliterate)

    assert "q1_roman" in roman_manifest.queries
    assert roman_manifest.queries["q1_roman"].text == "bharat"
    assert roman_manifest.queries["q1_roman"].script == "romanized"
    assert roman_manifest.queries["q1_roman"].source_query_id == "q1"
    assert "q1_roman" in roman_manifest.qrels
    assert roman_manifest.qrels["q1_roman"]["d1"] == 1
