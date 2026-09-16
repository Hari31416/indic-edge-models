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


def test_build_fast_miracl_manifests():
    """Verify build_fast_miracl_manifests retains all positive docs and samples distractors."""
    from indic_edge_embeddings.eval.build_fast_eval import build_fast_miracl_manifests

    queries = {
        "q1": Query(query_id="q1", text="भारत", language="hi", script="native"),
        "q2": Query(query_id="q2", text="दिल्ली", language="hi", script="native"),
    }
    corpus = {
        "d_pos1": CorpusDoc(doc_id="d_pos1", text="भारत दक्षिण एशिया का देश है।"),
        "d_pos2": CorpusDoc(doc_id="d_pos2", text="दिल्ली भारत की राजधानी है।"),
        "d_dis1": CorpusDoc(doc_id="d_dis1", text="अनावश्यक दस्तावेज 1"),
        "d_dis2": CorpusDoc(doc_id="d_dis2", text="अनावश्यक दस्तावेज 2"),
        "d_dis3": CorpusDoc(doc_id="d_dis3", text="अनावश्यक दस्तावेज 3"),
    }
    qrels = {"q1": {"d_pos1": 1}, "q2": {"d_pos2": 1}}

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        native = EvaluationManifest(
            name="miracl_hi_dev_native",
            language="hi",
            split="dev",
            queries=queries,
            corpus=corpus,
            qrels=qrels,
        )
        native.save(base_path / "miracl_hi_dev_native")

        roman = EvaluationManifest(
            name="miracl_hi_dev_romanized",
            language="hi",
            split="dev",
            queries={
                "q1_roman": Query(
                    query_id="q1_roman",
                    text="bharat",
                    language="hi",
                    script="romanized",
                )
            },
            corpus=corpus,
            qrels={"q1_roman": {"d_pos1": 1}},
        )
        roman.save(base_path / "miracl_hi_dev_romanized")

        fast_native_dir, fast_roman_dir = build_fast_miracl_manifests(
            eval_base_dir=base_path,
            target_corpus_size=3,
            seed=42,
            split="dev",
        )

        loaded_native = EvaluationManifest.load(fast_native_dir)
        assert len(loaded_native.corpus) == 3
        # Must contain all positive docs
        assert "d_pos1" in loaded_native.corpus
        assert "d_pos2" in loaded_native.corpus
        # Must contain exactly 1 distractor
        distractor_count = sum(
            1 for did in ["d_dis1", "d_dis2", "d_dis3"] if did in loaded_native.corpus
        )
        assert distractor_count == 1
