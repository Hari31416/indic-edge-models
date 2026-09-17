"""Offline tests for supervision, leakage protection, and retrieval integration."""

import copy
import json
from pathlib import Path

import pytest
import torch

from indic_edge_embeddings.data.build_quality_data import (
    adapt,
    build,
    load_lexicon,
    romanize,
)
from indic_edge_embeddings.train.dataset import (
    RetrievalDataset,
    RetrievalCollator,
    PairedRetrievalCollator,
    retrieval_candidates,
)
from indic_edge_embeddings.train.loss import InfoNCELoss, DualObjectiveLoss


def raw(i):
    return {
        "query_id": i,
        "query": f"भारत का नगर {i}",
        "Eng_Query": f"Indian city {i}",
        "Answer": f"उत्तर {i}",
        "Eng_Answer": f"Answer {i}",
        "passages": {
            "Translated_passages": [
                f"प्रासंगिक नगर {i}",
                f"दूसरा नगर {i}",
                "अनिर्णीत पाठ",
            ],
            "English_passages": [f"Relevant city {i}", f"Other city {i}", "unjudged"],
            "is_selected": [1, 1, 0],
        },
    }


def spec():
    return {
        "name": "hi",
        "family": "msmarco",
        "format": "msmarco-xi",
        "language": "hi",
        "split": "train",
        "max_queries": 20,
        "romanized_probability": 1,
        "files": [{"path": "source.jsonl"}],
    }


def setup(tmp_path):
    rows = [raw(i) for i in range(40)]
    (tmp_path / "source.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {
        "seed": 42,
        "matched_answers": True,
        "validation_fraction": 0.2,
        "inspection_samples": 4,
        "sources": [spec()],
    }


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_selected_passages_and_english():
    row = adapt(raw(1), spec())
    assert row["positives"] == ["प्रासंगिक नगर 1", "दूसरा नगर 1"]
    assert row["negatives"] == []
    assert row["answer"] not in row["positives"]
    en = adapt(raw(1), dict(spec(), language="en"))
    assert en["query"] == "Indian city 1"
    assert en["source_identity"] == row["source_identity"]
    bad = raw(1)
    bad["passages"]["is_selected"] = [1]
    with pytest.raises(ValueError, match="selection arrays"):
        adapt(bad, spec())


def test_miracl_negatives_and_rag_paragraph():
    p = lambda value: {"title": "", "text": value}
    row = adapt(
        {
            "query_id": "1",
            "query": "प्रश्न",
            "positive_passages": [p("सही")],
            "negative_passages": [p("गलत"), p("सही")],
        },
        {"name": "miracl", "format": "miracl"},
    )
    assert row["negatives"] == ["गलत"]
    rag = adapt(
        {
            "wiki_id": 1,
            "question": "प्रश्न",
            "answer": "उत्तर",
            "paragraph": "मूल लेख",
            "text": "DO NOT TRAIN ON GENERATED ANSWER",
        },
        {"name": "rag", "format": "indic-rag"},
    )
    assert rag["positives"] == ["मूल लेख"]
    assert rag["document_group"] == "wikipedia:hi:1"


def test_matched_reproducible_build_and_load(tmp_path):
    cfg = setup(tmp_path)
    report = build(cfg, tmp_path / "v1", tmp_path)
    cfg2 = copy.deepcopy(cfg)
    cfg2["inspection_samples"] = 1
    build(cfg2, tmp_path / "v2", tmp_path)
    sets = {}
    for split in ["train", "validation"]:
        path = tmp_path / "v1" / "passages" / f"{split}.jsonl"
        passage = read(path)
        answer = read(tmp_path / "v1" / "answers" / f"{split}.jsonl")
        assert [(r["query_id"], r["query"]) for r in passage] == [
            (r["query_id"], r["query"]) for r in answer
        ]
        assert all(not r["negatives"] for r in passage + answer)
        assert (
            path.read_bytes()
            == (tmp_path / "v2" / "passages" / f"{split}.jsonl").read_bytes()
        )
        ds = RetrievalDataset(path.with_suffix(".parquet"))
        assert len(ds) == len(passage)
        assert all(r["native_query"] for r in ds)
        assert all(len(r["positive_texts"]) == 2 for r in ds)
        sets[split] = passage
    for field in ["source_identity", "query", "split_group_id", "positive"]:
        assert {r[field] for r in sets["train"]}.isdisjoint(
            {r[field] for r in sets["validation"]}
        )
    selected_ids = [
        int(r["source_identity"].split(":")[1])
        for r in sets["train"] + sets["validation"]
    ]
    assert max(selected_ids) >= 20  # Whole-file sampling, not a truncated prefix.
    assert report["sources"]["hi"]["scanned"] == 40
    assert report["inputs"][0]["sha256"]
    with pytest.raises(FileExistsError):
        build(cfg, tmp_path / "v1", tmp_path)


def test_connected_splits_and_eval_exclusion(tmp_path):
    cfg = setup(tmp_path)
    cfg["sources"][0]["max_queries"] = 100
    en = dict(spec(), name="en", language="en", max_queries=100)
    cfg["sources"].append(en)
    (tmp_path / "exclude.jsonl").write_text(json.dumps({"query": raw(2)["query"]}))
    cfg["exclude"] = [{"path": "exclude.jsonl", "kind": "queries"}]
    build(cfg, tmp_path / "out", tmp_path)
    a = read(tmp_path / "out/passages/train.jsonl")
    b = read(tmp_path / "out/passages/validation.jsonl")
    assert all(r["source_identity"] != "msmarco:2" for r in a + b)
    assert {r["source_identity"] for r in a}.isdisjoint(
        {r["source_identity"] for r in b}
    )


def test_lexicon_preserves_english_and_numbers(tmp_path):
    path = tmp_path / "train.tsv"
    path.write_text("भारत\tbharat\t3\nभारत\tbhaarat\t1\n")
    import random

    roman = romanize("भारत API 123", load_lexicon(path), random.Random(42))
    assert roman in {"bharat API 123", "bhaarat API 123"}
    cfg = setup(tmp_path)
    cfg["dakshina_train"] = {"path": str(path), "split": "test"}
    with pytest.raises(ValueError, match="train lexicon"):
        build(cfg, tmp_path / "bad", tmp_path)


class Tokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.arange(len(texts)).reshape(-1, 1),
            "attention_mask": torch.ones(len(texts), 1, dtype=torch.long),
        }


def test_negatives_and_multi_positive_loss():
    batch = [
        {
            "query": "q",
            "positive": "p",
            "positive_texts": ["p", "also p"],
            "negatives": ["n", "p"],
        },
        {
            "query": "q roman",
            "native_query": "q",
            "positive": "also p",
            "negatives": ["n"],
        },
    ]
    for cls in [RetrievalCollator, PairedRetrievalCollator]:
        data = cls(Tokenizer())(batch)
        assert data["passages"]["input_ids"].shape[0] == 3
        assert data["positive_mask"].tolist() == [
            [True, True, False],
            [True, True, False],
        ]
        q = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        p = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        loss = InfoNCELoss()(q, p, data["positive_mask"], data["valid_mask"])
        assert loss.item() == pytest.approx(
            float(torch.log(torch.tensor(1.5))), abs=1e-5
        )
        loss.backward()
        assert p.grad[2].abs().sum() > 0  # Explicit negative contributes a gradient.
        dual, _ = DualObjectiveLoss(
            weight_script=0, weight_preserve=0, weight_retrieval=1
        )(
            torch.empty(0, 2),
            torch.empty(0, 2),
            [],
            q,
            p,
            data["positive_mask"],
            data["valid_mask"],
        )
        assert dual.item() == pytest.approx(loss.item())


def test_unknown_related_passages_masked():
    _, positives, valid = retrieval_candidates(
        [
            {"query": "q1", "positive": "p1", "split_group_id": "g"},
            {"query": "q2", "positive": "p2", "split_group_id": "g"},
        ]
    )
    assert positives.tolist() == [[True, False], [False, True]]
    assert valid.tolist() == positives.tolist()


@pytest.mark.parametrize("loss_type", ["infonce", "dual"])
def test_training_consumes_candidates_offline(tmp_path, loss_type):
    from tokenizers import Tokenizer as Backend
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from indic_edge_embeddings.train.trainer import RetrievalTrainer, TrainingConfig

    backend = Backend(
        WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "query": 2,
                "passage": 3,
                "native": 4,
                "roman": 5,
                "positive": 6,
                "negative": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    model = BertModel(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    teacher = tmp_path / "teacher"
    model.save_pretrained(teacher)
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "query_id": "q",
                    "source_query_id": "q",
                    "query": "native query",
                    "script": "native",
                    "positive": "positive passage",
                    "negatives": ["negative passage"],
                },
                {
                    "query_id": "qr",
                    "source_query_id": "q",
                    "query": "roman query",
                    "script": "romanized",
                    "positive": "positive passage",
                    "negatives": ["negative passage"],
                },
            ]
        )
    )
    dataset = RetrievalDataset(path)
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "out"),
        batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=1,
        fp16=False,
        loss_type=loss_type,
        teacher_model_name_or_path=str(teacher),
        logging_steps=1,
        eval_steps=1,
        checkpoints_per_epoch=0,
    )
    trainer = RetrievalTrainer(cfg, dataset, dataset, model=model, tokenizer=tokenizer)
    result = trainer.train()
    assert result["global_steps"] == 1
    assert torch.isfinite(torch.tensor(result["best_eval_loss"]))


def test_group_transitivity_and_roman_collisions():
    from indic_edge_embeddings.data.build_quality_data import group_records

    rows = [adapt(raw(i), spec()) for i in range(4)]
    rows[1]["positives"].append(rows[0]["positives"][0])
    rows[2]["english_query"] = rows[1]["english_query"]
    groups = group_records(rows)
    assert sorted(map(len, groups)) == [1, 3]
    rows[3]["variants"] = [("romanized", "same roman spelling")]
    rows[0]["variants"] = [("romanized", "same roman spelling")]
    assert len(group_records(rows)) == 1


def test_unreviewed_negatives_rejected_and_duplicate_records_skipped(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        adapt(
            {"query_id": "x", "query": "q", "positives": ["p"], "negatives": ["n"]},
            {"name": "reviewed", "format": "normalized"},
        )
    cfg = setup(tmp_path)
    with (tmp_path / "source.jsonl").open("a") as f:
        f.write("\n" + json.dumps(raw(0)))
    report = build(cfg, tmp_path / "out", tmp_path)
    assert report["sources"]["hi"]["duplicate_rows"] == 1


def test_passage_exclusions_and_floating_revision_rejected(tmp_path):
    from indic_edge_embeddings.data.build_quality_data import resolve_input

    with pytest.raises(ValueError, match="immutable"):
        resolve_input(
            {
                "repo_id": "example/data",
                "filename": "train.parquet",
                "revision": "main",
            },
            tmp_path,
        )
    cfg = setup(tmp_path)
    cfg["sources"][0]["max_queries"] = 100
    (tmp_path / "exclude.jsonl").write_text(json.dumps({"text": "प्रासंगिक नगर 2"}))
    cfg["exclude"] = [{"path": "exclude.jsonl", "kind": "passages"}]
    build(cfg, tmp_path / "out", tmp_path)
    rows = read(tmp_path / "out/passages/train.jsonl") + read(
        tmp_path / "out/passages/validation.jsonl"
    )
    assert all(r["source_identity"] != "msmarco:2" for r in rows)
