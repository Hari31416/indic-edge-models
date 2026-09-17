"""Unit tests for parameter-isolated LoRA training components."""

import json
from pathlib import Path
import tempfile
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from indic_edge_embeddings.eval.loader import CorpusDoc, EvaluationManifest, Query
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator
from indic_edge_embeddings.train.dataset import RetrievalCollator, RetrievalDataset
from indic_edge_embeddings.train.lora_trainer import (
    LoraRetrievalTrainer,
    LoraTrainingConfig,
)
from indic_edge_embeddings.train.run_lora_campaign import (
    evaluate_asymmetric_checkpoint,
    evaluate_symmetric_checkpoint,
)


def test_lora_config_defaults() -> None:
    """Verify LoraTrainingConfig default hyperparameters."""
    config = LoraTrainingConfig()
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert "query" in config.lora_target_modules
    assert "dense" in config.lora_target_modules
    assert config.learning_rate == 1e-4
    assert config.architecture_mode == "lora_symmetric"


def test_lora_parameter_counts() -> None:
    """Verify that LoRA only trains ~1.13% of parameters on multilingual-e5-small."""
    config = LoraTrainingConfig(
        model_name_or_path="intfloat/multilingual-e5-small",
        lora_r=16,
        lora_alpha=32,
        lora_target_modules=["query", "key", "value", "dense"],
    )

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps({"query_id": "q1", "query": "भारत", "positive": "भारत देश"})
            + "\n"
        )
        temp_data = f.name

    dataset = RetrievalDataset(temp_data)
    trainer = LoraRetrievalTrainer(config=config, train_dataset=dataset)

    trainable_params, all_params = trainer._count_parameters()
    trainable_pct = 100 * trainable_params / all_params

    # Expected: ~1.34M trainable out of ~119M total (approx 1.1% to 1.2%)
    assert trainable_params < 2_000_000
    assert 1.0 <= trainable_pct <= 1.3
    Path(temp_data).unlink()


def test_lora_symmetric_forward() -> None:
    """Verify forward pass and loss computation for symmetric LoRA."""
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    collator = RetrievalCollator(
        tokenizer=tokenizer,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=32,
    )

    batch_data = [
        {"query": "भारत की राजधानी", "positive": "नई दिल्ली राजधानी है"},
        {"query": "मौसम", "positive": "आज धूप है"},
    ]
    batch = collator(batch_data)

    config = LoraTrainingConfig(
        model_name_or_path="intfloat/multilingual-e5-small",
        architecture_mode="lora_symmetric",
        lora_r=8,
        lora_alpha=16,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(batch_data[0]) + "\n")
        temp_data = f.name

    dataset = RetrievalDataset(temp_data)
    trainer = LoraRetrievalTrainer(config=config, train_dataset=dataset)

    q_emb = trainer._encode_batch(batch["queries"])
    p_emb = trainer._encode_batch(batch["passages"])
    loss = trainer.loss_fn(q_emb, p_emb)

    assert q_emb.shape == (2, 384)
    assert p_emb.shape == (2, 384)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    Path(temp_data).unlink()


def test_lora_query_asymmetric_forward() -> None:
    """Verify forward pass and adapter disabling for query-asymmetric LoRA."""
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    collator = RetrievalCollator(
        tokenizer=tokenizer,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=32,
    )

    batch_data = [
        {"query": "bharat ki rajdhani", "positive": "नई दिल्ली राजधानी है"},
        {"query": "mausam", "positive": "आज धूप है"},
    ]
    batch = collator(batch_data)

    config = LoraTrainingConfig(
        model_name_or_path="intfloat/multilingual-e5-small",
        architecture_mode="lora_query_asymmetric",
        lora_r=8,
        lora_alpha=16,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(batch_data[0]) + "\n")
        temp_data = f.name

    dataset = RetrievalDataset(temp_data)
    trainer = LoraRetrievalTrainer(config=config, train_dataset=dataset)

    q_emb = trainer._encode_batch(batch["queries"])
    with trainer.model.disable_adapter():
        p_emb = trainer._encode_batch(batch["passages"])

    loss = trainer.loss_fn(q_emb, p_emb)

    assert q_emb.shape == (2, 384)
    assert p_emb.shape == (2, 384)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    Path(temp_data).unlink()


def test_lora_save_and_merge() -> None:
    """Verify saving adapter and merged model weights."""
    config = LoraTrainingConfig(
        model_name_or_path="intfloat/multilingual-e5-small",
        lora_r=8,
        lora_alpha=16,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps({"query_id": "q1", "query": "भारत", "positive": "भारत देश"})
            + "\n"
        )
        temp_data = f.name

    dataset = RetrievalDataset(temp_data)
    trainer = LoraRetrievalTrainer(config=config, train_dataset=dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir) / "test_model"
        trainer.save_model(target_dir)

        # Check adapter folder
        assert (target_dir / "adapter" / "adapter_config.json").exists()
        # Check merged model weights
        assert (target_dir / "config.json").exists()
        assert (target_dir / "model.safetensors").exists() or (
            target_dir / "pytorch_model.bin"
        ).exists()

        # Confirm AutoModel can load merged model
        loaded = AutoModel.from_pretrained(target_dir)
        assert loaded is not None
    Path(temp_data).unlink()


def test_lora_smoke_training_cpu() -> None:
    """Verify end-to-end LoRA training execution on CPU."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i in range(4):
            f.write(
                json.dumps(
                    {
                        "query_id": f"q_{i}",
                        "query": f"query sample {i}",
                        "positive": f"passage sample {i}",
                    }
                )
                + "\n"
            )
        train_file = f.name

    with tempfile.TemporaryDirectory() as out_dir:
        config = LoraTrainingConfig(
            model_name_or_path="intfloat/multilingual-e5-small",
            output_dir=out_dir,
            max_length=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            num_epochs=1,
            lora_r=8,
            lora_alpha=16,
            fp16=False,
            logging_steps=1,
            eval_steps=1,
            checkpoints_per_epoch=1,
        )

        dataset = RetrievalDataset(train_file)
        trainer = LoraRetrievalTrainer(
            config=config,
            train_dataset=dataset,
            eval_dataset=dataset,
        )

        summary = trainer.train()
        assert summary["global_steps"] == 2
        assert "best_eval_loss" in summary
        assert (Path(out_dir) / "final_model").exists()
        assert (Path(out_dir) / "training_report.md").exists()
        assert (Path(out_dir) / "training_summary.json").exists()

    Path(train_file).unlink()


def test_lora_evaluators_smoke() -> None:
    """Verify symmetric and asymmetric evaluation functions with mock manifest."""
    manifest = EvaluationManifest(
        name="test_manifest",
        language="hi",
        split="test",
        queries={
            "q1": Query(query_id="q1", text="भारत की राजधानी", language="hi"),
            "q2": Query(query_id="q2", text="दिल्ली", language="hi"),
        },
        corpus={
            "d1": CorpusDoc(doc_id="d1", text="नई दिल्ली भारत की राजधानी है।"),
            "d2": CorpusDoc(doc_id="d2", text="मुंबई महाराष्ट्र में है।"),
        },
        qrels={"q1": {"d1": 1}, "q2": {"d1": 1}},
    )

    base_evaluator = DenseRetrievalEvaluator(
        model_name_or_path="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=2,
        max_length=32,
        fp16=False,
    )

    manifests = {"hi": manifest}

    # 1. Symmetric eval
    sym_res = evaluate_symmetric_checkpoint(
        checkpoint_path="intfloat/multilingual-e5-small",
        manifests=manifests,
        cache_dir=None,
        batch_size=2,
        fp16=False,
    )
    assert "hi" in sym_res
    assert "ndcg@10" in sym_res["hi"]

    # 2. Asymmetric eval
    doc_ids = list(manifest.corpus.keys())
    doc_texts = [
        f"{manifest.corpus[did].title} {manifest.corpus[did].text}".strip()
        for did in doc_ids
    ]
    base_corpus_embeddings = {
        "hi": base_evaluator.encode(
            doc_texts,
            prefix=base_evaluator.passage_prefix,
            desc="Corpus",
            sort_by_length=False,
        )
    }
    asym_res = evaluate_asymmetric_checkpoint(
        query_checkpoint_path="intfloat/multilingual-e5-small",
        base_corpus_embeddings=base_corpus_embeddings,
        manifests=manifests,
        batch_size=2,
        fp16=False,
    )
    assert "hi" in asym_res
    assert "ndcg@10" in asym_res["hi"]
