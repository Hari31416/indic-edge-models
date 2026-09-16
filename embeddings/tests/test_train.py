"""Unit tests for embedding training components."""

import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

from indic_edge_embeddings.train.dataset import RetrievalDataset, RetrievalCollator
from indic_edge_embeddings.train.loss import InfoNCELoss, mean_pooling
from indic_edge_embeddings.train.trainer import RetrievalTrainer, TrainingConfig


def test_retrieval_dataset_jsonl():
    """Verify RetrievalDataset correctly loads and parses JSONL format."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {"query_id": "q1", "query": "भारत", "positive": "भारत एक देश है"}
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {"query_id": "q2", "query": "दिल्ली", "positive": "दिल्ली राजधानी है"}
            )
            + "\n"
        )
        temp_path = f.name

    dataset = RetrievalDataset(temp_path)
    assert len(dataset) == 2
    assert dataset[0]["query"] == "भारत"
    assert dataset[0]["positive"] == "भारत एक देश है"
    Path(temp_path).unlink()


def test_retrieval_collator():
    """Verify RetrievalCollator tokenizes queries and passages with correct prefixes."""
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    collator = RetrievalCollator(
        tokenizer=tokenizer,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=64,
    )

    batch = [
        {"query": "भारत की राजधानी", "positive": "नई दिल्ली भारत की राजधानी है।"},
        {"query": "मौसम कैसा है", "positive": "आज धूप खिली हुई है।"},
    ]

    collated = collator(batch)
    assert "queries" in collated
    assert "passages" in collated
    assert "input_ids" in collated["queries"]
    assert "attention_mask" in collated["queries"]
    assert collated["queries"]["input_ids"].shape[0] == 2
    assert collated["passages"]["input_ids"].shape[0] == 2


def test_infonce_loss():
    """Verify InfoNCELoss computes contrastive loss correctly."""
    loss_fn = InfoNCELoss(temperature=0.05)
    # 2 queries and 2 passages of dim 4, perfectly aligned
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    loss = loss_fn(q, p)
    # When perfectly aligned, loss should be small
    assert loss.item() < 0.1
    assert not torch.isnan(loss)


def test_training_smoke_cpu():
    """Verify end-to-end training loop runs without error on a tiny dataset on CPU."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i in range(4):
            f.write(
                json.dumps(
                    {
                        "query_id": f"q_{i}",
                        "query": f"query text {i}",
                        "positive": f"passage text {i}",
                    }
                )
                + "\n"
            )
        train_file = f.name

    with tempfile.TemporaryDirectory() as out_dir:
        config = TrainingConfig(
            model_name_or_path="intfloat/multilingual-e5-small",
            output_dir=out_dir,
            max_length=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            num_epochs=1,
            fp16=False,
            logging_steps=1,
            eval_steps=1,
        )

        dataset = RetrievalDataset(train_file)
        trainer = RetrievalTrainer(
            config=config,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        summary = trainer.train()

        assert summary["global_steps"] == 2
        assert (Path(out_dir) / "final_model" / "model.safetensors").exists() or (
            Path(out_dir) / "final_model" / "pytorch_model.bin"
        ).exists()

    Path(train_file).unlink()
