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


def test_script_alignment_loss():
    """Verify ScriptAlignmentLoss computes cosine distance between student and teacher."""
    from indic_edge_embeddings.train.loss import ScriptAlignmentLoss

    loss_fn = ScriptAlignmentLoss()
    # Identical vectors -> loss ~ 0
    s_roman = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    t_native = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    loss = loss_fn(s_roman, t_native)
    assert loss.item() < 1e-5

    # Orthogonal vectors -> loss ~ 1
    t_ortho = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    loss_ortho = loss_fn(s_roman, t_ortho)
    assert abs(loss_ortho.item() - 1.0) < 1e-5


def test_representation_preservation_loss():
    """Verify RepresentationPreservationLoss balances group means."""
    from indic_edge_embeddings.train.loss import RepresentationPreservationLoss

    loss_fn = RepresentationPreservationLoss()
    s1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # loss 0

    s2 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t2 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # loss 1

    loss = loss_fn([(s1, t1), (s2, t2)])
    # Average of 0.0 and 1.0 = 0.5
    assert abs(loss.item() - 0.5) < 1e-5


def test_dual_objective_loss():
    """Verify DualObjectiveLoss correctly scales and sums individual components."""
    from indic_edge_embeddings.train.loss import DualObjectiveLoss

    loss_fn = DualObjectiveLoss(
        weight_script=1.0,
        weight_preserve=2.0,
        weight_retrieval=0.5,
    )
    s_roman = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t_native = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # script loss = 1.0

    s_p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t_p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # preserve loss = 0.0

    total_loss, metrics = loss_fn(
        student_roman_emb=s_roman,
        teacher_native_emb=t_native,
        preservation_groups=[(s_p, t_p)],
        student_query_emb=s_roman,
        student_passage_emb=s_p,
    )

    assert "loss_script" in metrics
    assert "loss_preserve" in metrics
    assert "loss_retrieval" in metrics
    assert abs(metrics["loss_script"] - 1.0) < 1e-5
    assert abs(metrics["loss_preserve"] - 0.0) < 1e-5
    assert total_loss.item() > 0.0


def test_paired_retrieval_collator():
    """Verify PairedRetrievalCollator correctly produces student and teacher tokens."""
    from indic_edge_embeddings.train.dataset import PairedRetrievalCollator

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    collator = PairedRetrievalCollator(
        tokenizer=tokenizer,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=32,
        english_preservation_texts=["This is an English preservation text sample."],
    )

    batch = [
        {
            "query_id": "q1",
            "query": "bharat ki rajdhani",
            "native_query": "भारत की राजधानी",
            "positive": "नई दिल्ली",
            "script": "romanized",
        },
        {
            "query_id": "q2",
            "query": "दिल्ली का मौसम",
            "native_query": "दिल्ली का मौसम",
            "positive": "आज धूप है",
            "script": "native",
        },
    ]

    collated = collator(batch)
    assert "queries" in collated
    assert "teacher_queries" in collated
    assert "passages" in collated
    assert "roman_indices" in collated
    assert "native_indices" in collated
    assert "english_passages" in collated
    assert collated["roman_indices"].tolist() == [0]
    assert collated["native_indices"].tolist() == [1]


def test_intermediate_checkpoint_saving_and_resuming():
    """Verify RetrievalTrainer saves and resumes full state from checkpoint."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i in range(8):
            f.write(
                json.dumps(
                    {
                        "query_id": f"q_{i}",
                        "query": f"query text {i}",
                        "positive": f"passage text {i}",
                        "script": "native",
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
            checkpoints_per_epoch=2,
        )

        dataset = RetrievalDataset(train_file)
        trainer = RetrievalTrainer(config=config, train_dataset=dataset)
        summary = trainer.train()

        # Check that checkpoint directory exists
        ckpts = list(Path(out_dir).glob("checkpoint-*"))
        assert len(ckpts) >= 1
        first_ckpt = ckpts[0]
        assert (first_ckpt / "training_state.pt").exists()
        assert (first_ckpt / "training_config.json").exists()
        assert (first_ckpt / "config.json").exists()

        # Verify Hugging Face config.json was NOT overwritten by TrainingConfig
        with open(first_ckpt / "config.json", "r", encoding="utf-8") as f:
            hf_cfg = json.load(f)
        assert "hidden_size" in hf_cfg or "model_type" in hf_cfg
        assert "checkpoints_per_epoch" not in hf_cfg

        # Verify model can be loaded by AutoModel
        loaded_model = AutoModel.from_pretrained(first_ckpt)
        assert loaded_model is not None

        # Test resuming from the saved checkpoint
        resume_config = TrainingConfig(
            model_name_or_path="intfloat/multilingual-e5-small",
            output_dir=out_dir,
            resume_from_checkpoint=str(first_ckpt),
            max_length=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            num_epochs=1,
            fp16=False,
            logging_steps=1,
        )
        resume_trainer = RetrievalTrainer(config=resume_config, train_dataset=dataset)
        resume_summary = resume_trainer.train()
        assert resume_summary is not None

    Path(train_file).unlink()


def test_dataset_romanized_native_pairing_validation():
    """Verify RetrievalDataset matches native targets and raises ValueError when target is missing."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        # Romanized row whose source native query exists
        f.write(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "भारत",
                    "positive": "भारत देश",
                    "script": "native",
                    "source_query_id": "q1",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "query_id": "q1_roman",
                    "query": "bharat",
                    "positive": "भारत देश",
                    "script": "romanized",
                    "source_query_id": "q1",
                }
            )
            + "\n"
        )
        valid_file = f.name

    ds = RetrievalDataset(valid_file)
    assert len(ds) == 2
    assert ds[1]["query"] == "bharat"
    assert ds[1]["native_query"] == "भारत"
    Path(valid_file).unlink()

    # Test error when native query target is missing
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "query_id": "q2_roman",
                    "query": "dilli",
                    "positive": "दिल्ली",
                    "script": "romanized",
                    "source_query_id": "nonexistent_source_id",
                }
            )
            + "\n"
        )
        invalid_file = f.name

    with pytest.raises(ValueError, match="Native source query target"):
        RetrievalDataset(invalid_file)

    Path(invalid_file).unlink()
