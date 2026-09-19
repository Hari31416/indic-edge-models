"""Unit tests for ONNX export, INT8 quantization, and numerical parity checks."""

from pathlib import Path
import tempfile
from typing import List

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from indic_edge_embeddings.export.exporter import (
    EmbeddingExporter,
    EmbeddingWithPooling,
    verify_onnx_parity,
)


@pytest.fixture
def dummy_peft_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal PEFT adapter checkpoint directory for testing."""
    model_name = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
        lora_dropout=0.0,
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    ckpt_dir = tmp_path / "dummy_lora_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    return ckpt_dir


def test_embedding_wrapper_forward() -> None:
    """Test forward pass of PyTorch EmbeddingWithPooling wrapper."""
    model_name = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    wrapper = EmbeddingWithPooling(
        base_model=base_model, pooling="mean", normalize=True
    )
    wrapper.eval()

    inputs = tokenizer(
        ["Hello world", "नमस्ते दुनिया"],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        out = wrapper(inputs["input_ids"], inputs["attention_mask"])

    assert out.shape[0] == 2
    assert out.shape[1] == base_model.config.hidden_size
    # Check L2 normalization (unit length)
    norms = torch.norm(out, p=2, dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(2), atol=1e-5)


def test_export_and_quantize_pipeline(tmp_path: Path) -> None:
    """Test full ONNX export, INT8 dynamic quantization, and numerical parity check."""
    model_name = "hf-internal-testing/tiny-random-bert"
    exporter = EmbeddingExporter(
        model_name_or_path=model_name, pooling="mean", normalize=True
    )

    fp32_onnx = tmp_path / "model.onnx"
    exporter.export_onnx(fp32_onnx)
    assert fp32_onnx.exists()
    assert (tmp_path / "tokenizer_config.json").exists()

    int8_onnx = tmp_path / "model_int8.onnx"
    exporter.quantize_int8(fp32_onnx, int8_onnx)
    assert int8_onnx.exists()

    test_queries: List[str] = [
        "Machine learning dense retrieval",
        "नमस्ते दुनिया, यह परीक्षण वाक्य है।",
        "namaste duniya, testing romanized input.",
    ]

    # Verify FP32 parity
    fp32_report = verify_onnx_parity(
        exporter,
        fp32_onnx,
        test_sentences=test_queries,
        min_cosine_tol=0.999,
    )
    assert fp32_report.passed
    assert fp32_report.min_cosine_similarity >= 0.999

    # Verify INT8 parity
    int8_report = verify_onnx_parity(
        exporter,
        int8_onnx,
        test_sentences=test_queries,
        min_cosine_tol=0.95,
    )
    assert int8_report.passed
    assert int8_report.min_cosine_similarity >= 0.95


def test_peft_checkpoint_loading_and_export(
    dummy_peft_checkpoint: Path, tmp_path: Path
) -> None:
    """Test that EmbeddingExporter seamlessly loads and exports PEFT adapter directories."""
    exporter = EmbeddingExporter(
        model_name_or_path=str(dummy_peft_checkpoint),
        pooling="mean",
        normalize=True,
    )

    out_onnx = tmp_path / "peft_model.onnx"
    exporter.export_onnx(out_onnx)
    assert out_onnx.exists()

    report = verify_onnx_parity(
        exporter,
        out_onnx,
        test_sentences=["Testing PEFT adapter ONNX export."],
        min_cosine_tol=0.999,
    )
    assert report.passed
    assert report.min_cosine_similarity >= 0.999
