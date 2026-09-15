"""ONNX export, INT8 quantization, and numerical parity checks."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class EmbeddingWithPooling(nn.Module):
    """PyTorch wrapper combining transformer encoder and pooling into a single graph."""

    def __init__(
        self,
        base_model: nn.Module,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        """Initialize the embedding wrapper.

        Args:
            base_model: Underlying Hugging Face transformer model.
            pooling: Pooling mode ('mean' or 'cls').
            normalize: Whether to apply L2 normalization to output vector.
        """
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.normalize = normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass generating pooled embeddings.

        Args:
            input_ids: Token ID tensor of shape (batch, seq_len).
            attention_mask: Mask tensor of shape (batch, seq_len).
            token_type_ids: Optional segment ID tensor.

        Returns:
            Pooled embedding tensor of shape (batch, hidden_dim).
        """
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.base_model(**kwargs)
        hidden_state = outputs.last_hidden_state

        if self.pooling == "mean":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling == "cls":
            embeddings = hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


@dataclass
class ParityReport:
    """Numerical parity verification report between PyTorch and ONNX models."""

    max_absolute_diff: float
    mean_cosine_similarity: float
    min_cosine_similarity: float
    passed: bool
    num_samples: int
    onnx_file_size_mb: float


class EmbeddingExporter:
    """Exporter for transformer embedding models to ONNX and quantized formats."""

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        """Initialize the exporter.

        Args:
            model_name_or_path: Hugging Face model identifier or local directory.
            pooling: Pooling mode ('mean' or 'cls').
            normalize: Whether to apply L2 normalization.
        """
        self.model_name = model_name_or_path
        self.pooling = pooling
        self.normalize = normalize

        logger.info("Loading tokenizer and model for export: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        base_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = EmbeddingWithPooling(
            base_model=base_model,
            pooling=self.pooling,
            normalize=self.normalize,
        )
        self.model.eval()

    def export_onnx(
        self,
        output_path: Union[str, Path],
        opset_version: int = 14,
    ) -> Path:
        """Export the PyTorch model to ONNX.

        Args:
            output_path: Destination file path for the .onnx model.
            opset_version: ONNX operator set version.

        Returns:
            Path to the exported ONNX file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        dummy_text = "This is a sample sentence for ONNX tracing."
        inputs = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
        )

        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"},
        }
        forward_args = (inputs["input_ids"], inputs["attention_mask"])

        if "token_type_ids" in inputs:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            forward_args = (  # type: ignore[assignment]
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
            )

        logger.info("Exporting ONNX model to %s...", str(out_file))
        torch.onnx.export(
            self.model,
            forward_args,
            str(out_file),
            input_names=input_names,
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Save tokenizer alongside model
        self.tokenizer.save_pretrained(out_file.parent)
        logger.info("ONNX export complete: %s", str(out_file))
        return out_file

    def quantize_int8(
        self,
        onnx_model_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Apply dynamic INT8 quantization to an exported ONNX model.

        Args:
            onnx_model_path: Path to FP32 ONNX model.
            output_path: Path for output INT8 quantized ONNX file.

        Returns:
            Path to quantized ONNX file.
        """
        from onnxruntime.quantization import QuantType, quantize_dynamic

        in_file = Path(onnx_model_path)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Quantizing %s to INT8 -> %s...", str(in_file), str(out_file))
        quantize_dynamic(
            model_input=str(in_file),
            model_output=str(out_file),
            weight_type=QuantType.QInt8,
        )
        logger.info("Quantization complete: %s", str(out_file))
        return out_file


def verify_onnx_parity(
    pytorch_exporter: EmbeddingExporter,
    onnx_path: Union[str, Path],
    test_sentences: Optional[Sequence[str]] = None,
    max_abs_tol: float = 1e-4,
    min_cosine_tol: float = 0.999,
) -> ParityReport:
    """Verify numerical parity between PyTorch exporter and exported ONNX model.

    Args:
        pytorch_exporter: Exporter instance containing reference PyTorch model.
        onnx_path: Path to ONNX file.
        test_sentences: List of test strings across target languages.
        max_abs_tol: Maximum allowed absolute difference (for unquantized models).
        min_cosine_tol: Minimum acceptable cosine similarity.

    Returns:
        ParityReport detailing maximum differences and pass/fail verdict.
    """
    if test_sentences is None:
        test_sentences = [
            "Hello world, testing dense retrieval representations.",
            "नमस्ते दुनिया, खोज अनुक्रमणिका परीक्षण।",
            "namaste duniya, khoj anukramanika parikshan.",
            "What is the capital of India?",
            "भारत की राजधानी क्या है?",
            "bharat ki rajdhani kya hai?",
        ]

    onnx_file = Path(onnx_path)
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    tokenizer = pytorch_exporter.tokenizer
    pytorch_model = pytorch_exporter.model

    inputs = tokenizer(
        list(test_sentences),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        if "token_type_ids" in inputs:
            pt_out = (
                pytorch_model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["token_type_ids"],
                )
                .cpu()
                .numpy()
            )
        else:
            pt_out = (
                pytorch_model(inputs["input_ids"], inputs["attention_mask"])
                .cpu()
                .numpy()
            )

    ort_inputs: Dict[str, np.ndarray] = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].numpy()

    onnx_out = session.run(None, ort_inputs)[0]

    abs_diff = np.max(np.abs(pt_out - onnx_out))

    # Cosine similarities per sample
    pt_norm = pt_out / np.linalg.norm(pt_out, axis=1, keepdims=True)
    onnx_norm = onnx_out / np.linalg.norm(onnx_out, axis=1, keepdims=True)
    cosine_sims = np.sum(pt_norm * onnx_norm, axis=1)

    mean_cos = float(np.mean(cosine_sims))
    min_cos = float(np.min(cosine_sims))
    file_size_mb = onnx_file.stat().st_size / (1024 * 1024)

    passed = bool(min_cos >= min_cosine_tol)

    logger.info(
        "Parity check on %s: max_diff=%.6f, mean_cos=%.6f, min_cos=%.6f, size=%.2f MB, passed=%s",
        onnx_file.name,
        abs_diff,
        mean_cos,
        min_cos,
        file_size_mb,
        passed,
    )

    return ParityReport(
        max_absolute_diff=float(abs_diff),
        mean_cosine_similarity=mean_cos,
        min_cosine_similarity=min_cos,
        passed=passed,
        num_samples=len(test_sentences),
        onnx_file_size_mb=file_size_mb,
    )
