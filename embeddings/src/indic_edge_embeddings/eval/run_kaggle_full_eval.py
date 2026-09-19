"""Orchestrator script for remote Kaggle full evaluation campaign.

Executes all setup steps (dataset downloading, LoRA merging, ONNX INT8 export,
vocabulary pruning) and runs the 4-variant benchmark on full 506k document corpora.
"""

import logging
import os
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kaggle_full_eval")


def run_cmd(cmd: str, desc: str) -> None:
    """Execute shell command with timing and error handling."""
    logger.info(">>> Starting: %s", desc)
    logger.info("Command: %s", cmd)
    start_t = time.time()
    res = subprocess.run(cmd, shell=True, check=True)
    logger.info(">>> Completed: %s in %.1f seconds", desc, time.time() - start_t)


def main() -> None:
    """Run end-to-end full evaluation campaign."""
    eval_tmp = Path("/tmp/eval")
    models_tmp = Path("/tmp/models")
    cache_tmp = Path("/tmp/cache")
    output_dir = (
        Path("/kaggle/working/outputs/full_eval_campaign")
        if Path("/kaggle").exists()
        else Path("outputs/full_eval_campaign")
    )

    eval_tmp.mkdir(parents=True, exist_ok=True)
    models_tmp.mkdir(parents=True, exist_ok=True)
    cache_tmp.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate dataset if mounted on Kaggle
    dataset_dir = None
    for candidate in [
        Path("/kaggle/input/indic-edge-e5-models"),
        Path("/kaggle/input/indic_edge_e5_models"),
    ]:
        if candidate.exists():
            dataset_dir = candidate
            break

    if dataset_dir is None and Path("/kaggle/input").exists():
        matches = [
            m
            for m in Path("/kaggle/input").glob("**/indic-edge-e5-models")
            if m.is_dir()
        ]
        if matches:
            dataset_dir = matches[0]

    if dataset_dir:
        logger.info("Found mounted Kaggle dataset at %s", str(dataset_dir))
        lora_fp32_dir = dataset_dir / "lora_fp32" if (dataset_dir / "lora_fp32").exists() else None
        lora_int8_path = dataset_dir / "lora_int8" / "model_int8.onnx"
        lora_int8_pruned_path = dataset_dir / "lora_int8_pruned" / "model_int8.onnx"
        adapter_path = dataset_dir / "lora_adapter"
    else:
        logger.info(
            "No mounted dataset found; using local paths and in-command export..."
        )
        lora_fp32_dir = None
        adapter_path = Path(
            "experiments/2026-09-18-e5-lora-scaled/outputs/campaign_lora_scaled/01_lora_symmetric/final_model/adapter"
        )
        lora_int8_path = None
        lora_int8_pruned_path = None

    # Step 1: Build Full Evaluation Manifests
    run_cmd(
        f"{sys.executable} -m indic_edge_embeddings.data.build_eval_data --output-dir /tmp/eval --split dev",
        "Build full MIRACL Hindi dev evaluation manifests (Native & Romanized)",
    )
    run_cmd(
        f"{sys.executable} -m indic_edge_embeddings.eval.build_english_manifest --output-dir /tmp/eval --split test",
        "Build SciFact English test evaluation manifest",
    )

    # Step 2: Prepare LoRA FP32 Model
    if lora_fp32_dir is None or not lora_fp32_dir.exists():
        lora_fp32_dir = models_tmp / "lora_fp32"
        logger.info("Merging LoRA adapter into base model for FP32 PyTorch evaluation...")
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        base = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-small", trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        peft_m = PeftModel.from_pretrained(base, str(adapter_path))
        merged = peft_m.merge_and_unload()
        merged.save_pretrained(str(lora_fp32_dir))
        tok.save_pretrained(str(lora_fp32_dir))
        logger.info("LoRA FP32 model saved to %s", str(lora_fp32_dir))
    else:
        logger.info("Using pre-merged LoRA FP32 model from %s", str(lora_fp32_dir))

    # Step 3: Export ONNX models if not provided in dataset
    if lora_int8_path is None or not lora_int8_path.exists():
        lora_int8_dir = models_tmp / "lora_int8"
        run_cmd(
            f"{sys.executable} -m indic_edge_embeddings.export.export_cli --model-name {lora_fp32_dir} --output-dir {lora_int8_dir} --no-verify",
            "Export full-vocab LoRA model to ONNX dynamic INT8",
        )
        lora_int8_path = lora_int8_dir / "model_int8.onnx"

    if lora_int8_pruned_path is None or not lora_int8_pruned_path.exists():
        lora_pruned_dir = models_tmp / "lora_pruned"
        lora_int8_pruned_dir = models_tmp / "lora_int8_pruned"
        run_cmd(
            f"{sys.executable} -m indic_edge_embeddings.export.vocab_pruner --model-name {lora_fp32_dir} --output-dir {lora_pruned_dir}",
            "Prune vocabulary to Indic Devanagari & English (90.7k tokens)",
        )
        run_cmd(
            f"{sys.executable} -m indic_edge_embeddings.export.export_cli --model-name {lora_pruned_dir} --output-dir {lora_int8_pruned_dir} --no-verify",
            "Export pruned LoRA model to ONNX dynamic INT8 (~55 MB)",
        )
        lora_int8_pruned_path = lora_int8_pruned_dir / "model_int8.onnx"

    # Step 5: Execute Full 4-Model Evaluation Campaign
    run_cmd(
        f"{sys.executable} -m indic_edge_embeddings.eval.run_full_eval_campaign "
        f"--native-eval-dir /tmp/eval/miracl_hi_dev_native "
        f"--romanized-eval-dir /tmp/eval/miracl_hi_dev_romanized "
        f"--english-eval-dir /tmp/eval/scifact_en_test "
        f"--lora-fp32-path {lora_fp32_dir} "
        f"--lora-int8-path {lora_int8_dir}/model_int8.onnx "
        f"--lora-int8-pruned-path {lora_int8_pruned_dir}/model_int8.onnx "
        f"--output-dir {output_dir} "
        f"--cache-dir {cache_tmp} "
        f"--batch-size 512",
        "Execute 4-Model Evaluation Campaign across full datasets (506k corpus)",
    )

    logger.info(
        "All operations completed successfully! Outputs available in %s",
        str(output_dir),
    )


if __name__ == "__main__":
    main()
