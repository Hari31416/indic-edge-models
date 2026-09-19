"""Unified campaign runner executing parameter-isolated LoRA adaptation experiments and 3-way evaluation."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from indic_edge_embeddings.eval.loader import EvaluationManifest, resolve_manifest_path
from indic_edge_embeddings.eval.metrics import compute_retrieval_metrics
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator
from indic_edge_embeddings.train.dataset import RetrievalDataset, resolve_data_path
from indic_edge_embeddings.train.lora_trainer import (
    LoraRetrievalTrainer,
    LoraTrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lora_campaign")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the LoRA adaptation campaign."""
    parser = argparse.ArgumentParser(
        description="Run E5 LoRA adaptation experiment campaign with 3-way evaluation."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training parquet/jsonl file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation parquet/jsonl file",
    )
    parser.add_argument(
        "--native-eval-dir",
        type=str,
        default="eval/miracl_hi_dev_fast_native",
        help="Path to native Hindi evaluation manifest",
    )
    parser.add_argument(
        "--romanized-eval-dir",
        type=str,
        default="eval/miracl_hi_dev_fast_romanized",
        help="Path to Romanized Hindi evaluation manifest",
    )
    parser.add_argument(
        "--english-eval-dir",
        type=str,
        default="eval/scifact_en_test",
        help="Path to English evaluation manifest (e.g. SciFact)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/outputs/campaign_lora",
        help="Directory to store all campaign artifacts, checkpoints, and evaluations",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Base pretrained model",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="lora_symmetric",
        help="Comma-separated list of experiments to run (default: lora_symmetric)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for LoRA adapter (default: 1e-4)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs per experiment (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device training batch size (default: 32)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--checkpoints-per-epoch",
        type=int,
        default=1,
        help="Checkpoints saved per epoch (default: 1)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit on training rows for smoke testing",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit on validation rows for smoke testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/cache",
        help="Cache directory for document embeddings",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="Batch size for dense retrieval evaluation",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 precision",
    )
    parser.add_argument(
        "--native-baseline",
        type=float,
        default=None,
        help="Optional reference native nDCG@10 baseline (empirically measured if None)",
    )
    parser.add_argument(
        "--english-baseline",
        type=float,
        default=None,
        help="Optional reference English nDCG@10 baseline (empirically measured if None)",
    )
    return parser.parse_args()


def evaluate_symmetric_checkpoint(
    checkpoint_path: str,
    manifests: Dict[str, EvaluationManifest],
    cache_dir: str,
    batch_size: int = 512,
    fp16: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a symmetric checkpoint (queries and corpus both encoded with checkpoint).

    Args:
        checkpoint_path: Path to model checkpoint.
        manifests: Dict mapping task key to EvaluationManifest.
        cache_dir: Cache directory for corpus vectors.
        batch_size: Evaluation batch size.
        fp16: Whether to use FP16.

    Returns:
        Dict mapping task key to metric dictionary.
    """
    logger.info("Evaluating symmetric checkpoint %s...", checkpoint_path)
    evaluator = DenseRetrievalEvaluator(
        model_name_or_path=checkpoint_path,
        query_prefix="query: ",
        passage_prefix="passage: ",
        pooling="mean",
        max_length=512,
        batch_size=batch_size,
        cache_dir=cache_dir,
        fp16=fp16,
    )

    results: Dict[str, Dict[str, float]] = {}
    for task_name, manifest in manifests.items():
        logger.info("Running evaluation on %s (%s)...", task_name, manifest.name)
        metrics, _ = evaluator.evaluate(manifest)
        results[task_name] = {
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "recall@20": metrics.get("recall@20", 0.0),
            "mrr@10": metrics.get("mrr@10", 0.0),
        }
        logger.info(
            "  %s -> nDCG@10: %.4f | Recall@20: %.4f | MRR@10: %.4f",
            task_name,
            results[task_name]["ndcg@10"],
            results[task_name]["recall@20"],
            results[task_name]["mrr@10"],
        )

    return results


def evaluate_asymmetric_checkpoint(
    query_checkpoint_path: str,
    base_corpus_embeddings: Dict[str, np.ndarray],
    manifests: Dict[str, EvaluationManifest],
    batch_size: int = 512,
    fp16: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate an asymmetric checkpoint (queries encoded with LoRA, corpus frozen in base space).

    Args:
        query_checkpoint_path: Path to LoRA query model checkpoint.
        base_corpus_embeddings: Precomputed base model corpus embeddings per task.
        manifests: Dict mapping task key to EvaluationManifest.
        batch_size: Batch size for encoding.
        fp16: Whether to use FP16.

    Returns:
        Dict mapping task key to metric dictionary.
    """
    logger.info(
        "Evaluating query-asymmetric checkpoint (Query: %s | Frozen base corpus)...",
        query_checkpoint_path,
    )
    query_evaluator = DenseRetrievalEvaluator(
        model_name_or_path=query_checkpoint_path,
        query_prefix="query: ",
        passage_prefix="passage: ",
        pooling="mean",
        max_length=512,
        batch_size=batch_size,
        cache_dir=None,
        fp16=fp16,
    )

    results: Dict[str, Dict[str, float]] = {}
    for task_name, manifest in manifests.items():
        logger.info(
            "Running asymmetric evaluation on %s (%s)...", task_name, manifest.name
        )
        base_doc_embeddings = base_corpus_embeddings[task_name]
        doc_ids = list(manifest.corpus.keys())

        # 2. Encode queries using LoRA query model
        query_ids = list(manifest.queries.keys())
        query_texts = [manifest.queries[qid].text for qid in query_ids]
        lora_query_embeddings = query_evaluator.encode(
            query_texts,
            prefix=query_evaluator.query_prefix,
            desc=f"LoRA queries ({task_name})",
            sort_by_length=False,
        )

        # 3. Compute dot product similarity against frozen base embeddings
        similarity_matrix = np.dot(lora_query_embeddings, base_doc_embeddings.T)

        top_k = 100
        ranked_results: Dict[str, List[Tuple[str, float]]] = {}
        for q_idx, qid in enumerate(query_ids):
            scores = similarity_matrix[q_idx]
            top_indices = np.argsort(-scores)[:top_k]
            ranked_results[qid] = [
                (doc_ids[idx], float(scores[idx])) for idx in top_indices
            ]

        metrics = compute_retrieval_metrics(ranked_results, manifest.qrels)
        results[task_name] = {
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "recall@20": metrics.get("recall@20", 0.0),
            "mrr@10": metrics.get("mrr@10", 0.0),
        }
        logger.info(
            "  %s -> nDCG@10: %.4f | Recall@20: %.4f | MRR@10: %.4f",
            task_name,
            results[task_name]["ndcg@10"],
            results[task_name]["recall@20"],
            results[task_name]["mrr@10"],
        )

    return results


def run_lora_campaign() -> None:
    """Execute the parameter-isolated LoRA experimental campaign."""
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluation manifests
    logger.info("Loading evaluation manifests for 3-way assessment...")
    manifests: Dict[str, EvaluationManifest] = {
        "native": EvaluationManifest.load(resolve_manifest_path(args.native_eval_dir)),
        "romanized": EvaluationManifest.load(
            resolve_manifest_path(args.romanized_eval_dir)
        ),
        "english": EvaluationManifest.load(
            resolve_manifest_path(args.english_eval_dir)
        ),
    }
    logger.info(
        "Manifests loaded: Native (%d q), Romanized (%d q), English (%d q)",
        len(manifests["native"].queries),
        len(manifests["romanized"].queries),
        len(manifests["english"].queries),
    )

    # 2. Load training and validation datasets
    train_path = resolve_data_path(args.train_file)
    logger.info("Loading training data from %s...", str(train_path))
    train_dataset = RetrievalDataset(train_path, max_samples=args.max_train_samples)

    val_dataset = None
    if args.val_file:
        val_path = resolve_data_path(args.val_file)
        logger.info("Loading validation data from %s...", str(val_path))
        val_dataset = RetrievalDataset(val_path, max_samples=args.max_val_samples)

    # 3. Setup experiment configurations
    active_experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    logger.info("Active experiments scheduled: %s", active_experiments)

    experiment_definitions = {
        "lora_symmetric": {
            "title": f"LoRA Symmetric (r={args.lora_r}, alpha={args.lora_alpha}, LR {args.learning_rate})",
            "dir_name": "01_lora_symmetric",
            "architecture_mode": "lora_symmetric",
        },
        "lora_query_asymmetric": {
            "title": f"LoRA Query-Asymmetric (r={args.lora_r}, alpha={args.lora_alpha}, LR {args.learning_rate})",
            "dir_name": "02_lora_query_asymmetric",
            "architecture_mode": "lora_query_asymmetric",
        },
    }

    campaign_results: List[Dict[str, Any]] = []

    # 4. Measure base model reference baselines
    logger.info(
        "Evaluating base model '%s' to establish reference baselines...",
        args.base_model,
    )
    base_evaluator = DenseRetrievalEvaluator(
        model_name_or_path=args.base_model,
        query_prefix="query: ",
        passage_prefix="passage: ",
        pooling="mean",
        max_length=512,
        batch_size=args.eval_batch_size,
        cache_dir=args.cache_dir,
        fp16=not args.no_fp16,
    )

    base_eval: Dict[str, Dict[str, float]] = {}
    base_corpus_embeddings: Dict[str, np.ndarray] = {}
    for task_name, manifest in manifests.items():
        doc_ids = list(manifest.corpus.keys())
        doc_texts = [
            f"{manifest.corpus[did].title} {manifest.corpus[did].text}".strip()
            for did in doc_ids
        ]
        base_corpus_embeddings[task_name] = base_evaluator.encode(
            doc_texts,
            prefix=base_evaluator.passage_prefix,
            desc=f"Base corpus ({task_name})",
            sort_by_length=True,
        )
        metrics, _ = base_evaluator.evaluate(manifest)
        base_eval[task_name] = {
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "recall@20": metrics.get("recall@20", 0.0),
            "mrr@10": metrics.get("mrr@10", 0.0),
        }

    native_baseline = (
        args.native_baseline
        if args.native_baseline is not None
        else base_eval["native"]["ndcg@10"]
    )
    english_baseline = (
        args.english_baseline
        if args.english_baseline is not None
        else base_eval["english"]["ndcg@10"]
    )
    roman_baseline = base_eval["romanized"]["ndcg@10"]
    roman_target = 0.4000

    campaign_results.append(
        {
            "experiment": "base_model",
            "title": f"Base Model Reference ({args.base_model})",
            "checkpoint": "pretrained_base",
            "checkpoint_path": args.base_model,
            "native_ndcg@10": native_baseline,
            "native_recall@20": base_eval["native"]["recall@20"],
            "roman_ndcg@10": roman_baseline,
            "roman_recall@20": base_eval["romanized"]["recall@20"],
            "english_ndcg@10": english_baseline,
            "english_recall@20": base_eval["english"]["recall@20"],
            "decision": "REFERENCE BASELINE",
            "detailed_metrics": base_eval,
        }
    )

    logger.info(
        "Preservation gates: Native >= %.4f (baseline: %.4f) | English >= %.4f (baseline: %.4f) | Romanized target >= %.4f",
        native_baseline - 0.02,
        native_baseline,
        english_baseline - 0.02,
        english_baseline,
        roman_target,
    )

    start_total_time = time.time()

    for exp_key in active_experiments:
        if exp_key not in experiment_definitions:
            logger.warning("Skipping unknown experiment key '%s'", exp_key)
            continue

        exp_info = experiment_definitions[exp_key]
        exp_dir = output_root / exp_info["dir_name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("STARTING EXPERIMENT: %s", exp_info["title"])
        logger.info("Output directory: %s", str(exp_dir))
        logger.info("=" * 70)

        cfg = LoraTrainingConfig(
            model_name_or_path=args.base_model,
            output_dir=str(exp_dir),
            max_length=512,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            num_epochs=args.num_epochs,
            warmup_ratio=0.1,
            temperature=0.05,
            fp16=not args.no_fp16,
            seed=args.seed,
            logging_steps=50,
            eval_steps=250,
            checkpoints_per_epoch=args.checkpoints_per_epoch,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            architecture_mode=exp_info["architecture_mode"],
        )

        trainer = LoraRetrievalTrainer(
            config=cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        train_summary = trainer.train()

        checkpoints_to_eval = []
        best_model_path = exp_dir / "best_model"
        final_model_path = exp_dir / "final_model"

        if best_model_path.exists():
            checkpoints_to_eval.append(("best_model", best_model_path))
        if final_model_path.exists():
            checkpoints_to_eval.append(("final_model", final_model_path))
        if not checkpoints_to_eval:
            checkpoints_to_eval.append(("final_model", exp_dir))

        for ckpt_label, ckpt_path in checkpoints_to_eval:
            logger.info("Evaluating %s checkpoint at %s...", ckpt_label, str(ckpt_path))
            if exp_info["architecture_mode"] == "lora_query_asymmetric":
                eval_metrics = evaluate_asymmetric_checkpoint(
                    query_checkpoint_path=str(ckpt_path),
                    base_corpus_embeddings=base_corpus_embeddings,
                    manifests=manifests,
                    batch_size=args.eval_batch_size,
                    fp16=not args.no_fp16,
                )
            else:
                eval_metrics = evaluate_symmetric_checkpoint(
                    checkpoint_path=str(ckpt_path),
                    manifests=manifests,
                    cache_dir=args.cache_dir,
                    batch_size=args.eval_batch_size,
                    fp16=not args.no_fp16,
                )

            native_ndcg = eval_metrics["native"]["ndcg@10"]
            roman_ndcg = eval_metrics["romanized"]["ndcg@10"]
            en_ndcg = eval_metrics["english"]["ndcg@10"]

            native_pass = native_ndcg >= (native_baseline - 0.02)
            english_pass = en_ndcg >= (english_baseline - 0.02)
            roman_pass = roman_ndcg >= roman_target

            if native_pass and english_pass and roman_pass:
                decision = "ACCEPT (Native retained, English retained, Roman >= 0.40)"
            elif native_pass and english_pass:
                decision = f"PROGRESS (Preserved, Roman: {roman_ndcg:.4f})"
            else:
                failed_gates = []
                if not native_pass:
                    failed_gates.append(
                        f"Native drop: {native_ndcg - native_baseline:.4f}"
                    )
                if not english_pass:
                    failed_gates.append(
                        f"English drop: {en_ndcg - english_baseline:.4f}"
                    )
                decision = f"REJECT ({', '.join(failed_gates)})"

            record = {
                "experiment": exp_key,
                "title": f"{exp_info['title']} ({ckpt_label})",
                "checkpoint": ckpt_label,
                "checkpoint_path": str(ckpt_path),
                "native_ndcg@10": native_ndcg,
                "native_recall@20": eval_metrics["native"]["recall@20"],
                "roman_ndcg@10": roman_ndcg,
                "roman_recall@20": eval_metrics["romanized"]["recall@20"],
                "english_ndcg@10": en_ndcg,
                "english_recall@20": eval_metrics["english"]["recall@20"],
                "decision": decision,
                "detailed_metrics": eval_metrics,
                "training_summary": (
                    train_summary if ckpt_label == "final_model" else None
                ),
            }
            campaign_results.append(record)

            # Save individual checkpoint metrics JSON
            ckpt_metrics_file = output_root / f"{exp_key}_{ckpt_label}_metrics.json"
            with open(ckpt_metrics_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    total_duration = time.time() - start_total_time

    # 5. Output summary JSON and Markdown Table
    summary_file = output_root / "campaign_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_duration_seconds": total_duration,
                "results": campaign_results,
            },
            f,
            indent=2,
        )

    md_lines = [
        "# LoRA Adaptation Campaign Summary",
        "",
        f"Execution duration: {total_duration / 60:.1f} minutes",
        "",
        "| Experiment | Checkpoint | Native nDCG@10 | Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 | English nDCG@10 | English Recall@20 | Decision |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in campaign_results:
        md_lines.append(
            f"| {r['experiment']} | {r['checkpoint']} | {r['native_ndcg@10']:.4f} | {r['native_recall@20']:.4f} | {r['roman_ndcg@10']:.4f} | {r['roman_recall@20']:.4f} | {r['english_ndcg@10']:.4f} | {r['english_recall@20']:.4f} | {r['decision']} |"
        )

    md_lines.append("")
    md_lines.append("## Acceptance Criteria Reference")
    md_lines.append(
        f"- Native Hindi baseline: {native_baseline:.4f} (Gate: >= {native_baseline - 0.02:.4f})"
    )
    md_lines.append(
        f"- English SciFact baseline: {english_baseline:.4f} (Gate: >= {english_baseline - 0.02:.4f})"
    )
    md_lines.append(f"- Romanized target: >= {roman_target:.4f}")
    md_lines.append("")

    report_path = output_root / "campaign_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Campaign completed successfully in %.1f minutes.", total_duration / 60)
    logger.info("Summary report saved to %s", str(report_path))
    logger.info("Summary JSON saved to %s", str(summary_file))


if __name__ == "__main__":
    run_lora_campaign()
