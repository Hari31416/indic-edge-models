"""Unified campaign runner executing follow-up adaptation experiments and 3-way evaluation."""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from indic_edge_embeddings.eval.loader import EvaluationManifest, resolve_manifest_path
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator
from indic_edge_embeddings.train.dataset import RetrievalDataset, resolve_data_path
from indic_edge_embeddings.train.trainer import RetrievalTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("followup_campaign")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the multi-experiment campaign."""
    parser = argparse.ArgumentParser(
        description="Run E5 adaptation follow-up experiment campaign with 3-way evaluation."
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
        default="eval/miracl_hi_dev_native",
        help="Path to native Hindi evaluation manifest",
    )
    parser.add_argument(
        "--romanized-eval-dir",
        type=str,
        default="eval/miracl_hi_dev_romanized",
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
        default="/kaggle/working/outputs/campaign",
        help="Directory to store all campaign artifacts, checkpoints, and evaluations",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Base student model",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Frozen teacher model for distillation",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="conservative_control,teacher_align,dual_objective",
        help="Comma-separated list of experiments to run (default: conservative_control,teacher_align,dual_objective)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Initial learning rate for conservative schedule (default: 5e-6)",
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
        help="Checkpoints saved per epoch (default: 1 for epoch-end save)",
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
        "--eval-intermediate-checkpoints",
        action="store_true",
        default=False,
        help="Evaluate all intermediate checkpoints in addition to epoch-end final_model (default: False, evaluate epoch-end only)",
    )
    parser.add_argument(
        "--eval-final-only",
        action="store_true",
        default=True,
        help="Evaluate only epoch-end model (default: True)",
    )
    parser.add_argument(
        "--english-preservation-file",
        type=str,
        default=None,
        help="Optional path to file containing English preservation texts",
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


def evaluate_checkpoint(
    checkpoint_path: str,
    manifests: Dict[str, EvaluationManifest],
    cache_dir: str,
    batch_size: int = 512,
    fp16: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a specific checkpoint across loaded evaluation manifests.

    Args:
        checkpoint_path: Path to model checkpoint.
        manifests: Dict mapping task key to EvaluationManifest.
        cache_dir: Cache directory for corpus vectors.
        batch_size: Evaluation batch size.
        fp16: Whether to use FP16.

    Returns:
        Dict mapping task key to metric dictionary.
    """
    logger.info("Evaluating checkpoint %s...", checkpoint_path)
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


def run_campaign() -> None:
    """Execute the follow-up experimental campaign."""
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

    # 3. Supply English preservation texts for distillation
    english_preservation_texts: List[str] = []
    if args.english_preservation_file:
        p = Path(args.english_preservation_file)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        english_preservation_texts.append(line.strip())
    if not english_preservation_texts and "english" in manifests:
        # Extract corpus passages from the loaded English manifest
        for doc in manifests["english"].corpus.values():
            text = f"{doc.title} {doc.text}".strip()
            if len(text) > 25:
                english_preservation_texts.append(text)
            if len(english_preservation_texts) >= 10000:
                break
    logger.info(
        "Loaded %d English preservation texts for teacher distillation.",
        len(english_preservation_texts),
    )

    # 4. Setup experiment configurations
    active_experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    logger.info("Active experiments scheduled: %s", active_experiments)

    experiment_definitions = {
        "conservative_control": {
            "title": "Conservative Control (InfoNCE, LR 5e-6, 1 Epoch)",
            "dir_name": "01_conservative_control",
            "loss_type": "infonce",
            "weight_script": 0.0,
            "weight_preserve": 0.0,
            "weight_retrieval": 1.0,
        },
        "teacher_align": {
            "title": "Teacher Alignment + Preservation (L_script + L_preserve)",
            "dir_name": "02_teacher_alignment",
            "loss_type": "teacher_align",
            "weight_script": 1.0,
            "weight_preserve": 1.0,
            "weight_retrieval": 0.0,
        },
        "dual_objective": {
            "title": "Dual-Objective (L_script + L_preserve + L_retrieval)",
            "dir_name": "03_dual_objective",
            "loss_type": "dual",
            "weight_script": 1.0,
            "weight_preserve": 1.0,
            "weight_retrieval": 1.0,
        },
    }

    campaign_results: List[Dict[str, Any]] = []

    # 5. Measure or establish empirical baselines dynamically
    if args.native_baseline is not None and args.english_baseline is not None:
        native_baseline = args.native_baseline
        english_baseline = args.english_baseline
        roman_baseline = 0.0
    else:
        logger.info(
            "Evaluating base model '%s' to establish empirical baselines across manifests...",
            args.base_model,
        )
        base_eval = evaluate_checkpoint(
            checkpoint_path=args.base_model,
            manifests=manifests,
            cache_dir=args.cache_dir,
            batch_size=args.eval_batch_size,
            fp16=not args.no_fp16,
        )
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

    roman_target = 0.4000
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

        cfg = TrainingConfig(
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
            loss_type=exp_info["loss_type"],
            teacher_model_name_or_path=args.teacher_model,
            weight_script=exp_info["weight_script"],
            weight_preserve=exp_info["weight_preserve"],
            weight_retrieval=exp_info["weight_retrieval"],
            english_preservation_texts=english_preservation_texts,
        )

        trainer = RetrievalTrainer(
            config=cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        train_summary = trainer.train()

        # Gather checkpoints to evaluate (default: epoch-end final_model only)
        checkpoints_to_eval = []
        if not args.eval_intermediate_checkpoints:
            final_model_path = exp_dir / "final_model"
            if final_model_path.exists():
                checkpoints_to_eval.append(str(final_model_path))
            else:
                best_model_path = exp_dir / "best_model"
                if best_model_path.exists():
                    checkpoints_to_eval.append(str(best_model_path))
                else:
                    candidate_dirs = sorted(
                        list(exp_dir.glob("checkpoint-*")),
                        key=lambda p: (
                            int(p.name.split("-")[1])
                            if p.name.split("-")[1].isdigit()
                            else 0
                        ),
                    )
                    if candidate_dirs:
                        checkpoints_to_eval.append(str(candidate_dirs[-1]))
        else:
            # Sorted intermediate checkpoints + best + final
            candidate_dirs = sorted(
                list(exp_dir.glob("checkpoint-*")),
                key=lambda p: (
                    int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0
                ),
            )
            for c in candidate_dirs:
                checkpoints_to_eval.append(str(c))
            best_model_path = exp_dir / "best_model"
            if (
                best_model_path.exists()
                and str(best_model_path) not in checkpoints_to_eval
            ):
                checkpoints_to_eval.append(str(best_model_path))
            final_model_path = exp_dir / "final_model"
            if (
                final_model_path.exists()
                and str(final_model_path) not in checkpoints_to_eval
            ):
                checkpoints_to_eval.append(str(final_model_path))

        logger.info(
            "Found %d checkpoints to evaluate for experiment '%s'",
            len(checkpoints_to_eval),
            exp_key,
        )

        for ckpt in checkpoints_to_eval:
            ckpt_name = Path(ckpt).name
            eval_metrics = evaluate_checkpoint(
                checkpoint_path=ckpt,
                manifests=manifests,
                cache_dir=args.cache_dir,
                batch_size=args.eval_batch_size,
                fp16=not args.no_fp16,
            )

            native_ndcg = eval_metrics["native"]["ndcg@10"]
            roman_ndcg = eval_metrics["romanized"]["ndcg@10"]
            en_ndcg = eval_metrics["english"]["ndcg@10"]

            # Gate criteria verification
            native_pass = native_ndcg >= (native_baseline - 0.02)
            english_pass = en_ndcg >= (english_baseline - 0.02)
            roman_pass = roman_ndcg >= roman_target

            if native_pass and english_pass and roman_pass:
                decision = "ACCEPT"
            elif native_pass and english_pass:
                decision = "PROGRESS (Preserved, Roman < 0.40)"
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
                "title": exp_info["title"],
                "checkpoint": ckpt_name,
                "checkpoint_path": ckpt,
                "native_ndcg@10": native_ndcg,
                "native_recall@20": eval_metrics["native"]["recall@20"],
                "roman_ndcg@10": roman_ndcg,
                "roman_recall@20": eval_metrics["romanized"]["recall@20"],
                "english_ndcg@10": en_ndcg,
                "english_recall@20": eval_metrics["english"]["recall@20"],
                "decision": decision,
                "detailed_metrics": eval_metrics,
            }
            campaign_results.append(record)

    total_duration = time.time() - start_total_time

    # 4. Generate consolidated summary table and JSON
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

    # Format Markdown Table
    md_lines = [
        "# Follow-up Adaptation Campaign Summary",
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
    run_campaign()
