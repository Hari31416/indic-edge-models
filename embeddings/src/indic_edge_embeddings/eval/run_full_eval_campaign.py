"""Campaign runner for evaluating multiple embedding model variants on full benchmarks."""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from indic_edge_embeddings.eval.loader import EvaluationManifest, resolve_manifest_path
from indic_edge_embeddings.eval.metrics import compute_retrieval_metrics
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_campaign")


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format a standard markdown table surrounded by blank lines."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_line = (
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    )
    separator_line = (
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    )
    row_lines = [
        "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)) + " |"
        for r in rows
    ]
    return "\n" + "\n".join([header_line, separator_line] + row_lines) + "\n"


@dataclass
class ModelSpec:
    """Specification of a model variant to evaluate."""

    key: str
    display_name: str
    model_path: str
    is_query_asymmetric: bool = False
    is_onnx: bool = False
    adapter_path: Optional[str] = None


class FullCampaignEvaluator:
    """Orchestrates comprehensive evaluation of multiple model variants across full datasets."""

    def __init__(
        self,
        manifest_dirs: Dict[str, Union[str, Path]],
        output_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 512,
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        """Initialize campaign evaluator.

        Args:
            manifest_dirs: Dict mapping split key ('native', 'romanized', 'english') to manifest dir.
            output_dir: Output directory to store summary metrics and reports.
            cache_dir: Directory to cache corpus vectors.
            batch_size: Batch size for encoding passages and queries.
            device: Device for execution ('cuda', 'mps', 'cpu').
            fp16: Enable FP16 mixed precision.
        """
        self.manifest_dirs = manifest_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16

        # Pre-load manifests
        self.manifests: Dict[str, EvaluationManifest] = {}
        for split_key, path in self.manifest_dirs.items():
            resolved = resolve_manifest_path(path)
            logger.info(
                "Loading manifest for '%s' from %s...", split_key, str(resolved)
            )
            self.manifests[split_key] = EvaluationManifest.load(resolved)

    def evaluate_model(
        self,
        spec: ModelSpec,
    ) -> Dict[str, Any]:
        """Evaluate a single model specification across all manifests.

        Args:
            spec: ModelSpec defining the variant.

        Returns:
            Dict containing detailed evaluation metrics for each split.
        """
        logger.info("==================================================")
        logger.info("Evaluating variant: %s (%s)", spec.display_name, spec.model_path)
        logger.info("==================================================")

        results: Dict[str, Any] = {
            "key": spec.key,
            "display_name": spec.display_name,
            "model_path": spec.model_path,
            "splits": {},
        }

        evaluator = DenseRetrievalEvaluator(
            model_name_or_path=spec.model_path,
            query_prefix="query: ",
            passage_prefix="passage: ",
            pooling="mean",
            max_length=512,
            batch_size=self.batch_size,
            device=self.device,
            cache_dir=self.cache_dir,
            fp16=self.fp16,
        )

        for split_key, manifest in self.manifests.items():
            logger.info(
                "Running evaluation on split: %s (%d queries, %d corpus docs)...",
                split_key,
                len(manifest.queries),
                len(manifest.corpus),
            )
            start_t = time.time()
            metrics, _ = evaluator.evaluate(manifest)
            duration = time.time() - start_t
            logger.info(
                "[%s on %s] nDCG@10: %.4f, Recall@20: %.4f, MRR@10: %.4f (%.1fs)",
                spec.key,
                split_key,
                metrics.get("ndcg@10", 0.0),
                metrics.get("recall@20", 0.0),
                metrics.get("mrr@10", 0.0),
                duration,
            )
            results["splits"][split_key] = {
                "metrics": metrics,
                "duration_seconds": duration,
            }

        return results

    def run_campaign(
        self,
        model_specs: List[ModelSpec],
    ) -> Dict[str, Any]:
        """Run evaluation across all specified model variants and generate report."""
        campaign_results: List[Dict[str, Any]] = []

        start_time = time.time()
        for spec in model_specs:
            res = self.evaluate_model(spec)
            campaign_results.append(res)

            # Save individual model result
            spec_file = self.output_dir / f"{spec.key}_metrics.json"
            with open(spec_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

        total_time = time.time() - start_time

        # Build summary markdown table
        headers = [
            "Model Variant",
            "Native nDCG@10",
            "Native Recall@20",
            "Romanized nDCG@10",
            "Romanized Recall@20",
            "English nDCG@10",
            "English Recall@20",
        ]
        rows: List[List[str]] = []

        for res in campaign_results:
            splits = res["splits"]
            nat = splits.get("native", {}).get("metrics", {})
            rom = splits.get("romanized", {}).get("metrics", {})
            eng = splits.get("english", {}).get("metrics", {})

            rows.append(
                [
                    res["display_name"],
                    f"{nat.get('ndcg@10', 0.0):.4f}",
                    f"{nat.get('recall@20', 0.0):.4f}",
                    f"{rom.get('ndcg@10', 0.0):.4f}",
                    f"{rom.get('recall@20', 0.0):.4f}",
                    f"{eng.get('ndcg@10', 0.0):.4f}",
                    f"{eng.get('recall@20', 0.0):.4f}",
                ]
            )

        table_md = format_markdown_table(headers, rows)
        report_md = (
            "# Full Evaluation Campaign Summary\n\n"
            f"Evaluated on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"Total duration: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)\n\n"
            + table_md
            + "\n"
        )

        with open(self.output_dir / "campaign_summary.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        campaign_summary = {
            "timestamp": time.time(),
            "total_time_seconds": total_time,
            "models": campaign_results,
        }
        with open(
            self.output_dir / "campaign_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(campaign_summary, f, indent=2)

        logger.info(
            "Campaign completed in %.1f minutes. Summary report saved to %s",
            total_time / 60,
            str(self.output_dir),
        )
        return campaign_summary


def main() -> None:
    """Parse CLI arguments and run full evaluation campaign."""
    parser = argparse.ArgumentParser(
        description="Run evaluation campaign on full evaluation sets across 4 model variants"
    )
    parser.add_argument(
        "--native-eval-dir",
        type=str,
        default="data/eval/miracl_hi_dev_native",
        help="Path to native Devanagari evaluation manifest directory",
    )
    parser.add_argument(
        "--romanized-eval-dir",
        type=str,
        default="data/eval/miracl_hi_dev_romanized",
        help="Path to Romanized Hindi evaluation manifest directory",
    )
    parser.add_argument(
        "--english-eval-dir",
        type=str,
        default="data/eval/scifact_en_test",
        help="Path to English reference evaluation manifest directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to store campaign results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/cache",
        help="Cache directory for corpus encodings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--lora-fp32-path",
        type=str,
        default=None,
        help="Path to merged LoRA symmetric FP32 PyTorch model directory",
    )
    parser.add_argument(
        "--lora-int8-path",
        type=str,
        default=None,
        help="Path to LoRA symmetric full-vocab ONNX INT8 model file (.onnx)",
    )
    parser.add_argument(
        "--lora-int8-pruned-path",
        type=str,
        default=None,
        help="Path to LoRA symmetric pruned-vocab ONNX INT8 model file (.onnx)",
    )

    args = parser.parse_args()

    manifest_dirs = {
        "native": args.native_eval_dir,
        "romanized": args.romanized_eval_dir,
        "english": args.english_eval_dir,
    }

    # Define model specifications for the 4 target variants
    specs: List[ModelSpec] = [
        ModelSpec(
            key="base_e5",
            display_name="Base Pretrained E5 (FP32)",
            model_path="intfloat/multilingual-e5-small",
        ),
    ]

    if args.lora_fp32_path:
        specs.append(
            ModelSpec(
                key="lora_fp32",
                display_name="LoRA Symmetric (FP32 PyTorch)",
                model_path=args.lora_fp32_path,
            )
        )

    if args.lora_int8_path:
        specs.append(
            ModelSpec(
                key="lora_int8",
                display_name="LoRA Symmetric (ONNX INT8 ~114MB)",
                model_path=args.lora_int8_path,
                is_onnx=True,
            )
        )

    if args.lora_int8_pruned_path:
        specs.append(
            ModelSpec(
                key="lora_int8_pruned",
                display_name="LoRA Symmetric Pruned (ONNX INT8 ~55MB)",
                model_path=args.lora_int8_pruned_path,
                is_onnx=True,
            )
        )

    evaluator = FullCampaignEvaluator(
        manifest_dirs=manifest_dirs,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    evaluator.run_campaign(specs)


if __name__ == "__main__":
    main()
