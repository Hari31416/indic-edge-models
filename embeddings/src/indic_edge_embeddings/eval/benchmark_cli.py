"""Command line interface for running embedding model benchmarks."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from indic_edge_embeddings.eval.loader import EvaluationManifest
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_cli")


def run_benchmark(
    manifest_dir: str,
    model_name: str,
    query_prefix: str = "",
    passage_prefix: str = "",
    pooling: str = "mean",
    max_length: int = 512,
    batch_size: int = 512,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    output_json: Optional[str] = None,
    fp16: bool = True,
) -> None:
    """Load manifest, evaluate candidate model, and save/log results."""
    logger.info("Loading manifest from %s", manifest_dir)
    manifest = EvaluationManifest.load(manifest_dir)

    evaluator = DenseRetrievalEvaluator(
        model_name_or_path=model_name,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        pooling=pooling,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        fp16=fp16,
    )

    metrics, _ = evaluator.evaluate(manifest)

    logger.info("Evaluation Complete. Metrics for %s on %s:", model_name, manifest.name)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_name,
                    "manifest": manifest.name,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        logger.info("Saved results to %s", str(out_path))


def main() -> None:
    """Parse command line arguments and execute benchmark."""
    parser = argparse.ArgumentParser(
        description="Run dense retrieval benchmark on evaluation manifests"
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        required=True,
        help="Path to evaluation manifest directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default="",
        help="Prefix string for queries (e.g. 'query: ' for E5)",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="",
        help="Prefix string for passages (e.g. 'passage: ' for E5)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Pooling strategy",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 mixed precision on CUDA",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached corpus embeddings",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save output metrics JSON",
    )

    args = parser.parse_args()
    run_benchmark(
        manifest_dir=args.manifest_dir,
        model_name=args.model_name,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        output_json=args.output_json,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
