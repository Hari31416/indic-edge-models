"""Command line interface for exporting and quantizing embedding models."""

import argparse
import json
import logging
from pathlib import Path

from indic_edge_embeddings.export.exporter import (
    EmbeddingExporter,
    verify_onnx_parity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_cli")


def export_and_quantize(
    model_name: str,
    output_dir: str,
    pooling: str = "mean",
    quantize: bool = True,
    verify: bool = True,
) -> None:
    """Export model to ONNX, apply INT8 quantization, and run parity check."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = EmbeddingExporter(model_name_or_path=model_name, pooling=pooling)

    fp32_path = out_dir / "model.onnx"
    exporter.export_onnx(fp32_path)

    if quantize:
        int8_path = out_dir / "model_int8.onnx"
        exporter.quantize_int8(fp32_path, int8_path)

    if verify:
        logger.info("Running FP32 ONNX parity check...")
        fp32_report = verify_onnx_parity(exporter, fp32_path)
        logger.info(
            "FP32 parity pass: %s (mean_cos=%.6f)",
            fp32_report.passed,
            fp32_report.mean_cosine_similarity,
        )

        if quantize:
            logger.info("Running INT8 ONNX parity check...")
            int8_report = verify_onnx_parity(exporter, int8_path, min_cosine_tol=0.99)
            logger.info(
                "INT8 parity pass: %s (mean_cos=%.6f)",
                int8_report.passed,
                int8_report.mean_cosine_similarity,
            )

        report_file = out_dir / "export_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            report_data = {
                "model_name": model_name,
                "fp32_size_mb": fp32_report.onnx_file_size_mb,
                "fp32_mean_cosine": fp32_report.mean_cosine_similarity,
            }
            if quantize:
                report_data["int8_size_mb"] = int8_report.onnx_file_size_mb
                report_data["int8_mean_cosine"] = int8_report.mean_cosine_similarity
            json.dump(report_data, f, indent=2)
        logger.info("Export report saved to %s", str(report_file))


def main() -> None:
    """Parse CLI arguments and run export."""
    parser = argparse.ArgumentParser(
        description="Export Hugging Face embedding model to ONNX and INT8"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model ID or checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exported ONNX artifacts",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Pooling strategy",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical parity check",
    )

    args = parser.parse_args()
    export_and_quantize(
        model_name=args.model_name,
        output_dir=args.output_dir,
        pooling=args.pooling,
        quantize=not args.no_quantize,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
