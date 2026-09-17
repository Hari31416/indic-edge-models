"""Visualization and reporting utilities for training and retrieval evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_loss_curves(
    train_logs: Sequence[Dict[str, Any]],
    eval_logs: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Plot training loss and validation loss over optimization steps."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    if train_logs:
        steps = [entry["step"] for entry in train_logs]
        losses = [entry["train_loss"] for entry in train_logs]
        ax.plot(
            steps,
            losses,
            label="Train Loss",
            color="#1f77b4",
            linewidth=1.8,
            alpha=0.85,
        )

    if eval_logs:
        eval_steps = [entry["step"] for entry in eval_logs]
        eval_losses = [entry["eval_loss"] for entry in eval_logs]
        ax.plot(
            eval_steps,
            eval_losses,
            label="Validation Loss",
            color="#d62728",
            linewidth=2.2,
            marker="o",
            markersize=5,
        )

    ax.set_title(
        "Embedding Retrieval Training Loss Progression",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Optimization Steps", fontsize=11)
    ax.set_ylabel("InfoNCE Contrastive Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved loss curves plot to %s", str(out_file))
    return out_file


def plot_retrieval_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
    metrics: Sequence[str] = ("ndcg@10", "recall@20", "mrr@10"),
) -> Path:
    """Generate grouped bar chart comparing retrieval metrics across model variants."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    labels = list(comparison_data.keys())
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    for i, metric in enumerate(metrics):
        values = [comparison_data[lbl].get(metric, 0.0) for lbl in labels]
        metric_label = metric.upper()
        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=metric_label,
            color=colors[i % len(colors)],
            alpha=0.9,
        )

        for j, val in enumerate(values):
            ax.text(
                x[j] + (i - 1) * width,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Retrieval Score", fontsize=11)
    ax.set_title(
        "Retrieval Benchmark Comparison",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved retrieval comparison plot to %s", str(out_file))
    return out_file


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Format data rows as a markdown table with aligned columns."""
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        formatted_row = [str(cell) for cell in row]
        str_rows.append(formatted_row)
        for i, cell in enumerate(formatted_row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )
    separator_line = (
        "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    )
    data_lines = [
        "| "
        + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        + " |"
        for row in str_rows
    ]
    return header_line + "\n" + separator_line + "\n" + "\n".join(data_lines)
