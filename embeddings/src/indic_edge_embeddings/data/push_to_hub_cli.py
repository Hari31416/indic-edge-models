"""Upload prepared datasets to the Hugging Face Hub."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("push_to_hub_cli")


def push_dataset_directory(
    dataset_dir: str,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    """Load local parquet or jsonl dataset and push to Hugging Face Hub.

    Args:
        dataset_dir: Local directory containing parquet or jsonl splits.
        repo_id: Target HF dataset repository (e.g. 'username/indic-retrieval-pilot-hi').
        private: Whether the HF repo should be private.
        token: Optional Hugging Face authentication token.
    """
    dir_path = Path(dataset_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    parquet_files = list(dir_path.glob("*.parquet"))
    jsonl_files = list(dir_path.glob("*.jsonl"))

    splits: dict = {}
    if parquet_files:
        logger.info(
            "Found parquet files in %s: %s",
            dataset_dir,
            [f.name for f in parquet_files],
        )
        for pfile in parquet_files:
            split_name = pfile.stem
            splits[split_name] = load_dataset("parquet", data_files=str(pfile))["train"]
    elif jsonl_files:
        logger.info(
            "Found jsonl files in %s: %s", dataset_dir, [f.name for f in jsonl_files]
        )
        for jfile in jsonl_files:
            split_name = jfile.stem
            splits[split_name] = load_dataset("json", data_files=str(jfile))["train"]
    else:
        raise ValueError(f"No parquet or jsonl files found in {dataset_dir}")

    ds_dict = DatasetDict(splits)
    logger.info("Pushing DatasetDict to %s (private=%s)...", repo_id, private)
    ds_dict.push_to_hub(repo_id=repo_id, private=private, token=token)
    logger.info(
        "Successfully pushed dataset to https://huggingface.co/datasets/%s", repo_id
    )


def main() -> None:
    """Parse arguments and push dataset."""
    parser = argparse.ArgumentParser(
        description="Push local dataset folder to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Local dataset directory containing parquet or jsonl files",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face Hub repository (e.g. 'user/dataset-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HF Hub",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face write token (defaults to local huggingface-cli login)",
    )

    args = parser.parse_args()
    push_dataset_directory(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
