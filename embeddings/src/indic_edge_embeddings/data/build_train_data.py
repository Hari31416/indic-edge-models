"""Build and curate training datasets for Indic retrieval pilot."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from indic_edge_embeddings.data.transliterate import (
    transliterate_devanagari_to_roman,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_train_data")


def create_triplet_record(
    query_id: str,
    query: str,
    positive: str,
    negatives: List[str],
    language: str = "hi",
    script: str = "native",
    source: str = "",
) -> Dict[str, Any]:
    """Create a standardized retrieval triplet dictionary."""
    return {
        "query_id": query_id,
        "query": query,
        "positive": positive,
        "negatives": negatives,
        "language": language,
        "script": script,
        "source": source,
    }


def load_indic_swim_ir_records(max_count: int = 20_000) -> List[Dict[str, Any]]:
    """Load records from Indic SWIM-IR Hindi split."""
    records: List[Dict[str, Any]] = []
    logger.info("Fetching Indic SWIM-IR Hindi data from Hugging Face...")
    try:
        swim_path = hf_hub_download(
            repo_id="nthakur/indic-swim-ir-cross-lingual",
            repo_type="dataset",
            filename="hi/train-00000-of-00001.parquet",
        )
        table = pq.read_table(swim_path)
        for i in range(min(table.num_rows, max_count)):
            qid = str(table["_id"][i].as_py())
            q_text = str(table["query"][i].as_py()).strip()
            title = str(table["title"][i].as_py() or "").strip()
            text = str(table["text"][i].as_py() or "").strip()
            pos_text = f"{title} {text}".strip()
            if q_text and pos_text:
                records.append(
                    create_triplet_record(
                        query_id=f"swim_{qid}",
                        query=q_text,
                        positive=pos_text,
                        negatives=[],
                        language="hi",
                        script="native",
                        source="indic-swim-ir",
                    )
                )
        logger.info("Loaded %d records from Indic SWIM-IR", len(records))
    except Exception as e:
        logger.warning("Could not load Indic SWIM-IR: %s", e)
    return records


def is_valid_training_pair(query: str, positive: str) -> bool:
    """Check if query and passage form a valid, informative training pair."""
    q = query.strip()
    p = positive.strip()
    if len(q) < 5 or len(p) < 25:
        return False
    uninformative_markers = [
        "कोई उत्तर नहीं मिला",
        "no answer found",
        "कोई जवाब नहीं",
        "null",
    ]
    for marker in uninformative_markers:
        if marker in p.lower():
            return False
    return True


def load_msmarco_xi_records(
    needed_count: int,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Load records from MSMARCO-XI Hindi dataset with quality filtering."""
    records: List[Dict[str, Any]] = []
    logger.info("Fetching MSMARCO-XI Hindi data from Hugging Face...")
    try:
        msmarco_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            repo_type="dataset",
            filename="train/hintrain.parquet",
        )
        table = pq.read_table(
            msmarco_path,
            columns=["query_id", "query", "Answer", "Eng_Query", "Eng_Answer"],
        )
        curr_idx = offset
        while len(records) < needed_count and curr_idx < table.num_rows:
            qid = str(table["query_id"][curr_idx].as_py())
            q_text = str(table["query"][curr_idx].as_py() or "").strip()
            ans_text = str(table["Answer"][curr_idx].as_py() or "").strip()

            if is_valid_training_pair(q_text, ans_text):
                records.append(
                    create_triplet_record(
                        query_id=f"msmarco_hi_{qid}_{curr_idx}",
                        query=q_text,
                        positive=ans_text,
                        negatives=[],
                        language="hi",
                        script="native",
                        source="msmarco-xi",
                    )
                )
            curr_idx += 1
        logger.info(
            "Loaded %d valid records from MSMARCO-XI (scanned %d rows)",
            len(records),
            curr_idx - offset,
        )
    except Exception as e:
        logger.warning("Could not load MSMARCO-XI: %s", e)
    return records


def build_hindi_pilot_dataset(
    output_dir: str,
    max_samples: int = 100_000,
    val_samples: int = 5_000,
    romanized_ratio: float = 0.3,
    seed: int = 42,
) -> Path:
    """Build a curated training dataset from public Hindi retrieval sources.

    Args:
        output_dir: Destination folder.
        max_samples: Maximum training sample count.
        val_samples: Validation sample count.
        romanized_ratio: Proportion of queries to augment with Romanized variants.
        seed: Random seed for reproducibility.

    Returns:
        Path to saved dataset directory.
    """
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # 1. Indic SWIM-IR
    records.extend(load_indic_swim_ir_records())

    # 2. MSMARCO-XI to reach target count
    target_native_count = int((max_samples + val_samples) / (1.0 + romanized_ratio))
    needed_msmarco = max(0, target_native_count - len(records))
    if needed_msmarco > 0:
        records.extend(load_msmarco_xi_records(needed_count=needed_msmarco))

    logger.info("Total native base records collected: %d", len(records))

    # 3. Augment with synthetic Romanized queries
    augmented_records: List[Dict[str, Any]] = []
    for rec in records:
        augmented_records.append(rec)
        if random.random() < romanized_ratio:
            roman_query = transliterate_devanagari_to_roman(rec["query"])
            if roman_query and roman_query != rec["query"]:
                roman_rec = dict(rec)
                roman_rec["query_id"] = f"{rec['query_id']}_roman"
                roman_rec["query"] = roman_query
                roman_rec["script"] = "romanized"
                augmented_records.append(roman_rec)

    random.shuffle(augmented_records)

    # 4. Split train and validation
    val_set = augmented_records[:val_samples]
    train_set = augmented_records[val_samples : max_samples + val_samples]

    logger.info(
        "Final split counts: %d train, %d validation", len(train_set), len(val_set)
    )

    # 5. Save Parquet and JSONL
    train_table = pa.Table.from_pylist(train_set)
    val_table = pa.Table.from_pylist(val_set)

    pq.write_table(train_table, out_path / "train.parquet")
    pq.write_table(val_table, out_path / "validation.parquet")

    with open(out_path / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_path / "validation.jsonl", "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset_name": "indic_retrieval_pilot_hindi",
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "romanized_ratio": romanized_ratio,
        "seed": seed,
    }
    with open(out_path / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved curated training dataset to %s", str(out_path))
    return out_path


def main() -> None:
    """CLI entrypoint for training dataset generation."""
    parser = argparse.ArgumentParser(
        description="Build and curate Indic retrieval pilot training dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train/pilot_hindi_100k",
        help="Destination directory for training files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum training sample count",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=5_000,
        help="Validation sample count",
    )
    parser.add_argument(
        "--romanized-ratio",
        type=float,
        default=0.3,
        help="Proportion of Romanized queries to include",
    )

    args = parser.parse_args()
    build_hindi_pilot_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_samples=args.val_samples,
        romanized_ratio=args.romanized_ratio,
    )


if __name__ == "__main__":
    main()
