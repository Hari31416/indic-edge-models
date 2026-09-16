"""Build deterministic fast development evaluation slices with reduced distractor corpus."""

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple, Union

from indic_edge_embeddings.eval.loader import (
    CorpusDoc,
    EvaluationManifest,
    Query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_fast_eval")


def build_fast_miracl_manifests(
    eval_base_dir: Union[str, Path],
    target_corpus_size: int = 5000,
    seed: int = 42,
    split: str = "dev",
) -> Tuple[Path, Path]:
    """Generate deterministic fast native and Romanized evaluation manifests.

    Args:
        eval_base_dir: Base directory containing full evaluation manifests.
        target_corpus_size: Total number of documents in mini corpus (positives + distractors).
        seed: Random seed for deterministic distractor sampling.
        split: Dataset split ('dev').

    Returns:
        Tuple of (fast_native_dir, fast_romanized_dir).
    """
    base_path = Path(eval_base_dir)
    native_dir = base_path / f"miracl_hi_{split}_native"
    roman_dir = base_path / f"miracl_hi_{split}_romanized"

    if not native_dir.exists():
        raise FileNotFoundError(f"Source native manifest not found at: {native_dir}")
    if not roman_dir.exists():
        raise FileNotFoundError(f"Source Romanized manifest not found at: {roman_dir}")

    logger.info("Loading full evaluation manifests from %s...", str(base_path))
    native_manifest = EvaluationManifest.load(native_dir)
    roman_manifest = EvaluationManifest.load(roman_dir)

    # Collect all known positive document IDs
    positive_doc_ids: Set[str] = set()
    for qid, docs in native_manifest.qrels.items():
        for did, score in docs.items():
            if score > 0:
                positive_doc_ids.add(did)

    logger.info(
        "Found %d queries, %d total corpus docs, %d unique relevant docs",
        len(native_manifest.queries),
        len(native_manifest.corpus),
        len(positive_doc_ids),
    )

    if len(positive_doc_ids) > target_corpus_size:
        raise ValueError(
            f"Target corpus size ({target_corpus_size}) is smaller than required positive docs ({len(positive_doc_ids)})"
        )

    # Deterministically sample distractors
    rng = random.Random(seed)
    all_distractor_ids = [
        did for did in native_manifest.corpus if did not in positive_doc_ids
    ]
    # Sort first for cross-platform determinism before shuffling
    all_distractor_ids.sort()
    rng.shuffle(all_distractor_ids)

    needed_distractors = target_corpus_size - len(positive_doc_ids)
    selected_distractors = all_distractor_ids[:needed_distractors]
    selected_doc_ids = positive_doc_ids.union(selected_distractors)

    # Build shared subset corpus
    fast_corpus: Dict[str, CorpusDoc] = {
        did: native_manifest.corpus[did] for did in selected_doc_ids
    }

    # Create fast native manifest
    fast_native = EvaluationManifest(
        name=f"miracl_hi_{split}_fast_native",
        language=native_manifest.language,
        split=native_manifest.split,
        queries=native_manifest.queries,
        corpus=fast_corpus,
        qrels=native_manifest.qrels,
        metadata={
            **native_manifest.metadata,
            "subset_mode": "fast_development",
            "target_corpus_size": target_corpus_size,
            "seed": seed,
            "source_manifest": native_manifest.name,
        },
    )
    fast_native_dir = base_path / f"miracl_hi_{split}_fast_native"
    fast_native.save(fast_native_dir)

    # Create fast Romanized manifest with identical corpus
    fast_roman = EvaluationManifest(
        name=f"miracl_hi_{split}_fast_romanized",
        language=roman_manifest.language,
        split=roman_manifest.split,
        queries=roman_manifest.queries,
        corpus=fast_corpus,
        qrels=roman_manifest.qrels,
        metadata={
            **roman_manifest.metadata,
            "subset_mode": "fast_development",
            "target_corpus_size": target_corpus_size,
            "seed": seed,
            "source_manifest": roman_manifest.name,
        },
    )
    fast_roman_dir = base_path / f"miracl_hi_{split}_fast_romanized"
    fast_roman.save(fast_roman_dir)

    logger.info(
        "Successfully created fast evaluation manifests in %s and %s",
        str(fast_native_dir),
        str(fast_roman_dir),
    )
    return fast_native_dir, fast_roman_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fast evaluation manifests from full manifests"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="embeddings/data/eval",
        help="Base evaluation directory containing full manifests",
    )
    parser.add_argument(
        "--target-corpus-size",
        type=int,
        default=5000,
        help="Total number of documents in fast corpus (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for distractor sampling (default: 42)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split name (default: dev)",
    )

    args = parser.parse_args()
    build_fast_miracl_manifests(
        eval_base_dir=args.eval_dir,
        target_corpus_size=args.target_corpus_size,
        seed=args.seed,
        split=args.split,
    )


if __name__ == "__main__":
    main()
