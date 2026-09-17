"""Build and freeze English evaluation manifest from BEIR SciFact dataset."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Union

from indic_edge_embeddings.eval.loader import (
    CorpusDoc,
    EvaluationManifest,
    Query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_english_manifest")


def build_scifact_manifest(
    output_dir: Union[str, Path],
    split: str = "test",
) -> Path:
    """Download or load BEIR SciFact dataset and save frozen manifest.

    Args:
        output_dir: Directory where manifest directory will be saved.
        split: Split name (default: 'test').

    Returns:
        Path to the saved manifest directory.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "Hugging Face `datasets` package is required to build English manifest"
        )
        raise

    logger.info("Loading mteb/scifact split '%s'...", split)
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", split=split)

    logger.info(
        "Raw SciFact counts: %d corpus docs, %d queries in table, %d qrels in '%s' split",
        len(corpus_ds),
        len(queries_ds),
        len(qrels_ds),
        split,
    )

    # Filter qrels and gather unique query and doc IDs
    qrels_map: Dict[str, Dict[str, Union[int, float]]] = {}
    needed_qids = set()
    needed_docids = set()

    for item in qrels_ds:
        qid = str(item["query-id"])
        did = str(item["corpus-id"])
        score = float(item["score"])
        if score > 0:
            if qid not in qrels_map:
                qrels_map[qid] = {}
            qrels_map[qid][did] = int(score)
            needed_qids.add(qid)
            needed_docids.add(did)

    # Build queries map
    queries_map: Dict[str, Query] = {}
    for item in queries_ds:
        qid = str(item["_id"])
        if qid in needed_qids:
            queries_map[qid] = Query(
                query_id=qid,
                text=str(item["text"]).strip(),
                language="en",
                script="latin",
            )

    # Build corpus map (all documents in corpus for realistic retrieval ranking)
    corpus_map: Dict[str, CorpusDoc] = {}
    for item in corpus_ds:
        doc_id = str(item["_id"])
        corpus_map[doc_id] = CorpusDoc(
            doc_id=doc_id,
            text=str(item.get("text", "")).strip(),
            title=str(item.get("title", "")).strip(),
        )

    # Quality check: ensure all qrel queries exist in queries_map
    missing_queries = needed_qids - set(queries_map.keys())
    if missing_queries:
        logger.warning(
            "Found %d queries in qrels missing from queries table: %s",
            len(missing_queries),
            missing_queries,
        )

    manifest_name = f"scifact_en_{split}"
    manifest = EvaluationManifest(
        name=manifest_name,
        language="en",
        split=split,
        queries=queries_map,
        corpus=corpus_map,
        qrels=qrels_map,
        metadata={
            "source": "mteb/scifact",
            "benchmark": "BEIR",
            "language": "en",
            "split": split,
            "relevance_scale": "binary",
        },
    )

    out_path = Path(output_dir) / manifest_name
    manifest.save(out_path)
    logger.info(
        "SciFact English manifest saved successfully to %s (%d queries, %d docs, %d qrel queries)",
        str(out_path),
        len(queries_map),
        len(corpus_map),
        len(qrels_map),
    )
    return out_path


def main() -> None:
    """Parse command-line arguments and build English manifest."""
    parser = argparse.ArgumentParser(description="Build English evaluation manifest")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval",
        help="Target output directory for the manifest",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split (default: 'test')",
    )
    args = parser.parse_args()
    build_scifact_manifest(output_dir=args.output_dir, split=args.split)


if __name__ == "__main__":
    main()
