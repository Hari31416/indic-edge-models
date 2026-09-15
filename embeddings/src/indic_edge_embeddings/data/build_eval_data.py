"""Build and freeze evaluation datasets for native and Romanized Hindi retrieval."""

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from indic_edge_embeddings.data.transliterate import (
    transliterate_devanagari_to_roman,
    ROMANIZATION_VERSION,
)
from indic_edge_embeddings.eval.loader import (
    CorpusDoc,
    EvaluationManifest,
    Query,
    create_romanized_query_slice,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_eval_data")


def export_manifest_to_parquet(manifest: EvaluationManifest, output_dir: Path) -> None:
    """Export manifest queries, corpus, and qrels to Parquet format for HF Hub.

    Args:
        manifest: Evaluation manifest to export.
        output_dir: Destination directory.
    """
    parquet_dir = output_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Queries
    queries_data = [
        {
            "query_id": q.query_id,
            "text": q.text,
            "language": q.language,
            "script": q.script,
            "source_query_id": q.source_query_id or "",
        }
        for q in manifest.queries.values()
    ]
    pq.write_table(pa.Table.from_pylist(queries_data), parquet_dir / "queries.parquet")

    # Corpus
    corpus_data = [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": doc.text,
        }
        for doc in manifest.corpus.values()
    ]
    pq.write_table(pa.Table.from_pylist(corpus_data), parquet_dir / "corpus.parquet")

    # Qrels
    qrels_data = []
    for qid, docs in manifest.qrels.items():
        for did, score in docs.items():
            qrels_data.append(
                {
                    "query_id": qid,
                    "doc_id": did,
                    "score": int(score),
                }
            )
    pq.write_table(pa.Table.from_pylist(qrels_data), parquet_dir / "qrels.parquet")
    logger.info("Saved Parquet tables in %s", str(parquet_dir))


def load_miracl_tsv_queries(topics_file: str) -> Dict[str, Query]:
    """Parse MIRACL TSV topics file into Query objects."""
    queries: Dict[str, Query] = {}
    with open(topics_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                qid = row[0].strip()
                query_text = row[1].strip()
                queries[qid] = Query(
                    query_id=qid,
                    text=query_text,
                    language="hi",
                    script="native",
                )
    return queries


def load_miracl_tsv_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """Parse MIRACL TSV qrels file into nested dict."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                qid = row[0].strip()
                doc_id = row[2].strip()
                rel = int(row[3].strip())
                if rel > 0:
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = rel
    return qrels


def load_miracl_corpus_gz(corpus_gz_files: List[str]) -> Dict[str, CorpusDoc]:
    """Parse MIRACL gzipped JSONL corpus files into CorpusDoc objects."""
    corpus: Dict[str, CorpusDoc] = {}
    for gz_path in corpus_gz_files:
        logger.info("Reading corpus file %s...", Path(gz_path).name)
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    doc_id = str(item["docid"])
                    corpus[doc_id] = CorpusDoc(
                        doc_id=doc_id,
                        title=item.get("title", ""),
                        text=item.get("text", ""),
                    )
    return corpus


def build_miracl_hindi_eval(
    output_base_dir: str,
    split: str = "dev",
) -> Dict[str, Path]:
    """Download MIRACL Hindi split directly and create native and Romanized manifests.

    Args:
        output_base_dir: Base directory path to store evaluation manifests.
        split: MIRACL split to fetch (default: 'dev').

    Returns:
        Dict mapping manifest name to saved directory path.
    """
    base_path = Path(output_base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    if (base_path / f"miracl_hi_{split}_romanized").exists():
        raise FileExistsError(
            "Romanized dataset already exists; use a new output directory"
        )

    logger.info("Fetching MIRACL Hindi %s topics and qrels from Hugging Face...", split)
    topics_path = hf_hub_download(
        repo_id="miracl/miracl",
        repo_type="dataset",
        filename=f"miracl-v1.0-hi/topics/topics.miracl-v1.0-hi-{split}.tsv",
    )
    qrels_path = hf_hub_download(
        repo_id="miracl/miracl",
        repo_type="dataset",
        filename=f"miracl-v1.0-hi/qrels/qrels.miracl-v1.0-hi-{split}.tsv",
    )

    logger.info("Fetching MIRACL Hindi corpus shards from Hugging Face...")
    corpus_file_0 = hf_hub_download(
        repo_id="miracl/miracl-corpus",
        repo_type="dataset",
        filename="miracl-corpus-v1.0-hi/docs-0.jsonl.gz",
    )
    corpus_file_1 = hf_hub_download(
        repo_id="miracl/miracl-corpus",
        repo_type="dataset",
        filename="miracl-corpus-v1.0-hi/docs-1.jsonl.gz",
    )

    queries_map = load_miracl_tsv_queries(topics_path)
    qrels_map = load_miracl_tsv_qrels(qrels_path)
    corpus_map = load_miracl_corpus_gz([corpus_file_0, corpus_file_1])

    logger.info(
        "Loaded %d queries, %d judged queries, %d corpus documents",
        len(queries_map),
        len(qrels_map),
        len(corpus_map),
    )

    native_manifest = EvaluationManifest(
        name=f"miracl_hi_{split}_native",
        language="hi",
        split=split,
        queries=queries_map,
        corpus=corpus_map,
        qrels=qrels_map,
        metadata={
            "source": "miracl/miracl",
            "language": "hi",
            "script": "Devanagari",
        },
    )

    native_dir = base_path / f"miracl_hi_{split}_native"
    native_manifest.save(native_dir)
    export_manifest_to_parquet(native_manifest, native_dir)

    logger.info("Generating synthetic Romanized query slice...")
    roman_manifest = create_romanized_query_slice(
        native_manifest,
        transliterate_fn=transliterate_devanagari_to_roman,
        suffix="_roman",
    )
    roman_manifest.name = f"miracl_hi_{split}_romanized"
    roman_manifest.metadata["romanization_version"] = ROMANIZATION_VERSION
    roman_dir = base_path / f"miracl_hi_{split}_romanized"
    roman_manifest.save(roman_dir)
    export_manifest_to_parquet(roman_manifest, roman_dir)

    manifest_summary = {
        "miracl_native": str(native_dir),
        "miracl_romanized": str(roman_dir),
        "num_queries": len(queries_map),
        "num_corpus": len(corpus_map),
    }
    logger.info("Evaluation data creation complete: %s", manifest_summary)
    return {
        "native": native_dir,
        "romanized": roman_dir,
    }


def main() -> None:
    """CLI entrypoint for evaluation dataset generation."""
    parser = argparse.ArgumentParser(
        description="Build native and synthetic Romanized evaluation manifests"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval",
        help="Base directory to save evaluation datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Evaluation split to process",
    )

    args = parser.parse_args()
    build_miracl_hindi_eval(output_base_dir=args.output_dir, split=args.split)


if __name__ == "__main__":
    main()
