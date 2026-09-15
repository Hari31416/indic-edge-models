"""Build and freeze evaluation manifests from public retrieval datasets."""

import argparse
import logging
from pathlib import Path
from typing import Dict

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
logger = logging.getLogger("build_manifest")


def build_miracl_hindi_manifest(
    output_dir: str,
    split: str = "dev",
    create_romanized: bool = True,
) -> None:
    """Download or load MIRACL Hindi split and save frozen manifest.

    Args:
        output_dir: Directory where manifest will be saved.
        split: Split name ('dev').
        create_romanized: Whether to also save synthetic Romanized query slice.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "Hugging Face `datasets` package is required to build MIRACL manifest"
        )
        raise

    logger.info("Loading miracl/miracl Hindi split '%s'...", split)
    miracl_qrels = load_dataset("miracl/miracl", "hi", split=split)
    miracl_corpus = load_dataset("miracl/miracl-corpus", "hi", split="train")

    corpus_map: Dict[str, CorpusDoc] = {}
    for item in miracl_corpus:
        doc_id = str(item["docid"])
        corpus_map[doc_id] = CorpusDoc(
            doc_id=doc_id,
            text=item.get("text", ""),
            title=item.get("title", ""),
        )

    queries_map: Dict[str, Query] = {}
    qrels_map: Dict[str, Dict[str, int]] = {}

    for item in miracl_qrels:
        qid = str(item["query_id"])
        query_text = item["query"]
        if qid not in queries_map:
            queries_map[qid] = Query(
                query_id=qid,
                text=query_text,
                language="hi",
                script="native",
            )
            qrels_map[qid] = {}

        # Positive and negative judgments
        for pos_doc in item.get("positive_passages", []):
            did = str(pos_doc["docid"])
            qrels_map[qid][did] = 1

    manifest = EvaluationManifest(
        name=f"miracl_hi_{split}",
        language="hi",
        split=split,
        queries=queries_map,
        corpus=corpus_map,
        qrels=qrels_map,
        metadata={"source": "miracl/miracl", "language": "hi"},
    )

    out_path = Path(output_dir) / f"miracl_hi_{split}"
    manifest.save(out_path)

    if create_romanized:
        # Fallback character-level or phonetic transliterator if indic-transliteration not installed
        try:
            from indic_transliteration import sanscript

            def translit_fn(text: str) -> str:
                return sanscript.transliterate(
                    text, sanscript.DEVANAGARI, sanscript.ITRANS
                )

        except ImportError:
            logger.warning(
                "indic_transliteration not installed; using placeholder transliterator"
            )

            def translit_fn(text: str) -> str:
                return text

        roman_manifest = create_romanized_query_slice(manifest, translit_fn)
        roman_out = Path(output_dir) / f"miracl_hi_{split}_romanized"
        roman_manifest.save(roman_out)


def main() -> None:
    """CLI entrypoint for building evaluation manifests."""
    parser = argparse.ArgumentParser(
        description="Build and freeze evaluation manifests"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Target base directory for manifests",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Evaluation split (e.g. dev)",
    )
    parser.add_argument(
        "--no-romanized",
        action="store_true",
        help="Skip creating synthetic Romanized slice",
    )

    args = parser.parse_args()
    build_miracl_hindi_manifest(
        output_dir=args.output_dir,
        split=args.split,
        create_romanized=not args.no_romanized,
    )


if __name__ == "__main__":
    main()
