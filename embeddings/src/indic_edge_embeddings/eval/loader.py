"""Evaluation dataset loading and manifest management."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CorpusDoc:
    """A document in the evaluation corpus."""

    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)


@dataclass
class Query:
    """A search query in the evaluation dataset."""

    query_id: str
    text: str
    language: str
    script: str = "native"
    source_query_id: Optional[str] = None
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)


@dataclass
class EvaluationManifest:
    """Frozen evaluation dataset containing corpus, queries, and relevance judgments."""

    name: str
    language: str
    split: str
    queries: Dict[str, Query]
    corpus: Dict[str, CorpusDoc]
    qrels: Dict[str, Dict[str, Union[int, float]]]
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save the evaluation manifest to a directory.

        Args:
            output_dir: Target directory path.

        Returns:
            Path to the saved manifest directory.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        meta_file = out_path / "metadata.json"
        manifest_meta = {
            "name": self.name,
            "language": self.language,
            "split": self.split,
            "query_count": len(self.queries),
            "corpus_count": len(self.corpus),
            "qrels_query_count": len(self.qrels),
            "metadata": self.metadata,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

        queries_file = out_path / "queries.jsonl"
        with open(queries_file, "w", encoding="utf-8") as f:
            for q in self.queries.values():
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

        corpus_file = out_path / "corpus.jsonl"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for doc in self.corpus.values():
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

        qrels_file = out_path / "qrels.json"
        with open(qrels_file, "w", encoding="utf-8") as f:
            json.dump(self.qrels, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved evaluation manifest '%s' (%d queries, %d docs) to %s",
            self.name,
            len(self.queries),
            len(self.corpus),
            str(out_path),
        )
        return out_path

    @classmethod
    def load(cls, manifest_dir: Union[str, Path]) -> "EvaluationManifest":
        """Load an evaluation manifest from a directory.

        Args:
            manifest_dir: Path to manifest directory.

        Returns:
            Loaded EvaluationManifest object.
        """
        dir_path = Path(manifest_dir)
        meta_file = dir_path / "metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        queries: Dict[str, Query] = {}
        queries_file = dir_path / "queries.jsonl"
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries[data["query_id"]] = Query(**data)

        corpus: Dict[str, CorpusDoc] = {}
        corpus_file = dir_path / "corpus.jsonl"
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    corpus[data["doc_id"]] = CorpusDoc(**data)

        qrels_file = dir_path / "qrels.json"
        with open(qrels_file, "r", encoding="utf-8") as f:
            qrels = json.load(f)

        return cls(
            name=meta["name"],
            language=meta["language"],
            split=meta["split"],
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            metadata=meta.get("metadata", {}),
        )


def create_romanized_query_slice(
    manifest: EvaluationManifest,
    transliterate_fn,
    suffix: str = "_roman",
) -> EvaluationManifest:
    """Create a synthetic Romanized query slice preserving IDs and qrels.

    Args:
        manifest: Source EvaluationManifest with native script queries.
        transliterate_fn: Callable taking native text and returning Romanized text.
        suffix: Suffix appended to query IDs.

    Returns:
        New EvaluationManifest with Romanized queries.
    """
    romanized_queries: Dict[str, Query] = {}
    romanized_qrels: Dict[str, Dict[str, Union[int, float]]] = {}

    for qid, query in manifest.queries.items():
        new_qid = f"{qid}{suffix}"
        roman_text = transliterate_fn(query.text)
        romanized_queries[new_qid] = Query(
            query_id=new_qid,
            text=roman_text,
            language=query.language,
            script="romanized",
            source_query_id=qid,
            metadata={**query.metadata, "original_text": query.text},
        )
        if qid in manifest.qrels:
            romanized_qrels[new_qid] = manifest.qrels[qid]

    return EvaluationManifest(
        name=f"{manifest.name}_romanized",
        language=manifest.language,
        split=manifest.split,
        queries=romanized_queries,
        corpus=manifest.corpus,
        qrels=romanized_qrels,
        metadata={
            **manifest.metadata,
            "source_manifest": manifest.name,
            "transformation": "synthetic_romanization",
        },
    )
