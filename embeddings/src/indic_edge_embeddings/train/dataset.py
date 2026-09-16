"""Dataset and batch collation for retrieval fine-tuning."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def resolve_data_path(file_path: Union[str, Path]) -> Path:
    """Resolve file path, searching recursively in /kaggle/input if running on Kaggle.

    Args:
        file_path: Input file path.

    Returns:
        Resolved Path object.
    """
    path = Path(file_path)
    if path.exists():
        return path

    filename = path.name
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.glob(f"**/{filename}"))
        if matches:
            logger.info("Resolved data path %s to %s", str(file_path), str(matches[0]))
            return matches[0]

    return path


class RetrievalDataset(Dataset):
    """Dataset of query-positive passage pairs for contrastive retrieval training."""

    def __init__(
        self,
        file_path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> None:
        """Load dataset from a JSONL or Parquet file.

        Args:
            file_path: Path to .jsonl or .parquet dataset file.
            max_samples: Optional limit on the number of samples to load.
        """
        self.file_path = resolve_data_path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")


        self.records: List[Dict[str, Any]] = []
        self.native_map: Dict[str, str] = {}
        self._build_native_map()
        self._load_records(max_samples)
        self._resolve_native_pairs()
        logger.info(
            "Loaded %d retrieval records (%d native source references) from %s",
            len(self.records),
            len(self.native_map),
            str(self.file_path),
        )

    def _build_native_map(self) -> None:
        """Scan the dataset to index all native queries by query_id."""
        if self.file_path.suffix == ".parquet":
            try:
                table = pq.read_table(
                    self.file_path, columns=["query_id", "query", "script"]
                )
                qids = table["query_id"].to_pylist()
                queries = table["query"].to_pylist()
                scripts = table["script"].to_pylist()
                for qid, q, s in zip(qids, queries, scripts):
                    if s == "native":
                        self.native_map[str(qid)] = str(q)
            except Exception as e:
                logger.warning("Could not pre-scan parquet native queries: %s", e)
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("script") == "native":
                        self.native_map[str(item["query_id"])] = str(item["query"])

    def _load_records(self, max_samples: Optional[int]) -> None:
        """Internal method to read records from JSONL or Parquet."""
        if self.file_path.suffix == ".parquet":
            table = pq.read_table(self.file_path)
            rows = table.to_pylist()
            if max_samples:
                rows = rows[:max_samples]
            for row in rows:
                qid = str(row.get("query_id", ""))
                q_text = str(row.get("query", ""))
                script = str(row.get("script", "native"))
                sqid = str(row.get("source_query_id", qid))
                rec = {
                    "query_id": qid,
                    "query": q_text,
                    "positive": str(row.get("positive", "")),
                    "negatives": row.get("negatives", []) or [],
                    "language": str(row.get("language", "hi")),
                    "script": script,
                    "source_query_id": sqid,
                    "positive_texts": row.get("positive_texts") or [str(row.get("positive", ""))],
                    "split_group_id": row.get("split_group_id"),
                }
                self.records.append(rec)
                if script == "native":
                    self.native_map[qid] = q_text
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    qid = str(item.get("query_id", ""))
                    q_text = str(item.get("query", ""))
                    script = str(item.get("script", "native"))
                    sqid = str(item.get("source_query_id", qid))
                    rec = {
                        "query_id": qid,
                        "query": q_text,
                        "positive": str(item.get("positive", "")),
                        "negatives": item.get("negatives", []) or [],
                        "language": str(item.get("language", "hi")),
                        "script": script,
                        "source_query_id": sqid,
                        "positive_texts": item.get("positive_texts") or [str(item.get("positive", ""))],
                        "split_group_id": item.get("split_group_id"),
                    }
                    self.records.append(rec)
                    if script == "native":
                        self.native_map[qid] = q_text
                    if max_samples and len(self.records) >= max_samples:
                        break

    def _resolve_native_pairs(self) -> None:
        """Populate native_query on each record using source_query_id lookup."""
        for rec in self.records:
            if rec.get("script") == "romanized":
                sqid = rec.get("source_query_id")
                if sqid and sqid in self.native_map:
                    rec["native_query"] = self.native_map[sqid]
                else:
                    raise ValueError(
                        f"Native source query target '{sqid}' not found in dataset for Romanized query '{rec['query_id']}'."
                    )
            else:
                rec["native_query"] = rec["query"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def retrieval_candidates(batch):
    """Encode positives first, then unique vetted negatives; mask known positives.

    All labeled positives present in the candidate list receive probability mass.
    Related split-group passages with unknown relevance are excluded from loss.
    """
    import unicodedata

    def key(value):
        return " ".join(unicodedata.normalize("NFC", value).casefold().split())

    candidates = [item["positive"] for item in batch]
    seen = {key(p) for p in candidates}
    for item in batch:
        for negative in item.get("negatives", []):
            if key(negative) not in seen:
                candidates.append(negative)
                seen.add(key(negative))
    candidate_keys = [key(p) for p in candidates]
    known_by_query = {}
    for item in batch:
        query = key(item.get("native_query", item["query"]))
        known_by_query.setdefault(query, set()).update(
            key(p) for p in item.get("positive_texts", [item["positive"]])
        )
        known_by_query[query].add(key(item["positive"]))
    positive_mask = torch.zeros((len(batch), len(candidates)), dtype=torch.bool)
    valid_mask = torch.ones_like(positive_mask)
    for i, item in enumerate(batch):
        known = known_by_query[key(item.get("native_query", item["query"]))]
        positive_mask[i] = torch.tensor([p in known for p in candidate_keys])
        group = item.get("split_group_id")
        if group:
            for j, other in enumerate(batch):
                if other.get("split_group_id") == group and not positive_mask[i, j]:
                    valid_mask[i, j] = False
    return candidates, positive_mask, valid_mask


class RetrievalCollator:
    """Collate batch of retrieval examples with dynamic tokenization and formatting."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        max_length: int = 512,
    ) -> None:
        """Initialize collator.

        Args:
            tokenizer: Pretrained tokenizer for encoding texts.
            query_prefix: Prefix prepended to query texts.
            passage_prefix: Prefix prepended to passage texts.
            max_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_length = max_length

    def __call__(
        self, batch: Sequence[Dict[str, Any]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Tokenize queries and positive passages in the batch.

        Args:
            batch: Sequence of record dicts containing query and positive.

        Returns:
            Dict containing queries and passages batch tensors.
        """
        queries = [f"{self.query_prefix}{item['query']}".strip() for item in batch]
        candidate_texts, positive_mask, valid_mask = retrieval_candidates(batch)
        passages = [f"{self.passage_prefix}{p}".strip() for p in candidate_texts]

        query_enc = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        passage_enc = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "queries": query_enc,
            "passages": passage_enc,
            "positive_mask": positive_mask,
            "valid_mask": valid_mask,
        }


class PairedRetrievalCollator:
    """Collate batch of paired retrieval examples with teacher targets and preservation items."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        max_length: int = 512,
        english_preservation_texts: Optional[List[str]] = None,
    ) -> None:
        """Initialize paired collator.

        Args:
            tokenizer: Pretrained tokenizer for encoding texts.
            query_prefix: Prefix prepended to query texts.
            passage_prefix: Prefix prepended to passage texts.
            max_length: Maximum sequence length for tokenization.
            english_preservation_texts: Optional list of English texts for representation preservation.
        """
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_length = max_length
        self.english_preservation_texts = english_preservation_texts or []

    def __call__(
        self, batch: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tokenize batch into student queries, teacher native targets, and passages.

        Args:
            batch: Sequence of record dicts.

        Returns:
            Dict containing batch encodings and index tensors.
        """
        student_queries = [f"{self.query_prefix}{item['query']}".strip() for item in batch]
        teacher_native_queries = [
            f"{self.query_prefix}{item.get('native_query', item['query'])}".strip()
            for item in batch
        ]
        candidate_texts, positive_mask, valid_mask = retrieval_candidates(batch)
        passages = [f"{self.passage_prefix}{p}".strip() for p in candidate_texts]

        is_romanized = [item.get("script") == "romanized" for item in batch]
        roman_indices = [i for i, is_r in enumerate(is_romanized) if is_r]
        native_indices = [i for i, is_r in enumerate(is_romanized) if not is_r]

        student_query_enc = self.tokenizer(
            student_queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        teacher_query_enc = self.tokenizer(
            teacher_native_queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        passage_enc = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        res: Dict[str, Any] = {
            "queries": student_query_enc,
            "teacher_queries": teacher_query_enc,
            "passages": passage_enc,
            "positive_mask": positive_mask,
            "valid_mask": valid_mask,
            "roman_indices": torch.tensor(roman_indices, dtype=torch.long),
            "native_indices": torch.tensor(native_indices, dtype=torch.long),
        }

        if self.english_preservation_texts:
            import random

            sample_size = min(len(batch), len(self.english_preservation_texts))
            en_samples = random.sample(self.english_preservation_texts, sample_size)
            en_prefixed = [f"{self.passage_prefix}{t}".strip() for t in en_samples]
            res["english_passages"] = self.tokenizer(
                en_prefixed,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        return res
