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
        self._load_records(max_samples)
        logger.info(
            "Loaded %d retrieval records from %s",
            len(self.records),
            str(self.file_path),
        )

    def _load_records(self, max_samples: Optional[int]) -> None:
        """Internal method to read records from JSONL or Parquet."""
        if self.file_path.suffix == ".parquet":
            table = pq.read_table(self.file_path)
            rows = table.to_pylist()
            if max_samples:
                rows = rows[:max_samples]
            for row in rows:
                self.records.append(
                    {
                        "query_id": str(row.get("query_id", "")),
                        "query": str(row.get("query", "")),
                        "positive": str(row.get("positive", "")),
                        "negatives": row.get("negatives", []) or [],
                        "language": str(row.get("language", "hi")),
                        "script": str(row.get("script", "native")),
                    }
                )
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    self.records.append(
                        {
                            "query_id": str(item.get("query_id", "")),
                            "query": str(item.get("query", "")),
                            "positive": str(item.get("positive", "")),
                            "negatives": item.get("negatives", []) or [],
                            "language": str(item.get("language", "hi")),
                            "script": str(item.get("script", "native")),
                        }
                    )
                    if max_samples and len(self.records) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


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
        passages = [
            f"{self.passage_prefix}{item['positive']}".strip() for item in batch
        ]

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
        }


