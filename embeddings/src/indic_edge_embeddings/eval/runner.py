"""Evaluation runners for dense and lexical retrieval."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from indic_edge_embeddings.eval.loader import EvaluationManifest
from indic_edge_embeddings.eval.metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling on token embeddings weighted by attention mask.

    Args:
        model_output: Last hidden state tensor of shape (batch, seq_len, hidden_dim).
        attention_mask: Attention mask tensor of shape (batch, seq_len).

    Returns:
        Pooled embedding tensor of shape (batch, hidden_dim).
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    )
    sum_embeddings = torch.sum(model_output * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class DenseRetrievalEvaluator:
    """High-throughput dense retrieval evaluator with length sorting and caching."""

    def __init__(
        self,
        model_name_or_path: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        pooling: str = "mean",
        normalize: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 512,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize dense evaluator.

        Args:
            model_name_or_path: Hugging Face model identifier or local checkpoint path.
            query_prefix: Optional prefix string prepended to queries (e.g. 'query: ').
            passage_prefix: Optional prefix string prepended to passages (e.g. 'passage: ').
            pooling: Pooling strategy ('mean' or 'cls').
            normalize: Whether to L2-normalize output embeddings.
            max_length: Maximum token length for tokenization.
            device: Execution device ('cpu', 'cuda', 'mps', or auto-detected).
            batch_size: Batch size for encoding (default: 512).
            cache_dir: Directory to cache encoded corpus vectors.
        """
        self.model_name = model_name_or_path
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Loading model: %s on %s (batch_size=%d, max_length=%d)",
            self.model_name,
            self.device,
            self.batch_size,
            self.max_length,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: Sequence[str],
        prefix: str = "",
        desc: str = "Encoding",
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """Encode a sequence of text strings into embedding vectors in batches.

        Args:
            texts: List of input strings.
            prefix: Prefix prepended to each string before tokenization.
            desc: Progress bar description.
            sort_by_length: Sort by string length to minimize batch padding overhead.

        Returns:
            Numpy array of shape (num_texts, embedding_dim) matching original text order.
        """
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        if sort_by_length and len(prefixed_texts) > 1:
            lengths = np.array([len(t) for t in prefixed_texts])
            sort_indices = np.argsort(lengths)
            ordered_texts = [prefixed_texts[i] for i in sort_indices]
        else:
            sort_indices = None
            ordered_texts = prefixed_texts

        all_embeddings: List[np.ndarray] = []
        total_batches = (len(ordered_texts) + self.batch_size - 1) // self.batch_size

        with torch.inference_mode():
            for i in tqdm(
                range(0, len(ordered_texts), self.batch_size),
                total=total_batches,
                desc=desc,
                ncols=85,
            ):
                batch_texts = ordered_texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encoded)

                if self.pooling == "mean":
                    embeddings = mean_pooling(
                        outputs.last_hidden_state, encoded["attention_mask"]
                    )
                elif self.pooling == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

                if self.normalize:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=1
                    )

                all_embeddings.append(embeddings.cpu().numpy())

        stacked = np.vstack(all_embeddings)
        if sort_indices is not None:
            # Restore original sequence order
            revert_indices = np.argsort(sort_indices)
            return stacked[revert_indices]
        return stacked

    def evaluate(
        self,
        manifest: EvaluationManifest,
        top_k: int = 100,
    ) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
        """Run dense retrieval evaluation over a manifest.

        Args:
            manifest: Evaluation manifest containing queries, corpus, and qrels.
            top_k: Number of top candidate documents to rank per query.

        Returns:
            Tuple of (computed metrics dict, raw ranked results dict).
        """
        doc_ids = list(manifest.corpus.keys())
        doc_texts = [
            f"{manifest.corpus[did].title} {manifest.corpus[did].text}".strip()
            for did in doc_ids
        ]

        cached = False
        cache_file: Optional[Path] = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            model_hash = hashlib.md5(
                f"{self.model_name}_{self.passage_prefix}_{len(doc_texts)}_{self.max_length}".encode(
                    "utf-8"
                )
            ).hexdigest()[:10]
            cache_file = self.cache_dir / f"corpus_emb_{model_hash}.npy"
            if cache_file.exists():
                logger.info("Loading cached corpus embeddings from %s", str(cache_file))
                doc_embeddings = np.load(cache_file)
                cached = True

        if not cached:
            logger.info(
                "Encoding %d corpus documents with prefix '%s' (batch_size=%d, length-sorted)...",
                len(doc_texts),
                self.passage_prefix,
                self.batch_size,
            )
            doc_embeddings = self.encode(
                doc_texts,
                prefix=self.passage_prefix,
                desc="Corpus docs",
                sort_by_length=True,
            )
            if cache_file:
                logger.info("Saving corpus embeddings to cache: %s", str(cache_file))
                np.save(cache_file, doc_embeddings)

        query_ids = list(manifest.queries.keys())
        query_texts = [manifest.queries[qid].text for qid in query_ids]

        logger.info(
            "Encoding %d queries with prefix '%s'...",
            len(query_texts),
            self.query_prefix,
        )
        query_embeddings = self.encode(
            query_texts,
            prefix=self.query_prefix,
            desc="Queries",
            sort_by_length=False,
        )

        logger.info("Computing similarity matrix and ranking top-%d results...", top_k)
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for q_idx, qid in enumerate(query_ids):
            scores = similarity_matrix[q_idx]
            top_indices = np.argsort(-scores)[:top_k]
            results[qid] = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]

        metrics = compute_retrieval_metrics(results, manifest.qrels)
        logger.info("Evaluation results on %s: %s", manifest.name, metrics)
        return metrics, results
