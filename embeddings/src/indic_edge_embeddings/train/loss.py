"""Contrastive loss functions for retrieval fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Apply attention-masked mean pooling to token embeddings.

    Args:
        last_hidden_state: Output tensor of shape (batch, seq_len, hidden_dim).
        attention_mask: Attention mask tensor of shape (batch, seq_len).

    Returns:
        Pooled embedding tensor of shape (batch, hidden_dim).
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class InfoNCELoss(nn.Module):
    """InfoNCE / Multiple Negatives Ranking loss with in-batch negatives."""

    def __init__(self, temperature: float = 0.05) -> None:
        """Initialize InfoNCE loss.

        Args:
            temperature: Softmax scaling temperature (default: 0.05, scaling factor 20.0).
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute in-batch contrastive loss between query and passage vectors.

        Args:
            query_embeddings: Tensor of shape (batch_size, hidden_dim), L2-normalized.
            passage_embeddings: Tensor of shape (batch_size, hidden_dim), L2-normalized.

        Returns:
            Scalar loss tensor.
        """
        # Cosine similarity matrix of shape (batch_size, batch_size)
        similarity = torch.matmul(query_embeddings, passage_embeddings.T)
        similarity = similarity / self.temperature

        # Targets are diagonal indices (0, 1, ..., batch_size - 1)
        targets = torch.arange(
            query_embeddings.size(0),
            device=query_embeddings.device,
            dtype=torch.long,
        )

        return self.cross_entropy(similarity, targets)
