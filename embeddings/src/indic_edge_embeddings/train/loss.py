"""Contrastive loss functions for retrieval fine-tuning."""

from typing import Dict, List, Optional, Tuple, Union

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
        positive_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
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

        if positive_mask is not None:
            positive_mask = positive_mask.to(similarity.device, dtype=torch.bool)
            valid_mask = (
                torch.ones_like(positive_mask)
                if valid_mask is None
                else valid_mask.to(similarity.device, dtype=torch.bool)
            )
            if (
                positive_mask.shape != similarity.shape
                or valid_mask.shape != similarity.shape
            ):
                raise ValueError("Retrieval masks must match query/candidate scores")
            if (
                not positive_mask.any(dim=1).all()
                or (positive_mask & ~valid_mask).any()
            ):
                raise ValueError("Every query needs at least one valid positive")
            # Float32 logsumexp avoids fp16 overflow in multi-positive reductions.
            scores = similarity.float()
            numerator = scores.masked_fill(~positive_mask, float("-inf")).logsumexp(
                dim=1
            )
            denominator = scores.masked_fill(~valid_mask, float("-inf")).logsumexp(
                dim=1
            )
            return (denominator - numerator).mean()

        # Targets are diagonal indices (0, 1, ..., batch_size - 1)
        targets = torch.arange(
            query_embeddings.size(0),
            device=query_embeddings.device,
            dtype=torch.long,
        )

        return self.cross_entropy(similarity, targets)


class ScriptAlignmentLoss(nn.Module):
    """Cosine distance loss aligning student Romanized queries to frozen teacher native queries."""

    def forward(
        self,
        student_roman_embeddings: torch.Tensor,
        teacher_native_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute script alignment loss: mean(1 - cosine(S(q_roman), T(q_native))).

        Args:
            student_roman_embeddings: Tensor of shape (N, D), L2-normalized.
            teacher_native_embeddings: Tensor of shape (N, D), L2-normalized.

        Returns:
            Scalar loss tensor.
        """
        if student_roman_embeddings.numel() == 0:
            return torch.tensor(
                0.0, device=student_roman_embeddings.device, requires_grad=True
            )

        cos_sim = torch.sum(
            student_roman_embeddings * teacher_native_embeddings, dim=-1
        )
        return torch.mean(1.0 - cos_sim)


class RepresentationPreservationLoss(nn.Module):
    """Group-averaged cosine distance loss preserving representation against frozen teacher."""

    def forward(
        self,
        groups: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute group-balanced preservation loss: mean_by_group(1 - cosine(S(text), T(text))).

        Args:
            groups: List of (student_embeddings, teacher_embeddings) tuples, each (N_g, D), L2-normalized.

        Returns:
            Scalar loss tensor averaging over valid groups.
        """
        group_losses = []
        for student_emb, teacher_emb in groups:
            if student_emb.numel() == 0:
                continue
            cos_sim = torch.sum(student_emb * teacher_emb, dim=-1)
            group_losses.append(torch.mean(1.0 - cos_sim))

        if not group_losses:
            # Fallback if no valid groups provided
            first_dev = groups[0][0].device if groups else torch.device("cpu")
            return torch.tensor(0.0, device=first_dev, requires_grad=True)

        return torch.stack(group_losses).mean()


class DualObjectiveLoss(nn.Module):
    """Combined objective balancing script alignment, representation preservation, and retrieval."""

    def __init__(
        self,
        weight_script: float = 1.0,
        weight_preserve: float = 1.0,
        weight_retrieval: float = 0.0,
        temperature: float = 0.05,
    ) -> None:
        """Initialize DualObjectiveLoss.

        Args:
            weight_script: Scaling factor for L_script.
            weight_preserve: Scaling factor for L_preserve.
            weight_retrieval: Scaling factor for InfoNCE retrieval loss (0.0 for alignment-only).
            temperature: Softmax scaling temperature for InfoNCE.
        """
        super().__init__()
        self.weight_script = weight_script
        self.weight_preserve = weight_preserve
        self.weight_retrieval = weight_retrieval
        self.script_loss_fn = ScriptAlignmentLoss()
        self.preserve_loss_fn = RepresentationPreservationLoss()
        self.retrieval_loss_fn = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        student_roman_emb: torch.Tensor,
        teacher_native_emb: torch.Tensor,
        preservation_groups: list[tuple[torch.Tensor, torch.Tensor]],
        student_query_emb: Optional[torch.Tensor] = None,
        student_passage_emb: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss and return loss breakdown.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        device = (
            student_roman_emb.device
            if student_roman_emb.numel() > 0
            else (
                preservation_groups[0][0].device
                if preservation_groups
                else torch.device("cpu")
            )
        )
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}

        if self.weight_script > 0 and student_roman_emb.numel() > 0:
            l_script = self.script_loss_fn(student_roman_emb, teacher_native_emb)
            total_loss = total_loss + self.weight_script * l_script
            metrics["loss_script"] = l_script.item()
        else:
            metrics["loss_script"] = 0.0

        if self.weight_preserve > 0 and preservation_groups:
            l_preserve = self.preserve_loss_fn(preservation_groups)
            total_loss = total_loss + self.weight_preserve * l_preserve
            metrics["loss_preserve"] = l_preserve.item()
        else:
            metrics["loss_preserve"] = 0.0

        if (
            self.weight_retrieval > 0
            and student_query_emb is not None
            and student_passage_emb is not None
            and student_query_emb.numel() > 0
        ):
            l_retrieval = self.retrieval_loss_fn(
                student_query_emb, student_passage_emb, positive_mask, valid_mask
            )
            total_loss = total_loss + self.weight_retrieval * l_retrieval
            metrics["loss_retrieval"] = l_retrieval.item()
        else:
            metrics["loss_retrieval"] = 0.0

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
