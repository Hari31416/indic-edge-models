"""Training engine for fine-tuning embedding models on retrieval data."""

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from indic_edge_embeddings.train.dataset import RetrievalCollator, RetrievalDataset
from indic_edge_embeddings.train.loss import InfoNCELoss, mean_pooling
from indic_edge_embeddings.train.plots import (
    format_markdown_table,
    plot_loss_curves,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for embedding model fine-tuning."""

    model_name_or_path: str = "intfloat/multilingual-e5-small"
    output_dir: str = "outputs/pilot_e5_adapted"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_length: int = 512
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    temperature: float = 0.05
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 250
    save_total_limit: int = 2


class RetrievalTrainer:
    """Trainer for embedding models using in-batch negative contrastive loss."""

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        """Initialize RetrievalTrainer.

        Args:
            config: Training configuration.
            train_dataset: Training dataset instance.
            eval_dataset: Optional validation dataset instance.
            model: Optional pre-loaded model instance.
            tokenizer: Optional pre-loaded tokenizer instance.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self._set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = config.fp16 and (self.device.type == "cuda")

        logger.info(
            "Initializing trainer on device %s (fp16=%s, batch_size=%d, grad_accum=%d)",
            self.device,
            self.fp16,
            config.batch_size,
            config.gradient_accumulation_steps,
        )

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModel.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.loss_fn = InfoNCELoss(temperature=config.temperature)

        self.collator = RetrievalCollator(
            tokenizer=self.tokenizer,
            query_prefix=config.query_prefix,
            passage_prefix=config.passage_prefix,
            max_length=config.max_length,
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        """Set random seed across all frameworks for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _encode_batch(self, batch_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through transformer with mean pooling and L2 normalization."""
        inputs = {k: v.to(self.device) for k, v in batch_tensors.items()}
        outputs = self.model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def evaluate(
        self, eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Compute validation loss over the evaluation dataset."""
        if eval_dataloader is None:
            if self.eval_dataset is None:
                return {}
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collator,
                drop_last=False,
            )

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.fp16
            else torch.inference_mode()
        )

        with torch.inference_mode():
            for batch in eval_dataloader:
                with autocast_ctx:
                    q_emb = self._encode_batch(batch["queries"])
                    p_emb = self._encode_batch(batch["passages"])
                    loss = self.loss_fn(q_emb, p_emb)

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        self.model.train()
        return {"eval_loss": avg_loss}

    def train(self) -> Dict[str, Any]:
        """Execute full training loop with checkpointing and evaluation."""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            drop_last=True,
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.config.gradient_accumulation_steps
        )
        max_train_steps = self.config.num_epochs * num_update_steps_per_epoch
        warmup_steps = int(max_train_steps * self.config.warmup_ratio)

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.config.learning_rate
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

        logger.info("***** Running Training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.config.num_epochs)
        logger.info("  Batch size per device = %d", self.config.batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d",
            self.config.gradient_accumulation_steps,
        )
        logger.info("  Total optimization steps = %d", max_train_steps)
        logger.info("  Warmup steps = %d", warmup_steps)

        global_step = 0
        best_eval_loss = float("inf")
        train_logs: list[dict[str, Any]] = []
        eval_logs: list[dict[str, Any]] = []

        start_time = time.time()
        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            accum_loss = 0.0
            step_in_epoch = 0

            for step, batch in enumerate(train_dataloader):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.fp16
                ):
                    q_emb = self._encode_batch(batch["queries"])
                    p_emb = self._encode_batch(batch["passages"])
                    loss = self.loss_fn(q_emb, p_emb)
                    loss = loss / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                    step + 1 == len(train_dataloader)
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    step_in_epoch += 1

                    if global_step % self.config.logging_steps == 0:
                        lr_curr = scheduler.get_last_lr()[0]
                        logger.info(
                            "Epoch %d/%d | Step %d/%d | Loss: %.4f | LR: %.2e",
                            epoch + 1,
                            self.config.num_epochs,
                            global_step,
                            max_train_steps,
                            accum_loss,
                            lr_curr,
                        )
                        train_logs.append(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "train_loss": accum_loss,
                                "lr": lr_curr,
                            }
                        )

                    if (
                        self.eval_dataset is not None
                        and global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        eval_loss = eval_metrics.get("eval_loss", float("inf"))
                        logger.info(
                            "Step %d Eval Results: %s", global_step, eval_metrics
                        )
                        eval_logs.append(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "eval_loss": eval_loss,
                            }
                        )
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            logger.info(
                                "New best validation loss: %.4f. Saving best checkpoint...",
                                best_eval_loss,
                            )
                            self.save_model(self.output_dir / "best_model")

                    accum_loss = 0.0

            # End of epoch evaluation
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics.get("eval_loss", float("inf"))
                logger.info(
                    "End of Epoch %d Eval Loss: %.4f", epoch + 1, eval_loss
                )
                eval_logs.append(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "eval_loss": eval_loss,
                    }
                )
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.save_model(self.output_dir / "best_model")

        total_time = time.time() - start_time
        throughput = (
            len(self.train_dataset) * self.config.num_epochs / max(1e-3, total_time)
        )
        logger.info(
            "Training completed in %.2f seconds (%.1f examples/s). Best eval loss: %.4f",
            total_time,
            throughput,
            best_eval_loss,
        )

        # Save final model
        self.save_model(self.output_dir / "final_model")

        # Generate plots and summary tables
        try:
            plot_loss_curves(
                train_logs, eval_logs, self.output_dir / "loss_curve.png"
            )
        except Exception as e:
            logger.warning("Failed to generate loss curve plot: %s", e)

        # Format markdown training report table
        headers = ["Parameter / Metric", "Value"]
        rows = [
            ["Base Model", self.config.model_name_or_path],
            ["Training Examples", str(len(self.train_dataset))],
            ["Validation Examples", str(len(self.eval_dataset)) if self.eval_dataset else "N/A"],
            ["Epochs", str(self.config.num_epochs)],
            ["Per-Device Batch Size", str(self.config.batch_size)],
            ["Gradient Accumulation", str(self.config.gradient_accumulation_steps)],
            ["Effective Batch Size", str(self.config.batch_size * self.config.gradient_accumulation_steps)],
            ["Learning Rate", f"{self.config.learning_rate:.2e}"],
            ["Total Optimization Steps", str(global_step)],
            ["Training Duration (s)", f"{total_time:.2f}"],
            ["Throughput (examples/s)", f"{throughput:.1f}"],
            ["Best Validation Loss", f"{best_eval_loss:.4f}" if best_eval_loss < float("inf") else "N/A"],
        ]
        report_md = "# Training Summary Report\n\n" + format_markdown_table(headers, rows) + "\n"
        with open(self.output_dir / "training_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        # Save training summary JSON
        summary = {
            "config": asdict(self.config),
            "total_time_seconds": total_time,
            "throughput_examples_per_sec": throughput,
            "global_steps": global_step,
            "best_eval_loss": best_eval_loss,
            "train_logs": train_logs,
            "eval_logs": eval_logs,
        }
        with open(
            self.output_dir / "training_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_model(self, save_dir: Union[str, Path]) -> None:
        """Save model weights, tokenizer, and config to directory."""
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        logger.info("Saved model and tokenizer to %s", str(target))
