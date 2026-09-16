"""Training engine for fine-tuning embedding models on retrieval data."""

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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

from indic_edge_embeddings.train.dataset import (
    PairedRetrievalCollator,
    RetrievalCollator,
    RetrievalDataset,
)
from indic_edge_embeddings.train.loss import (
    DualObjectiveLoss,
    InfoNCELoss,
    mean_pooling,
)
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

    # Checkpoint and follow-up campaign settings
    checkpoints_per_epoch: int = 4
    save_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # Distillation and alignment settings
    loss_type: str = "infonce"  # "infonce", "teacher_align", "dual"
    teacher_model_name_or_path: Optional[str] = "intfloat/multilingual-e5-small"
    weight_script: float = 1.0
    weight_preserve: float = 1.0
    weight_retrieval: float = 1.0
    english_preservation_texts: Optional[List[str]] = None


class RetrievalTrainer:
    """Trainer for embedding models with contrastive and teacher distillation losses."""

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
            "Initializing trainer on device %s (fp16=%s, batch_size=%d, grad_accum=%d, loss_type=%s)",
            self.device,
            self.fp16,
            config.batch_size,
            config.gradient_accumulation_steps,
            config.loss_type,
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

        # Loss function setup
        self.loss_type = config.loss_type
        if self.loss_type == "infonce":
            self.loss_fn = InfoNCELoss(temperature=config.temperature)
            self.dual_loss_fn = None
            self.collator = RetrievalCollator(
                tokenizer=self.tokenizer,
                query_prefix=config.query_prefix,
                passage_prefix=config.passage_prefix,
                max_length=config.max_length,
            )
            self.teacher_model = None
        else:
            # teacher_align or dual
            w_retrieval = config.weight_retrieval if self.loss_type == "dual" else 0.0
            self.dual_loss_fn = DualObjectiveLoss(
                weight_script=config.weight_script,
                weight_preserve=config.weight_preserve,
                weight_retrieval=w_retrieval,
                temperature=config.temperature,
            )
            self.loss_fn = None
            self.collator = PairedRetrievalCollator(
                tokenizer=self.tokenizer,
                query_prefix=config.query_prefix,
                passage_prefix=config.passage_prefix,
                max_length=config.max_length,
                english_preservation_texts=config.english_preservation_texts,
            )
            # Load frozen teacher model
            teacher_path = config.teacher_model_name_or_path or config.model_name_or_path
            logger.info("Loading frozen teacher model from %s...", teacher_path)
            self.teacher_model = AutoModel.from_pretrained(
                teacher_path, trust_remote_code=True
            )
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        """Set random seed across all frameworks for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _encode_batch(
        self,
        batch_tensors: Dict[str, torch.Tensor],
        model: Optional[PreTrainedModel] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer with mean pooling and L2 normalization."""
        target_model = model if model is not None else self.model
        inputs = {k: v.to(self.device) for k, v in batch_tensors.items()}
        outputs = target_model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        step: int,
        epoch: int,
        batch_in_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
    ) -> Path:
        """Save a fully resumable checkpoint including weights, optimizer, and RNG state."""
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)

        state = {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if self.fp16 else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(state, target / "training_state.pt")

        config_dict = asdict(self.config)
        # Save training configuration separately from the Hugging Face model config.json
        with open(target / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(
            "Saved resumable checkpoint to %s (step %d, epoch %d, batch %d)",
            str(target),
            step,
            epoch,
            batch_in_epoch,
        )
        return target

    def resume_from_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
    ) -> Tuple[int, int, int]:
        """Restore training state and weights from a saved checkpoint.

        Returns:
            Tuple of (global_step, start_epoch, resume_batch_in_epoch).
        """
        target = Path(checkpoint_dir)
        state_file = target / "training_state.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Training state not found at {state_file}")

        logger.info("Restoring model weights from %s...", str(target))
        self.model = AutoModel.from_pretrained(target, trust_remote_code=True).to(self.device)
        self.model.train()

        logger.info("Restoring training state from %s...", str(state_file))
        state = torch.load(state_file, map_location=self.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.fp16 and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])

        rng_state = state.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available() and rng_state["cuda"]:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

        step = int(state.get("step", 0))
        epoch = int(state.get("epoch", 0))
        batch_in_epoch = int(state.get("batch_in_epoch", 0))
        logger.info("Resumed from step %d, epoch %d, batch %d", step, epoch, batch_in_epoch)
        return step, epoch, batch_in_epoch

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
                    if self.loss_fn is not None:
                        loss = self.loss_fn(q_emb, p_emb, batch.get("positive_mask"), batch.get("valid_mask"))
                    else:
                        # For teacher alignment / dual loss, evaluation uses retrieval loss
                        eval_retrieval_fn = InfoNCELoss(temperature=self.config.temperature)
                        loss = eval_retrieval_fn(q_emb, p_emb, batch.get("positive_mask"), batch.get("valid_mask"))

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

        # Checkpoint cadence calculation (default quarter-epoch)
        if self.config.save_steps:
            checkpoint_step_interval = self.config.save_steps
        elif self.config.checkpoints_per_epoch > 0:
            checkpoint_step_interval = max(
                1, num_update_steps_per_epoch // self.config.checkpoints_per_epoch
            )
        else:
            checkpoint_step_interval = num_update_steps_per_epoch

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

        global_step = 0
        start_epoch = 0
        resume_batch = 0

        if self.config.resume_from_checkpoint:
            global_step, start_epoch, resume_batch = self.resume_from_checkpoint(
                self.config.resume_from_checkpoint, optimizer, scheduler, scaler
            )

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
        logger.info("  Checkpoint step interval = %d", checkpoint_step_interval)

        best_eval_loss = float("inf")
        train_logs: List[Dict[str, Any]] = []
        eval_logs: List[Dict[str, Any]] = []
        saved_checkpoints: List[str] = []

        start_time = time.time()
        self.model.train()

        for epoch in range(start_epoch, self.config.num_epochs):
            accum_loss = 0.0
            last_metrics: Dict[str, float] = {}
            current_resume_batch = resume_batch if epoch == start_epoch else 0

            for step, batch in enumerate(train_dataloader):
                if step < current_resume_batch:
                    continue
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.fp16
                ):
                    q_emb = self._encode_batch(batch["queries"])
                    p_emb = self._encode_batch(batch["passages"])

                    if self.loss_type == "infonce":
                        loss = self.loss_fn(q_emb, p_emb, batch.get("positive_mask"), batch.get("valid_mask"))
                        last_metrics = {"loss": loss.item()}
                    else:
                        # Teacher distillation pass
                        with torch.no_grad():
                            t_q_emb = self._encode_batch(
                                batch["teacher_queries"], model=self.teacher_model
                            )
                            t_p_emb = self._encode_batch(
                                batch["passages"], model=self.teacher_model
                            )
                            t_en_emb = (
                                self._encode_batch(
                                    batch["english_passages"], model=self.teacher_model
                                )
                                if "english_passages" in batch
                                else None
                            )

                        s_en_emb = (
                            self._encode_batch(batch["english_passages"])
                            if "english_passages" in batch
                            else None
                        )

                        roman_idx = batch["roman_indices"]
                        native_idx = batch["native_indices"]

                        if roman_idx.numel() > 0:
                            s_roman = q_emb[roman_idx]
                            t_native = t_q_emb[roman_idx]
                        else:
                            s_roman = torch.empty(0, q_emb.shape[1], device=self.device)
                            t_native = torch.empty(0, t_q_emb.shape[1], device=self.device)

                        preservation_groups = []
                        if native_idx.numel() > 0:
                            preservation_groups.append((q_emb[native_idx], t_q_emb[native_idx]))
                        preservation_groups.append((p_emb, t_p_emb))
                        if s_en_emb is not None and t_en_emb is not None:
                            preservation_groups.append((s_en_emb, t_en_emb))

                        loss, last_metrics = self.dual_loss_fn(
                            student_roman_emb=s_roman,
                            teacher_native_emb=t_native,
                            preservation_groups=preservation_groups,
                            student_query_emb=q_emb,
                            student_passage_emb=p_emb,
                            positive_mask=batch.get("positive_mask"),
                            valid_mask=batch.get("valid_mask"),
                        )

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

                    if global_step % self.config.logging_steps == 0:
                        lr_curr = scheduler.get_last_lr()[0]
                        metrics_str = " | ".join(
                            f"{k}: {v:.4f}" for k, v in last_metrics.items()
                        )
                        logger.info(
                            "Epoch %d/%d | Step %d/%d | Loss: %.4f | LR: %.2e | %s",
                            epoch + 1,
                            self.config.num_epochs,
                            global_step,
                            max_train_steps,
                            accum_loss,
                            lr_curr,
                            metrics_str,
                        )
                        train_logs.append(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "train_loss": accum_loss,
                                "lr": lr_curr,
                                **last_metrics,
                            }
                        )

                    # Validation loss evaluation
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

                    # Intermediate resumable checkpointing
                    if global_step % checkpoint_step_interval == 0:
                        ckpt_dir = self.output_dir / f"checkpoint-{global_step}"
                        self.save_checkpoint(
                            ckpt_dir,
                            step=global_step,
                            epoch=epoch,
                            batch_in_epoch=step + 1,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                        )
                        saved_checkpoints.append(str(ckpt_dir))

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
        final_dir = self.output_dir / "final_model"
        self.save_model(final_dir)
        saved_checkpoints.append(str(final_dir))

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
            ["Loss Type", self.config.loss_type],
            ["Training Examples", str(len(self.train_dataset))],
            ["Validation Examples", str(len(self.eval_dataset)) if self.eval_dataset else "N/A"],
            ["Epochs", str(self.config.num_epochs)],
            ["Per-Device Batch Size", str(self.config.batch_size)],
            ["Gradient Accumulation", str(self.config.gradient_accumulation_steps)],
            ["Effective Batch Size", str(self.config.batch_size * self.config.gradient_accumulation_steps)],
            ["Learning Rate", f"{self.config.learning_rate:.2e}"],
            ["Total Optimization Steps", str(global_step)],
            ["Saved Checkpoints", str(len(saved_checkpoints))],
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
            "saved_checkpoints": saved_checkpoints,
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
