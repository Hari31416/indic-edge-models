"""Parameter-isolated LoRA training engine for retrieval embedding models."""

import copy
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
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
    RetrievalCollator,
    RetrievalDataset,
)
from indic_edge_embeddings.train.loss import (
    InfoNCELoss,
    mean_pooling,
)
from indic_edge_embeddings.train.plots import (
    format_markdown_table,
    plot_loss_curves,
)

logger = logging.getLogger(__name__)


@dataclass
class LoraTrainingConfig:
    """Configuration for LoRA fine-tuning on retrieval data."""

    model_name_or_path: str = "intfloat/multilingual-e5-small"
    output_dir: str = "outputs/lora_adapted"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_length: int = 512
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    temperature: float = 0.05
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 250
    checkpoints_per_epoch: int = 1
    save_steps: Optional[int] = None

    # LoRA specific hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    architecture_mode: str = (
        "lora_symmetric"  # "lora_symmetric" or "lora_query_asymmetric"
    )


class LoraRetrievalTrainer:
    """Trainer for LoRA adaptation of retrieval embedding models."""

    def __init__(
        self,
        config: LoraTrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[Union[PreTrainedModel, PeftModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        """Initialize LoraRetrievalTrainer.

        Args:
            config: LoRA training configuration.
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
            "Initializing LoRA trainer on device %s (fp16=%s, batch_size=%d, grad_accum=%d, mode=%s, r=%d, alpha=%d)",
            self.device,
            self.fp16,
            config.batch_size,
            config.gradient_accumulation_steps,
            config.architecture_mode,
            config.lora_r,
            config.lora_alpha,
        )

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer

        if model is None:
            base_model = AutoModel.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_target_modules),
                lora_dropout=config.lora_dropout,
                bias="none",
            )
            self.model: PeftModel = get_peft_model(base_model, lora_config)
        else:
            self.model = model  # type: ignore

        self.model.to(self.device)

        # Log trainable parameters
        trainable_params, all_params = self._count_parameters()
        trainable_pct = 100 * trainable_params / max(1, all_params)
        logger.info(
            "LoRA Model Parameters: %s trainable || %s total || %.4f%% trainable",
            f"{trainable_params:,}",
            f"{all_params:,}",
            trainable_pct,
        )

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
        """Set random seed across all libraries for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters in the model.

        Returns:
            Tuple of (trainable_param_count, total_param_count).
        """
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, all_params

    def _encode_batch(
        self,
        batch_tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through transformer with mean pooling and L2 normalization."""
        inputs = {k: v.to(self.device) for k, v in batch_tensors.items()}
        outputs = self.model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def evaluate(
        self, eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Compute validation loss over the evaluation dataset.

        Args:
            eval_dataloader: Optional pre-constructed DataLoader.

        Returns:
            Dictionary containing 'eval_loss'.
        """
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

                    if self.config.architecture_mode == "lora_query_asymmetric":
                        with self.model.disable_adapter():
                            p_emb = self._encode_batch(batch["passages"])
                    else:
                        p_emb = self._encode_batch(batch["passages"])

                    loss = self.loss_fn(
                        q_emb,
                        p_emb,
                        batch.get("positive_mask"),
                        batch.get("valid_mask"),
                    )

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        self.model.train()
        return {"eval_loss": avg_loss}

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
        """Save LoRA adapter checkpoint, optimizer state, and configuration."""
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights
        adapter_dir = target / "adapter"
        self.model.save_pretrained(adapter_dir)
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
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        torch.save(state, target / "training_state.pt")

        config_dict = asdict(self.config)
        with open(target / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(
            "Saved LoRA checkpoint to %s (step %d, epoch %d, batch %d)",
            str(target),
            step,
            epoch,
            batch_in_epoch,
        )
        return target

    def save_model(self, save_dir: Union[str, Path]) -> None:
        """Save both LoRA adapter and merged standalone model weights."""
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)

        # 1. Save adapter weights
        adapter_dir = target / "adapter"
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

        # 2. Save merged standalone model weights
        logger.info(
            "Merging LoRA weights into base model for export to %s...", str(target)
        )
        merged_model = copy.deepcopy(self.model).merge_and_unload()
        merged_model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)

        logger.info("Saved LoRA adapter and merged model to %s", str(target))

    def train(self) -> Dict[str, Any]:
        """Execute full LoRA training loop with evaluation and checkpointing."""
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

        if self.config.save_steps:
            checkpoint_step_interval = self.config.save_steps
        elif self.config.checkpoints_per_epoch > 0:
            checkpoint_step_interval = max(
                1, num_update_steps_per_epoch // self.config.checkpoints_per_epoch
            )
        else:
            checkpoint_step_interval = num_update_steps_per_epoch

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

        global_step = 0
        trainable_count, total_count = self._count_parameters()

        logger.info("***** Running LoRA Training *****")
        logger.info("  Architecture mode = %s", self.config.architecture_mode)
        logger.info(
            "  LoRA Rank (r) = %d, Alpha = %d",
            self.config.lora_r,
            self.config.lora_alpha,
        )
        logger.info(
            "  Trainable params = %d (%.2f%%)",
            trainable_count,
            100 * trainable_count / total_count,
        )
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.config.num_epochs)
        logger.info("  Batch size per device = %d", self.config.batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d (Effective batch size = %d)",
            self.config.gradient_accumulation_steps,
            self.config.batch_size * self.config.gradient_accumulation_steps,
        )
        logger.info("  Total optimization steps = %d", max_train_steps)
        logger.info("  Warmup steps = %d", warmup_steps)
        logger.info("  Learning rate = %.2e", self.config.learning_rate)

        best_eval_loss = float("inf")
        train_logs: List[Dict[str, Any]] = []
        eval_logs: List[Dict[str, Any]] = []
        saved_checkpoints: List[str] = []

        start_time = time.time()
        self.model.train()

        for epoch in range(self.config.num_epochs):
            accum_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.fp16
                ):
                    q_emb = self._encode_batch(batch["queries"])

                    if self.config.architecture_mode == "lora_query_asymmetric":
                        # Keep passage representations 100% frozen in base pretrained space
                        with self.model.disable_adapter():
                            p_emb = self._encode_batch(batch["passages"])
                    else:
                        # Symmetric LoRA: queries and passages share LoRA representation
                        p_emb = self._encode_batch(batch["passages"])

                    loss = self.loss_fn(
                        q_emb,
                        p_emb,
                        batch.get("positive_mask"),
                        batch.get("valid_mask"),
                    )
                    loss = loss / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                    step + 1 == len(train_dataloader)
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

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

                    # Periodic validation evaluation
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
                logger.info("End of Epoch %d Eval Loss: %.4f", epoch + 1, eval_loss)
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
            "LoRA Training completed in %.2f seconds (%.1f examples/s). Best eval loss: %.4f",
            total_time,
            throughput,
            best_eval_loss,
        )

        # Save final model
        final_dir = self.output_dir / "final_model"
        self.save_model(final_dir)
        saved_checkpoints.append(str(final_dir))

        # Generate loss curves
        try:
            plot_loss_curves(train_logs, eval_logs, self.output_dir / "loss_curve.png")
        except Exception as e:
            logger.warning("Failed to generate loss curve plot: %s", e)

        # Format markdown report
        headers = ["Parameter / Metric", "Value"]
        rows = [
            ["Base Model", self.config.model_name_or_path],
            ["Architecture Mode", self.config.architecture_mode],
            ["LoRA Rank (r)", str(self.config.lora_r)],
            ["LoRA Alpha (alpha)", str(self.config.lora_alpha)],
            ["LoRA Dropout", str(self.config.lora_dropout)],
            ["Target Modules", ", ".join(self.config.lora_target_modules)],
            [
                "Trainable Parameters",
                f"{trainable_count:,} ({100 * trainable_count / total_count:.2f}%)",
            ],
            ["Total Parameters", f"{total_count:,}"],
            ["Training Examples", str(len(self.train_dataset))],
            [
                "Validation Examples",
                str(len(self.eval_dataset)) if self.eval_dataset else "N/A",
            ],
            ["Epochs", str(self.config.num_epochs)],
            ["Per-Device Batch Size", str(self.config.batch_size)],
            ["Gradient Accumulation", str(self.config.gradient_accumulation_steps)],
            [
                "Effective Batch Size",
                str(self.config.batch_size * self.config.gradient_accumulation_steps),
            ],
            ["Learning Rate", f"{self.config.learning_rate:.2e}"],
            ["Total Optimization Steps", str(global_step)],
            ["Training Duration (s)", f"{total_time:.2f}"],
            ["Throughput (examples/s)", f"{throughput:.1f}"],
            [
                "Best Validation Loss",
                f"{best_eval_loss:.4f}" if best_eval_loss < float("inf") else "N/A",
            ],
        ]
        report_md = (
            "# LoRA Training Summary Report\n\n"
            + format_markdown_table(headers, rows)
            + "\n"
        )
        with open(self.output_dir / "training_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        # Save training summary JSON
        summary = {
            "config": asdict(self.config),
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
            "trainable_percent": 100 * trainable_count / total_count,
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
