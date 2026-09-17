"""Command-line interface for embedding model training."""

import argparse
import logging
import sys
from pathlib import Path

from indic_edge_embeddings.train.dataset import RetrievalDataset
from indic_edge_embeddings.train.trainer import RetrievalTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_cli")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Indic embedding models on retrieval data"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training data file (.jsonl or .parquet)",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation data file (.jsonl or .parquet)",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Base model checkpoint or Hugging Face model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pilot_e5_adapted",
        help="Directory to save trained model artifacts and checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per device (default: 32)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Total number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup proportion of total training steps (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Softmax temperature for InfoNCE loss (default: 0.05)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit on training samples for smoke testing",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit on validation samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Logging interval in optimization steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Validation evaluation interval in optimization steps",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 mixed precision",
    )
    parser.add_argument(
        "--checkpoints-per-epoch",
        type=int,
        default=4,
        help="Number of intermediate checkpoints to save per epoch (default: 4 for quarter-epoch)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Optional exact interval in steps to save intermediate checkpoints",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Directory of checkpoint to resume training from",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="infonce",
        choices=["infonce", "teacher_align", "dual"],
        help="Loss objective: 'infonce' (contrastive), 'teacher_align' (L_script + L_preserve), 'dual' (align + preserve + retrieval)",
    )
    parser.add_argument(
        "--teacher-model-name-or-path",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Teacher model checkpoint or Hugging Face ID for distillation/alignment",
    )
    parser.add_argument(
        "--weight-script",
        type=float,
        default=1.0,
        help="Weight for script alignment loss L_script (default: 1.0)",
    )
    parser.add_argument(
        "--weight-preserve",
        type=float,
        default=1.0,
        help="Weight for representation preservation loss L_preserve (default: 1.0)",
    )
    parser.add_argument(
        "--weight-retrieval",
        type=float,
        default=1.0,
        help="Weight for retrieval loss in dual mode (default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint for embedding model training."""
    args = parse_args()

    config = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        fp16=not args.no_fp16,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        checkpoints_per_epoch=args.checkpoints_per_epoch,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        loss_type=args.loss_type,
        teacher_model_name_or_path=args.teacher_model_name_or_path,
        weight_script=args.weight_script,
        weight_preserve=args.weight_preserve,
        weight_retrieval=args.weight_retrieval,
    )

    logger.info("Loading training dataset from %s...", args.train_file)
    train_dataset = RetrievalDataset(
        args.train_file, max_samples=args.max_train_samples
    )

    eval_dataset = None
    if args.val_file:
        logger.info("Loading validation dataset from %s...", args.val_file)
        eval_dataset = RetrievalDataset(args.val_file, max_samples=args.max_val_samples)

    trainer = RetrievalTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    logger.info("Training process completed successfully.")


if __name__ == "__main__":
    main()
