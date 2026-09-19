"""Vocabulary pruning for compressing multilingual embedding models.

Prunes the XLM-RoBERTa / SentencePiece Unigram embedding matrix and tokenizer
to retain only tokens required for target languages (e.g., Devanagari Hindi,
Romanized Hindi, English, numbers, punctuation, and byte-level fallbacks),
reducing INT8 model footprint by 75-85% (e.g., 114 MB -> ~27 MB).
"""

import copy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger("vocab_pruner")

# Unicode ranges for major Indic scripts
UNICODE_RANGES: Dict[str, Tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

# Special Unicode code points to always retain (ZWJ, ZWNJ, Danda, Double Danda, NBSP, Rupee symbol)
SPECIAL_CODEPOINTS: Set[int] = {
    0x200C,  # ZWNJ (Zero Width Non-Joiner)
    0x200D,  # ZWJ (Zero Width Joiner)
    0x0964,  # Devanagari Danda (।)
    0x0965,  # Devanagari Double Danda (॥)
    0x00A0,  # Non-breaking space
    0x20B9,  # Indian Rupee Sign (₹)
    0x2581,  # Lower One Eighth Block (SentencePiece space symbol  )
}

BYTE_TOKEN_PATTERN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


@dataclass
class PruneReport:
    """Summary statistics from vocabulary pruning."""

    original_vocab_size: int
    pruned_vocab_size: int
    reduction_percent: float
    original_embedding_params: int
    pruned_embedding_params: int
    estimated_fp32_saving_mb: float
    estimated_int8_saving_mb: float
    validation_mean_cosine: float
    validation_passed: bool


class VocabPruner:
    """Prunes tokenizer and embedding matrix for XLM-RoBERTa architectures."""

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        target_scripts: Optional[List[str]] = None,
        include_latin_extended: bool = False,
    ) -> None:
        """Initialize pruner with a model path and target scripts.

        Args:
            model_name_or_path: Path or Hugging Face model ID.
            target_scripts: List of script names to keep (default: ['devanagari']).
            include_latin_extended: Whether to retain accented European Latin characters.
        """
        self.model_path = Path(model_name_or_path)
        self.target_scripts = target_scripts or ["devanagari"]
        self.include_latin_extended = include_latin_extended

        logger.info(
            "Loading model and tokenizer for pruning from: %s", str(self.model_path)
        )
        resolved_path = (
            str(self.model_path.resolve())
            if self.model_path.exists()
            else str(self.model_path)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
        self.orig_model = AutoModel.from_pretrained(
            resolved_path, trust_remote_code=True
        )
        self.orig_model.eval()

    def _is_character_allowed(self, char: str) -> bool:
        """Check if a single character belongs to allowed alphabets or scripts."""
        code = ord(char)
        if code in SPECIAL_CODEPOINTS:
            return True
        # Basic ASCII (controls, Latin letters A-Z, a-z, digits 0-9, standard punctuation)
        if code < 128:
            return True
        if self.include_latin_extended:
            if 0x00A0 <= code <= 0x024F:
                return True
            if 0x2000 <= code <= 0x206F:
                return True
        # Target Indic scripts
        for script_name in self.target_scripts:
            if script_name in UNICODE_RANGES:
                start, end = UNICODE_RANGES[script_name]
                if start <= code <= end:
                    return True
        return False

    def _is_token_allowed(self, token_str: str) -> bool:
        """Check if all characters in a token string belong to allowed scripts."""
        if BYTE_TOKEN_PATTERN.match(token_str):
            return True
        for char in token_str:
            if not self._is_character_allowed(char):
                return False
        return True

    def find_kept_token_indices(
        self,
        corpus_texts: Optional[List[str]] = None,
    ) -> List[int]:
        """Find the list of token indices from the original vocab to keep.

        Args:
            corpus_texts: Optional list of raw texts to guarantee token coverage.

        Returns:
            Sorted list of integer token indices to preserve.
        """
        kept_set: Set[int] = set()

        # Always preserve all special tokens
        for sp_id in self.tokenizer.all_special_ids:
            kept_set.add(sp_id)

        # Always preserve byte-fallback tokens and allowed script tokens
        vocab_dict = self.tokenizer.get_vocab()
        for token_str, token_id in vocab_dict.items():
            if self._is_token_allowed(token_str):
                kept_set.add(token_id)

        # Include all tokens activated by corpus texts
        if corpus_texts:
            logger.info(
                "Tokenizing %d corpus texts to preserve activated tokens...",
                len(corpus_texts),
            )
            for text in corpus_texts:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                for tid in token_ids:
                    kept_set.add(tid)

        kept_list = sorted(list(kept_set))
        logger.info(
            "Identified %d tokens to keep out of %d (%.2f%% reduction)",
            len(kept_list),
            len(vocab_dict),
            100.0 * (1.0 - len(kept_list) / len(vocab_dict)),
        )
        return kept_list

    def prune_and_save(
        self,
        output_dir: Union[str, Path],
        corpus_texts: Optional[List[str]] = None,
    ) -> PruneReport:
        """Prune model embeddings and tokenizer, saving the result to output_dir.

        Args:
            output_dir: Output directory for pruned model and tokenizer.
            corpus_texts: Optional list of domain texts to include in token preservation.

        Returns:
            PruneReport with size and validation metrics.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        kept_indices = self.find_kept_token_indices(corpus_texts)
        old_vocab_size = len(self.tokenizer)
        new_vocab_size = len(kept_indices)

        # 1. Map old token IDs to new token IDs
        old_to_new: Dict[int, int] = {
            old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)
        }

        # 2. Extract and prune word embeddings weight matrix
        old_embed = self.orig_model.embeddings.word_embeddings.weight.data
        hidden_size = old_embed.shape[1]
        new_embed = old_embed[kept_indices, :].clone()

        logger.info(
            "Pruning embedding matrix: %s -> %s",
            list(old_embed.shape),
            list(new_embed.shape),
        )

        # 3. Create pruned model copy and update embedding layer
        pruned_config = copy.deepcopy(self.config)
        pruned_config.vocab_size = new_vocab_size

        pruned_model = copy.deepcopy(self.orig_model)
        pruned_model.config.vocab_size = new_vocab_size

        new_embed_layer = torch.nn.Embedding(
            new_vocab_size,
            hidden_size,
            padding_idx=old_to_new.get(self.tokenizer.pad_token_id, 1),
        )
        new_embed_layer.weight.data = new_embed
        pruned_model.embeddings.word_embeddings = new_embed_layer

        # 4. Construct pruned tokenizer
        tok_json_path = self.model_path / "tokenizer.json"
        if not tok_json_path.exists():
            self.tokenizer.save_pretrained(out_path)
            tok_json_path = out_path / "tokenizer.json"

        with open(tok_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)

        if "model" in tok_data and "vocab" in tok_data["model"]:
            orig_vocab = tok_data["model"]["vocab"]
            pruned_vocab = [orig_vocab[idx] for idx in kept_indices]
            tok_data["model"]["vocab"] = pruned_vocab

        # Update added_tokens mappings if any
        if "added_tokens" in tok_data:
            new_added_tokens = []
            for item in tok_data["added_tokens"]:
                old_id = item.get("id")
                if old_id in old_to_new:
                    item_copy = dict(item)
                    item_copy["id"] = old_to_new[old_id]
                    new_added_tokens.append(item_copy)
            tok_data["added_tokens"] = new_added_tokens

        # Save pruned model and tokenizer configs
        pruned_model.save_pretrained(out_path)
        self.tokenizer.save_pretrained(out_path)

        # Re-write tokenizer.json with pruned vocab
        pruned_tok_json = out_path / "tokenizer.json"
        with open(pruned_tok_json, "w", encoding="utf-8") as f:
            json.dump(tok_data, f, indent=2, ensure_ascii=False)

        # Update tokenizer_config.json
        tok_cfg_path = out_path / "tokenizer_config.json"
        if tok_cfg_path.exists():
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            tok_cfg["vocab_size"] = new_vocab_size
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(tok_cfg, f, indent=2)

        # 5. Numerical Parity Validation on Probe Sentences
        validation_mean_cos, val_passed = self._validate_pruned_model(
            out_path, old_to_new
        )

        orig_params = old_vocab_size * hidden_size
        pruned_params = new_vocab_size * hidden_size
        fp32_saving_mb = (orig_params - pruned_params) * 4 / (1024 * 1024)
        int8_saving_mb = (orig_params - pruned_params) * 1 / (1024 * 1024)

        report = PruneReport(
            original_vocab_size=old_vocab_size,
            pruned_vocab_size=new_vocab_size,
            reduction_percent=100.0 * (1.0 - new_vocab_size / old_vocab_size),
            original_embedding_params=orig_params,
            pruned_embedding_params=pruned_params,
            estimated_fp32_saving_mb=fp32_saving_mb,
            estimated_int8_saving_mb=int8_saving_mb,
            validation_mean_cosine=validation_mean_cos,
            validation_passed=val_passed,
        )

        logger.info(
            "Pruning Report: %d -> %d tokens (%.1f%% reduction). Saved ~%.1f MB (INT8). Parity: cos=%.6f (pass=%s)",
            report.original_vocab_size,
            report.pruned_vocab_size,
            report.reduction_percent,
            report.estimated_int8_saving_mb,
            report.validation_mean_cosine,
            report.validation_passed,
        )

        # Save pruning summary JSON
        with open(out_path / "prune_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "original_vocab_size": report.original_vocab_size,
                    "pruned_vocab_size": report.pruned_vocab_size,
                    "reduction_percent": report.reduction_percent,
                    "estimated_fp32_saving_mb": report.estimated_fp32_saving_mb,
                    "estimated_int8_saving_mb": report.estimated_int8_saving_mb,
                    "validation_mean_cosine": report.validation_mean_cosine,
                    "validation_passed": report.validation_passed,
                },
                f,
                indent=2,
            )

        return report

    def _validate_pruned_model(
        self,
        pruned_model_dir: Path,
        old_to_new: Dict[int, int],
    ) -> Tuple[float, bool]:
        """Verify that pruned model embeddings match original model embeddings."""
        probe_texts = [
            "query: What causes rainfall in the monsoon season?",
            "passage: Monsoon winds carry moisture from the Indian Ocean.",
            "query: भारत की राजधानी क्या है?",
            "passage: नई दिल्ली भारत की राजधानी और केंद्र शासित प्रदेश है।",
            "query: bharat ki rajdhani kya hai aur kab bani?",
            "passage: nai delhi bharat ki rajdhani hai jise 1911 me ghoshit kiya gaya tha.",
            "query: Machine learning models running in client browsers on edge devices.",
        ]

        pruned_tok = AutoTokenizer.from_pretrained(
            str(pruned_model_dir.resolve()), trust_remote_code=True
        )
        pruned_mod = AutoModel.from_pretrained(
            str(pruned_model_dir.resolve()), trust_remote_code=True
        )
        pruned_mod.eval()

        cos_sims: List[float] = []

        with torch.no_grad():
            for text in probe_texts:
                orig_enc = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                pruned_enc = pruned_tok(
                    text, return_tensors="pt", padding=True, truncation=True
                )

                # Check token IDs map 1:1
                orig_ids = orig_enc["input_ids"][0].tolist()
                expected_new_ids = [old_to_new[oid] for oid in orig_ids]
                actual_new_ids = pruned_enc["input_ids"][0].tolist()

                if expected_new_ids != actual_new_ids:
                    logger.warning(
                        "Tokenization divergence on '%s': expected %s, got %s",
                        text,
                        expected_new_ids,
                        actual_new_ids,
                    )

                orig_out = self.orig_model(**orig_enc).last_hidden_state
                # Mean pool
                mask = (
                    orig_enc["attention_mask"]
                    .unsqueeze(-1)
                    .expand(orig_out.size())
                    .float()
                )
                orig_vec = torch.sum(orig_out * mask, 1) / torch.clamp(
                    mask.sum(1), min=1e-9
                )
                orig_vec = torch.nn.functional.normalize(orig_vec, p=2, dim=1)

                pruned_out = pruned_mod(**pruned_enc).last_hidden_state
                p_mask = (
                    pruned_enc["attention_mask"]
                    .unsqueeze(-1)
                    .expand(pruned_out.size())
                    .float()
                )
                pruned_vec = torch.sum(pruned_out * p_mask, 1) / torch.clamp(
                    p_mask.sum(1), min=1e-9
                )
                pruned_vec = torch.nn.functional.normalize(pruned_vec, p=2, dim=1)

                cos = torch.cosine_similarity(orig_vec, pruned_vec, dim=1).item()
                cos_sims.append(cos)

        mean_cos = sum(cos_sims) / len(cos_sims)
        passed = mean_cos >= 0.999
        return mean_cos, passed


def main() -> None:
    """Parse CLI arguments and run vocabulary pruning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune XLM-RoBERTa vocabulary and embedding table"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path to Hugging Face model or merged checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for pruned model and tokenizer",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=["devanagari"],
        help="Target Indic scripts to retain (e.g. devanagari bengali tamil)",
    )
    parser.add_argument(
        "--include-latin-extended",
        action="store_true",
        help="Retain Latin extended / European accented characters",
    )

    args = parser.parse_args()

    pruner = VocabPruner(
        model_name_or_path=args.model_name,
        target_scripts=args.scripts,
        include_latin_extended=args.include_latin_extended,
    )
    report = pruner.prune_and_save(output_dir=args.output_dir)
    logger.info("Vocabulary pruning complete: %s", str(report))


if __name__ == "__main__":
    main()
