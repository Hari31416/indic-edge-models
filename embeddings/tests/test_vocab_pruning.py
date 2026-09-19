"""Unit tests for vocabulary pruning and embedding compression."""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from indic_edge_embeddings.export.vocab_pruner import VocabPruner


def test_vocab_pruner_parity(tmp_path):
    """Verify that vocabulary pruning preserves exact numerical embeddings for target language probes."""
    model_name = "intfloat/multilingual-e5-small"

    pruner = VocabPruner(
        model_name_or_path=model_name,
        target_scripts=["devanagari"],
    )

    out_dir = tmp_path / "pruned_model"
    report = pruner.prune_and_save(output_dir=out_dir)

    # Check pruning metrics
    assert report.original_vocab_size == 250002
    assert report.pruned_vocab_size < report.original_vocab_size
    assert report.reduction_percent > 50.0
    assert report.validation_passed
    assert report.validation_mean_cosine >= 0.999

    # Load pruned model and tokenizer
    pruned_tok = AutoTokenizer.from_pretrained(str(out_dir))
    pruned_model = AutoModel.from_pretrained(str(out_dir))
    pruned_model.eval()

    orig_tok = AutoTokenizer.from_pretrained(model_name)
    orig_model = AutoModel.from_pretrained(model_name)
    orig_model.eval()

    probes = [
        "query: What is machine learning?",
        "passage: Machine learning algorithms build a model based on sample data.",
        "query: भारत का संविधान कब लागू हुआ था?",
        "passage: भारत का संविधान 26 जनवरी 1950 को लागू हुआ था।",
        "query: bharat ka samvidhan kab lagu hua tha?",
    ]

    with torch.no_grad():
        for probe in probes:
            orig_inputs = orig_tok(
                probe, return_tensors="pt", padding=True, truncation=True
            )
            pruned_inputs = pruned_tok(
                probe, return_tensors="pt", padding=True, truncation=True
            )

            orig_out = orig_model(**orig_inputs).last_hidden_state
            mask = (
                orig_inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(orig_out.size())
                .float()
            )
            orig_vec = torch.sum(orig_out * mask, 1) / torch.clamp(
                mask.sum(1), min=1e-9
            )
            orig_vec = torch.nn.functional.normalize(orig_vec, p=2, dim=1)

            pruned_out = pruned_model(**pruned_inputs).last_hidden_state
            p_mask = (
                pruned_inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(pruned_out.size())
                .float()
            )
            pruned_vec = torch.sum(pruned_out * p_mask, 1) / torch.clamp(
                p_mask.sum(1), min=1e-9
            )
            pruned_vec = torch.nn.functional.normalize(pruned_vec, p=2, dim=1)

            cos_sim = torch.cosine_similarity(orig_vec, pruned_vec, dim=1).item()
            assert cos_sim >= 0.999, f"Divergence on probe '{probe}': cos_sim={cos_sim}"
