# Pair-specific Indic browser translation

Build a compact translator for one selected language pair, starting with one direction, that runs entirely inside a browser tab.

## Status and priority

Deferred until the [embedding project](../embeddings/README.md) is complete. This folder contains planning documents only. Translation is independent of the reranking model; their order after embeddings remains open.

No pair, direction, base model, training recipe, or download budget has been finalized. English to Hindi is a provisional example, not a committed choice. Supporting the reverse direction is a separate extension.

Start with [STEPS.md](STEPS.md) when this project becomes active, and revise it using baseline results.

Agent workflow is in [AGENTS.md](AGENTS.md). Run outputs and resulting decisions belong in [experiments/](experiments/README.md).

## Goal and constraints

Deliver better translation at a comparable browser cost, or comparable translation with a substantially smaller download, memory footprint, or latency.

- Solo developer with a Mac M4 Pro and intermittent access to two 16 GB T4s.
- Existing checkpoints and public parallel text; no private corpus collection.
- Begin with short sentences and paragraphs in one direction.
- All translation and preprocessing inference must run locally in the browser.
- Model quality and reproducible evaluation come before a larger application.

The earlier sub-100 MB target is a hypothesis to investigate. Fine-tuning alone does not shrink a model, and a model's parameter count does not establish its complete browser bundle size.

## Existing work to reuse

- [IndicTrans2 export repository](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/README.md)
- [Export fixes and parity checks](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/EXPORT_ISSUES.md)
- [Size optimization analysis](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/ONNX_SIZE_OPTIMIZATION.md)
- [Browser demo](/Users/hari/Desktop/sandbox/indictrans2-onnx-browser-demo/README.md)

Reuse validated tokenizer mappings, wrappers, and fixtures where appropriate. Reconcile the existing browser benchmark document's conflicting status claims against raw runs before using its numbers as baselines. Small parity fixture sets establish implementation agreement, not translation quality.

## Model candidates

Hugging Face sources checked on 15 September 2026. Recheck availability, revisions, terms, and runtime compatibility when work starts. These are baseline candidates, not measured recommendations for a selected pair.

| Candidate | Initial role | Qualification |
| --- | --- | --- |
| [Helsinki-NLP/opus-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi) | English to Hindi baseline and possible fine-tuning base | Marian architecture; card lists Apache 2.0. |
| [Xenova/opus-mt-en-hi](https://huggingface.co/Xenova/opus-mt-en-hi/tree/main/onnx) | Existing browser baseline | Listed INT8 encoder plus merged decoder total about 110 MB before tokenizer/runtime assets. Count only the files the chosen runtime actually loads. |
| [Helsinki-NLP/opus-mt-hi-en](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) | Baseline if Hindi to English is selected | Separate checkpoint and evaluation from the forward direction; card lists Apache 2.0. |
| [ai4bharat/indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) | English to Indic quality reference; possible compression base or teacher | Reuse existing export experience. Selecting one output language does not remove unused weights automatically. |
| [ai4bharat/indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) | Indic to English reference | Use the direction-specific tokenizer, language tags, and preprocessing. |
| [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en) | Alternative baseline if Bengali to English is selected | Example of a different pair to investigate; not an additional initial deliverable. |

A stronger teacher is optional and must earn its compute cost in evaluation. If building a smaller student, choose a supported encoder-decoder architecture with adequate source and target token coverage. An English T5-small checkpoint is not a ready Indic translator.

## Public data candidates

| Source | Use | Qualification |
| --- | --- | --- |
| [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | Main source of training pairs | Inspect the selected language and component, prioritizing suitable human-translated seed/Wiki/Daily data before adding mined text. Record exact subset names and terms. |
| [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) | Optional mined parallel data | Overlaps BPCC sources; do not concatenate blindly or count duplicates as new data. |
| [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) | Held-out general-domain evaluation | Keep out of training and teacher-generated training targets. |
| [IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) | Held-out conversational evaluation | Report separately from general-domain text. |

Use a disjoint slice of training sources for development. Public availability and a dataset-level license label do not settle every component's provenance or redistribution terms.

## Expected release

- A direction-specific checkpoint, tokenizer, browser bundle, and usage example.
- Reproducible training, data selection, decoding, and evaluation configurations.
- Translation quality, meaning-preservation checks, and browser measurements by input length.
- A browser demo and model card with scope, provenance, limitations, and measured device support.

Add implementation and experiment records here later. Keep downloaded corpora, caches, and large checkpoints separate from tracked source files.

## Background

- [Project context](../context.md)
- [Validation and options](../VALIDATION_AND_OPTIONS.md)
- [Astra suggestions](../ASTRA_SUGGESTIONS.md)
- [Grok suggestions](../GROK_SUGGESTIONS.md)
