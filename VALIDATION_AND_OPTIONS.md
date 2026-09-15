# Indic edge models: validation and options

Research date: 15 September 2026.

## Recommendation

Keep the embedder first and make the reranker conditional. Run a small comparison before choosing E5 as the training base. Test two opportunities separately: improved retrieval for Romanized queries, and similar retrieval quality in a substantially smaller language-specific browser download.

Astra is the better starting plan. Grok identifies useful failure conditions, but treats several hypotheses as established facts and understates both your existing translation work and the alternatives.

This is a source and feasibility assessment. I inspected local documentation and current primary sources; I did not train models, rerun your exports, or independently reproduce model-card scores. Suggested budgets and thresholds below are experiment choices, not measured results.

## What you want and already have

Your context specifies a model-first project, fully local browser inference, public training data, a solo developer, an M4 Pro, and intermittent access to two 16 GB T4s. It excludes a new foundation LLM, private-corpus verticals, speech, and a cloud API as the main release. Nothing requires an 80 MB ceiling or mobile-first deployment; those were suggestions. Mac RAM, target browser/device, languages you can inspect, and intended release terms remain unspecified.

The related repositories contain useful prior work:

- [Export issues and fixes](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/EXPORT_ISSUES.md) documents successful 200M/320M exports, manual ONNX wrappers, tokenizer ID remapping, and small parity fixture sets.
- [Browser demo](/Users/hari/Desktop/sandbox/indictrans2-onnx-browser-demo/README.md) documents a client-side translation implementation and published model collections.
- [Size optimization analysis](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/ONNX_SIZE_OPTIMIZATION.md) already investigates duplicated tied weights, decoder duplication, tokenizer payloads, and direction-specific vocabulary pruning.
- [Browser benchmarks](/Users/hari/Desktop/sandbox/indictrans2-onnx-export/LIVE_BENCHMARKS.md) contains conflicts: its prose says INT8 WebGPU, base FP32 WASM, and 1B FP32 WebGPU configurations were excluded, while its table marks those configurations completed. Those results need reconciliation against raw runs before reuse.

Thus, Grok's description of another failed 1B export is unsupported. You have substantial implementation work. Fixture parity establishes agreement on tested examples, not translation quality or universal browser compatibility.

## Validation of the two suggestions

| Proposal or claim                                         | Assessment                 | Consequence                                                                                          |
| --------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| Fine-tune an existing small retriever                     | Sound                      | Establish an unchanged, exported baseline first.                                                     |
| Romanized and mixed-script retrieval is promising         | Plausible, unproven        | Measure the gap; do not assume every current model fails.                                            |
| Only Romanized specialization can win                     | Too strong                 | Compression, latency, and specific cross-language directions can also matter.                        |
| Start with English and Hindi                              | Sensible provisional scope | Prefer a language you can inspect; add a second script after a useful result.                        |
| Grok's 80 MB budget alongside its E5-small recommendation | Unresolved size mismatch   | E5's published INT8 ONNX alone is about 118 MB; the budget needs measurement and a compression plan. |
| Reduce MiniLM layers to get a tiny download               | Incomplete                 | Vocabulary can dominate storage even when transformer compute is small.                              |
| Aksharantar and BPCC supply retrieval training            | Only indirectly            | Transliteration and translation pairs provide alignment, not search relevance.                       |
| Dakshina is gold retrieval evaluation                     | Incorrect                  | It supplies human Romanizations, not query-to-passage relevance judgments.                           |
| Use MIRACL as independent evaluation                      | Needs qualification        | Its held-out split is useful, but E5 trained on its training split.                                  |
| Add a reranker after examining top-20 recall              | Sound                      | It only helps when relevant passages enter the candidate set.                                        |
| A 40–80 MB translator can rival IndicTrans2               | An ambitious target        | Requires actual compression/distillation and evaluation, not ordinary fine-tuning.                   |
| There is no credible third project                        | Unsupported                | Code-mixed tagging and bounded OCR are feasible alternatives.                                        |

### Browser size changes the training choice

[E5's published ONNX files](https://huggingface.co/Xenova/multilingual-e5-small/tree/main/onnx) list approximately 118 MB for INT8 and 399 MB for the Q4 variant. A quantization label does not guarantee a smaller total artifact. Count all files actually downloaded, including the tokenizer and runtime.

[E5's configuration](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/config.json) has 250,037 vocabulary entries and width 384. Multiplication gives about 96 million token-embedding parameters, roughly 82% of the approximately 118M total. This is a derived storage estimate, not a benchmark. Dropping half its transformer layers leaves most of that table intact.

A reduced vocabulary is therefore worth testing. Illustratively, 50,000 retained rows at width 384 require 19.2M weights, plus roughly 22M other weights if the rest of E5 stays unchanged. That suggests a much smaller INT8 weight budget. It does not prove 50,000 tokens cover your languages adequately or establish the final ONNX size.

Start by retaining existing token rows and remapping IDs. Preserve special tokens, English coverage, digits, punctuation, and representative names. Measure changed tokenization, unknowns, sequence inflation, and retrieval. A new tokenizer changes segmentation and needs more adaptation. Good native-script coverage can still fail on English names and informal Roman spelling.

### Evaluation is the main risk

[E5's training disclosure](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md) explicitly includes MIRACL and Mr. TyDi training splits. Their held-out data remains useful for comparisons, but it is in-distribution evaluation, not evidence of zero-shot generalization. Training-set exposure is not itself test-set leakage.

[Dakshina](https://github.com/google-research-datasets/dakshina) gives attested word spellings and sentence Romanizations. Use it for script robustness and transliteration evaluation. Retrieving the original sentence from its Romanization is an alignment task; it is easier and different from answering a short search query.

Transliterating held-out retrieval queries preserves an existing search task, provided meaning survives, but creates a synthetic query distribution. Pure Romanized Hindi, Hindi-English code-mixing, and mixed Unicode scripts are different test conditions. A model can improve one while regressing on another.

There is some natural code-mixed retrieval work. The [CMIR-2025 system paper](https://ceur-ws.org/Vol-4173/T3-3.pdf) describes English-Bengali retrieval with only 20 unique queries. That is evidence that the task exists, but too small and language-specific to settle broad Hinglish claims.

Your no-private-dataset constraint still permits reproducible selection and transformation of public data. It does not remove the work of checking examples. If you also want zero manual relevance labeling, keep the first claim narrow: improved performance on named public and synthetic benchmarks. Natural Hinglish search superiority remains unproven without appropriate public relevance judgments or a later annotation effort.

### The transliteration baseline requires real local inference

[IndicXlit](https://github.com/AI4Bharat/IndicXlit) is a roughly 11M word-level model. It is a good quality baseline, but per-word inference and dictionaries can add latency and payload. Its linked [React component](https://github.com/AI4Bharat/indic-transliterate-js) explicitly uses an API. That package alone does not satisfy your all-in-browser requirement.

For early quality experiments, run IndicXlit locally in Python. Before treating it as the shipping fallback, verify a local browser implementation and measure the whole pipeline. Preserve the original query and English tokens; blindly transliterating every Latin token is a weak code-mixed baseline.

### Reranking remains a conditional second model

Test a retrieval-trained small cross-encoder such as [mMARCO mMiniLMv2](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1), alongside the larger teachers named in the suggestions. Export support, actual package size, and target-language quality still need measurement.

Score top-5, top-10, and top-20 candidates. Report incremental latency and combined download/memory. If relevant results are missing entirely, improve candidate retrieval or test lexical+dense fusion first. If a cheap transliteration pipeline wins, that does not automatically justify spending the remaining GPU budget on a reranker.

## Current work that changes the baseline list

### Direct retrieval competitors

1. [IBM Granite embedding 97M multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2). Released in 2026; explicit retrieval training includes Hindi, Bengali, Marathi, Telugu, and Urdu. Its [Transformers.js ONNX export](https://huggingface.co/onnx-community/granite-embedding-97m-multilingual-r2-ONNX/tree/main/onnx) lists a 97.9 MB INT8 graph. Add this alongside E5 before choosing the student.
2. [Bekko embedding v1 a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m). The card provides browser instructions and public training resources. The name counts active transformer parameters; the full model is about 106M parameters. It is a relevant throughput comparator, not an 8 MB download. Its aggregate scores are author-reported, not evidence of Hindi or Romanized superiority.
3. [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m). Keep as a larger on-device reference. Its card supports reduced embedding dimensions and lists Gemma terms, rather than an MIT license. Smaller output vectors reduce index storage, not automatically model storage.
4. [Jina v5 text nano](https://jina.ai/en-US/models/jina-embeddings-v5-text-nano/). Astra's 239M figure and noncommercial weight-license description are supported. It is useful as a reference, but not interchangeable with a permissively licensed release base.

Prioritize the first two additions and E5 for the pilot. Running every available model would consume time without resolving the core question.

### Indic-specific development

[IndicBERT-v3-270M](https://huggingface.co/ai4bharat/IndicBERT-v3-270M) is a newer bidirectional Gemma-based encoder. Its card says dedicated retrieval versions are forthcoming. Do not treat the current base model as a ready retrieval teacher without task validation. It does show why older references to only IndicBERT/MuRIL are incomplete.

[Quanfire's government retrieval adapter](https://huggingface.co/quanfire-ai/embed-gov-indic) is an existing E5-small specialization for parallel government releases. Its results are self-reported and narrow; it is not proof of general search quality. It does invalidate the assumption that compact Indic retrieval specialization is unexplored. Its small adapter still requires the base model.

### Translation competition

[Sarvam Edge's technical announcement](https://www.sarvam.ai/blogs/sarvam-edge) describes a 150M translation model and about 334 MB on-device memory. This is evidence of serious native-device work, not proof of a publicly downloadable browser checkpoint. Treat vendor latency and quality claims as claims until reproduced.

[Chrome's Translator API](https://developer.chrome.com/docs/ai/translator-api) already lists Hindi and Bengali and currently says it works on desktop, not mobile. Offline/private operation alone is therefore insufficient differentiation on supported Chrome desktops. Portable browser support, smaller downloads, or demonstrated improvements on a specific language pair are stronger targets.

## Public data: what each source can actually support

| Source                                                                                                                           | Appropriate use                                    | Main qualification                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [Indic SWIM-IR](https://huggingface.co/datasets/nthakur/indic-swim-ir-cross-lingual)                                             | Indic-query to English-passage training            | Synthetic queries; inspect actual configurations and counts rather than copying the full-family totals.                            |
| [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)                                                               | Native and cross-language retrieval training       | Translated supervision; use selected passage labels carefully and retain original IDs. It refers to original MS MARCO terms.       |
| [MIRACL](https://huggingface.co/datasets/miracl/miracl)                                                                          | Held-out native retrieval checks                   | E5 trained on its training split; avoid tuning on the final evaluation partition.                                                  |
| [Hindi-BEIR](https://aclanthology.org/2025.naacl-long.220/)                                                                      | Broader Hindi retrieval checks                     | Mix of translated, existing, and synthetic tasks. Report by task and check source overlap.                                         |
| Aksharantar / Dakshina                                                                                                           | Script alignment and Romanization robustness       | Neither supplies general search relevance supervision.                                                                             |
| [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC)                                                                           | Translation and auxiliary cross-language alignment | About 230M bitext pairs overall; the source table includes 19.4M Samanantar pairs, not necessarily the entire original collection. |
| [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen), [IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) | Held-out translation evaluation                    | Keep out of training and teacher-generated training targets.                                                                       |

Split by underlying query/document identity before translation or transliteration. Keep all versions of an example in the same partition. Public availability does not guarantee permission for every redistribution or use; record the exact source terms and model revisions in the experiment manifest.

## Additional projects that fit your constraints

### 1. Language-specific compressed retriever

This is my preferred extension of your plan. Deliver an English-plus-one-language model with demonstrated size savings and controlled quality loss. Test vocabulary pruning first, then distillation or fewer layers if compute latency also needs improvement. Add 128/256-dimensional training only if index size matters.

The [Granite R2 paper](https://arxiv.org/abs/2605.13521) explicitly uses pruning and vocabulary selection for its compact model. The technique is established; your contribution would be measured Indic/browser specialization.

Success can be similar quality at a much smaller download. It need not depend on obtaining a new natural Hinglish relevance dataset.

### 2. Small code-mixed token tagger

Train a compact model to identify Hindi/English tokens and extract names, organizations, and locations. This can support selective transliteration, local document tagging, and search filters. A shared encoder with separate task heads is a reasonable experiment, but do not assume joint training beats separate models.

[COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA) contains train/test resources for token language identification, entity recognition, and other tasks. Some outputs are model-assisted and manually refined. Its README and repository license label disagree, so resolve the exact terms before making redistribution claims. [Naamapadam](https://arxiv.org/abs/2212.10168) supplies additional Indic entity supervision, and [IndicNER](https://huggingface.co/ai4bharat/IndicNER) is an existing baseline.

Evaluate entity-span F1, token macro-F1, and English-token preservation. Select a student only after counting its full vocabulary and exported bytes. This is a more tractable public-label project than open-ended generation. It is not automatically a reliable PII redactor; the labels and error costs are different.

### 3. Context-aware query normalization

Learn which Romanized tokens to normalize while preserving English words, names, and numbers. Compare token tagging plus IndicXlit with a small end-to-end normalizer. COMI-LINGUA's normalization and language-ID resources make a bounded experiment possible, subject to inspection of their actual task scope.

Measure both normalization errors and downstream retrieval. This becomes worthwhile only if analysis shows query normalization causes enough search failures. It is a concrete extension beyond another word transliterator, but does not yet have evidence of superiority.

### 4. One-script printed OCR

Public recognition labels do exist. [IIIT's Mozhi resources](https://ilocr.iiit.ac.in/dataset/) and [deployable OCR paper](https://ilocr.iiit.ac.in/dataset/static/assets/img/publication/printed/deployable_ocr.pdf) include cropped recognition data. This improves the feasibility assessment relative to Grok's suggestion. [IndicDLP](https://github.com/AI4Bharat/IndicDLP) addresses layout, so it cannot substitute for text transcription labels.

Start with one script and cropped printed words or lines, using the labels actually available. Compare a compact CTC recognizer, which predicts character sequences without a text-generation decoder, with relevant OCR baselines. TrOCR-small need not be the default. Evaluate character/word error, artifact size, latency, and unfamiliar fonts. Real page OCR additionally requires detection and reading order. Dataset access and component licenses need inspection before committing to this route.

I rank OCR below retrieval and code-mixed tagging because its deployment and evaluation pipeline is larger.

### 5. Pair-specific translation using your existing exports

This remains your strongest alternative if you want a visibly useful standalone model. First measure one direction with existing distilled IndicTrans2 and [OPUS-MT's browser export](https://huggingface.co/Xenova/opus-mt-en-hi/tree/main/onnx). Then inspect language-specific vocabulary pruning and export weight deduplication before funding a new student.

If distilling, choose a target-language-capable tokenizer. Plain English T5-small is not a ready Indic translator. A supported encoder-decoder architecture does not guarantee good target-script coverage. Report chrF, a documented BLEU setup, meaning preservation, and end-to-end browser sentence latency. A 40–80 MB package is a research target, not a likely outcome established by these notes.

## Bounded first experiment

### Stage 1: baseline and export check

- Provisionally use English and Hindi. Keep the scope to a language you can inspect.
- Compare E5-small, Granite 97M R2, and Bekko a8m on the same public retrieval slices. Use each model's required pooling, prompts, and normalization.
- Include lexical retrieval and E5 plus local transliteration. Compare original-query plus normalized-query fusion where practical.
- Separate native, synthetic Romanized, code-mixed, and cross-language results. Do not combine them into a single headline.
- Run on the actual exported precision. Measure graph/tokenizer/runtime bytes, cold download, cached load, warm median/p95 query latency, indexing throughput, and memory where measurable.
- Test WASM and WebGPU separately. Model-name suffixes do not establish operator support or numerical parity.

### Stage 2: one training change at a time

1. Fine-tune unchanged E5 on a bounded 100k–300k selection of public relevance examples, with retained English/native examples and controlled Romanized variants.
2. Compare against the unchanged model. Inspect failures before adding teacher complexity.
3. Test vocabulary pruning separately, then a recovery fine-tune if necessary.
4. Add distillation only when the teacher improves the target task. For script robustness, a useful experiment is native-script teacher input paired with Romanized student input; judge whether meaning survives the transformation.

The T4 setup plausibly supports short-sequence training in this size range, but batch size and throughput require a profiling run. Two GPUs do not pool their VRAM automatically. Ordinary gradient accumulation does not enlarge the set of in-batch contrastive negatives; use explicit negatives or a tested cached-loss method if needed. Cache bounded teacher outputs and checkpoint optimizer/sampler state to survive session interruptions. Mac teacher throughput and memory remain unmeasured.

### Stage 3: decide using a frozen test

Example acceptance criteria, to set before final evaluation:

- Quality route: at least +0.03 absolute nDCG@10 on the declared target slice versus the strongest eligible browser baseline, with no more than 0.02 absolute loss on each main native/English regression suite.
- Size route: at least 30% smaller complete model/tokenizer download at similar quality and acceptable latency.
- Bootstrap uncertainty over original queries, grouping script variants together. Repeat only the promising training configuration with another seed.
- Add reranking only after top-k recall and an oracle ordering analysis show useful headroom.

These numbers are proposed decision thresholds. A few Kaggle weeks is not a scientific stopping rule; stop after the bounded ablations show no useful quality, size, or latency advantage.

## Final choice

Build a narrowly scoped retriever whose browser cost and Indic quality are measured together. Start with current baselines, then compare script adaptation and vocabulary compression as separate experiments. Keep translation as the strongest reuse of your existing work. If you prefer a smaller public-label project, code-mixed token tagging is the most credible additional option I found.
