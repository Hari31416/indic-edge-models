# Embedding project steps

Status: 16 September 2026. [E5 pilot training](experiments/2026-09-15-e5-pilot-training/README.md) is complete: synthetic Romanized nDCG@10 improved from 0.0006 to 0.1059 (+176x), but native nDCG@10 regressed from 0.5513 to 0.3921 (-0.1592). Next: iterate on Step 4 with dual-objective training (distillation preservation loss + query alignment) before proceeding to Step 5 compression. Browser validation in step 3 remains incomplete.

This sequence will change as results arrive. Complete the next useful experiment, record what it establishes, and revise the remaining steps. Later stages are conditional; a useful result does not require implementing every technique listed here.

## 1. Define the first experiment

- Choose the first Indic language and query-to-passage directions. English plus Hindi is the current starting assumption.
- Choose an initial device and browser. Record Mac RAM and runtime versions; do not infer phone performance from a laptop run.
- Define provisional download, memory, and latency budgets. The earlier 80 MB suggestion is not a project requirement.
- Decide the intended model/data release terms and whether synthetic query transformations are acceptable.
- Select one primary quality metric and explicit regression checks. Use nDCG@10 for ranking quality and Recall@20 to measure whether relevant passages enter the candidate set.

**Output:** A short experiment specification with scope, budgets, metrics, and unresolved questions.

## 2. Establish usable public evaluation data

- Inspect suitable Hindi-BEIR and MIRACL splits for native retrieval; consider other public benchmarks for the selected language.
- Separate development data from a final held-out test before tuning.
- Keep original query and document identifiers. Group translations and script variants with their source examples when splitting.
- Create a clearly labeled synthetic Romanized evaluation slice from existing retrieval queries if appropriate. Check that transformations preserve meaning.
- Use Dakshina for Romanization robustness, not as a replacement for search relevance judgments.
- Include natural code-mixed retrieval only when suitable public relevance labels are available. Otherwise record this as an evaluation gap. Linguistically grounded mixed-script evaluation (entity/loanword preservation, phrase-level code-switching) is deferred to later roadmap phases.


E5 has seen MIRACL training data. Its held-out splits remain useful comparisons, but do not establish performance on an entirely unseen task distribution.

**Output:** Versioned evaluation manifests and a small example audit. Know what claims the data can support before training.

## 3. Measure existing models in the browser

- Compare the candidates in the README on identical corpora and query sets, using each model's correct prompts, pooling, and normalization.
- Include lexical retrieval. Test local transliteration followed by embedding, preserving English tokens and comparing against the original query.
- Run a small export smoke test early. Confirm tokenization, embedding outputs, and ranking agreement against the reference implementation within stated tolerances.
- Measure the selected precision on WASM and WebGPU where supported.
- Record model/tokenizer/runtime download bytes, cold download, cached load, warm median and p95 query latency, indexing throughput, and memory where measurable.
- Inspect failures by query type. Distinguish irrelevant matches, script failures, missing candidates, and truncation.

A Python transliteration baseline is useful for quality testing. It is not a browser deployment result until its inference also runs in the tab.

**Output:** Baseline table, reproducible runtime settings, and a shortlist of failures worth addressing.

**Decision:** Select the training base and primary hypothesis from measured results. If existing models meet the intended need, narrow the target before spending GPU time.

## 4. Build a bounded training pilot

The [v3 builder and training integration](experiments/2026-09-16-dataset-builder-v3/README.md) are implemented and tested offline. [Build configurations and commands](configs/README.md) cover the matched answer/passage comparison and a later mixed-source pilot. Production data inspection and retrieval evaluation remain pending.

[Training data quality inspection](experiments/2026-09-16-training-data-quality/README.md) found that 92.8% of pilot rows use MSMARCO-XI answers rather than selected passages. Retain the current dataset as a control; a future data comparison should inspect selected passage supervision, translation quality, English retention, and negatives before scaling volume.

- Inspect language-specific slices of Indic SWIM-IR and MSMARCO-XI for relevance training. Confirm terms, schema, and actual available examples.
- Begin with a manageable subset. The earlier 100,000 to 300,000 example range is an experiment budget, not a required dataset size.
- Retain native-script and English examples alongside controlled Romanized variants. Translation or word-transliteration pairs are auxiliary alignment data, not substitutes for relevance labels.
- Add explicit difficult negatives and check for unlabeled relevant passages. Ordinary gradient accumulation does not create a larger set of in-batch contrastive negatives.
- Profile a short training run before fixing batch size, sequence length, or the use of both GPUs.
- Save configurations, seeds, data manifests, and resumable checkpoints for interrupted Kaggle sessions.

**Output:** One reproducible pilot run that changes as few things as possible.

## 5. Test adaptation and compression separately

First measure targeted retrieval fine-tuning without architecture changes. Compare its gains and regressions on development data against the unchanged base.

Then choose compression experiments based on the measured bottleneck:

- If download size dominates, test vocabulary pruning with token-ID remapping and coverage checks.
- If inference compute dominates, test a smaller student or fewer layers.
- If vector index storage dominates, investigate smaller trained embedding dimensions. Shorter vectors do not automatically shrink model weights.
- Add teacher distillation only after demonstrating that the teacher improves the target task. Cache a bounded set of teacher outputs.

For vocabulary changes, inspect names, English words, digits, punctuation, Roman spellings, unknown tokens, and increased token counts. Evaluate compression alone before combining it with adaptation, then use a recovery fine-tune if necessary.

**Output:** A comparison showing which change caused each quality, size, or latency difference.

**Decision:** Keep the smallest set of changes that provides a useful advantage. Revise or stop branches that do not help.

## 6. Validate the candidate and decide whether to release

- Freeze acceptance thresholds before final testing. Choose a quality improvement at comparable browser cost, or a meaningful cost reduction with bounded quality loss.
- Test the exact exported, quantized artifact on the held-out evaluation.
- Report native, Romanized, code-mixed, and cross-language results separately, with per-language regressions.
- Estimate uncertainty over original queries, grouping related script variants together. Repeat a promising training setup with another seed where practical.
- Verify fully local operation after the required assets are cached. Include cold-start and indexing costs in the deployment assessment.
- Test additional devices only before claiming support for them.

**Output:** A release decision supported by quality and deployment measurements. If the result misses the target, return to the relevant earlier step rather than repeatedly tuning against the final test.

## 7. Package the first release

- Publish the checkpoint, tokenizer, browser artifacts, usage example, model card, and reproducible evaluation instructions.
- Document training-data provenance, terms, preprocessing, supported languages, and known failure cases.
- Include a small browser demonstration comparing the model with the strongest relevant baseline.
- Record why this model is useful and what the measurements do not establish.

Only then consider another language or a reranker. A reranker needs evidence that relevant passages are retrieved but ranked too low, plus an acceptable browser latency budget.

## Updating this roadmap

Record each substantive experiment or completed stage in [experiments/](experiments/README.md), following the [shared record format](../AGENTS.md#experiment-records). Keep this file focused on the current sequence and revise it when recorded evidence changes the next step.
