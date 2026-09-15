# Reranking project steps

Initial roadmap, 15 September 2026. Deferred until the embedding project is complete. These steps are conditional and should change with results.

## 1. Review the embedding results

- Pin the chosen retriever, tokenizer, corpus, chunking, and query preprocessing.
- Measure candidate recall at 5, 10, and 20 passages.
- Estimate the best possible ranking within those candidates using available relevance judgments.
- Inspect whether failures come from missing passages or poor ordering.

**Output:** A decision on whether reranking offers enough benefit to pursue.

## 2. Define evaluation and budgets

- Reuse public evaluation splits and example identities from the embedding project. Keep a final test held out from tuning.
- Choose a primary ranking metric, such as nDCG@10, and per-language regression checks.
- Set limits for additional model download, memory, and end-to-end search latency.
- Freeze candidate lists for controlled model comparisons. Record candidate depth and the passage truncation policy.
- Label synthetic Romanized evaluation separately from naturally occurring code-mixed data.

**Output:** An evaluation manifest and provisional acceptance criteria.

## 3. Benchmark existing rerankers

- Compare the unchanged retrieval order with multilingual MiniLM reranking.
- Use one larger model from the README to estimate achievable quality, provided local inference is practical.
- Measure top-5, top-10, and top-20 reranking with controlled batch sizes and sequence lengths.
- Test a small browser export early. Check pair tokenization, special tokens, score extraction, and ranking agreement with the reference implementation.
- Record cached load, warm median/p95 latency, and combined pipeline memory. Test WASM and WebGPU separately where supported.

**Decision:** Select a base model and a specific failure to address. Keep the project deferred if the added cost outweighs the improvement.

## 4. Prepare a bounded public-data pilot

- Select relevance examples from the README's sources for the chosen languages.
- Split by original query/document identity before translating, transliterating, or mining candidates.
- Mine difficult candidates using the actual retriever. Include known positives and inspect likely false negatives.
- Retain English and native-script examples alongside any script variants.
- Choose an initial pointwise or pairwise relevance objective. Keep the first experiment simple enough to attribute changes.
- Profile one short training run and save resumable checkpoints for Kaggle interruptions.

**Output:** A reproducible small fine-tuning run and error analysis.

## 5. Improve quality or reduce cost

- Compare fine-tuning with the unchanged base before changing architecture.
- Distill teacher scores or preferences only if the teacher improves the relevant development tasks. Cache scores for training examples only.
- Test vocabulary reduction if storage dominates, or fewer layers/a smaller student if scoring latency dominates.
- Quantize the exported candidate and repeat quality checks. Compression and fine-tuning should first be measured separately.
- Test whether fewer reranked candidates preserve most of the gain.

Do not assume the embedding model and reranker can share weights or runtime memory merely because their tokenizers or architectures resemble each other.

**Output:** A quality-versus-cost comparison and a candidate worth final validation.

## 6. Validate and release

- Freeze the model and thresholds, then evaluate the exact browser artifact on held-out data.
- Report gains relative to the same candidate lists, plus performance of the complete deployed pipeline.
- Estimate uncertainty over queries, grouping related script variants. Repeat a promising training run with another seed where practical.
- Verify cached offline operation and test each device class before claiming support.
- Publish artifacts, usage instructions, training/evaluation recipes, and known failure cases.

**Decision:** Release only if the measured benefit justifies the extra browser cost. Otherwise retain the findings and revise the scope.

## Updating the roadmap

Record each substantive experiment or completed stage in [experiments/](experiments/README.md), following the [shared record format](../AGENTS.md#experiment-records). Revise this roadmap when recorded evidence changes the next step. Use development data for iteration rather than repeatedly selecting against the final test.
