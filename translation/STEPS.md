# Translation project steps
 
Status: Active (21 September 2026). Activated after concluding the embedding project. Initial focus is on Step 1 and Step 2.


## 1. Choose one direction and define success

- Choose a pair using usefulness, ability to inspect both languages, public training data, and baseline quality.
- Use English to Hindi as the initial example only. Treat the reverse direction as a separate budget and evaluation decision.
- Choose device/browser targets and provisional download, memory, and sentence-latency limits.
- Specify input lengths, expected domains, and intended release terms.
- Define success as a measured quality or browser-cost improvement. Recheck existing in-browser translation options at that time.

**Output:** A direction-specific experiment specification.

## 2. Prepare public data and evaluation

- Inspect the selected BPCC language/component and its provenance. Use a bounded subset rather than downloading the entire collection by default.
- Filter malformed, duplicate, wrong-language, and badly aligned pairs. Record filtering rules and counts.
- Keep document/source identities where available and remove overlap across training, development, and evaluation.
- Reserve IN22-Gen and IN22-Conv for final evaluation; use separate development data during iteration.
- Define chrF and a reproducible BLEU setup. Record normalization, tokenization, and metric signatures.
- Plan bilingual inspection of public examples for names, numbers, dates, negation, omissions, and additions.

**Output:** Data manifests, an inspected sample, and a frozen evaluation protocol. Do not claim human-validated quality without inspection by someone competent in both languages.

## 3. Establish reference and browser baselines

- Run the selected OPUS-MT checkpoint and direction-appropriate distilled IndicTrans2 model on identical development examples.
- Record decoding parameters, including beam width and output limits. Compare matched settings before separately tuning for latency.
- Reproduce relevant existing exports and tokenizer fixtures. Resolve conflicting old benchmark claims using raw evidence.
- Verify a small end-to-end browser run before training. Check language tags, script processing, token IDs, decoder cache behavior, and termination.
- Measure complete download, cold/cached load, time to first output, total sentence latency, and memory where measurable. Separate results by input/output length and WASM/WebGPU backend.

**Decision:** Identify the actual gap: quality, storage, runtime memory, or generation latency. If existing artifacts already meet the goal, choose a narrower improvement before training.

## 4. Test the least expensive useful change

- If packaging dominates, inspect duplicated tied weights, separate decoder graphs, and unnecessary files.
- If multilingual vocabulary dominates, investigate pruning for the selected direction. Remap embeddings, output-head rows, token IDs, and special tokens consistently.
- Measure changed segmentation, unknown tokens, and coverage of names, English borrowings, and numbers.
- Test supported quantization settings and compare outputs with the reference model.
- If quality is the main gap and the model already fits, run a bounded fine-tuning pilot on public parallel data.

**Output:** An isolated comparison for each change. Export cleanup may avoid training, while pruning may require a recovery fine-tune.

## 5. Distill a smaller student only if needed

- Choose a student architecture with a plausible browser export path and adequate tokenizer coverage.
- Confirm that a candidate teacher improves the chosen task, then profile local generation throughput and memory.
- Generate targets for a bounded training subset, retaining original references and source IDs. Keep final evaluation examples out of this process.
- Compare training on public reference translations with training that includes teacher outputs. Teacher mistakes should not silently replace good human references.
- Profile short-sequence training on one T4 before choosing batch size or multi-GPU execution. Two 16 GB GPUs do not automatically provide one 32 GB memory pool.
- Save resumable model, optimizer, and data-sampling state for session interruptions.

**Decision:** Continue only if the student provides a useful quality-versus-cost tradeoff. A small parameter count alone is insufficient.

## 6. Validate the final browser artifact

- Freeze the candidate, decoding settings, and acceptance criteria before final testing.
- Evaluate the exported precision on IN22-Gen and IN22-Conv separately, with uncertainty estimates where practical.
- Inspect meaning preservation on a documented sample. Report fluency separately from correctness.
- Test long inputs within the declared limits, Unicode handling, empty inputs, truncation, and generation stopping.
- Verify offline operation after caching, and measure repeated translations for memory growth.
- Validate every claimed device class. A laptop result does not establish phone support.

**Output:** A release decision with quality, failure analysis, and deployment evidence. Do not repeatedly select models against the final test.

## 7. Release one direction

- Publish model/tokenizer/browser artifacts and explicit source/target language codes.
- Include preprocessing, decoding, evaluation, and data-selection recipes with pinned revisions.
- Provide a small local browser demo and a model card describing provenance and limitations.
- Report the actual bundle size and supported input lengths; identify any remaining quality gap to the reference model.

Expand to the reverse direction or another pair only after this result is useful. Do not turn the first pilot into full multilingual translation.

## Updating the roadmap

Record each substantive experiment or completed stage in [experiments/](experiments/README.md), following the [shared record format](../AGENTS.md#experiment-records). Keep this file focused on the current sequence and revise it when recorded evidence changes the next step. Preserve earlier outcomes and decisions in the experiment history.
