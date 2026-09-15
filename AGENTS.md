# Indic edge models

## Scope and context

- Before changing project scope, read [context.md](context.md) for goals and hardware/data constraints. User instructions take precedence over historical planning notes.
- Before selecting models or datasets, read the relevant subproject README. Consult [SUGGESTIONS_AND_OPTIONS.md](SUGGESTIONS_AND_OPTIONS.md) for the consolidated assessment and deferred options, then verify current source information when needed.
- Work inside the relevant subproject and read its `AGENTS.md`. Embeddings is the first implementation priority; reranking and translation remain deferred until embeddings is complete or the user changes the order.
- Evaluate the actual browser artifact. Distinguish reference-model results, export parity, and measured browser performance.

## Experiment records

Keep a dedicated `experiments/` folder inside each subproject. Before starting or resuming experimental work, read that folder's README and the records relevant to the current step.

For each substantive experiment or roadmap step, create `experiments/YYYY-MM-DD-short-name/README.md`. Use a unique suffix for repeated runs. Start the record before execution and update it when the run finishes, fails, or is interrupted.

Each record must contain:

1. **Question and criterion:** the hypothesis or step, its roadmap link, and the condition for success.
2. **Setup:** model/data revisions, split identifiers, configuration, seed, code revision where available, hardware, and runtime versions. Mark inapplicable fields explicitly.
3. **Execution:** commands or notebook references, changes being compared, and actual run status.
4. **Outputs:** metrics, baseline comparisons, inspected examples, and paths or links to logs and artifacts. Label partial results and missing measurements.
5. **Interpretation:** what the evidence supports, limitations, and unresolved questions.
6. **Decision:** what to keep, change, stop, or investigate next, and which outputs justify it. Record an inconclusive result explicitly.
7. **Next step:** the follow-up experiment or roadmap revision, including any resume instructions.

Store small reports/configurations beside the record. Reference large datasets, caches, and weights by reproducible locations instead of committing them. Preserve failed runs and superseded decisions; link later records rather than rewriting history as if the outcome were known beforehand.

A step is complete when its record contains the actual status, outputs, and resulting decision; the subproject's experiment index links to it; and any affected `STEPS.md` or README status has been updated. An interrupted run remains incomplete with a recorded resume point.

## Data Creation Guidelines

- Whenever downloading or creating datasets for training or evaluation, make sure to analyze some random sample of raw data and understand its quality, format, and characteristics and confirm that it make sense.

## Remote Kaggle execution

- **Use runner CLI:** Dispatch workloads with `scripts/run_kaggle.py push` (non-blocking), stream live logs with `status --stream`, and download results with `pull`.
- **Isolate scratch data in `/tmp`:** Kaggle bundles everything in `/kaggle/working/` into downloadable artifacts. Direct raw datasets and caches to `/tmp/` and save only final models/metrics in `/kaggle/working/outputs/`.
- **Generate remote data in-command:** Ephemeral containers start clean; chain data creation commands before execution (e.g. `python -m ...build_data --output-dir /tmp/data && python -m ...train --data-dir /tmp/data`) or attach a dataset with `--datasets`.
- **Avoid `DataParallel` on modern encoders:** Submodule device lookups in modern architectures (ModernBERT, mmBERT, RoPE) fail under `torch.nn.DataParallel`. Use single-device GPU or `torchrun` with `--multi-gpu` for PyTorch DDP.
- **Maintain flat experiment outputs:** Record kernel IDs and statuses in `experiments/YYYY-MM-DD-<name>/README.md`, and pull result JSONs directly to the experiment directory without nested folder structures.