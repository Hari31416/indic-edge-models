# Translation project

Before working here, read [README.md](README.md) for scope, candidate resources, and existing export work. Read [STEPS.md](STEPS.md) for the current sequence and [../AGENTS.md](../AGENTS.md) for shared priority and experiment-record rules.

## Decisions to preserve

- Begin implementation after the embedding project is complete unless the user changes priority.
- Select one language pair and one direction from evidence. English to Hindi remains a provisional example.
- Reproduce relevant existing exports and reconcile conflicting benchmark claims before reusing their numbers.
- Keep IN22 final evaluation separate from training, teacher-generated targets, and iterative model selection.
- Check translation meaning as well as automatic metrics. Record decoding settings and evaluate complete browser bundles, including tokenizers and decoder graphs.

## Experiment history

Keep this project's records in [experiments/](experiments/README.md). Before a new run, inspect relevant earlier outputs and decisions. Record every substantive experiment or roadmap step there using the shared format, with direction-specific setup and results.
