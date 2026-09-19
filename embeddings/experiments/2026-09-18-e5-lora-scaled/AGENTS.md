# Embedding project

Before working here, read [README.md](README.md) for scope and [STEPS.md](STEPS.md) for the current sequence. Read [../AGENTS.md](../AGENTS.md) for shared rules and the experiment-record format.

## Decisions to preserve

- Select the base model and language set from baseline evidence. Candidate names in the README are provisional.
- Measure script adaptation and compression separately before combining them.
- Keep native, Romanized, code-mixed, and cross-language evaluation distinct. Synthetic script variants support a narrower claim than naturally typed search queries.
- Preserve original query/document identities across transformations and splits. Use development data for iteration and reserve final evaluation for a frozen candidate.

## Experiment history

Keep this project's records in [experiments/](experiments/README.md). Before a new run, inspect relevant earlier outputs and decisions. Record every substantive experiment or roadmap step there using the shared format, and update the roadmap when evidence changes the next step.
