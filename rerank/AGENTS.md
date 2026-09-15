# Reranking project

Before working here, read [README.md](README.md) for scope and [STEPS.md](STEPS.md) for the current sequence. Read [../AGENTS.md](../AGENTS.md) for shared priority and experiment-record rules.

## Decisions to preserve

- Begin implementation after the embedding project is complete unless the user changes priority.
- Establish candidate recall and the best possible ordering under available relevance judgments before choosing a training plan.
- Pin the retriever, corpus, preprocessing, and candidate lists for controlled reranker comparisons.
- Measure ranking gains and the complete retrieval-plus-reranking cost. Unjudged passages may be relevant; inspect mined negatives accordingly.

## Experiment history

Keep this project's records in [experiments/](experiments/README.md). Before a new run, inspect relevant earlier outputs and decisions, including the embedding results used to generate candidates. Record every substantive experiment or roadmap step there using the shared format.
