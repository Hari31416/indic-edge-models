# Training data v3

The new builder keeps the historical pilot unchanged. Run from the repository root with the embeddings virtualenv. Its first use should be the matched answer/passage comparison, then a separate mixed-source experiment.

## Matched comparison

Obtain the upstream **training** Parquet, not our existing pilot Parquet. Our pilot discarded the selected-passage fields and cannot reconstruct them. Place it at `/tmp/indic-data/hintrain.parquet`, or edit `quality_matched.json` to point at it.

For a reproducible Hugging Face download, replace the local `files` entry with:

```json
{
  "repo_id": "ai4bharat/MSMARCO-XI",
  "filename": "train/hintrain.parquet",
  "revision": "REPLACE_WITH_THE_40_CHARACTER_DATASET_COMMIT"
}
```

Get the commit from the dataset repository history or `HfApi().dataset_info("ai4bharat/MSMARCO-XI").sha`. The builder rejects floating revisions and records file hashes. Local paths are also hashed. Keep downloaded files/caches under `/tmp` on Kaggle.

```bash
PYTHONPATH=embeddings/src embeddings/.venv/bin/python \
  -m indic_edge_embeddings.data.build_quality_data \
  --config embeddings/configs/quality_matched.json \
  --output-dir embeddings/data/train/quality_matched_v3
```

Outputs are `answers/{train,validation}.{jsonl,parquet}` and `passages/{train,validation}.{jsonl,parquet}`. Both arms have identical query IDs, order, native/Romanized variants and split membership. They use only queries with both an informative answer and a selected passage. Both disable explicit negatives to isolate the supervision change. This is a new matched control, not an exact recreation of the historical pilot.

One selected passage per query is chosen deterministically; `positive_texts` preserves every selected passage for false-negative masking. Additional positives are not all independently sampled as targets in this implementation.

The unchanged training command accepts either arm:

```bash
PYTHONPATH=embeddings/src embeddings/.venv/bin/python \
  -m indic_edge_embeddings.train.train_cli \
  --train-file embeddings/data/train/quality_matched_v3/passages/train.parquet \
  --val-file embeddings/data/train/quality_matched_v3/passages/validation.parquet \
  --model-name-or-path intfloat/multilingual-e5-small \
  --output-dir /tmp/passage-control \
  --learning-rate 5e-6 --num-epochs 1 --batch-size 32
```

Use the same model revision, training settings, update budget and retrieval evaluation for the answer arm. On Kaggle dispatch through `scripts/run_kaggle.py push` per project instructions; keep final training artifacts under `/kaggle/working/outputs/`.

## Mixed-source configuration

`quality_mixed.json` is a second-stage recipe. Its `/tmp/indic-data` inputs must be supplied before execution. It includes MSMARCO-XI Hindi selected passages, original English query/passage pairs from the same file, MIRACL Hindi train, Indic-RAG-Suite Hindi train, and a small SWIM Hindi-to-English slice. It does not download missing sources silently or substitute another dataset.

Before validation allocation and filtering, the caps target approximately 45k native Hindi-to-Hindi, 30k Romanized Hindi-to-Hindi, 20k English-to-English and 5k Hindi/Romanized-to-English rows. These are approximate counts, not enforced sampling weights. Source availability, connected split groups, exclusions and conversion failures change actual counts, which are reported. All native rows remain available for teacher pairing.

Supported JSONL or Parquet input schemas:

| `format`     | Required fields                                                      | Notes                                                                                                                                                                                                                                                                                         |
| ------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `msmarco-xi` | `query_id`, `query`, `Answer`, `Eng_Query`, `Eng_Answer`, `passages` | `passages` contains equally sized `is_selected`, `Translated_passages`/`English_passages`. Only selected texts are positives. Unselected passages are never automatically negatives. Set `family: msmarco` for both languages.                                                                |
| `miracl`     | `query_id`, `query`, `positive_passages`, `negative_passages`        | A train-split export using MIRACL's documented loaded schema. Passage objects have `title` and `text`. Judged negatives are retained, capped at two by default. Raw topics/qrels TSVs must first be joined to corpus passages; they are not this schema.                                      |
| `indic-rag`  | `wiki_id`, `question`, `answer`, `paragraph`                         | Uses the source paragraph, never the generated question/answer/reasoning `text`. Query identity hashes article ID, question and paragraph. Article IDs group related examples across splits.                                                                                                  |
| `swim`       | `_id`, `query`, `title`, `text`                                      | Hindi queries with English positive passages; tracked as a separate direction.                                                                                                                                                                                                                |
| `normalized` | `query_id`, `query`, `positives`                                     | Optional `answer`, `english_query`, `document_group`, `negatives`. Negatives require `negative_provenance: human_judged` or `reviewed`. Use original MS MARCO IDs and `family: msmarco` when importing mMARCO. Consolidate multiple rows for a query into one positive list before importing. |

Every source must declare `split: train`, a unique `name`, `format`, `max_queries`, and `files`. Declarations do not verify an upstream split: supply the correct training file. Each `files` entry accepts a local path or pinned HF file reference. Relative paths resolve against the config directory. `max_queries` caps a reservoir sample drawn across every input row, not a prefix. Identical repeated records are skipped; conflicting source IDs fail visibly.

Source references: [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), [MIRACL](https://huggingface.co/datasets/miracl/miracl), [Indic-RAG-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite), [SWIM](https://huggingface.co/datasets/nthakur/indic-swim-ir-cross-lingual), [mMARCO](https://github.com/unicamp-dl/mMARCO).

## Romanization and split protection

The optional `dakshina_train` entry accepts the original training lexicon TSV, `native<TAB>roman<TAB>attestation_count`, and requires `split: train`. Attestation counts weight spelling choices. Uncovered Hindi words use our existing v2 transliterator. English tokens and numbers remain intact. Do not feed the published sentence dev/test data into this augmentation. The generated queries remain synthetic; this does not create natural code-mixed retrieval labels. [Dakshina](https://github.com/google-research-datasets/dakshina)

Connected groups keep source identities, normalized native/English/Romanized query texts and exact normalized positive texts together. Matched arms also group identical answer texts. Indic-RAG groups articles. Shared positives can create large connected groups; inspect `largest_group` and actual validation counts. A build fails if it cannot create nonempty splits.

`exclude` entries have `kind: queries` or `kind: passages`, and read local JSONL/Parquet or pinned HF files. Query entries accept `query` or `text`; passage entries accept `positives`, `positive` or `text`. The example configs exclude our evaluation query texts, including generated Romanized exact matches. Add other held-out sources explicitly. This catches exact normalized overlap, not semantic paraphrases or unknown upstream contamination. Excluding the entire MIRACL corpus would also exclude legitimate MIRACL training passages; choose that policy deliberately.

## Quality inspection and training negatives

Each build writes:

- `report.json`: input hashes/revisions, builder hash, seed/configuration, rejection counts, heuristic review flags, source/direction counts, connected-group sizes and output hashes.
- `raw_samples.json`: an independent seeded reservoir of up to 200 raw rows per source for inspection. Changing its size does not change training selection.
- `rejected_samples.json`: bounded examples of automatic rejections.
- `romanization_failures.json`: conversions that failed and retained only the native row.

Inspect the raw examples for translation meaning, query completeness, answer support and domain coverage before training. Short queries are flagged rather than rejected. The code does not claim to detect semantic translation errors automatically. Review tokenizer truncation separately; whitespace length flags are not token measurements. Save the human findings in the experiment record.

The collators append explicit negatives to encoded passages. The loss sums probability over known positives and masks related-group passages of unknown relevance. It handles duplicate positives and same-native-query variants. Old datasets without these metadata still load. The current implementation does not mine or semantically judge new negatives; supply MIRACL judgments or a reviewed normalized input. With two negatives per row, passage encoding may approach three times the original batch size, so profile GPU memory before a full run.

## Validation

```bash
PYTHONPATH=embeddings/src HF_HUB_OFFLINE=1 embeddings/.venv/bin/python -m pytest \
  embeddings/tests/test_quality_data.py embeddings/tests/test_data.py -q
```

Tests cover selected-passage parsing, matched arms, full-input sampling, exclusion propagation, script pairing, deterministic lexicon spelling, multi-positive masking, explicit-negative gradients, and one-step offline training/evaluation for InfoNCE and the dual loss. They use fixtures and tiny local models, not a production data-quality or retrieval measurement.
