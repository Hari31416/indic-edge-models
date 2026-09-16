"""Versioned passage supervision, matched controls, and source inspection.

Run with --config PATH --output-dir NEW_DIRECTORY. Inputs are streamed, then
reservoir sampled across the complete input. No unselected passage is silently
promoted to a negative. See configs/README.md for supported source schemas.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import platform
import subprocess
import re
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from .transliterate import transliterate_devanagari_to_roman, ROMANIZATION_VERSION


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Expected a text string")
    return " ".join(unicodedata.normalize("NFC", value).split())


def informative(value):
    return bool(value) and normalize(value).strip(".। !") not in {
        "null",
        "none",
        "n/a",
        "no answer found",
        "कोई उत्तर नहीं मिला",
        "कोई जवाब नहीं",
    }


def unique_texts(values):
    return list({normalize(v): v for v in values if informative(v)}.values())


def read_rows(path):
    if Path(path).suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    else:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def resolve_input(spec, base):
    if "path" in spec:
        path = (base / spec["path"]).resolve()
        provenance = {"path": str(path)}
    else:
        revision = spec["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                "HF inputs require an immutable 40-character commit revision"
            )
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                filename=spec["filename"],
                revision=revision,
            )
        )
        provenance = {k: spec[k] for k in ("repo_id", "filename", "revision")}
    provenance["sha256"] = file_hash(path)
    return path, provenance


def adapt(row, spec):
    """Return query-level supervision; preserve all known positive passages."""
    fmt = spec["format"]
    lang = spec.get("language", "hi")
    doc_lang = spec.get("passage_language", lang)
    if fmt == "swim":
        qid = str(row["_id"])
    elif fmt == "indic-rag":
        qid = digest(str(row["wiki_id"]) + ":" + text(row["question"]) + ":" + text(row["paragraph"]))
    else:
        qid = str(row["query_id"])
    family = spec.get("family", spec["name"])
    english_query = ""
    answer = ""
    negatives = []
    if fmt == "msmarco-xi":
        english = lang == "en"
        query = text(row["Eng_Query"] if english else row["query"])
        english_query = text(row["Eng_Query"])
        answer = text(row["Eng_Answer"] if english else row["Answer"])
        passages = row["passages"]
        values = passages["English_passages" if english else "Translated_passages"]
        selected = passages["is_selected"]
        if len(values) != len(selected) or any(v not in (0, 1) for v in selected):
            raise ValueError("Invalid passage selection arrays")
        positives = [text(p) for p, flag in zip(values, selected) if flag == 1]
        # Unselected MS MARCO passages are not certified irrelevant.
    elif fmt == "miracl":
        query = text(row["query"])

        def passage(p):
            return text((p.get("title", "") + " " + p["text"]).strip())

        positives = [passage(p) for p in row["positive_passages"]]
        negatives = [passage(p) for p in row["negative_passages"]]
    elif fmt == "swim":
        query = text(row["query"])
        positives = [text((row.get("title", "") + " " + row["text"]).strip())]
        doc_lang = "en"
    elif fmt == "indic-rag":
        query = text(row["question"])
        answer = text(row["answer"])
        # The combined `text` field includes the answer; never use it as a passage.
        positives = [text(row["paragraph"])]
    elif fmt == "normalized":
        query = text(row["query"])
        positives = [text(p) for p in row["positives"]]
        answer = text(row.get("answer", ""))
        english_query = text(row.get("english_query", ""))
        negatives = [text(p) for p in row.get("negatives", [])]
        if negatives and row.get("negative_provenance") not in {
            "human_judged",
            "reviewed",
        }:
            raise ValueError(
                "Normalized negatives need human_judged or reviewed provenance"
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    positives = unique_texts(positives)
    negative_texts = unique_texts(negatives)
    positive_keys = {normalize(p) for p in positives}
    negative_texts = [p for p in negative_texts if normalize(p) not in positive_keys]
    if not informative(query) or not positives:
        return None
    return dict(
        source=spec["name"],
        source_identity=f"{family}:{qid}",
        query=query,
        positives=positives,
        answer=answer,
        negatives=negative_texts,
        language=lang,
        passage_language=doc_lang,
        english_query=english_query,
        document_group=(f"wikipedia:{lang}:{row['wiki_id']}" if fmt == "indic-rag" else row.get("document_group", "")),
        negative_provenance=(
            "human_judged"
            if fmt == "miracl"
            else row.get("negative_provenance", "none")
        ),
    )


def load_lexicon(path):
    """Dakshina TRAIN lexicon: native<TAB>roman<TAB>attestation count."""
    lexicon = defaultdict(Counter)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            native, roman, count = line.rstrip("\n").split("\t")
            if roman.isascii() and roman.strip() and int(count) > 0:
                lexicon[unicodedata.normalize("NFC", native)][roman] += int(count)
    return dict(lexicon)


def romanize(query, lexicon, rng):
    def replace(match):
        word = match.group()
        choices = lexicon.get(unicodedata.normalize("NFC", word))
        if choices:
            return rng.choices(list(choices), weights=list(choices.values()), k=1)[0]
        return transliterate_devanagari_to_roman(word)

    result = re.sub(r"[\u0900-\u0963\u0971-\u097f]+", replace, query)
    return transliterate_devanagari_to_roman(result)


def group_records(records, include_answers=True):
    """Connected components across identities, queries and known positive texts."""
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for i, row in enumerate(records):
        keys = ["id:" + row["source_identity"], "q:" + normalize(row["query"])]
        keys += ["q:" + normalize(q) for _, q in row.get("variants", [])]
        if row.get("document_group"):
            keys.append("doc:" + row["document_group"])
        if row["english_query"]:
            keys.append("en:" + normalize(row["english_query"]))
        keys += ["p:" + normalize(p) for p in row["positives"]]
        if include_answers and informative(row["answer"]):
            keys.append("p:" + normalize(row["answer"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    return list(groups.values())


def write_rows(path, rows):
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))


def build(config, output_dir, base=Path(".")):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 42)
    rng = random.Random(seed)
    matched = config.get("matched_answers", True)
    validation_fraction = config.get("validation_fraction", 0.05)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not isinstance(config.get("max_negatives", 2), int) or config.get("max_negatives", 2) < 0:
        raise ValueError("max_negatives must be a nonnegative integer")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {"status": "in_progress", "config": config, "inputs": [], "sources": {},
              "python": platform.python_version(), "platform": platform.platform(),
              "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
              "builder_sha256": file_hash(__file__)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    excluded_queries, excluded_passages = set(), set()
    for item in config.get("exclude", []):
        path, provenance = resolve_input(item, base)
        report["inputs"].append(provenance)
        for row in read_rows(path):
            if item.get("kind") == "queries":
                excluded_queries.add(normalize(text(row.get("query", row.get("text", "")))))
            elif item.get("kind") == "passages":
                for p in row.get("positives", [row.get("positive", row.get("text", ""))]):
                    excluded_passages.add(normalize(text(p)))
            else:
                raise ValueError("Exclusions must declare kind=queries or passages")
    lexicon = {}
    if config.get("dakshina_train"):
        spec = config["dakshina_train"]
        if spec.get("split") != "train":
            raise ValueError("Dakshina augmentation accepts the train lexicon only")
        path, provenance = resolve_input(spec, base)
        report["inputs"].append(provenance)
        lexicon = load_lexicon(path)
    records = []
    blocked_identities = set()
    raw_samples, rejected_samples = [], []
    names = set()
    for spec in config["sources"]:
        if spec["name"] in names:
            raise ValueError("Source names must be unique")
        names.add(spec["name"])
        if spec.get("split") != "train":
            raise ValueError("Only explicitly declared train sources are accepted")
        cap = spec["max_queries"]
        if not isinstance(cap, int) or cap <= 0:
            raise ValueError("max_queries must be a positive integer")
        stats = Counter()
        sampled = []
        inspected = []
        seen = {}
        source_rng = random.Random(f"{seed}:{spec['name']}")
        inspect_rng = random.Random(f"inspection:{seed}:{spec['name']}")
        for input_spec in spec["files"]:
            path, provenance = resolve_input(input_spec, base)
            report["inputs"].append(dict(provenance, source=spec["name"]))
            for raw in read_rows(path):
                stats["scanned"] += 1
                # Separate RNG means changing review sample size cannot change the dataset.
                review = {"source": spec["name"], "raw": raw}
                size = config.get("inspection_samples", 200)
                j = inspect_rng.randrange(stats["scanned"])
                if len(inspected) < size:
                    inspected.append(review)
                elif j < size:
                    inspected[j] = review
                try:
                    row = adapt(raw, spec)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Schema error in {spec['name']} row {stats['scanned']}: {exc}"
                    ) from exc
                reason = None
                if row is None:
                    reason = "missing_query_or_selected_positive"
                elif matched and not informative(row["answer"]):
                    reason = "missing_answer_for_matched_control"
                elif (
                    normalize(row["query"]) in excluded_queries
                    or normalize(row["english_query"]) in excluded_queries
                ):
                    reason = "evaluation_query_overlap"
                elif any(normalize(p) in excluded_passages for p in row["positives"] + [row["answer"]]):
                    reason = "evaluation_passage_overlap"
                if reason:
                    if reason.startswith("evaluation_"):
                        blocked_identities.add(row["source_identity"])
                    stats[reason] += 1
                    if len(rejected_samples) < 200:
                        rejected_samples.append(
                            {"source": spec["name"], "reason": reason, "raw": raw}
                        )
                    continue
                key = (row["source_identity"], row["language"], row["passage_language"])
                fingerprint = digest(json.dumps(row, ensure_ascii=False, sort_keys=True))
                if key in seen:
                    if seen[key] == fingerprint:
                        stats["duplicate_rows"] += 1
                        continue
                    raise ValueError(f"Conflicting query identity {key}; consolidate its positives before building")
                seen[key] = fingerprint
                stats["eligible"] += 1
                if len(row["query"].split()) <= 2:
                    stats["review_short_query"] += 1
                if any(len(p.split()) > 400 for p in row["positives"]):
                    stats["review_long_positive"] += 1
                if row["language"] == "hi" and not re.search(r"[\u0900-\u097f]", row["query"]):
                    stats["review_hindi_query_without_devanagari"] += 1
                j = source_rng.randrange(stats["eligible"])
                if len(sampled) < cap:
                    sampled.append(row)
                elif j < cap:
                    sampled[j] = row
        stats["sampled_queries"] = len(sampled)
        report["sources"][spec["name"]] = dict(stats)
        raw_samples.extend(inspected)
        records.extend(sampled)
    if not records:
        raise ValueError("No eligible queries")
    source_options = {s["name"]: s for s in config["sources"]}
    conversion_failures = []
    for row in records:
        spec = source_options[row["source"]]
        qid = digest(row["source"] + ":" + row["source_identity"] + ":" + row["language"])
        local_rng = random.Random(f"{seed}:{qid}")
        probability = spec.get("romanized_probability", 0)
        if not 0 <= probability <= 1:
            raise ValueError("romanized_probability must be in [0, 1]")
        row["variants"] = [("native", row["query"])]
        if row["language"] == "hi" and local_rng.random() < probability:
            try:
                roman = romanize(row["query"], lexicon, local_rng)
                if roman != row["query"]:
                    row["variants"].append(("romanized", roman))
            except ValueError as exc:
                conversion_failures.append({"query_id": qid, "error": str(exc)})
        if any(normalize(q) in excluded_queries for _, q in row["variants"]):
            blocked_identities.add(row["source_identity"])
    groups = group_records(records, include_answers=matched)
    before = len(records)
    groups = [g for g in groups if not any(r["source_identity"] in blocked_identities for r in g)]
    records = [r for g in groups for r in g]
    report["excluded_related_queries"] = before - len(records)
    rng.shuffle(groups)
    # Whole components only, with actual source/direction counts reported below.
    val_target = max(1, round(len(records) * validation_fraction))
    validation_count = 0
    splits = {"train": [], "validation": []}
    for group in groups:
        split = "validation" if validation_count + len(group) <= val_target else "train"
        splits[split].append(group)
        if split == "validation":
            validation_count += len(group)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Cannot form nonempty disjoint splits; inspect component sizes or increase input size")
    outputs = {}
    for split, components in splits.items():
        passage_rows, answer_rows = [], []
        for group in components:
            group_id = digest("\n".join(sorted({r["source_identity"] for r in group})))
            for row in group:
                spec = source_options[row["source"]]
                qid = digest(
                    row["source"] + ":" + row["source_identity"] + ":" + row["language"]
                )
                local_rng = random.Random(f"{seed}:{qid}")
                positives = sorted(row["positives"], key=normalize)
                positive = local_rng.choice(positives)
                negs = list(row["negatives"])
                local_rng.shuffle(negs)
                negs = negs[: config.get("max_negatives", 2)]
                for script, query in row["variants"]:
                    record = dict(
                        query_id=qid if script == "native" else qid + "_roman",
                        source_query_id=qid,
                        source_identity=row["source_identity"],
                        split_group_id=group_id,
                        query=query,
                        native_query=row["query"],
                        positive=positive,
                        positive_texts=positives,
                        negatives=negs,
                        source=row["source"],
                        language=row["language"],
                        passage_language=row["passage_language"],
                        script=script,
                        negative_provenance=row["negative_provenance"],
                        romanization_version=(
                            ("dakshina_train+" if lexicon else "")
                            + ROMANIZATION_VERSION
                            if script == "romanized"
                            else None
                        ),
                    )
                    passage_rows.append(record)
                    if matched:
                        # Same query exposure and no passage negatives in either control arm.
                        passage_rows[-1] = dict(record, negatives=[])
                        answer_rows.append(
                            dict(
                                record,
                                positive=row["answer"],
                                positive_texts=[row["answer"]],
                                negatives=[],
                            )
                        )
        order = list(range(len(passage_rows)))
        rng.shuffle(order)
        for arm, rows in [("passages", passage_rows)] + (
            [("answers", answer_rows)] if matched else []
        ):
            rows = [rows[i] for i in order]
            folder = out / arm
            folder.mkdir(exist_ok=True)
            write_rows(folder / split, rows)
            outputs[f"{arm}/{split}"] = {
                "rows": len(rows),
                "directions": dict(
                    Counter(
                        f"{r['language']}/{r['script']}->{r['passage_language']}"
                        for r in rows
                    )
                ),
                "sources": dict(Counter(r["source"] for r in rows)),
                "negative_rows": sum(bool(r["negatives"]) for r in rows),
                "sha256": file_hash(folder / f"{split}.jsonl"),
            }
    for name, value in [
        ("raw_samples", raw_samples),
        ("rejected_samples", rejected_samples),
        ("romanization_failures", conversion_failures),
    ]:
        (out / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )
    report.update(
        status="complete",
        outputs=outputs,
        source_queries=len(records),
        connected_groups=len(groups),
        largest_group=max(map(len, groups)),
        romanization_failures=len(conversion_failures),
        limitations=[
            "Automatic filters do not establish translation or relevance quality.",
            "Inspect raw_samples.json before training.",
            "Word-lexicon augmentation is synthetic, not natural code-mixed supervision.",
            "Ratios are measured outputs; caps and probabilities do not enforce exact row weights.",
        ],
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        report = build(config, args.output_dir, args.config.resolve().parent)
    except Exception as exc:
        # Never alter a pre-existing build on refusal to overwrite.
        if (
            not isinstance(exc, FileExistsError)
            and (args.output_dir / "report.json").exists()
        ):
            path = args.output_dir / "report.json"
            report = json.loads(path.read_text())
            report.update(status="failed", error=str(exc))
            path.write_text(json.dumps(report, indent=2))
        raise
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    main()
