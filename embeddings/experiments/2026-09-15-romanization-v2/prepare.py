"""Rebuild versioned local data without downloading or modifying the originals."""

import hashlib, json, pathlib, random, re, shutil
import pyarrow as pa
import pyarrow.parquet as pq
from indic_edge_embeddings.data.transliterate import (
    ROMANIZATION_VERSION,
    transliterate_devanagari_to_roman,
)
from indic_edge_embeddings.data.build_train_data import split_and_augment_records

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "embeddings/data"


def read(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


report = {
    "version": ROMANIZATION_VERSION,
    "seed": 42,
    "sample_seed": 20260915,
    "input_hashes": {},
}
source = DATA / "eval/miracl_hi_dev_native"
target = DATA / "eval/miracl_hi_dev_romanized_v2"
if target.exists():
    raise FileExistsError(target)
target.mkdir()
queries = read(source / "queries.jsonl")
qrels = json.loads((source / "qrels.json").read_text())
new = []
for q in queries:
    new.append(
        dict(
            q,
            query_id=q["query_id"] + "_roman_v2",
            source_query_id=q["query_id"],
            text=transliterate_devanagari_to_roman(q["text"]),
            script="romanized",
            metadata={
                "original_text": q["text"],
                "romanization_version": ROMANIZATION_VERSION,
            },
        )
    )
(target / "queries.jsonl").write_text(
    "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in new)
)
dump(target / "qrels.json", {q["query_id"]: qrels[q["source_query_id"]] for q in new})
# Reuse exactly the native corpus, preserving row order.
shutil.copyfile(source / "corpus.jsonl", target / "corpus.jsonl")
meta = json.loads((source / "metadata.json").read_text())
meta.update(name=target.name)
meta["metadata"].update(
    script="Latin",
    transformation="synthetic_romanization",
    romanization_version=ROMANIZATION_VERSION,
    source_manifest=source.name,
)
dump(target / "metadata.json", meta)
report["eval"] = {
    "path": str(target.relative_to(ROOT)),
    "queries": len(new),
    "residual_devanagari": sum(
        bool(re.search("[\u0900-\u097f]", q["text"])) for q in new
    ),
    "samples": random.Random(20260915).sample(new, 20),
}
old = DATA / "train/pilot_hindi_100k"
train_out = DATA / "train/pilot_hindi_100k_romanized_v2"
if train_out.exists():
    raise FileExistsError(train_out)
all_old = {s: read(old / (s + ".jsonl")) for s in ["train", "validation"]}
report["old_source_overlap"] = len(
    {r["query_id"].removesuffix("_roman") for r in all_old["train"]}
    & {r["query_id"].removesuffix("_roman") for r in all_old["validation"]}
)
native = [r for rows in all_old.values() for r in rows if r["script"] == "native"]
valid = []
rejected = []
for r in native:
    try:
        transliterate_devanagari_to_roman(r["query"])
    except ValueError as e:
        rejected.append({"query_id": r["query_id"], "query": r["query"], "reason": str(e)})
        continue
    valid.append(r)
train, val = split_and_augment_records(valid, 100000, 5000, 0.3, 42)
train_out.mkdir()
for name, rows in [("train", train), ("validation", val)]:
    for row in rows:
        row.setdefault("romanization_version", None)
    (train_out / (name + ".jsonl")).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    pq.write_table(pa.Table.from_pylist(rows), train_out / (name + ".parquet"))
report["training"] = {
    "path": str(train_out.relative_to(ROOT)),
    "source_native_rows": len(native),
    "rejected_count": len(rejected),
    "train_rows": len(train),
    "validation_rows": len(val),
    "source_overlap": len(
        {r["source_query_id"] for r in train} & {r["source_query_id"] for r in val}
    ),
    "normalized_query_group_overlap": len(
        {r["split_group_id"] for r in train} & {r["split_group_id"] for r in val}
    ),
    "romanized_rows": sum(r["script"] == "romanized" for r in train + val),
    "residual_devanagari": sum(
        bool(re.search("[\u0900-\u097f]", r["query"]))
        for r in train + val
        if r["script"] == "romanized"
    ),
    "samples": random.Random(20260915).sample(
        [r for r in train if r["script"] == "romanized"], 20
    ),
}
dump(OUT / "rejected_training_rows.json", rejected)
for p in [
    source / "queries.jsonl",
    source / "qrels.json",
    source / "corpus.jsonl",
    old / "train.jsonl",
    old / "validation.jsonl",
]:
    report["input_hashes"][str(p.relative_to(ROOT))] = sha(p)
report["output_hashes"] = {
    str(p.relative_to(ROOT)): sha(p)
    for d in [target, train_out]
    for p in d.iterdir()
    if p.is_file()
}
dump(
    train_out / "dataset_metadata.json",
    {
        "romanization_version": ROMANIZATION_VERSION,
        "seed": 42,
        "source": "local pilot_hindi_100k native rows",
        "split_policy": "source_and_normalized_query_group",
        "counts": {k: v for k, v in report["training"].items() if k != "samples"},
        "source_hashes": report["input_hashes"],
    },
)
dump(OUT / "data_report.json", report)
print(
    json.dumps(
        {k: v for k, v in report["training"].items() if k != "samples"}, indent=2
    )
)
