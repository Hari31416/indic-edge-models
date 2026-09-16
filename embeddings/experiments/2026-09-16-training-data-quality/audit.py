import json, random, collections, hashlib, sys, platform
from pathlib import Path

root = Path("embeddings/data/train/pilot_hindi_100k")
out = Path("embeddings/experiments/2026-09-16-training-data-quality")
sets = {
    s: [json.loads(l) for l in (root / f"{s}.jsonl").open()]
    for s in ["train", "validation"]
}
stats = {
    "python": sys.version,
    "platform": platform.platform(),
    "hashes": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.jsonl")
    },
}
for s, rows in sets.items():
    c = collections.Counter(r["positive"] for r in rows if r["script"] == "native")
    stats[s] = {
        "rows": len(rows),
        "source_script": dict(
            collections.Counter(r["source"] + " / " + r["script"] for r in rows)
        ),
        "language": dict(collections.Counter(r["language"] for r in rows)),
        "explicit_negative_rows": sum(bool(r["negatives"]) for r in rows),
        "unique_sources": len({r.get("source_query_id", r["query_id"]) for r in rows}),
        "native_duplicate_positive_excess": sum(n - 1 for n in c.values()),
        "positive_no_devanagari": sum(
            not any("\u0900" <= x <= "\u097f" for x in r["positive"]) for r in rows
        ),
        "positive_words_quantiles": {
            str(q): sorted(len(r["positive"].split()) for r in rows)[
                int(q * (len(rows) - 1))
            ]
            for q in [0, 0.1, 0.5, 0.9, 0.99, 1]
        },
    }
for field in ["source_query_id", "split_group_id", "positive", "query"]:
    stats["overlap_" + field] = len(
        {r.get(field) for r in sets["train"]}
        & {r.get(field) for r in sets["validation"]}
    )
rng = random.Random(20260916)
samples = []
for key in sorted({(r["source"], r["script"]) for r in sets["train"]}):
    subset = [r for r in sets["train"] if (r["source"], r["script"]) == key]
    samples.extend(rng.sample(subset, min(15, len(subset))))
(out / "stats.json").write_text(json.dumps(stats, indent=2))
(out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
print(json.dumps(stats, indent=2))
for i, r in enumerate(samples):
    print(
        json.dumps(
            {
                "i": i,
                "id": r["query_id"],
                "source": r["source"],
                "script": r["script"],
                "q": r["query"],
                "p": r["positive"],
            },
            ensure_ascii=False,
        )
    )
