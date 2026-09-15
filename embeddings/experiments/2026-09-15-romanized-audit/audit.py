"""Run from repository root with the embeddings environment. Uses local assets only."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import hashlib, json, pathlib, random, re, subprocess, importlib.metadata
import numpy as np
from indic_edge_embeddings.data.transliterate import transliterate_devanagari_to_roman
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator
from indic_edge_embeddings.eval.metrics import compute_retrieval_metrics

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
BASE = ROOT / "embeddings/data/eval"


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


native = BASE / "miracl_hi_dev_native"
roman = BASE / "miracl_hi_dev_romanized"
n = rows(native / "queries.jsonl")
r = rows(roman / "queries.jsonl")
nd = {q["query_id"]: q for q in n}
nq = json.loads((native / "qrels.json").read_text())
rq = json.loads((roman / "qrels.json").read_text())
report = {
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "seed": 20260915,
    "runtime": {
        p: importlib.metadata.version(p)
        for p in ["torch", "transformers", "numpy", "indic-transliteration"]
    },
    "hashes": {
        str(p.relative_to(ROOT)): sha(p)
        for d in [native, roman]
        for p in [d / "queries.jsonl", d / "qrels.json", d / "corpus.jsonl"]
    },
}
report["data"] = {
    "native_count": len(n),
    "romanized_count": len(r),
    "unique_roman_ids": len({q["query_id"] for q in r}),
    "mapping_errors": sum(
        q["source_query_id"] not in nd
        or q["metadata"]["original_text"] != nd[q["source_query_id"]]["text"]
        or rq.get(q["query_id"]) != nq.get(q["source_query_id"])
        for q in r
    ),
    "qrels_without_query": len(set(rq) - {q["query_id"] for q in r}),
    "empty_roman_queries": sum(not q["text"].strip() for q in r),
    "residual_devanagari_queries": sum(
        bool(re.search("[\u0900-\u097f]", q["text"])) for q in r
    ),
    "current_transliteration_mismatches": sum(
        q["text"] != transliterate_devanagari_to_roman(nd[q["source_query_id"]]["text"])
        for q in r
    ),
}
report["samples"] = [
    {
        "native": nd[q["source_query_id"]]["text"],
        "romanized": q["text"],
        "id": q["query_id"],
    }
    for q in random.Random(20260915).sample(r, 20)
]
corpus = []
with (native / "corpus.jsonl").open() as f:
    for line in f:
        corpus.append(json.loads(line))
ids = [d["doc_id"] for d in corpus]
idset = set(ids)
report["data"]["missing_relevant_docs"] = len(
    {d for ds in nq.values() for d in ds} - idset
)
cache = ROOT / "embeddings/data/cache/corpus_emb_8e0da0ebae.npy"
vectors = np.load(cache, mmap_mode="r")
report["cache"] = {
    "path": str(cache.relative_to(ROOT)),
    "shape": list(vectors.shape),
    "sha256": sha(cache),
}
report["model_snapshots"] = [
    p.name
    for p in (
        pathlib.Path.home()
        / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots"
    ).iterdir()
]
(OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report["data"]), flush=True)
eval = DenseRetrievalEvaluator(
    "intfloat/multilingual-e5-small", device="cpu", batch_size=32, cache_dir=None
)
# Check the saved corpus row alignment against fresh encodings at dispersed positions.
positions = random.Random(20260915).sample(range(len(corpus)), 24)
fresh = eval.encode(
    [(corpus[i]["title"] + " " + corpus[i]["text"]).strip() for i in positions],
    prefix="passage: ",
)
saved = np.array(vectors[positions])
cos = (fresh * saved).sum(1) / (
    np.linalg.norm(fresh, axis=1) * np.linalg.norm(saved, axis=1)
)
report["cache"]["fresh_row_cosine"] = {
    "min": float(cos.min()),
    "mean": float(cos.mean()),
}
report["evaluations"] = {}
for label, queries, qrels, prefix in [
    ("native", n, nq, "query: "),
    ("romanized", r, rq, "query: "),
    ("romanized_no_prefix", r, rq, ""),
]:
    emb = eval.encode([q["text"] for q in queries], prefix=prefix, sort_by_length=False)
    results = {}
    for start in range(0, len(queries), 16):
        scores = emb[start : start + 16] @ vectors.T
        for j, s in enumerate(scores):
            top = np.argpartition(-s, 19)[:20]
            top = top[np.argsort(-s[top])]
            results[queries[start + j]["query_id"]] = [
                (ids[k], float(s[k])) for k in top
            ]
    metrics = compute_retrieval_metrics(results, qrels)
    report["evaluations"][label] = metrics
    (OUT / (label + "_rankings.json")).write_text(json.dumps(results))
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(label, metrics, flush=True)
# Independent binary/linear-gain nDCG calculation from saved rankings.
rankings = json.loads((OUT / "romanized_rankings.json").read_text())
values = {}
for query in r:
    qid = query["query_id"]
    rel = rq[qid]
    gains = np.array([rel.get(doc, 0) for doc, _ in rankings[qid][:10]])
    ideal = sorted(rel.values(), reverse=True)[:10]
    values[qid] = float(
        np.sum(gains / np.log2(np.arange(2, 12)))
        / np.sum(np.array(ideal) / np.log2(np.arange(2, len(ideal) + 2)))
    )
report["independent_ndcg_check"] = float(np.mean(list(values.values())))
report["romanized_subsets"] = {}
for name, present in [("residual_devanagari", True), ("no_devanagari", False)]:
    subset = [
        values[q["query_id"]]
        for q in r
        if bool(re.search("[\u0900-\u097f]", q["text"])) == present
    ]
    report["romanized_subsets"][name] = {
        "count": len(subset),
        "ndcg@10": float(np.mean(subset)),
    }
(OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
