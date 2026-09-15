"""Compare frozen old/new queries against the audited unchanged E5 corpus."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
import hashlib, json, pathlib
import numpy as np
from indic_edge_embeddings.eval.runner import DenseRetrievalEvaluator
from indic_edge_embeddings.eval.metrics import compute_retrieval_metrics

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "embeddings/data"
audit = json.loads((OUT.parent / "2026-09-15-romanized-audit/audit.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


cache = DATA / "cache/corpus_emb_8e0da0ebae.npy"
assert sha(cache) == audit["cache"]["sha256"]
corpus = DATA / "eval/miracl_hi_dev_native/corpus.jsonl"
assert (
    sha(corpus)
    == audit["hashes"]["embeddings/data/eval/miracl_hi_dev_native/corpus.jsonl"]
)
assert sha(DATA / "eval/miracl_hi_dev_romanized_v2/corpus.jsonl") == sha(corpus)
ids = [json.loads(line)["doc_id"] for line in corpus.open()]
vec = np.load(cache, mmap_mode="r")
assert len(ids) == len(vec)
evaluator = DenseRetrievalEvaluator(
    "intfloat/multilingual-e5-small", device="cpu", batch_size=32
)
report = {
    "model_revision": audit["model_snapshots"][0],
    "runtime": audit["runtime"],
    "cache_sha256": audit["cache"]["sha256"],
    "metrics": {},
}
for name in [
    "miracl_hi_dev_native",
    "miracl_hi_dev_romanized",
    "miracl_hi_dev_romanized_v2",
]:
    base = DATA / "eval" / name
    qs = [json.loads(line) for line in (base / "queries.jsonl").open()]
    rel = json.loads((base / "qrels.json").read_text())
    emb = evaluator.encode(
        [q["text"] for q in qs], prefix="query: ", sort_by_length=False
    )
    results = {}
    for start in range(0, len(qs), 16):
        for j, s in enumerate(emb[start : start + 16] @ vec.T):
            top = np.argpartition(-s, 19)[:20]
            top = top[np.argsort(-s[top])]
            results[qs[start + j]["query_id"]] = [(ids[k], float(s[k])) for k in top]
    report["metrics"][name] = compute_retrieval_metrics(results, rel)
    (OUT / (name + "_rankings.json")).write_text(json.dumps(results))
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(name, report["metrics"][name], flush=True)
