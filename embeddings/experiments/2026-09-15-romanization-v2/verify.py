"""Verify final generated rows, split boundaries, and frozen evaluation identity."""
import hashlib,json,pathlib,re
from indic_edge_embeddings.data.transliterate import transliterate_devanagari_to_roman
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'embeddings/data'
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
report=json.loads((OUT/'data_report.json').read_text())
for name,digest in report['input_hashes'].items():assert sha(ROOT/name)==digest
for name,digest in report['output_hashes'].items():assert sha(ROOT/name)==digest
native={r['query_id']:r for split in ['train','validation'] for r in read(DATA/'train/pilot_hindi_100k'/(split+'.jsonl')) if r['script']=='native'}
sets={s:read(DATA/'train/pilot_hindi_100k_romanized_v2'/(s+'.jsonl')) for s in ['train','validation']}
for key in ['source_query_id','split_group_id']:
 assert {r[key] for r in sets['train']}.isdisjoint({r[key] for r in sets['validation']})
for rows in sets.values():
 assert len({r['query_id'] for r in rows})==len(rows)
 for r in rows:
  source=native[r['source_query_id']]
  assert r['positive']==source['positive'] and r['negatives']==source['negatives']
  if r['script']=='romanized':
   assert r['query']==transliterate_devanagari_to_roman(source['query'])
   assert not re.search('[\u0900-\u097f]',r['query'])
   assert all(token in r['query'] for token in re.findall(r'[A-Za-z][A-Za-z0-9_-]*',source['query']))
base=DATA/'eval';n={r['query_id']:r for r in read(base/'miracl_hi_dev_native/queries.jsonl')}
r=read(base/'miracl_hi_dev_romanized_v2/queries.jsonl');nr=json.loads((base/'miracl_hi_dev_native/qrels.json').read_text());rr=json.loads((base/'miracl_hi_dev_romanized_v2/qrels.json').read_text())
assert len(r)==len(n)==350
for q in r:
 assert rr[q['query_id']]==nr[q['source_query_id']]
 assert q['text']==transliterate_devanagari_to_roman(n[q['source_query_id']]['text'])
 assert not re.search('[\u0900-\u097f]',q['text'])
 assert all(token in q['text'] for token in re.findall(r'[A-Za-z][A-Za-z0-9_-]*',n[q['source_query_id']]['text']))
assert sha(base/'miracl_hi_dev_native/corpus.jsonl')==sha(base/'miracl_hi_dev_romanized_v2/corpus.jsonl')
initial=json.loads((OUT/'initial_data_report.json').read_text())
assert all(report['output_hashes'][k]==v for k,v in initial['output_hashes'].items() if '/eval/' in k)
result={'status':'passed','checked_training_rows':sum(map(len,sets.values())),'checked_eval_queries':len(r),'checks':['original inputs preserved','output hashes match','source and normalized query split disjointness','unique query IDs','training passage/negative labels unchanged','Romanization matches current implementation','no residual Devanagari','English tokens preserved','evaluation relevance IDs unchanged','evaluation corpus unchanged','final evaluation data identical to scored initial v2']}
(OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
