# Grok suggestions

Notes from the Sep 2026 planning chat. Opinionated. Not a training recipe.

## Verdict

The embedder then reranker plan is the right *model* bet if the in-browser bar stays. It only wins if you specialize on romanized and mixed-script Indic retrieval. A generic small Indic embedder loses to models that already run in the tab.

Park the chatbot.

## What you have to beat

`intfloat/multilingual-e5-small` (118M, 384-d) already has Xenova ONNX and Transformers.js support. It saw MIRACL and Mr. TyDi. Native-script Telugu Wikipedia retrieval is already strong. Hindi is weaker. How Indians actually type (Hinglish, Tanglish, mixed Unicode) is the hole.

Other names:

- `paraphrase-multilingual-MiniLM-L12-v2`: speed baseline, STS not search
- `l3cube-pune/indic-sentence-similarity-sbert`: Indic STS on MuRIL, wrong objective, fat for a tab
- `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`: teachers, not shipping weights (reranker int8 ONNX is still ~544 MB)
- Snowflake arctic-embed-m-v2.0: Transformers.js exists, still hundreds of MB

Browser budget to take seriously: about 80 MB q8 before the tab feels heavy.

## How to run the chosen plan

Continue from e5-small or Multilingual-MiniLM-L12-H384. Do not start from MuRIL if the artifact has to be a small ONNX file. Do not pretrain from scratch.

Distill from bge-m3 or multilingual-e5-large. Train the student on Kaggle. Run teacher scoring, ONNX export, and WebGPU QA on the M4 Pro.

Then distill `bge-reranker-v2-m3` into mMiniLM-L6 or L12. The 568M teacher does not belong in the tab.

Language order:

- Phase 1: English, Hindi, romanized Hindi
- Phase 2: Tamil, Telugu, Bengali, Marathi plus romanization
- Later: Gujarati, Kannada, Malayalam, Punjabi, Odia only if the demo still needs them

Twenty-two scheduled languages is a scorecard, not a product.

## Public data (no scraping)

You said you cannot collate a custom corpus. You do not need to. You do need a mixing script and a romanized-query eval (transliterate queries, keep native passages).

- Aksharantar: 26M transliteration pairs, 21 languages, 12 scripts. This is the multi-script objective.
- Dakshina: human romanization, including sentence-level Latin to native. Use as eval gold.
- BPCC (`ai4bharat/BPCC`): ~230M bitext across 22 scheduled languages. Prefer BPCC-Human (~2.2M) before noisy mined pairs. On AIKosh via IndicTrans2 / Samanantar.
- Samanantar: 49.6M En-Indic. Older slice, already inside BPCC-Mined.
- MIRACL: real retrieval qrels for Hindi, Bengali, Telugu, and others. Hindi train is only 1,169 queries. Treat as eval plus a little fine-tune, not as the whole training set.
- Hindi-BEIR, mMARCO-hi, `mteb/IndicQARetrieval`: native-script Hindi retrieval. Check you did not regress vs e5-small.
- Mr. TyDi: e5 already trained on this. Contamination-aware eval.
- IN22-Gen / IN22-Conv: domain-diverse parallel sentences, not retrieval labels.

AIKosh catalog pages hydrate poorly in the browser. The useful sets already live on Hugging Face and GitHub.

## Kill criterion

After a few Kaggle weeks, you must beat both:

- e5-small alone
- IndicXlit (or Dakshina-style transliterate) then e5-small

on romanized queries. Stay within about 2 nDCG of e5-small on native Hindi-BEIR.

If the cheap pipeline wins, stop training a new embedder. Ship e5-small plus a tiny transliterator. Spend remaining GPU time only on the reranker.

## Hardware split

Kaggle 2x T4 (2x16 GB):

- Full fine-tune of 30-118M dual encoders
- Distillation of a MiniLM-class cross-encoder
- QLoRA on a 2B chat model later, if you insist, in painful 9-hour sessions

M4 Pro:

- Teacher inference
- ONNX / Transformers.js
- Browser QA

Full fine-tune of IndicBERT / MuRIL (~278M) is tight and the exported model is fat. Skip from-scratch pretraining.

## Chatbot (later)

`sarvamai/sarvam-1` is a 2B completion model for 10 Indic languages plus English. Community Q4_K_M GGUF is about 1.55 GB. That is a laptop, not a tab. Cloud Indic chat already exists. Fine-tuning on Kaggle is possible later. It is not how you beat in-browser baselines.

Sarvam Edge (ASR 74M, TTS 24M, translation 150M) is closer to an on-device story and is being built with device OEMs. Do not compete there without a retrieval-specific angle.

## If you do something else instead

The next best *model* that still fits public data and T4s: distill IndicTrans2 into a tiny, Optimum-exportable translator for a few pairs (`en↔hi`, then `ta` or `bn`). IndicTrans2 is ~1B and ugly to get into ONNX. BPCC is the training set. IN22-Conv is the eval. Success is a 40-80 MB q8 file that beats Helsinki `opus-mt` and is close enough to IndicTrans2 that someone would download it. Google Translate still wins on quality. You win on private and offline.

You already have `indictrans2-onnx-export` and `indictrans2-onnx-browser-demo` under Desktop/sandbox. That work is evidence the export hole is real. Distilling to Marian or T5-small is the part that was missing, not another failed 1B ONNX attempt.

Other options, ranked:

- Indic line OCR (fine-tune TrOCR-small). Browser TrOCR is Latin. Indian scans are mixed script. `ai4bharat/indicdlp` is layout boxes, not transcripts, so you still need line-OCR labels. More useful than a chatbot, more annoying to eval.
- Hinglish to English seq2seq on PHINC / HinGE / aggregated HF sets. T5-small hobby checkpoints already exist. Quality vs a cloud LLM is the problem. I would not spend a year here.
- A Transformers.js toolkit with no new weights: e5-small plus IndicXlit plus a distilled `en-hi` translator, with honest native / romanized / code-mixed numbers. More useful than a mediocre new checkpoint.

## Do not do these

- Generic Indic LLM or chat as the flagship
- ASR / TTS (IndicVoices, Bhashini, Sarvam Edge)
- Full 22-language MT
- Legal or health RAG
- Another IndicSBERT

## What I would ship

An ONNX plus Transformers.js demo that indexes a few thousand Indic Wikipedia or government-style pages in the tab, accepts Hinglish, and shows side-by-side numbers against e5-small. Publish eval for native, romanized, code-mixed, and cross-lingual. That is useful without a company.

There is not a third idea in this hardware box that is both differentiated and shippable. Pick the romanized retriever, or pick the tiny translator. Do not collect both as "research."
