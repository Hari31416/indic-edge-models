# Context

What I want to build, and the limits I am working under. Written so a later session can start without rereading the chat.

## Goal

I want to fine-tune models for Indic languages in India. The model comes first. A product can wait.

Each component should run entirely on the edge, in a browser tab, and beat current in-browser options that do the same job.

## Chosen direction

- Small Indic multi-script embedder
- Then a reranker
- Much later, use or fine-tune an existing Indic model as a chatbot

The chatbot is far away. It is not the current project.

## Constraints

- Solo developer
- No GPU cluster
- Hardware is a Mac M4 Pro plus Kaggle 2x T4 (16 GB each, short sessions)
- I do not want to collect or collate a private dataset
- Training data should come from public sources (Hugging Face, AIKosh, GitHub)
- Usefulness matters more than a paper
- I am not trying to cover all 22 scheduled languages on day one unless that is actually required

## Non-goals

- Training an Indic LLM from scratch
- Legal, health, or other verticals that need private corpora
- Speech, unless the plan explicitly changes
- Shipping a cloud API as the main artifact
