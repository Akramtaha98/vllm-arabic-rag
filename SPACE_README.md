---
title: Arabic RAG Optimizer
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
license: mit
short_description: Semantic pruning for faster Arabic RAG on vLLM
---

# ⚡ Arabic RAG Optimizer

A live demo of **Semantic-Driven Context Pruning** for Arabic Retrieval-Augmented
Generation. Retrieved Arabic passages are scored sentence-by-sentence with a
cross-encoder and pruned before being sent to the LLM — shrinking the prompt
(and the KV-cache it consumes) without losing the answer.

Try the example questions in the app, or type your own. Adjust the
compression ratio in the sidebar to see the retrieval → pruning →
generation pipeline trade off context size against latency in real time.

Full source, benchmarks, and the research writeup: see the
[GitHub repository](https://github.com/YOUR_USERNAME/vllm-arabic-rag).

**Note on this Space:** it calls a hosted LLM API configured via secrets
(`VLLM_API_URL`, `VLLM_MODEL_NAME`, `VLLM_API_KEY`). It does not run vLLM
itself — vLLM's KV-cache/throughput benefits are demonstrated in the
project's benchmark suite against a self-hosted vLLM server, not in this
lightweight hosted demo.
