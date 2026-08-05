<div align="center">

# ⚡ Arabic RAG Optimizer

**Semantic-Driven Context Pruning for Memory-Constrained vLLM Deployments**

Shrink retrieved Arabic context at the *sentence* level before it ever reaches
your LLM — less KV-cache pressure, faster generation, same answer quality.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![vLLM](https://img.shields.io/badge/inference-vLLM-6C4CF1)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![HF Spaces](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-yellow)](#deploy-your-own-demo)

</div>

---

Arabic's morphological complexity means tokenizers routinely produce **1.5–2×
more tokens** than English for equivalent meaning. In a RAG pipeline, that
inflates the KV-cache footprint on every request — under concurrent traffic,
it's the first thing to bottleneck a vLLM deployment.

**Arabic RAG Optimizer** sits between retrieval and generation: a
**Lightweight Semantic Pruning Middleware (LSPM)** scores every retrieved
sentence against the query with a cross-encoder, keeps only what's relevant,
and reassembles a compact, coherent context — before it ever touches the
model's KV-cache.

## How it works

```
 User Query
     │
     ▼
 Vector DB (Chroma)  ───────────►  Top-K raw Arabic passages
     │
     ▼
 LSPM Middleware
   • split into sentences
   • cross-encoder relevance scoring (query × sentence)
   • keep top-N%, reassemble in original order
   • ratio: fixed slider OR dynamic (reads vLLM's live GPU load)
     │
     ▼
 vLLM Server  ─────────────────►  OpenAI-compatible /v1/chat/completions
     │                            PagedAttention KV-cache, now under less pressure
     ▼
 Streamlit UI
```

## Features

- **Sentence-level pruning**, not chunk-level — finer granularity than typical
  RAG chunking, so less semantic content is thrown away per token saved
- **Two pruning models** built in: a fast multilingual MiniLM for sub-second
  pruning, or BGE-reranker-v2-m3 for maximum relevance accuracy
- **Dynamic compression** — optionally reads vLLM's `/metrics`
  (`vllm:gpu_cache_usage_perc`) and tightens pruning automatically as GPU load
  rises, relaxing it when the GPU is idle
- **Streamed answers** in the UI for fast perceived latency
- **Backend-agnostic** — point it at a self-hosted vLLM server or at a hosted
  OpenAI-compatible endpoint (e.g. NVIDIA NIM) with zero code changes
- **Full benchmark suite** — Locust load tests, tokenization-disparity
  analysis, and ROUGE-L/BLEU/BERTScore fidelity scoring, ready to reproduce
  the throughput/KV-cache/quality claims for a paper

## Demo

🔗 **[Live demo on Hugging Face Spaces](#)** — replace with your Space URL once deployed (see below).

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/vllm-arabic-rag.git
cd vllm-arabic-rag

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with a backend — pick **one**:

```bash
# Option A: self-hosted vLLM (requires a CUDA GPU)
VLLM_API_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Option B: hosted, no GPU needed — NVIDIA NIM free tier
VLLM_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
VLLM_MODEL_NAME=meta/llama-3.1-8b-instruct
VLLM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Get a free NIM key at [build.nvidia.com](https://build.nvidia.com) → pick a
> model → **Get API Key**. Smaller/mid-size models (8B-class) return in
> seconds on the free tier; the newest flagship models can queue for minutes —
> stick to 8B-class for a responsive demo.

Then run it:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Try one of the example questions or type
your own Arabic query.

## Project layout

```
vllm-arabic-rag/
├── app.py                        # Streamlit UI
├── middleware/
│   ├── pruning.py                 # SemanticPruner + DynamicRatioController (LSPM core)
│   ├── vllm_client.py             # OpenAI-compatible HTTP client (streaming + TTFT tracking)
│   └── retriever.py               # Chroma wrapper + in-memory mock corpus
├── eval/
│   ├── tokenization_disparity.py  # AR vs EN token-count disparity analysis
│   ├── semantic_fidelity.py       # ROUGE-L / BLEU / BERTScore: pruned vs raw
│   └── requirements-eval.txt
├── benchmark/
│   ├── locustfile.py              # Throughput/TTFT stress test (raw vs pruned)
│   ├── plot_results.py            # Charts from Locust CSV output
│   └── requirements-benchmark.txt
├── tests/
│   └── test_pruning.py
├── scripts/
│   ├── start_vllm.sh              # Launch a local/rented-GPU vLLM server
│   └── deploy_hf.sh               # One-command push to a Hugging Face Space
├── SPACE_README.md                # README used for the HF Space (has YAML frontmatter)
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Deploy your own demo

**Hugging Face Spaces (free, CPU):**

```bash
# 1. Create an empty Space at https://huggingface.co/new-space
#    SDK: Streamlit · Hardware: CPU basic
# 2. Get a write-scope token: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxx HF_USERNAME=yourname HF_SPACE=arabic-rag-optimizer \
  bash scripts/deploy_hf.sh
```

Then set `VLLM_API_URL` / `VLLM_MODEL_NAME` / `VLLM_API_KEY` as **secrets**
in the Space's Settings tab. That's it — the Space rebuilds and serves the
UI automatically.

**Docker:**

```bash
docker build -t arabic-rag-optimizer .
docker run -p 8501:8501 --env-file .env arabic-rag-optimizer
```

## Reproducing the benchmarks

<details>
<summary><b>Tokenization disparity (Arabic vs English)</b></summary>

```bash
pip install -r eval/requirements-eval.txt
python eval/tokenization_disparity.py \
  --pairs data/parallel_pairs.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --out results/tokenization_disparity.csv
```
</details>

<details>
<summary><b>Throughput / TTFT under load (baseline vs. pruned)</b></summary>

```bash
pip install -r benchmark/requirements-benchmark.txt

# Baseline: raw, unpruned context
RAG_MODE=raw locust -f benchmark/locustfile.py --host http://<vllm-host>:8000 \
  --users 50 --spawn-rate 5 --run-time 3m --headless --csv=results/baseline

# LSPM: pruned context, ratio=0.5
RAG_MODE=pruned COMPRESSION_RATIO=0.5 locust -f benchmark/locustfile.py \
  --host http://<vllm-host>:8000 --users 50 --spawn-rate 5 --run-time 3m \
  --headless --csv=results/lspm_r05

python benchmark/plot_results.py \
  --baseline results/baseline_stats_history.csv \
  --pruned results/lspm_r05_stats_history.csv \
  --out results/throughput_comparison.png
```

Also scrape vLLM's `/metrics` during each run to log
`vllm:gpu_cache_usage_perc` over time for the KV-cache savings figure.
</details>

<details>
<summary><b>Semantic fidelity (ROUGE-L / BLEU / BERTScore)</b></summary>

Generate `raw_answer` and `pruned_answer` per question on a fixed eval set
(e.g. the Arabic Reading Comprehension Dataset — ARCD), save to
`data/eval_set.jsonl`, then:

```bash
python eval/semantic_fidelity.py --data data/eval_set.jsonl --out results/fidelity.csv
```
</details>

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Roadmap

- [ ] Populate `data/parallel_pairs.jsonl` and `data/eval_set.jsonl` from a real corpus (e.g. ARCD)
- [ ] Run the full benchmark matrix: ratios `{0.2, 0.5, 0.8, 1.0}` × concurrency `{1, 10, 25, 50, 100}`
- [ ] Add a LangChain-vanilla-RAG baseline alongside the raw-vLLM baseline
- [ ] Write up results for submission (target: Q1/Q2 NLP or systems venue)

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this project is useful in your research, please cite the accompanying
paper once published. In the meantime:

```bibtex
@misc{arabic-rag-optimizer,
  title  = {Semantic-Driven Context Pruning for Optimizing Arabic RAG Systems
            in Memory-Constrained vLLM Deployments},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/vllm-arabic-rag}
}
```
