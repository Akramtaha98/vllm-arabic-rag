#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the demo / benchmarks.
# Requires: pip install vllm  (needs a CUDA GPU)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${2:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching
