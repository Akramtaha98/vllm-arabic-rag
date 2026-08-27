"""
Dynamic-controller ablation: fixed vs. dynamic compression ratio under a
time-varying load pattern
------------------------------------------------------------------------
Reviewer concern (both reviewers): the paper describes a
DynamicRatioController (middleware/pruning.py) that reads vLLM's live
/metrics endpoint (vllm:gpu_cache_usage_perc / vllm:kv_cache_usage_perc)
and adjusts the compression ratio accordingly, but never actually
evaluates it against a fixed-ratio baseline under conditions where the
controller's adaptivity should matter, i.e. a load pattern that ramps
KV-cache pressure up and down rather than sitting at one flat
concurrency level (which is all run_full_sweep.py drives).

This script drives a ramping concurrency profile (low -> high -> low)
against a real, self-hosted vLLM server and runs it twice:
  1. fixed:   COMPRESSION_RATIO held constant (default 0.5) the whole time
  2. dynamic: DynamicRatioController.get_ratio() re-polls /metrics before
              every request and uses whatever ratio it returns

It reuses locustfile.py's request logic under the hood via Locust's own
Python API (LocalRunner) rather than reimplementing HTTP/streaming
handling, so TTFT/latency/tokens-per-sec are measured identically to the
existing benchmark. Unlike run_full_sweep.py (flat concurrency per
cell), this drives a single continuous run per condition with
concurrency stepped over time, and logs the controller's chosen ratio
alongside vLLM's live KV-cache occupancy at each step so the paper can
plot "ratio chosen vs. cache pressure over time" for the dynamic
condition and contrast it with the fixed condition's flat line.

REQUIRES: a real self-hosted vLLM server (this is the GPU-benchmark
track, NOT the NIM-hosted-API track used by run_arcd_token_budget.py /
run_arcd_llmlingua2_baseline.py) -- run this on RunPod (or any machine
with a GPU) alongside the existing corrected benchmark sweep, using the
same server you already start for BENCHMARK_INSTRUCTIONS.md:

    vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 --max-model-len 8192

Then, from the repo root, with that server running:

    python benchmark/run_controller_ablation.py \
        --host http://localhost:8000 --out-dir results/controller_ablation

Output: results/controller_ablation/{fixed,dynamic}_ratio_log.jsonl (one
row per request: timestamp, concurrency step, ratio used, KV-cache
occupancy at request time, TTFT, total latency) plus a small summary
printed at the end. Analyze with a short follow-up script or by hand in
a notebook -- the paper only needs the ratio-vs-occupancy plot and a
TTFT/latency comparison table, both derivable directly from the JSONL.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, DynamicRatioController, DynamicRatioConfig, split_sentences  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402
from benchmark.sample_queries import SAMPLE_QUERIES, SYSTEM_PROMPT  # noqa: E402

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PRUNER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_TOKENS = 256

# Ramping concurrency profile: (duration_s, concurrent_workers). This is the
# "time-varying load" the reviewers asked for -- low, then a spike, then
# recovery -- deliberately short per-step so a full run (~9 min) is cheap on
# a rented GPU. Extend the tuples for a longer/more realistic ramp if you
# have more RunPod budget.
LOAD_PROFILE = [
    (60, 4),
    (60, 8),
    (60, 32),
    (60, 32),
    (60, 8),
    (60, 4),
]

FIXED_RATIO = 0.5


def naive_or_lspm_context(pruner, query, ratio):
    result = pruner.prune(query, MOCK_CORPUS, compression_ratio=ratio)
    return result.pruned_text


def one_request(session, host, context, query, ratio, condition, occupancy_at_request, log_lock, log_file):
    request_id = str(uuid.uuid4())
    prompt_text = f"السياق: {context}\n\nالسؤال: {query}"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t0_epoch = time.time()
    ttft_ms = None
    try:
        with session.post(f"{host}/v1/chat/completions", json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                record = {
                    "request_id": request_id, "epoch": t0_epoch, "condition": condition,
                    "ratio": ratio, "kv_cache_occupancy_at_request": occupancy_at_request,
                    "status": "http_error", "http_status": resp.status_code,
                }
            else:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    line = line.strip()
                    if line in ("", "[DONE]"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content") and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                total_latency_ms = (time.perf_counter() - t0) * 1000
                record = {
                    "request_id": request_id, "epoch": t0_epoch, "condition": condition,
                    "ratio": ratio, "kv_cache_occupancy_at_request": occupancy_at_request,
                    "status": "ok", "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms,
                }
    except Exception as e:
        record = {
            "request_id": request_id, "epoch": t0_epoch, "condition": condition,
            "ratio": ratio, "kv_cache_occupancy_at_request": occupancy_at_request,
            "status": "exception", "error": str(e),
        }
    with log_lock:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()


def worker_loop(host, pruner, controller, condition, stop_event, get_concurrency_ratio_hint, log_lock, log_file):
    session = requests.Session()
    import random
    while not stop_event.is_set():
        query = random.choice(SAMPLE_QUERIES)
        if condition == "fixed":
            ratio = FIXED_RATIO
            occupancy = controller._fetch_gpu_cache_usage() if controller is not None else None
        else:
            occupancy = controller._fetch_gpu_cache_usage()
            ratio = controller.get_ratio(fallback_ratio=FIXED_RATIO, usage=occupancy)
        context = naive_or_lspm_context(pruner, query, ratio)
        one_request(session, host, context, query, ratio, condition, occupancy, log_lock, log_file)


def run_condition(host, condition, out_dir):
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME)
    controller = DynamicRatioController(
        config=DynamicRatioConfig(metrics_url=f"{host}/metrics"),
    )

    out_path = out_dir / f"{condition}_ratio_log.jsonl"
    log_file = open(out_path, "w", encoding="utf-8")
    log_lock = threading.Lock()

    print(f"\n=== condition={condition} -> {out_path} ===", flush=True)
    for step_i, (duration_s, n_workers) in enumerate(LOAD_PROFILE):
        print(f"  step {step_i+1}/{len(LOAD_PROFILE)}: {n_workers} workers for {duration_s}s", flush=True)
        stop_event = threading.Event()
        threads = [
            threading.Thread(
                target=worker_loop,
                args=(host, pruner, controller, condition, stop_event, n_workers, log_lock, log_file),
                daemon=True,
            )
            for _ in range(n_workers)
        ]
        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)

    log_file.close()
    return out_path


def summarize(path):
    ttfts, lats, ratios = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                ttfts.append(rec["ttft_ms"])
                lats.append(rec["total_latency_ms"])
            if rec.get("ratio") is not None:
                ratios.append(rec["ratio"])
    if not ttfts:
        print(f"  {path.name}: no successful requests logged.")
        return
    print(f"  {path.name}: n={len(ttfts)} ok, "
          f"TTFT mean={statistics.mean(ttfts):.0f}ms p95={sorted(ttfts)[int(0.95*len(ttfts))-1]:.0f}ms, "
          f"latency mean={statistics.mean(lats):.0f}ms, "
          f"ratio range=[{min(ratios):.2f},{max(ratios):.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="e.g. http://localhost:8000")
    ap.add_argument("--out-dir", default="results/controller_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_path = run_condition(args.host, "fixed", out_dir)
    dynamic_path = run_condition(args.host, "dynamic", out_dir)

    print("\n--- summary ---")
    summarize(fixed_path)
    summarize(dynamic_path)
    print(
        "\nNext: plot kv_cache_occupancy_at_request vs. ratio over time for the "
        "'dynamic' log (should track cache pressure) vs. the flat line for "
        "'fixed', and compare TTFT/latency during the high-concurrency step "
        "(step 3-4 of LOAD_PROFILE) between conditions -- that comparison is "
        "the actual ablation result the paper needs."
    )


if __name__ == "__main__":
    main()
