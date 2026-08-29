"""
Dynamic-controller ablation v2: redesigned to actually enter the
controller's adaptive range
------------------------------------------------------------------------
Why v2 exists: the first ablation run (benchmark/run_controller_ablation.py)
completed cleanly but never observed the controller adapt -- KV-cache
occupancy peaked at 2.1% against a high_load_threshold of 75%, so
DynamicRatioController correctly, but uninterestingly, returned its
max_ratio ceiling on all 2,302 requests. Two design choices caused that:
(1) it used the tiny synthetic MOCK_CORPUS (~240 tokens total) instead of
realistic-length context, and (2) it ran against a full-size KV-cache pool
(default --gpu-memory-utilization ~0.9 on a 24 GiB card), so even 32
concurrent requests barely dented capacity.

v2 fixes both honestly, without inventing load that doesn't reflect the
paper's real workload:
  1. Uses REAL retrieved contexts from the ARCD eval set (the same
     140-question pool used throughout this project), unpruned, which
     average ~1,486 prompt tokens each (measured this session in
     results/arcd_prompt_tokens.jsonl) -- about 6x the old synthetic
     corpus's entire size, per single request.
  2. You start the vLLM server for THIS run with a deliberately
     constrained --gpu-memory-utilization (0.3 instead of the default
     ~0.9), documented plainly in the paper as an intentional way to
     exercise the controller's full adaptive range within a feasible
     request volume on a single rented GPU, rather than requiring an
     unrealistic number of concurrent connections against a full-size
     cache pool. This is a disclosed methodological choice, not a hidden
     one -- state it exactly this way in the manuscript.

Run (from repo root), against a server started with a SMALLER cache pool
than the main sweep uses:

    vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 --max-model-len 8192 \
        --served-model-name meta-llama/Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.3

    python benchmark/run_controller_ablation_v2.py \
        --host http://localhost:8000 --out-dir results/controller_ablation_v2

Output: results/controller_ablation_v2/{fixed,dynamic}_ratio_log.jsonl,
same schema as v1 (request_id, epoch, condition, ratio,
kv_cache_occupancy_at_request, ttft_ms, total_latency_ms), plus a summary
printed at the end showing the ratio range actually explored.
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, DynamicRatioController, DynamicRatioConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PRUNER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_TOKENS = 256

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

# Higher concurrency ramp than v1 (which topped out at 32) -- combined with
# the constrained cache pool above, this is what actually pushes occupancy
# into the controller's threshold range. Six 90s steps (~9 min/condition).
LOAD_PROFILE = [
    (90, 10),
    (90, 30),
    (90, 80),
    (90, 80),
    (90, 30),
    (90, 10),
]

FIXED_RATIO = 0.5


def load_arcd_pool():
    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    # (question, list-of-6-real-documents) pairs -- same pool used by
    # scripts/run_arcd_pilot.py and results/arcd_prompt_tokens.jsonl.
    return [(item["question"], item["documents"]) for item in eval_set]


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_context(pruner, query, docs, ratio):
    """ratio == 1.0 means raw/unpruned (used implicitly when the controller
    or fixed setting returns a ratio of 1.0 -- in practice both conditions
    here always prune, since FIXED_RATIO=0.5 and the controller's ceiling
    is < 1.0, but this keeps the function correct if that ever changes)."""
    if ratio >= 1.0:
        return " ".join(docs)
    result = pruner.prune(query, docs, compression_ratio=ratio)
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


def worker_loop(host, pool, pruner, controller, condition, stop_event, log_lock, log_file):
    session = requests.Session()
    while not stop_event.is_set():
        query, docs = random.choice(pool)
        docs = naive_retrieve_order(query, docs, top_k=6)
        if condition == "fixed":
            ratio = FIXED_RATIO
            occupancy = controller._fetch_gpu_cache_usage()
        else:
            occupancy = controller._fetch_gpu_cache_usage()
            ratio = controller.get_ratio(fallback_ratio=FIXED_RATIO, usage=occupancy)
        context = build_context(pruner, query, docs, ratio)
        one_request(session, host, context, query, ratio, condition, occupancy, log_lock, log_file)


def run_condition(host, pool, condition, out_dir):
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME)
    controller = DynamicRatioController(config=DynamicRatioConfig(metrics_url=f"{host}/metrics"))

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
                args=(host, pool, pruner, controller, condition, stop_event, log_lock, log_file),
                daemon=True,
            )
            for _ in range(n_workers)
        ]
        for t in threads:
            t.start()
        # Print a mid-step occupancy sample so you can watch it live and
        # confirm the constrained cache pool is actually being stressed
        # before committing to the full ~18-minute run.
        time.sleep(duration_s / 2)
        occ = controller._fetch_gpu_cache_usage()
        print(f"    mid-step occupancy sample: {occ}", flush=True)
        time.sleep(duration_s / 2)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)

    log_file.close()
    return out_path


def summarize(path):
    ttfts, lats, ratios, occs = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                ttfts.append(rec["ttft_ms"])
                lats.append(rec["total_latency_ms"])
            if rec.get("ratio") is not None:
                ratios.append(rec["ratio"])
            if rec.get("kv_cache_occupancy_at_request") is not None:
                occs.append(rec["kv_cache_occupancy_at_request"])
    if not ttfts:
        print(f"  {path.name}: no successful requests logged.")
        return
    print(f"  {path.name}: n={len(ttfts)} ok, "
          f"TTFT mean={statistics.mean(ttfts):.0f}ms p95={sorted(ttfts)[int(0.95*len(ttfts))-1]:.0f}ms, "
          f"latency mean={statistics.mean(lats):.0f}ms, "
          f"ratio range=[{min(ratios):.2f},{max(ratios):.2f}], "
          f"occupancy range=[{min(occs):.4f},{max(occs):.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="e.g. http://localhost:8000")
    ap.add_argument("--out-dir", default="results/controller_ablation_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_arcd_pool()
    print(f"Loaded {len(pool)} real ARCD (question, 6-doc pool) pairs for load generation.")

    fixed_path = run_condition(args.host, pool, "fixed", out_dir)
    dynamic_path = run_condition(args.host, pool, "dynamic", out_dir)

    print("\n--- summary ---")
    summarize(fixed_path)
    summarize(dynamic_path)
    print(
        "\nIf 'dynamic' ratio range is still pinned at a single value, the "
        "cache pool is still not constrained enough -- rerun with a lower "
        "--gpu-memory-utilization (e.g. 0.2) or a higher step in LOAD_PROFILE "
        "before concluding the controller doesn't adapt. If the range now "
        "spans multiple values, plot kv_cache_occupancy_at_request vs. ratio "
        "over time for 'dynamic' against the flat line for 'fixed', and "
        "compare TTFT/latency during the high-concurrency step between "
        "conditions -- that comparison is the actual ablation result."
    )


if __name__ == "__main__":
    main()
