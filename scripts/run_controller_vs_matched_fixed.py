"""
Dynamic controller vs. a MATCHED fixed-ratio baseline -- closes the gap
flagged in review: run_controller_ablation_v2.py already shows the
dynamic controller's ratio adapts (up to 99.2% KV-cache occupancy, full
ratio range exercised), but it compares "dynamic" only against a fixed
ratio of 0.5 chosen a priori, not against a fixed ratio matched on the
SAME average number of retained tokens the dynamic controller actually
used. Without that match, any latency/throughput/quality difference
between conditions is confounded with the two conditions simply
retaining different amounts of context on average -- so this script
cannot, by itself, support a claim that ADAPTING outperforms a
comparably-sized fixed policy. It can only support (or fail to support)
that claim once the two arms are matched.

Design (two-pass, both against a live vLLM server):

  Pass 1 -- "dynamic": identical to run_controller_ablation_v2.py's
  dynamic condition (same LOAD_PROFILE, same constrained-cache-pool
  server, same real ARCD pool). Logs every request's realized ratio.

  Pass 2 -- "matched_fixed": computes r_matched = mean(ratio) from pass
  1's log, then re-runs the IDENTICAL load profile and query sequence
  (same manifest order) with the ratio pinned at r_matched for every
  request. This is the actual apples-to-apples comparison: both arms
  retain the same average fraction of context; only one of them varies
  that fraction request-to-request in response to live cache pressure.

  Answer-quality pass: for a fixed subset of ARCD questions (default 40,
  same eval set used throughout the paper), each arm's realized
  per-request ratio is replayed OFFLINE against the ARCD pool to
  generate an answer at that exact ratio, scored with the paper's own
  token-F1 scorer (scripts/score_arcd.py, imported directly so the
  metric definition cannot drift from the one already validated against
  results/arcd_stats_summary.csv). This measures whether the dynamic
  policy's per-request ratio CHOICES help or hurt answer quality
  relative to the matched constant choice, independent of serving load.

Run (from repo root), against the SAME constrained-cache-pool server
setup as run_controller_ablation_v2.py:

    vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 --max-model-len 8192 \
        --served-model-name meta-llama/Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.3

    python scripts/run_controller_vs_matched_fixed.py \
        --host http://localhost:8000 --out-dir results/controller_vs_matched

Output: results/controller_vs_matched/{dynamic,matched_fixed}_log.jsonl
(serving-side, same schema as run_controller_ablation_v2.py) and
results/controller_vs_matched/quality_{dynamic,matched_fixed}.jsonl
(answer-quality replay), plus a printed summary comparing throughput,
p95 TTFT/latency, and mean F1 between the two arms at the matched
average ratio.

Not yet run: requires a live GPU and the NIM/vLLM chat endpoint used
elsewhere in this project for generation. This script is the
pre-registered design for that run -- committing it before running it
means the comparison cannot be quietly reshaped after seeing results.
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
from scripts.score_arcd import score  # noqa: E402  (paper's own validated F1 scorer)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PRUNER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_TOKENS = 256

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

# Identical serving load profile to run_controller_ablation_v2.py, so the
# two scripts' "dynamic" arms are directly comparable if both are run.
LOAD_PROFILE = [
    (90, 10),
    (90, 30),
    (90, 80),
    (90, 80),
    (90, 30),
    (90, 10),
]

N_QUALITY_QUESTIONS = 40


def load_arcd_pool():
    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    return [(item["id"], item["question"], item["documents"], item["gold_answers"]) for item in eval_set]


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_context(pruner, query, docs, ratio):
    if ratio >= 1.0:
        return " ".join(docs)
    result = pruner.prune(query, docs, compression_ratio=ratio)
    return result.pruned_text


# --------------------------------------------------------------------------
# Pass 1/2: serving-side load, logs realized ratio per request
# --------------------------------------------------------------------------

def one_request(session, host, context, query, ratio, condition, occupancy, log_lock, log_file):
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
                record = {"request_id": request_id, "epoch": t0_epoch, "condition": condition,
                          "ratio": ratio, "kv_cache_occupancy_at_request": occupancy,
                          "status": "http_error", "http_status": resp.status_code}
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
                record = {"request_id": request_id, "epoch": t0_epoch, "condition": condition,
                          "ratio": ratio, "kv_cache_occupancy_at_request": occupancy,
                          "status": "ok", "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms}
    except Exception as e:
        record = {"request_id": request_id, "epoch": t0_epoch, "condition": condition,
                  "ratio": ratio, "kv_cache_occupancy_at_request": occupancy,
                  "status": "exception", "error": str(e)}
    with log_lock:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()


def worker_loop(host, pool, pruner, controller, condition, fixed_ratio, stop_event, log_lock, log_file, seed):
    session = requests.Session()
    rng = random.Random(seed)
    while not stop_event.is_set():
        _, query, docs, _ = rng.choice(pool)
        docs = naive_retrieve_order(query, docs, top_k=6)
        if condition == "matched_fixed":
            ratio = fixed_ratio
            occupancy = controller._fetch_gpu_cache_usage()
        else:
            occupancy = controller._fetch_gpu_cache_usage()
            ratio = controller.get_ratio(fallback_ratio=fixed_ratio, usage=occupancy)
        context = build_context(pruner, query, docs, ratio)
        one_request(session, host, context, query, ratio, condition, occupancy, log_lock, log_file)


def run_condition(host, pool, condition, fixed_ratio, out_dir):
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME)
    controller = DynamicRatioController(config=DynamicRatioConfig(metrics_url=f"{host}/metrics"))
    out_path = out_dir / f"{condition}_log.jsonl"
    log_file = open(out_path, "w", encoding="utf-8")
    log_lock = threading.Lock()

    print(f"\n=== condition={condition} (fixed_ratio={fixed_ratio}) -> {out_path} ===", flush=True)
    for step_i, (duration_s, n_workers) in enumerate(LOAD_PROFILE):
        print(f"  step {step_i+1}/{len(LOAD_PROFILE)}: {n_workers} workers for {duration_s}s", flush=True)
        stop_event = threading.Event()
        threads = [
            threading.Thread(target=worker_loop,
                              args=(host, pool, pruner, controller, condition, fixed_ratio,
                                    stop_event, log_lock, log_file, 1000 * step_i + w),
                              daemon=True)
            for w in range(n_workers)
        ]
        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)
    log_file.close()
    return out_path


def summarize_serving(path):
    ttfts, lats, ratios = [], [], []
    n_ok = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                n_ok += 1
                ttfts.append(rec["ttft_ms"])
                lats.append(rec["total_latency_ms"])
            if rec.get("ratio") is not None:
                ratios.append(rec["ratio"])
    if not ttfts:
        return None
    p95 = lambda xs: sorted(xs)[max(0, int(0.95 * len(xs)) - 1)]
    return {
        "n_ok": n_ok,
        "mean_ratio": statistics.mean(ratios),
        "ttft_mean": statistics.mean(ttfts), "ttft_p95": p95(ttfts),
        "latency_mean": statistics.mean(lats), "latency_p95": p95(lats),
    }


# --------------------------------------------------------------------------
# Quality replay: offline, using the same NIM chat client as run_arcd_pilot.py
# --------------------------------------------------------------------------

def quality_replay(host_chat_client, pool, condition, realized_ratios, out_dir, n_questions):
    """realized_ratios: list of per-request ratios actually logged for this
    condition during the serving pass, used here as an empirical
    distribution to draw from -- for 'matched_fixed' this collapses to a
    single constant, for 'dynamic' it reproduces the real ratio spread the
    controller exhibited."""
    from middleware.pruning import SemanticPruner
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME)
    rng = random.Random(42)
    subset = pool[:n_questions]
    out_path = out_dir / f"quality_{condition}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f_out:
        for qid, question, docs, gold in subset:
            ratio = rng.choice(realized_ratios)
            docs_ranked = naive_retrieve_order(question, docs, top_k=6)
            context = build_context(pruner, question, docs_ranked, ratio)
            prompt = f"السياق: {context}\n\nالسؤال: {question}"
            resp = host_chat_client.chat(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
            answer = resp.text
            s = score(answer, gold)
            f_out.write(json.dumps({
                "id": qid, "question": question, "condition": condition, "ratio": ratio,
                "answer": answer, "gold_answers": gold, "em": s["em"], "f1": s["f1"],
            }, ensure_ascii=False) + "\n")
            f_out.flush()
    return out_path


def summarize_quality(path):
    f1s = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            f1s.append(json.loads(line)["f1"])
    return statistics.mean(f1s) if f1s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="vLLM serving host, e.g. http://localhost:8000")
    ap.add_argument("--out-dir", default="results/controller_vs_matched")
    ap.add_argument("--skip-quality", action="store_true",
                     help="Skip the answer-quality replay pass (requires VLLM_API_KEY / NIM access).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_arcd_pool()
    print(f"Loaded {len(pool)} real ARCD (id, question, 6-doc pool, gold) tuples.")

    # Pass 1: dynamic, to learn its realized mean ratio.
    dynamic_path = run_condition(args.host, pool, "dynamic", fixed_ratio=0.5, out_dir=out_dir)
    dyn_stats = summarize_serving(dynamic_path)
    if dyn_stats is None:
        print("No successful 'dynamic' requests logged -- aborting before running the matched arm.")
        return
    r_matched = round(dyn_stats["mean_ratio"], 3)
    print(f"\nDynamic arm realized mean ratio = {r_matched}. "
          f"Running 'matched_fixed' at this SAME average ratio next.")

    # Pass 2: matched_fixed, pinned at dynamic's own realized average.
    matched_path = run_condition(args.host, pool, "matched_fixed", fixed_ratio=r_matched, out_dir=out_dir)
    matched_stats = summarize_serving(matched_path)

    print("\n--- serving-side summary (matched on average retained context) ---")
    for name, stats in (("dynamic", dyn_stats), ("matched_fixed", matched_stats)):
        if stats is None:
            print(f"  {name}: no successful requests logged.")
            continue
        print(f"  {name:14s}: n={stats['n_ok']:4d} mean_ratio={stats['mean_ratio']:.3f} "
              f"TTFT mean/p95={stats['ttft_mean']:.0f}/{stats['ttft_p95']:.0f}ms "
              f"latency mean/p95={stats['latency_mean']:.0f}/{stats['latency_p95']:.0f}ms")

    if args.skip_quality:
        print("\n--skip-quality set: not running the answer-quality replay pass.")
        return

    from middleware.vllm_client import VLLMClient
    import os
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        print("\nVLLM_API_KEY not set -- skipping answer-quality replay pass "
              "(re-run with it set, or pass --skip-quality to suppress this message).")
        return
    chat_client = VLLMClient(
        api_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="meta/llama-3.1-8b-instruct", api_key=api_key, timeout_s=60,
    )

    dyn_ratios = [json.loads(l)["ratio"] for l in open(dynamic_path, encoding="utf-8")
                  if json.loads(l).get("ratio") is not None]
    matched_ratios = [r_matched]

    print(f"\nRunning answer-quality replay ({N_QUALITY_QUESTIONS} questions/arm)...")
    q_dyn_path = quality_replay(chat_client, pool, "dynamic", dyn_ratios, out_dir, N_QUALITY_QUESTIONS)
    q_matched_path = quality_replay(chat_client, pool, "matched_fixed", matched_ratios, out_dir, N_QUALITY_QUESTIONS)

    f1_dyn = summarize_quality(q_dyn_path)
    f1_matched = summarize_quality(q_matched_path)
    print("\n--- answer-quality summary ---")
    print(f"  dynamic:       mean F1 = {f1_dyn:.4f}" if f1_dyn is not None else "  dynamic: no data")
    print(f"  matched_fixed: mean F1 = {f1_matched:.4f}" if f1_matched is not None else "  matched_fixed: no data")
    print(
        "\nReport this comparison exactly as run: if dynamic's F1/latency does not "
        "clearly beat matched_fixed's, state plainly in the manuscript that the "
        "controller was not shown to outperform a matched fixed policy on this "
        "sample -- do not report only the serving-side ablation (which shows "
        "adaptation occurs) as if it also showed adaptation helps."
    )


if __name__ == "__main__":
    main()
