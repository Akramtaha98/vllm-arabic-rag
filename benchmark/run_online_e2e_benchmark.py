"""
True end-to-end ("online") serving benchmark: unlike run_full_sweep.py
(which reads every context from benchmark/precomputed_contexts.json, see
precompute_contexts.py), this script performs sentence splitting,
cross-encoder scoring, and order-preserving context reconstruction live,
inside the timed request path, for every LSPM request. It exists to close
a specific gap flagged in review: the paper's main throughput/latency
sweep excludes LSPM's cross-encoder cost (measured separately in Section
3.1 at a mean of 506.1 ms per request) because it runs against
precomputed contexts. That is a deliberate, disclosed choice to isolate
vLLM-side serving cost from client-side pruning cost -- but it means the
paper's "within 3% of raw throughput" headline is not, on its own,
an end-to-end online-serving claim. This script produces that claim.

What it measures, per request, for each of raw / lspm / naive:
  - pruning_latency_ms: wall-clock time for split + cross-encoder score +
    rank + reconstruct (0 for raw and naive, which are not cross-encoder
    scored; naive's split cost is included since it uses the same
    splitter, see naive_truncate() below).
  - ttft_ms: time from the moment this request's context-build STARTS
    (not from when the HTTP POST is sent) to the first streamed token.
    This is the key methodological difference from locustfile.py, whose
    TTFT clock starts only after build_context() already returned.
  - total_latency_ms: same start point, through the final streamed token.
  - completion tokens/sec, from the server's usage field when available.

This intentionally sacrifices Locust's process-pool concurrency model: a
small, fixed pool of Python threads issues requests directly against the
vLLM server's OpenAI-compatible endpoint, so that pruning time and
network time are attributed to the same wall clock and cannot be hidden
behind Locust's gevent scheduler (the original confound documented in
precompute_contexts.py's docstring, Section 5.8/Appendix G). Concurrency
here means "number of live worker threads issuing requests back-to-back
with think time," not Locust's spawn-rate semantics -- report it as such.

Run (from repo root), against a normally-configured vLLM server (default
--gpu-memory-utilization, i.e. the SAME server configuration as the main
sweep in run_full_sweep.py, not the constrained pool used in
run_controller_ablation_v2.py):

    vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 --max-model-len 8192 \
        --served-model-name meta-llama/Llama-3.1-8B-Instruct

    python benchmark/run_online_e2e_benchmark.py \
        --host http://localhost:8000 \
        --concurrency 1 5 10 20 \
        --ratio 0.5 \
        --requests-per-cell 60 \
        --out-dir results/online_e2e

Output: results/online_e2e/{raw,lspm,naive}_c{N}.jsonl (one line per
request) plus results/online_e2e/summary.csv (one row per
method x concurrency cell: mean/p95 pruning_latency_ms, mean/p95 ttft_ms,
mean/p95 total_latency_ms, mean tokens/sec, n_ok, n_error). Feed
summary.csv into analyze_sweep_results.py-style plotting, or directly
into the manuscript's Section 4/Table numbers -- this script does not
overwrite any existing table, it produces the NEW online-serving numbers
the reviewer asked for, to be added alongside the existing
precomputed-context sweep, not in place of it. The two are different,
both-legitimate measurements: the existing sweep isolates vLLM serving
cost; this one reports the fully honest cost a real deployment would pay.

Not yet run: requires a live GPU. Committing this script documents the
exact, pre-registered methodology before that run happens, so the
eventual numbers cannot be quietly cherry-picked after the fact.
"""
import argparse
import csv
import json
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402
from benchmark.sample_queries import SAMPLE_QUERIES, SYSTEM_PROMPT  # noqa: E402

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PRUNER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_TOKENS = 256


def naive_truncate(documents, compression_ratio):
    """Identical definition to locustfile.py's naive_truncate -- kept in
    sync by hand, see that file's docstring note. Its split_sentences()
    call is included inside the timed window below since it is real
    per-request work, even though it is far cheaper than cross-encoder
    scoring."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    n = len(all_sentences)
    num_to_keep = max(1, int(round(n * compression_ratio)))
    num_to_keep = min(num_to_keep, n)
    return " ".join(all_sentences[:num_to_keep])


def build_context_timed(pruner, mode, query, docs, ratio):
    """Returns (context, pruning_latency_ms). Every branch is timed,
    including raw's trivial join, so the reported pruning_latency_ms for
    raw is a true (near-zero) baseline rather than an assumed zero."""
    t0 = time.perf_counter()
    if mode == "lspm":
        result = pruner.prune(query, docs, compression_ratio=ratio)
        context = result.pruned_text
    elif mode == "naive":
        context = naive_truncate(docs, ratio)
    else:
        context = " ".join(docs)
    pruning_latency_ms = (time.perf_counter() - t0) * 1000
    return context, pruning_latency_ms


def one_request(session, host, mode, ratio, pruner, query, docs, log_lock, log_file):
    request_id = str(uuid.uuid4())
    t_build_start = time.perf_counter()
    context, pruning_latency_ms = build_context_timed(pruner, mode, query, docs, ratio)

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

    ttft_ms = None
    usage = None
    completion_tokens_est = 0
    t0_epoch = time.time()
    try:
        with session.post(f"{host}/v1/chat/completions", json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                record = {
                    "request_id": request_id, "epoch": t0_epoch, "mode": mode, "ratio": ratio,
                    "status": "http_error", "http_status": resp.status_code,
                    "pruning_latency_ms": pruning_latency_ms,
                }
                _write(log_lock, log_file, record)
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[len("data: "):]
                line = line.strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    if ttft_ms is None:
                        # Clock starts at t_build_start, BEFORE the HTTP call --
                        # this is the methodological difference from
                        # locustfile.py's TTFT, which starts after build_context()
                        # already returned.
                        ttft_ms = (time.perf_counter() - t_build_start) * 1000
                    completion_tokens_est += 1
        total_latency_ms = (time.perf_counter() - t_build_start) * 1000
    except Exception as e:
        record = {
            "request_id": request_id, "epoch": t0_epoch, "mode": mode, "ratio": ratio,
            "status": "exception", "error": str(e), "pruning_latency_ms": pruning_latency_ms,
        }
        _write(log_lock, log_file, record)
        return

    if usage:
        completion_tokens = usage.get("completion_tokens")
        token_count_source = "usage_field"
    else:
        completion_tokens = completion_tokens_est
        token_count_source = "estimated_chunk_count"

    record = {
        "request_id": request_id, "epoch": t0_epoch, "mode": mode, "ratio": ratio,
        "status": "ok", "pruning_latency_ms": pruning_latency_ms,
        "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms,
        "completion_tokens": completion_tokens, "token_count_source": token_count_source,
    }
    _write(log_lock, log_file, record)


def _write(log_lock, log_file, record):
    with log_lock:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()


def run_cell(host, mode, ratio, n_concurrency, n_requests, out_dir):
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME) if mode == "lspm" else None
    docs = MOCK_CORPUS
    out_path = out_dir / f"{mode}_c{n_concurrency}.jsonl"
    log_file = open(out_path, "w", encoding="utf-8")
    log_lock = threading.Lock()

    requests_per_worker = [n_requests // n_concurrency + (1 if i < n_requests % n_concurrency else 0)
                            for i in range(n_concurrency)]

    def worker(n_reqs):
        session = requests.Session()
        for i in range(n_reqs):
            query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            one_request(session, host, mode, ratio, pruner, query, docs, log_lock, log_file)

    print(f"  {mode} @ concurrency={n_concurrency}: {n_requests} requests -> {out_path}", flush=True)
    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in requests_per_worker]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t_start
    log_file.close()
    return out_path, wall_s


def summarize_cell(path, wall_s, n_concurrency):
    prune_lat, ttfts, lats, toks = [], [], [], []
    n_ok = n_err = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                n_ok += 1
                prune_lat.append(rec["pruning_latency_ms"])
                ttfts.append(rec["ttft_ms"])
                lats.append(rec["total_latency_ms"])
                if rec.get("completion_tokens"):
                    toks.append(rec["completion_tokens"])
            else:
                n_err += 1
    if not ttfts:
        return None
    p95 = lambda xs: sorted(xs)[max(0, int(0.95 * len(xs)) - 1)]
    total_tokens = sum(toks)
    return {
        "mode": path.stem.rsplit("_c", 1)[0],
        "concurrency": n_concurrency,
        "n_ok": n_ok,
        "n_error": n_err,
        "pruning_latency_ms_mean": round(statistics.mean(prune_lat), 1),
        "pruning_latency_ms_p95": round(p95(prune_lat), 1),
        "ttft_ms_mean": round(statistics.mean(ttfts), 1),
        "ttft_ms_p95": round(p95(ttfts), 1),
        "total_latency_ms_mean": round(statistics.mean(lats), 1),
        "total_latency_ms_p95": round(p95(lats), 1),
        "throughput_req_per_s": round(n_ok / wall_s, 3) if wall_s > 0 else 0,
        "throughput_tokens_per_s": round(total_tokens / wall_s, 1) if wall_s > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--requests-per-cell", type=int, default=60)
    ap.add_argument("--out-dir", default="results/online_e2e")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in ("raw", "lspm", "naive"):
        for c in args.concurrency:
            path, wall_s = run_cell(args.host, mode, args.ratio, c, args.requests_per_cell, out_dir)
            row = summarize_cell(path, wall_s, c)
            if row:
                rows.append(row)

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {summary_path}")
    print("\n--- summary ---")
    for row in rows:
        print(f"  {row['mode']:6s} c={row['concurrency']:3d}: "
              f"prune_ms={row['pruning_latency_ms_mean']:.1f} "
              f"ttft_ms={row['ttft_ms_mean']:.1f} "
              f"latency_ms={row['total_latency_ms_mean']:.1f} "
              f"throughput={row['throughput_req_per_s']:.2f} req/s")
    print(
        "\nCompare lspm's throughput_req_per_s / total_latency_ms_mean against "
        "raw's at matched concurrency: this is the true end-to-end comparison "
        "including cross-encoder cost. If lspm's advantage over raw shrinks or "
        "reverses relative to the precomputed-context sweep in "
        "results/sweep_summary_v2.csv, report both numbers side by side in the "
        "manuscript rather than only the more favorable one."
    )


if __name__ == "__main__":
    main()
