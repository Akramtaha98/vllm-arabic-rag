# Network-Aware Edge-Cloud Experiment: Implementation Status

Covers Tasks 1-7 of the network-aware conversion instruction. Read this
alongside `paper/NETWORK_SPECIALIZATION_PLAN.md` (the design document from
the previous round). **No GPU is currently rented and no edge/cloud network
link exists to test against, so the validation grid (Task 6) has not been
executed.** Everything below is either (a) code that is written and
unit-verified against synthetic/local inputs without a GPU, or (b) honestly
marked as blocked pending GPU rental. Nothing in this document is a
fabricated or estimated experimental result.

## 1. What was implemented, in the order requested

**Task 1 (repair the existing harness).** `benchmark/request_manifest.py`
(new) generates a fixed, seeded query sequence shared identically across
every cell — verified reproducible (same seed produces byte-identical
output). `benchmark/locustfile.py` and the new `benchmark/locustfile_edge.py`
consume this manifest instead of `random.choice`, and now request
`stream_options: {"include_usage": true}` so prompt/completion token
counts come from vLLM's own `usage` field rather than an SSE-chunk-count
estimate (falls back to the old estimate, explicitly flagged
`token_count_source: "estimated_chunk_count"`, only if the server doesn't
send it). `benchmark/metrics_scraper.py` now records an absolute epoch
timestamp on every KV-cache sample (previously only a scraper-relative
`t_seconds`), so KV-cache samples can be joined post-hoc against the edge
gateway's and locust's own per-request logs by wall-clock time — this is
the "synchronized KV-cache sampling" fix. `benchmark/warmup.py` (new) runs
a fixed number of sequential, unmeasured requests immediately before each
timed cell, in both direct-vLLM and gateway modes. Repeats (>=3) and
randomized cell order were already present in the existing v2 harness
(`run_full_sweep.py`) from a prior round and are reused, not rebuilt.

**Task 2 (client-edge-cloud prototype).** `benchmark/edge_gateway.py` (new,
FastAPI): hosts retrieval (reusing the existing mock corpus), pruning
(raw/naive/LSPM), and the controller decision; forwards to the cloud vLLM
server; streams the response internally and returns one JSON response with
TTFT, total latency, token counts, and byte counts already computed at the
edge. `benchmark/locustfile_edge.py` (new) is the client, calling the
gateway's single `/generate` endpoint. Centralized logging: the gateway
writes one JSONL line per request (network inputs, controller decision,
timing, byte sizes, tokens) to `REQUEST_LOG_PATH`.

**Task 3 (network emulation).** `benchmark/network_profiles.py` (new)
defines the four required profiles (`edge_lan`, `good_cloud`,
`moderate_wan`, `constrained_wireless`) as `tc netem` parameter sets, with
`apply_profile()` (raises rather than silently continuing on permission or
interface errors) and `verify_profile()`, which pings the real target and
records the **measured** RTT mean/stddev and packet loss — not just the
requested parameters — specifically because the paper's own integrity rule
requires reporting what was actually measured, not just what was
requested.

**Task 4 (five conditions).** `raw`, `naive`, `fixed_lspm` reuse the
existing, already-validated pruning code. `kv_aware` wraps
`middleware.pruning.DynamicRatioController` — the paper's Section 3.4
design, which had never been run before this implementation pass — in a
logging shim (`benchmark/network_controller.py:KVOnlyDecision`) so its
decisions are recorded in the same schema as the new controller.
`network_aware` is the new controller described next.

**Task 5 (network-aware controller).** `benchmark/network_controller.py`
(new): a transparent, rule-based controller — explicitly not
reinforcement learning, per instruction. Five inputs (bandwidth, RTT,
queue length, concurrency, KV-cache utilization), each min-max normalized
against a documented placeholder calibration range, combined into a
weighted pressure score, mapped linearly to a ratio, then snapped to the
paper's already fidelity-validated ratios (0.3/0.5/0.7, Section 5.4) so
the controller can never select a ratio whose answer-quality effect is
unmeasured. Every decision is logged with every raw input, every
normalized input, the weights used, and the resulting ratio.

**Task 6 (small validation grid).** `benchmark/run_validation_grid.py`
(new): orchestrates the exact grid specified — edge_lan and
constrained_wireless profiles x concurrency {1, 10, 50} x five methods x 3
repeats (90 cells) — with mandatory warm-up before each cell, mandatory
profile verification before any cell under a profile runs, and per-cell
failure handling that never fabricates or silently substitutes a missing
measurement (a failed cell is retried up to `--max-retries` and then
recorded as failed with a specific reason, never dropped or interpolated).

## 2. What was actually verified, without a GPU, and how

Everything below was executed for real in this working environment (no
GPU present, no root network access exercised on the shared interface) —
these are not simulated or hand-traced results, they are real program runs:

- **`request_manifest.py`**: generated a 200-entry manifest twice with the
  same seed; byte-identical output confirmed reproducibility.
- **`network_controller.py`**: unit-tested the `NetworkAwareController`
  against synthetic inputs — best-case network/server conditions select
  the loosest validated ratio (0.7), worst-case selects the tightest
  (0.3), missing inputs fall back to a neutral score without crashing,
  the ratio is confirmed monotonically non-increasing as RTT alone rises
  (holding all other inputs at their best value), and malformed weight
  configurations (not summing to 1.0) are rejected at construction time.
- **`network_profiles.py`**: the ping-output parser was unit-tested
  against a realistic canned `ping` transcript (correct RTT mean and 0%
  loss recovered); `tc qdisc del ... dev lo` was run against loopback
  (safe, no-op) to confirm command construction without touching the
  shared sandbox network interface.
- **`metrics_scraper.py`**: ran a real 2-second scrape loop against an
  intentionally unreachable URL with `--gpu-util` set (no GPU present) —
  confirmed it degrades gracefully, writes a fully-populated CSV with
  explicit `error`/`gpu_error` reasons on every row, and never crashes or
  silently omits columns.
- **`warmup.py`**: ran against an unreachable host — confirmed it reports
  every failure individually, exits non-zero, and refuses to let a caller
  proceed to a measurement run against an unverified server.
- **`edge_gateway.py`**: imported the module directly and exercised
  `build_context()`, the request tracker's queue/in-flight state machine,
  and `decide_ratio()` for all five methods against an unreachable cloud
  URL — confirmed graceful fallback (`fallback_used=True`) rather than a
  crash for `kv_aware`/`network_aware`. Then started it as a **real
  running FastAPI/uvicorn server** on a local port and hit `/health` and
  `/generate` with real HTTP requests — confirmed correct JSON responses,
  including a clean, structured error response (not a 500) when the
  (deliberately unreachable) cloud tier fails, and confirmed the failure
  was correctly written to the JSONL request log.
- **`run_validation_grid.py`**: ran it end-to-end against a nonexistent
  network interface and unreachable hosts — confirmed it fails fast per
  profile with a specific, logged reason (`profile_apply_failed`), covers
  all 30 cells under that profile with honest failure records rather than
  hanging or crashing uncontrolled, and — after a bug found during this
  exact test (profile-apply/verify failures were being recorded in memory
  but never written to the JSONL log file) was fixed and re-verified — now
  persists every cell's outcome to disk correctly.

**One real bug was found and fixed during this verification pass**,
described above; it is disclosed here rather than silently corrected,
consistent with the integrity requirement not to conceal problems found
during development.

## 3. Task 7's required status report

- **Is the instrumentation reliable?** The pieces testable without a GPU
  (manifest determinism, controller decision logic and its edge cases,
  network-profile application/verification/failure handling, metrics
  scraper resilience, warm-up failure handling, and the edge gateway's
  request/response/logging cycle including its failure path) all passed
  real, executed checks, including one bug caught and fixed. The pieces
  that require a live vLLM server, a live network link, and root/
  CAP_NET_ADMIN on a real interface (the actual `tc netem` shaping taking
  effect on real traffic, the controller's live KV-cache/bandwidth/RTT
  fetch path under real load, and the full locust-driven concurrency
  behavior) are **unverified** — they are implemented and passed every
  test that could be run without that infrastructure, but "reliable under
  real load" cannot be claimed until they are.
- **Does the previous high-concurrency KV-cache reversal (Section 5.8)
  remain?** **Not yet determined.** This requires a live vLLM server and
  has not been re-run. `paper/NETWORK_SPECIALIZATION_PLAN.md` Section 1.4
  already extracted what can be said from the *existing* v1 data without a
  new run (naive truncation's KV-cache grows far faster than raw's or
  LSPM's at high concurrency, consistent with — but not proof of — the
  already-disclosed run-order confound); nothing new can be added to that
  until the validation grid actually runs.
- **TTFT and p95 end-to-end latency:** not measured (no GPU).
- **Transmitted bytes:** not measured (no live network link), though the
  gateway now logs `request_bytes`/`response_bytes` for every request once
  one is made.
- **Throughput and KV-cache utilization:** not measured (no GPU).
- **Answer correctness and faithfulness:** not measured, and would in any
  case require pointing the manifest at the ARCD ground-truth pool instead
  of the 5-query mock corpus first (documented as a known next step in
  `edge_gateway.py`'s closing comment) — the mock corpus has no gold
  answers to score against.
- **Failures and unresolved confounds:** the one code bug described in
  Section 2 (now fixed). No experimental confounds to report yet, since no
  experiment has run.

## 4. What is needed to move from "implemented" to "run"

1. Rent a GPU for the cloud tier (same vLLM setup as
   `benchmark/BENCHMARK_INSTRUCTIONS.md`).
2. A second machine, VM, or network namespace for the edge tier, with
   root/CAP_NET_ADMIN on the interface facing the cloud tier (a single
   box with two network namespaces joined by a `veth` pair, one shaped
   with `tc netem`, is the cheapest way to do this without a second
   physical machine or a second cloud instance).
3. Generate a manifest sized for the grid
   (`python benchmark/request_manifest.py --length 6000`).
4. Start the cloud vLLM server, then the edge gateway
   (`CLOUD_VLLM_URL=... uvicorn benchmark.edge_gateway:app --host 0.0.0.0 --port 9000`).
5. Run `python benchmark/run_validation_grid.py --edge-host http://localhost:9000 --cloud-metrics-url http://<cloud-host>:8000/metrics --edge-iface <iface> --ping-target <cloud-host> ...`.
6. Send me the resulting `results/validation_grid/` directory; I will
   analyze it and report Task 7's items with real numbers, extending
   `run_validation_grid.py`'s output with the ARCD-pool answer-quality
   pass described in `edge_gateway.py`'s closing note.

Nothing above should be summarized to a committee or reviewer as
"completed" until step 6 has actually happened.
