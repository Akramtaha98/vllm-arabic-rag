# Repositioning the Paper for "Computer Networks and Intelligent Systems": Gap Analysis, Design, and Execution Plan

**Prepared for:** Akram Taha
**Subject paper:** "Semantic-Driven Context Pruning for Arabic RAG Systems: Toward Memory-Efficient vLLM-Based Deployment" (current Round-8 draft, `paper_draft_v2.md`)
**Purpose of this document:** Answer the 10 tasks and produce the 10 deliverables requested, honestly. **This document is a plan and a partial analysis, not a rewritten paper.** Section 0 explains why, and exactly what is and is not usable right now.

---

## 0. Read this first: what is real, what is proposed, and why the two must stay separate

Everything below that describes an experiment, a measurement, or a number either (a) comes from data your paper already has archived (the Round-7 v1 GPU sweep, `results/sweep_summary.csv`, reproduced in the current paper's Appendix G), and is labeled **[EXISTING DATA]**, or (b) is a design/proposal that has not been run, and is labeled **[PROPOSED — NOT YET RUN]**. Nothing in this document should be copied into the paper as a completed result unless it is labeled **[EXISTING DATA]**.

Three hard constraints govern what can honestly be claimed right now:

1. **No GPU is currently rented.** The RunPod pod used for the v1 sweep is stopped. Nothing that requires a live vLLM server (any new controller test, any edge/cloud latency comparison, any concurrency sweep) can be executed until a GPU is rented again.
2. **No edge device or second physical network is available in this environment.** Everything called "edge" or "cloud" in Task 2/4 has to be **emulated** on rented cloud GPU(s) using a network-emulation tool (`tc`/`netem` on Linux, or a proxy like `toxiproxy`/`comcast`) between a client machine and the vLLM server. This is a legitimate, standard systems-research method, but it must be disclosed as emulation in every place it is reported, per your own instruction (Task 4, "do not describe an emulated environment as a real production network").
3. **A genuinely new controller and a genuinely new architecture are software-engineering work, not just paper-writing work.** Implementing the network-aware controller (Task 3), instrumenting the client/gateway/server to measure RTT/jitter/bandwidth/queue length (Task 6), and running the concurrency x network-condition grid (Task 4) is a multi-day implementation-and-experiment effort, not something that can be fabricated or shortcut. This document gives you the exact design and the exact commands, but running it is the next phase, gated on GPU rental.

Given this, the honest scope of *this* response is: complete Deliverables 1, 2, 3, 4, 5, 6, 7, 9, 10 as design/analysis/planning artifacts now (all of these are things a reviewer would expect to see designed before code is written), and give you a ready-to-execute runbook for the actual experiments, which I can help you run the moment a GPU is rented. Deliverable 8 (a section-by-section revision plan) is included, but the revision itself is not performed until real results exist to revise with, consistent with your Task 10 instruction not to make superficial or premature changes.

---

## Deliverable 1: Gap Analysis — Is This Currently a Computer-Networks Paper?

### 1.1 What intelligent-systems components already exist in the paper

Reading the current manuscript in full, the intelligent-systems contribution is real and already substantive:

- **RAG pipeline** (retriever + generator, Section 3) — standard architecture, TF-IDF/keyword retriever, LLM generation via a hosted Llama-3.1-8B-Instruct endpoint and, separately, a self-hosted vLLM instance for the GPU benchmark.
- **LSPM (Lightweight Semantic Pruning Middleware)** (Section 3) — a cross-encoder sentence-relevance scorer that prunes retrieved context before it reaches the generator. This is the paper's core intelligent-systems contribution: a learned relevance model making a content-selection decision.
- **A *design* for a dynamic compression-ratio controller** (Section 3.4) — reads vLLM's `vllm:gpu_cache_usage_perc` metric and is meant to tighten the pruning ratio as KV-cache pressure rises. The paper is explicit, correctly, that **this controller has never been evaluated against live traffic** — it is a described mechanism, not a tested one. This is the single component closest to "network/systems-aware control," and it is currently unimplemented and untested.
- **A real, measured vLLM GPU benchmark** (Section 5.8) — throughput, TTFT, KV-cache occupancy vs. concurrency, for one server, one GPU, no network variable at all (client and server are effectively co-located; Locust runs on the same or a low-latency link to the server).

### 1.2 Why the current connection to computer networks is weak

Being direct about this, since you asked for a defensible answer, not a comfortable one:

1. **There is no network variable anywhere in the current experiments.** Every experiment — the tokenization study, the fidelity pilot, the ARCD re-test, the end-to-end retrieval check, and the GPU benchmark — either calls a hosted API over an ordinary internet connection with no controlled or measured network condition, or runs Locust against a server on the same machine/rack. RTT, jitter, bandwidth, and packet loss are never measured, never varied, and never appear as an independent variable in any table.
2. **The compression-ratio controller is KV-cache-only by design, and untested even on that axis.** It cannot currently be called "network-aware" because it has no network input and has never run.
3. **"Deployment" language in the title and abstract is aspirational, not demonstrated.** The paper is explicit and honest about this (Section 1's scope note), which is good research integrity, but it also means there is currently no basis for network-engineering framing — the paper measures a compression method's effect on one GPU's KV-cache and throughput, not a distributed system's behavior under network variation.
4. **No edge/cloud, no multi-tier deployment, no distributed component exists anywhere in the code or the experiments.** The reference implementation is a single-process pipeline (retriever, LSPM, generator client) calling one endpoint.

In short: the paper is a legitimate **information-retrieval / NLP-systems** paper with one real systems measurement (Section 5.8) and one unimplemented systems-control idea (Section 3.4). It is not currently, in any defensible sense, a computer-networks paper. Retitling it without adding the missing experimental content (which is exactly what you have instructed me not to do) would not survive a networks-track reviewer or a specialization committee that checks the methodology, not just the title.

### 1.3 Claims that currently need additional experimental support to carry network/distributed-systems weight

| # | Claim in current paper | What's missing to support a networks claim | Section |
|---|---|---|---|
| 1 | LSPM reduces KV-cache occupancy (single GPU, c=1 only) | No transmitted-bytes measurement, no RTT/bandwidth variation, no multi-tier deployment | 5.8 |
| 2 | Dynamic controller design (KV-cache-only) | Never implemented against live traffic; no network input; no comparison against fixed ratios under varying conditions | 3.4 |
| 3 | "vLLM-based deployment" (title) | No deployment topology tested at all — single co-located client/server | Title, Abstract |
| 4 | GPU benchmark's high-concurrency KV-cache reversal (c≥25) | Root cause not isolated; only two candidate causes hypothesized, not tested individually (Section 8 of this document develops a targeted plan) | 5.8, 7, 8 |
| 5 | LSPM's overhead is "not obviously disqualifying" (57.8 ms CPU, single-request) | No measurement under concurrent load, no measurement of network-transmission savings from a smaller context (fewer bytes over the wire matters for a networks claim, and is currently never measured) | 5.6 |

### 1.4 Investigating the KV-cache reversal at high concurrency, using only existing data

**[EXISTING DATA]** — this analysis uses only numbers already in your paper's Appendix G (`results/sweep_summary.csv`, v1 single run). No new experiment was run to produce the table below; it is a re-reading of data you already have.

| Concurrency | Raw KV mean/peak (%) | LSPM r=0.3 KV mean/peak (%) | Naive r=0.3 KV mean/peak (%) | Naive Tok/s | Raw Tok/s |
|---|---|---|---|---|---|
| 1 | 0.15 / 0.75 | 0.11 / 0.50 | 0.21 / 0.90 | 50.81 | 50.98 |
| 10 | 0.77 / 1.26 | 0.70 / 1.40 | 1.30 / 2.66 | 47.59 | 48.17 |
| 25 | 1.13 / 1.79 | 1.32 / 2.30 | 2.83 / 4.92 | 43.31 | 44.89 |
| 50 | 1.67 / 2.51 | 2.06 / 3.41 | 5.02 / 9.58 | 38.19 | 39.96 |
| 100 | 2.93 / 4.45 | 3.70 / 5.96 | 10.66 / 17.44 | 25.01 | 29.47 |

Two things are visible directly in numbers you already have, without needing new data:

- **Naive truncation's KV-cache grows far faster than either raw's or LSPM's as concurrency rises** (10.66/17.44% at c=100, vs. raw's 2.93/4.45% and LSPM's 3.70/5.96% — naive is roughly 3.6-4x raw at c=100 despite starting from a *shorter* input than raw at every ratio). If input length reduction were the only mechanism at work, naive truncation (which also shortens input) should track LSPM, not exceed raw by a wide and growing margin. It does not.
- **Naive truncation's completion tokens/second degrades faster too** (25.01 tok/s at c=100 vs. raw's 29.47 and LSPM's 27.29), and naive's cells ran **last** in the fixed, non-randomized block order your paper already discloses (Section 5.8: "naive truncation, run last, is roughly 35-55 minutes into a single ~60-minute continuous run").

This is *consistent with*, but does not prove, the run-order confound your paper already names as the leading candidate cause (no server restart between method blocks, so naive's cells inherit whatever server-side state — fragmented KV-cache blocks, scheduler queue backlog, thermal/clock effects — accumulated during the prior ~35-55 minutes of raw and LSPM traffic). It is also consistent with naive truncation producing systematically longer or more repetitive generated output at high concurrency (a plausible but currently untested effect of handing the model a context truncated mid-sentence, which can degrade generation coherence and, in some decoding regimes, generation length). **Distinguishing these two candidate causes requires new data** (per-request generated-token-length distributions, and a randomized-order re-run) that does not exist yet — this is exactly what Deliverable 6 (the diagnostic plan) is designed to produce. Reporting a definitive cause today, from the existing aggregate data alone, would be over-claiming; the honest current status is "one candidate cause is directly supported by the run-order pattern in the data you have; a second candidate cause (output-length differences) is plausible but unverified without new per-request data."

---

## Deliverable 2: Mapping Every Proposed Addition to Your Specializations

| Proposed addition | General specialization: Computer Network Engineering & Wireless Communications | Specific specialization: Computer Networks and Intelligent Systems |
|---|---|---|
| Edge-cloud two-tier architecture (Deliverable 3) | Directly a network-topology and deployment-engineering question: where computation sits relative to the network, and what that costs in latency/bandwidth | Combines a distributed-systems topology with an ML inference pipeline — canonical "intelligent systems over networks" design |
| Network-aware controller (Deliverable 4) | Uses RTT, jitter, bandwidth, packet loss as first-class control inputs — core network-engineering measurement and control-theory content | The controller itself is a small online decision-maker (a learned or heuristic policy) — the "intelligent" half of the specific specialization |
| Network emulation harness (`tc`/`netem`) | Standard network-engineering methodology for reproducing WAN/wireless conditions (bandwidth caps, latency, jitter, loss) in a controlled lab setting | Provides the controlled independent variable needed to evaluate the intelligent controller's adaptivity |
| Client-server and edge-cloud concurrency experiments | Queueing, throughput, and congestion behavior under concurrent load are classical network/distributed-systems measurement problems | Ties system-level network behavior back to the AI system's output quality (Deliverable 7) — the joint evaluation is the specific specialization's core method |
| Transmitted-bytes / bandwidth-consumption measurement | A direct network-engineering metric absent from the current paper entirely | Connects the intelligent pruning decision to a measurable network resource saving, not just a GPU-memory saving |
| Concurrency-reversal root-cause diagnosis (Deliverable 6) | Scheduling, queueing, and batching are systems/networking concerns (continuous batching is itself a network-style multiplexing scheduler) | Demonstrates rigorous systems debugging of an intelligent-serving system, expected of this specific specialization |

---

## Deliverable 3: Proposed Edge-Cloud RAG Architecture

**[PROPOSED — NOT YET IMPLEMENTED]**

```
 +------------------+       Arabic query (HTTPS/gRPC)      +---------------------------+
 |  Client 1..N     |-------------------------------------->|      Edge Gateway         |
 |  (Locust-driven, |                                       |  (co-located with, or     |
 |  emulated network|<--------------------------------------|   1 hop from, clients)    |
 |  condition per   |         streamed answer + metadata    |                           |
 |  run: LAN/WAN/   |                                       |  - TF-IDF/keyword         |
 |  constrained)    |                                       |    retriever              |
 +------------------+                                       |  - Network Monitor        |
                                                              |    (RTT/jitter/BW probe   |
                                                              |     to cloud tier, queue  |
                                                              |     length, req rate)     |
                                                              |  - LSPM (cross-encoder    |
                                                              |    pruning, runs on edge  |
                                                              |    CPU/small GPU)         |
                                                              |  - Network-Aware          |
                                                              |    Controller (Deliv. 4)  |
                                                              +-------------+-------------+
                                                                            |
                                                        pruned context + prompt
                                                        (bytes measured per request)
                                                                            |
                                                                            v
                                                              +---------------------------+
                                                              |      Cloud Tier           |
                                                              |  - vLLM server             |
                                                              |    (PagedAttention,        |
                                                              |     continuous batching)   |
                                                              |  - /metrics endpoint       |
                                                              |    (KV-cache, GPU mem,     |
                                                              |     queue depth)           |
                                                              |  - Monitoring Layer        |
                                                              |    aggregates network      |
                                                              |    + server + GPU +        |
                                                              |    answer-quality logs     |
                                                              +---------------------------+
```

**Component placement and responsibilities:**

- **Client (emulated network edge):** issues Arabic queries via Locust (reusing your existing harness, Section 4.6/8), tagged with a network-condition profile (LAN, WAN, bandwidth-constrained) applied via `tc netem` on the path to the edge gateway or, if edge and cloud are collapsed onto one machine for the first phase, on the path to the vLLM server directly.
- **Edge gateway:** hosts the retriever, the Network Monitor, LSPM, and the new controller. This is the tier closest to the client, where cutting bytes before the WAN hop actually matters — pruning at the edge, before the cloud hop, is the only placement where LSPM's byte reduction produces a real network-bandwidth saving rather than only a GPU-memory saving.
- **Network Monitor (new component):** periodically (e.g., every 1-5 s) measures RTT and available bandwidth to the cloud tier (lightweight active probes, or passive measurement from recent request round-trip times — design choice to make explicit in the implementation, since active probing itself consumes bandwidth), and tracks local queue length and request rate.
- **Cloud tier:** unchanged vLLM server plus its existing `/metrics` endpoint (already used in Section 5.8), which the Monitoring Layer scrapes alongside the edge's network measurements so that a single per-request record ties together network condition, server state, and eventual answer quality.
- **Monitoring Layer:** a log aggregator (can be as simple as a shared CSV/SQLite sink, consistent with your existing `analyze_sweep_results.py` pattern) collecting one row per request with every field in Deliverable 6's metric list.

**Deployment note for the resource-constrained reality of this project:** with one rented GPU and no separate edge hardware, "edge" and "cloud" are two logical roles on the same or two rented machines, separated by an emulated network link. This must be stated explicitly in the paper (per your Task 4 instruction) — e.g., "the edge tier and cloud tier were emulated on [N] machines connected via a `tc netem`-shaped link; no physical edge device or production WAN was used." That is a legitimate systems-research setup, used routinely in edge-computing papers, provided it is disclosed as such.

---

## Deliverable 4: Network-Aware Controller Design

**[PROPOSED — NOT YET IMPLEMENTED]**

### 4.1 Inputs

| Symbol | Input | Source |
|---|---|---|
| $B$ | Available bandwidth (edge→cloud) | Active probe or passive EWMA of recent transfer sizes/times |
| $R$ | RTT (ms) | Active probe (ICMP or application-level ping) or passive from recent request round-trips |
| $J$ | Jitter (ms, RTT stddev over a rolling window) | Derived from $R$ history |
| $L$ | Packet loss rate (if applicable, emulated via `netem`) | `netem` config (known) in emulated trials; passive retransmit counting if TCP-level access is available |
| $C$ | Context size before pruning (bytes/tokens) | Computed at retrieval time |
| $U$ | Concurrent users / in-flight requests | Edge gateway request counter |
| $Q$ | Request queue length at edge and/or cloud | Edge queue depth; cloud `/metrics` scheduler queue if exposed |
| $K$ | KV-cache utilization (mean, peak) | vLLM `/metrics` (existing, Section 4.6) |
| $G$ | GPU memory utilization | `nvidia-smi` or vLLM `/metrics` |
| $T$ | Recent token-generation speed (tok/s) | Rolling average from completed requests |
| $\lambda$ | Request rate (req/s) | Edge gateway counter |

### 4.2 Proposed formulation

Define a **network pressure score** $N \in [0,1]$ and a **server pressure score** $S \in [0,1]$, each a normalized weighted combination of their respective inputs, and combine them into a single compression-ratio decision. Normalization matters and must be calibrated empirically (Deliverable 5's baseline runs establish the min/max operating range for each raw input before this formula can be tuned) — the weights below are a starting design, not a claimed-optimal one.

$$N = w_1 \cdot \widehat{R} + w_2 \cdot \widehat{J} + w_3 \cdot (1 - \widehat{B}) + w_4 \cdot \widehat{L}, \qquad \sum w_i = 1$$

$$S = v_1 \cdot \widehat{K} + v_2 \cdot \widehat{G} + v_3 \cdot \widehat{Q} + v_4 \cdot \widehat{U}, \qquad \sum v_i = 1$$

where $\widehat{x} = \text{clip}\left(\dfrac{x - x_{\min}}{x_{\max} - x_{\min}}, 0, 1\right)$ is min-max normalization against an operating range established from Deliverable 5's baseline sweep (this is exactly the kind of calibration step that requires real measured data before the controller can be trusted — using unfitted or guessed ranges here would itself be a form of unvalidated claim).

**Combined pressure and ratio decision:**

$$P = \max(N, S) \quad \text{or} \quad P = \alpha N + (1-\alpha) S$$

(both variants — worst-of and weighted-sum — should be implemented and compared against each other, since which one better matches real system behavior is itself an empirical question, not one to assume.)

$$r = r_{\max} - (r_{\max} - r_{\min}) \cdot P$$

where $r$ is the pruning ratio actually applied (higher $r$ = more content kept; the existing paper's convention), and $r_{\min}, r_{\max}$ bound the controller's output to the ratios already validated for fidelity in Section 5.4 (r = 0.3, 0.5, 0.7) — the controller should not be allowed to select a ratio outside the range the paper has already shown preserves fidelity, since doing so would introduce a fidelity risk the paper has not tested.

### 4.3 Pseudocode

```python
def compute_ratio(metrics, weights, calib):
    # metrics: dict with rtt_ms, jitter_ms, bw_mbps, loss_pct,
    #          kv_mean_pct, gpu_mem_pct, queue_len, concurrent_users
    # weights: {w1..w4, v1..v4, alpha}
    # calib:   per-input (min, max) from the Deliverable 5 baseline sweep

    def norm(x, key):
        lo, hi = calib[key]
        return min(1.0, max(0.0, (x - lo) / (hi - lo)))

    N = (weights['w1'] * norm(metrics['rtt_ms'], 'rtt_ms')
         + weights['w2'] * norm(metrics['jitter_ms'], 'jitter_ms')
         + weights['w3'] * (1 - norm(metrics['bw_mbps'], 'bw_mbps'))
         + weights['w4'] * norm(metrics['loss_pct'], 'loss_pct'))

    S = (weights['v1'] * norm(metrics['kv_mean_pct'], 'kv_mean_pct')
         + weights['v2'] * norm(metrics['gpu_mem_pct'], 'gpu_mem_pct')
         + weights['v3'] * norm(metrics['queue_len'], 'queue_len')
         + weights['v4'] * norm(metrics['concurrent_users'], 'concurrent_users'))

    P = weights['alpha'] * N + (1 - weights['alpha']) * S   # or max(N, S), compare both

    r_min, r_max = 0.3, 0.7   # bounded to already-validated ratios, Sec. 5.4
    return r_max - (r_max - r_min) * P
```

### 4.4 Comparison conditions this controller must be run against

This is not optional polish — without these comparisons, "network-aware" is an unvalidated label:

1. **Fixed ratio** (r = 0.3, 0.5, 0.7 individually) — the existing baseline.
2. **KV-cache-only dynamic controller** — i.e., implement the existing Section 3.4 design for the first time (it has never been run), using only $S$ (no $N$ term).
3. **Proposed network-aware controller** — full $N$ and $S$.

Each must be evaluated across the same network-condition x concurrency grid (Deliverable 5) so the comparison is fair and the network-awareness specifically (not just the dynamic-vs-fixed distinction) is isolated.

---

## Deliverable 5: Detailed Experimental Plan

**[PROPOSED — NOT YET RUN]**

### 5.1 Baselines (Task 5)

| # | Condition | Notes |
|---|---|---|
| 1 | Unpruned RAG (raw) | Existing |
| 2 | Naive truncation | Existing, r = 0.3/0.5/0.7 |
| 3 | Fixed-ratio LSPM | Existing, r = 0.3/0.5/0.7 |
| 4 | Dynamic KV-aware LSPM | **New implementation** — Section 3.4's design, run for the first time |
| 5 | Network-aware LSPM (proposed) | **New**, Deliverable 4 |
| 6 | Edge deployment vs. cloud deployment | **New**, if two logical tiers can be stood up (see Deliverable 3's resource note) |

### 5.2 Independent variables

- **Network condition** (emulated via `tc netem` on the edge↔cloud link): LAN (~1 ms RTT, no shaping), WAN (~40-80 ms RTT, typical intercontinental), bandwidth-constrained (e.g., 5 Mbps cap, representative of constrained mobile/rural links), each with a jitter component (e.g., ±10% of base RTT) and an optional packet-loss condition (e.g., 1%) for a fourth "degraded" profile.
- **Concurrency**: 1, 5, 10, 25, 50 simulated users (matches your instruction; 100 retained from the existing v1 sweep only if hardware allows).
- **Method** (the 6 conditions above).
- **Ratio** (0.3/0.5/0.7 for the fixed/dynamic-KV conditions; controller-selected for the network-aware condition).

### 5.3 Fixed variables (held constant for fair comparison, per Task 5)

Same model (Llama-3.1-8B-Instruct, bf16), same retriever, same 5-query sample set or the existing ARCD pool (recommend re-using the ARCD 140-question pool for continuity with Section 5.4's already-validated fidelity results), same generation settings (temperature, max tokens) as Section 4.6.

### 5.4 Repetitions and statistics

Directly reusing your v2 harness design (already built, `benchmark/run_full_sweep.py --repeats`, `analyze_sweep_results.py`'s `mean_ci95()`): **repeats ≥ 3 per (method, ratio, network-condition, concurrency) cell**, randomized execution order (already implemented, fixes the v1 confound), t-distribution 95% CIs. Given the added network-condition axis, the full grid is larger than v1's 35 cells (6 methods x up to 3 ratios x 4 network conditions x 5 concurrency levels is large; a reduced, clearly justified subset — e.g., fixing ratio to r = 0.3 for the network-condition sweep, and only sweeping ratio at one representative network condition — should be pre-registered in the paper rather than run exhaustively and selectively reported).

### 5.5 Estimated cost

Scaling from your existing v2 estimate (~3 hours for 35 cells x 3 repeats): a reduced grid of, e.g., 6 methods x 4 network conditions x 5 concurrency levels x 3 repeats at r = 0.3 only (360 cell-runs) at 90 s/cell is roughly 9 hours of GPU time — a multi-session rental, and the reason this must be scoped and approved before running, not assumed.

---

## Deliverable 6: Metrics and the High-Concurrency Diagnostic Plan

### 6.1 Metrics to collect (Task 6), all logged per request where feasible

End-to-end latency, TTFT, prefill latency (if separable from vLLM's internal timing/logs), inter-token latency, requests/s, tokens/s, p50/p95/p99 for every latency metric (not means alone, per your explicit instruction), bytes transmitted per request (edge→cloud and cloud→edge, measurable via `tc`/packet capture or simply `len()` of the serialized request/response), bandwidth consumption, queueing delay (edge and cloud, if the cloud scheduler exposes it), GPU memory utilization, KV-cache utilization, CPU/system memory on the edge tier. Energy consumption only if a reliable meter (e.g., `nvidia-smi --query-gpu=power.draw` or a wall meter) is available — do not estimate it otherwise, consistent with your instruction.

### 6.2 Diagnosing the high-concurrency KV-cache reversal (Task 8)

**[PROPOSED — NOT YET RUN]**, building directly on the existing-data analysis in Deliverable 1.4. A minimal, targeted diagnostic sequence, ordered from cheapest/fastest to most expensive, each isolating one candidate cause from your list:

1. **Randomized-order re-run at c ≥ 25 only** (already-built v2 harness capability) — if the reversal shrinks or disappears once naive truncation no longer runs last, the run-order/server-state confound is confirmed as at least a major contributor. This is the single highest-value, lowest-cost test and should run first.
2. **Log per-request generated-token count** (trivial addition to the existing Locust script) for raw/LSPM/naive at a fixed concurrency (e.g., c = 50) — directly tests the "differences in generated-output length" candidate cause by comparing distributions, not just aggregates.
3. **Move LSPM's cross-encoder scoring out of the load generator** (already built in v2, `precompute_contexts.py`) and separately **re-time naive truncation's (trivial) computation** — isolates "cross-encoder semantic-pruning overhead" and "load-generator CPU bottleneck" from server-side effects, since v2 already removes this confound for LSPM but the same precomputation should be confirmed for naive truncation's truncation step too (currently near-zero cost, but should be measured, not assumed).
4. **Restart the vLLM server between method blocks** (or interleave methods within a single randomized run, which the v2 harness already does) — tests "warm-up/cold-start" and "prefix-caching state carried over between methods" as causes; vLLM's automatic prefix caching (if enabled) could plausibly cause one method's requests to benefit from KV blocks cached by a structurally similar preceding request from a different method, which would be a genuine, reportable, non-obvious systems finding if confirmed.
5. **Log vLLM's internal scheduler/batching state** (if exposed via `/metrics` or verbose logs) at each concurrency level — directly tests "continuous batching" and "scheduling policy" as causes, by checking whether naive truncation's shorter-but-more-numerous(?) sequences are being batched differently than raw's.
6. **Confirm KV-cache measurement points are equivalent across methods** — a scrape-timing/aggregation-window check (are all three methods' KV-cache samples drawn from directly comparable wall-clock windows relative to their own request stream, not just the same wall-clock window overall) — a purely diagnostic, no-new-data check on the existing v1 logging code.

Each step either confirms or rules out a specific candidate cause; the plan is designed so that step 1 alone (cheapest) may already explain most of the effect, per the existing-data pattern already identified in Deliverable 1.4. **If, after this sequence, the cause remains ambiguous, the paper must report it as unresolved** (per your explicit Task 8 instruction), not settle on a plausible-sounding but unconfirmed explanation.

---

## Deliverable 7: Preserving Intelligent-System Quality Under Network-Aware Pruning

**[PROPOSED — NOT YET RUN]**

Reuse the already-validated metrics and, where possible, the already-built pipelines: EM/F1 against ARCD gold answers (Section 4.4's existing pipeline, re-run for any new ratio the network-aware controller selects), ROUGE-L/BERTScore (existing pilot pipeline), and the already-built, already-run blind human evaluation protocol (Section 4.10/5.9) extended to cover network-aware-controller-selected ratios specifically, since a network-aware controller could in principle select an aggressive ratio under poor network conditions that the existing fidelity results (Section 5.4) were not tested at in combination with high concurrency. Abstention rate and hallucination/unsupported-claim rate are not currently measured anywhere in the paper and would need new annotation criteria added to the existing `HUMAN_EVAL_PROTOCOL.md` rubric (the faithfulness dimension already partially covers hallucination; abstention would need to be added as a new category, e.g. "0 = incorrect, including refusal/non-answer" is currently folded into the existing 0-2 correctness scale and would benefit from being split out explicitly for this extension).

**Required trade-off analysis** (Task 7's final ask): a single figure/table per network condition plotting (bytes-transmitted reduction, TTFT reduction, KV-cache reduction) against (EM/F1 change, faithfulness change) across the ratio range the network-aware controller actually selects in practice — this is the figure that would make the network-awareness contribution legible to a reviewer, and it does not exist until Deliverable 5's experiments are run.

---

## Deliverable 8 (revision plan) is provided as Section-by-Section below; Deliverable 9's revised title/abstract/contributions follow.

## Section-by-Section Manuscript Revision Plan (only to be executed once Deliverables 3-7 produce real data)

| Section | Change, gated on real results |
|---|---|
| Title | Switch to the proposed networks title **only after** the edge-cloud experiments (Deliverable 5) produce real, reportable network-condition results; until then, keep the current title |
| Abstract | Add one finding sentence per genuinely new, measured result (network-aware controller vs. baselines; edge vs. cloud latency/bandwidth) — no new sentence without a number behind it |
| Introduction | New paragraph motivating the network/edge angle: WAN/mobile deployment of Arabic RAG, bandwidth as a first-class constraint alongside GPU memory |
| Contributions | Add: network-aware controller design and evaluation; edge-cloud emulated architecture; concurrency-reversal root-cause diagnosis (report whatever Deliverable 6 actually finds, including "unresolved" if that is the honest outcome) |
| Related Work | New subsection: Edge AI / edge inference, distributed RAG, network-aware LLM serving, adaptive resource management — real citations only, each verified against its original source before being added (see Deliverable 10's integrity note) |
| System Architecture | Replace/extend Section 3 with Deliverable 3's diagram, once implemented (not just designed) |
| Controller formulation | New subsection with Deliverable 4's equations, fitted calibration constants from real data (not the placeholder ranges in this document) |
| Test environment | New subsection disclosing exact `tc netem` parameters, machine specs, and explicitly stating the emulated (not physical) nature of the edge/cloud split |
| Methodology | Extend Section 4 with the network-condition x concurrency grid actually run (a subset of Deliverable 5's design, scoped to what GPU budget allows) |
| Statistical analysis | Extend existing CI/significance-testing approach (already used for ARCD, Section 4.4/5.4) to the new controller comparison; no significance claim without the corresponding test, per Task 10 |
| Results | New section(s) reporting exactly what was run — including negative or null results on the network-aware controller if that is what the data shows |
| Threats to validity | Emulated-vs-physical network, single-GPU/single-vendor generalizability, controller weight calibration on a single workload |
| Limitations | Explicitly carry over anything Deliverable 5/6 could not complete due to GPU budget, exactly as the current paper already does for its own open items |
| Conclusion | Rewritten only once real results exist to summarize |

---

## Deliverable 9: Revised Title, Abstract, and Contributions — Conditional Draft, Not Yet Adoptable

**Per your own Task 9 instruction** ("use the proposed new title only if genuine network or Edge-Cloud experiments have been completed"), the following is a **template to fill in once Deliverable 5's experiments produce numbers** — it is deliberately left with bracketed placeholders rather than invented figures, and should not be pasted into the manuscript as-is:

**Title (conditional):** "Network-Aware Semantic Context Pruning for Memory- and Latency-Efficient Arabic RAG Services in Edge-Cloud Systems"

**Abstract addition (template, not to be used until filled with real numbers):** "We further introduce a network-aware controller that selects the pruning ratio from real-time RTT, jitter, bandwidth, and server load, and evaluate it against fixed-ratio and KV-cache-only dynamic baselines across [N] emulated network conditions and concurrency levels [1-50]. The network-aware controller achieves [X]% lower bytes-transmitted and [Y]ms lower TTFT than the KV-only controller at matched fidelity ([EM/F1/faithfulness delta]), while [state the high-concurrency KV-cache finding honestly, including if it remains partially unresolved]."

**Contribution list addition (template):** "A network-aware compression-ratio controller, evaluated against fixed and KV-cache-only baselines across [N] emulated network conditions, showing [state the actual, measured comparative result — not before it exists]."

---

## Deliverable 10 (part 1): Recommended Journal Categories

Once Deliverables 3-7 are actually executed and the paper genuinely carries network/distributed-systems content, venues whose official scope explicitly covers computer networks, intelligent systems, distributed systems, or computer engineering (verify current scope statements directly on each journal's site before submitting, since scope statements are occasionally updated):

- **IEEE Transactions on Network and Service Management** — network-aware adaptive systems, service management.
- **IEEE Internet of Things Journal** / **IEEE Transactions on Mobile Computing** — edge-cloud, network-condition-aware inference.
- **Journal of Network and Computer Applications (Elsevier)** — directly covers network-aware distributed application design.
- **Future Generation Computer Systems (Elsevier)** — distributed/edge computing systems, resource-aware scheduling.
- **IEEE Access** — broad computer engineering scope, covers both networks and intelligent systems tracks, often used for interdisciplinary edge-AI work.
- **Applied Intelligence (Springer)** — your current target; retains fit if the paper keeps a strong intelligent-systems core, but its scope leans AI/ML methods more than networks, so the network-specialization case is stronger at the venues above if that is the primary goal.

---

## Deliverable 10 (part 2): Arabic Summary for the University Scientific Committee

**ملخص باللغة العربية للجنة العلمية**

**عنوان البحث:** تقليم السياق الدلالي الواعي بحالة الشبكة لخدمات الاسترجاع المعزز بالتوليد (RAG) للغة العربية في أنظمة الحوسبة الطرفية-السحابية (Edge-Cloud)، بكفاءة في الذاكرة والزمن.

**سبب انتماء البحث المكتمل إلى تخصص "شبكات الحاسوب والأنظمة الذكية":**

يجمع هذا البحث، بعد استكمال التجارب المذكورة في هذه الخطة، بين محورين أساسيين لتخصص شبكات الحاسوب والأنظمة الذكية:

أولاً، من الناحية الشبكية، يُصمَّم النظام كبنية موزعة على طبقتين (طرفية وسحابية)، ويُقاس أداؤه تحت ظروف شبكية متحكم بها فعلياً (زمن الاستجابة RTT، والتذبذب Jitter، وعرض النطاق الترددي المتاح، ونسبة فقدان الحزم)، باستخدام أدوات محاكاة شبكية معتمدة علمياً (`tc netem`)، مع الإفصاح الصريح عن كون البيئة الشبكية محاكاة وليست بيئة إنتاج فعلية. كما يقيس البحث مقاييس شبكية مباشرة لم تكن موجودة في النسخة الأصلية من البحث، مثل حجم البيانات المرسلة لكل طلب، واستهلاك عرض النطاق الترددي، وزمن الانتظار في طابور الشبكة.

ثانياً، من الناحية الذكاء الاصطناعي/الأنظمة الذكية، يُطوَّر البحث متحكماً واعياً بحالة الشبكة والخادم (Network-Aware Controller) يقرر نسبة تقليم السياق الدلالي بشكل ديناميكي بناءً على مدخلات شبكية وحاسوبية مشتركة (RTT، عرض النطاق، حالة ذاكرة KV-cache، طول طابور الانتظار)، ويقارن هذا المتحكم علمياً مع أساليب تقليم ثابتة النسبة ومع متحكم سابق يعتمد فقط على حالة ذاكرة GPU، مع الحفاظ على جودة الإجابة (الدقة، والالتزام بالسياق المسترجع) عبر تقييم آلي وتقييم بشري أعمى (Blind Human Evaluation) مطبق بالفعل في نسخة سابقة من البحث.

هذا الدمج بين قياس شبكي حقيقي متحكم به تجريبياً، وقرار ذكي تكيفي (Adaptive Intelligent Decision) مبني على تلك القياسات، هو جوهر تخصص "شبكات الحاسوب والأنظمة الذكية" تحديداً، وليس مجرد إضافة مصطلحات شبكية إلى بحث ذكاء اصطناعي قائم. تلتزم هذه الخطة، ويلتزم الباحث، بعدم اعتماد العنوان أو الملخص المقترحين أعلاه إلا بعد تنفيذ التجارب الفعلية الموصوفة، حفاظاً على النزاهة العلمية للبحث.

---

## What happens next

This document completes Deliverables 1, 2, 3, 4, 5, 6, 7, 8, 9 (as design/plan), and 10, honestly scoped as design and analysis rather than completed experiments, per Task 10's integrity requirement. The next concrete step is entirely gated on your call: renting a GPU (and, ideally, a second small machine or VM to act as the emulated edge tier — a single machine with two network namespaces shaped by `netem` also works and is cheaper) so I can implement the Network Monitor, the controller code from Deliverable 4, and run the Deliverable 5/6 experiment grid for real. Tell me when you're ready to rent, and I'll prepare the exact environment setup (this time making sure raw per-request logs are pushed to GitHub before the pod is stopped, closing the v1 data-availability gap at the same time).
