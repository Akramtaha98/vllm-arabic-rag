# Minor-Revision Strengthening Plan — Status

Tracking the 6 reviewer-gap items, against what you told me you have
available (GPU: yes; extra API budget before submission: no; bilingual
evaluators: yes, 1-2 people).

## Done now (no resources needed)

**#6 — Formal Zenodo data citation.** Added as reference `[43]` in both
`paper_draft_v2.md` and the LaTeX build, cited from the Data Availability
sentence. Recompiled — 32 pages, no errors, `[43]` renders correctly as
the last entry in the bibliography.

## Ready for you to run (harness built and tested; needs your GPU)

**#1 — Real vLLM GPU benchmark.** This is the highest-impact item, and
you have a GPU, so it's the one to prioritize. I extended the existing
benchmark harness (`benchmark/`) to add what it was missing:
- A third condition, **naive truncation**, alongside raw and LSPM (same
  truncation function as the accuracy experiments, so both experiments
  share one baseline definition).
- Ratios changed from a generic {0.2, 0.5, 0.8} sweep to the paper's
  actual headline ratios **{0.3, 0.5, 0.7}**.
- Real **TTFT** measurement via streaming responses (previously only
  total latency and a throughput estimate were captured).
- The sweep runner now auto-skips completed cells, so it's safe to
  interrupt and resume.

I stress-tested `analyze_sweep_results.py` against fabricated locust/
metrics CSVs to confirm the parsing and plotting logic is correct before
handing it to you — what's unverified is only the real GPU run, which
needs your hardware. Full instructions: `benchmark/BENCHMARK_INSTRUCTIONS.md`.
Runtime estimate: ~55-60 minutes at the default 90s/cell.

**When you have results:** send me `results/sweep_summary.csv` and
`results/sweep_comparison.png` and I'll replace the analytical KV-cache
projection with the measured numbers, add a new results subsection, and
rebuild the PDF.

## Ready for your evaluators (kit built and tested; needs 1-2 people)

**#5 — Human evaluation.** Built and dry-ran the full pipeline:
- `scripts/build_human_eval_sample.py` — already run; produced
  `data/human_eval_sample_blind.csv` (90 rows: 30 ARCD questions × 3
  conditions at r=0.3, fully shuffled, blind opaque IDs) and
  `data/human_eval_codebook.json` (private method mapping — do not share
  with evaluators until rating is done).
- `HUMAN_EVAL_PROTOCOL.md` — the rubric and instructions to hand your
  evaluators directly: correctness (0/1/2) and faithfulness (0/1),
  judged against the full source context so blinding isn't broken by
  showing the pruned/truncated context each method actually saw.
- `scripts/analyze_human_eval.py` — computes per-method mean scores with
  bootstrap 95% CIs and Cohen's kappa between raters. I tested this
  end-to-end with synthetic ratings to confirm the statistics compute
  correctly; real output waits on your evaluators.

**When ratings are done:** run `python scripts/analyze_human_eval.py` and
send me `results/human_eval_summary.csv`. I'll add it as a new subsection
and update the Limitations section (currently says "automatic metrics
only" — this closes that specific gap).

## Deliberately not started — you said no additional API spend before submission

**#2 — Scale the retrieval experiment**, **#3 — add a competing
compression baseline (e.g. LLMLingua)**, **#4 — test a second generator
model**: all three require many more live calls against your NVIDIA NIM
endpoint (or a second provider for #4), which costs money and time on
your account. I didn't build or run anything for these to avoid
surprising you with API usage you didn't approve.

If you want to revisit any of these before or during a revision round,
tell me a rough call budget and I'll scope a specific experiment size
(e.g. "#2: expand from 140 to 220 questions" or "#3: implement LLMLingua
as a fourth condition at r=0.3 only, ~420 calls") before running anything.
Given your reviewer list ranks #1 and #2 as the two that matter most for
minor-revision odds, and #2 is the one still blocked on budget, that's
the most likely next thing to greenlight if you decide to spend more.

## Bottom line

Once you run the GPU sweep (#1) and get evaluator ratings back (#5), send
both results sets and I'll integrate them, rebuild the LaTeX/PDF, and
that's the two highest-leverage items from your list closed with real
measurements instead of projections — which was the single biggest
reviewer-facing risk in the current draft.
