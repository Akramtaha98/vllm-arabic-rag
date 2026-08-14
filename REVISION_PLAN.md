# Minor-Revision Strengthening Plan — Status

Tracking the 6 reviewer-gap items, against what you told me you have
available (GPU: yes; extra API budget before submission: no; bilingual
evaluators: yes, 1-2 people).

## Done now (no resources needed)

**#6 — Formal Zenodo data citation.** Added as reference `[43]` in both
`paper_draft_v2.md` and the LaTeX build, cited from the Data Availability
sentence. Recompiled — 32 pages, no errors, `[43]` renders correctly as
the last entry in the bibliography.

## #1 status: v1 done and honestly scoped in the manuscript; v2 (corrected re-run) is the top open item

**v1 — first GPU sweep.** Ran the full 35-cell sweep (raw + LSPM + naive
truncation × r={0.3,0.5,0.7} × concurrency={1,10,25,50,100}) on a rented
RTX 3090. Two real issues surfaced, both caught before submission rather
than after:

- The `e2e_p50/p95/p99` columns in `sweep_summary.csv` are a measurement
  artifact (Locust's timer stops at response headers, not full stream
  completion) — dropped from the paper entirely.
- Between-method KV-cache/throughput comparisons are only clean at the
  single c=1 cell; already mixed at c=10 (LSPM's peak KV-cache exceeds
  raw's at every ratio there) and reverse outright by c≥25. This was
  caught on review (my initial draft of Section 5.8 overclaimed a
  "c=1, c=10" confirmation that Table 8's own numbers didn't support —
  corrected to a single-cell, exploratory claim only).

**v2 — corrected re-run (not yet executed).** Built and validated
(synthetic-data-tested, same standard as v1) three harness fixes:
`benchmark/precompute_contexts.py` moves LSPM's cross-encoder scoring out
of the Locust load-generator process entirely; `run_full_sweep.py` now
shuffles cell execution order (seeded) instead of running fixed method
blocks in sequence; `--repeats 3` (new) runs each cell independently 3x
so `analyze_sweep_results.py` can report mean ± 95% CI instead of a
point estimate. Full instructions in `benchmark/BENCHMARK_INSTRUCTIONS.md`.
**Estimated cost: ~3 hours of GPU rental** (vs. v1's ~1 hour) at the
default settings — this is the next thing to run when you're ready.

Section 4.5/4.6 and 5.8 in `paper_draft_v2.md` (mirrored into LaTeX) now
describe the actual executed v1 experiment and its scope honestly, with
the v2 fix specified as required next steps rather than optional polish.
Full 35-cell v1 table added as Appendix G. PDF rebuilt: 36 pages, compiles
clean, Figure 6 regenerated larger (3 stacked panels) and readable.

**Also still needed, separate from the re-run itself:** a new Zenodo
version (not a new record — same concept DOI, versioned) archiving the
v2 data/scripts/updated PDF, since the currently-cited DOI (Aug 6) predates
all of this. See the message accompanying this plan for exact steps once
you're ready to push.

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

## Round 3 external review (this round): three more real bugs, all fixed

1. **Abstract/body desync, root-caused.** `sn-article-taha.tex` hand-copied
   the abstract separately from `paper_draft_v2.md`, and that copy hadn't
   been touched since round 6 — it still said "absent GPU access... not
   yet measured" on page 1 while Section 5.8 reported a real benchmark.
   Fixed the immediate text, and fixed the root cause: `build_latex.py`
   now generates `abstract_generated.tex` from the same markdown source as
   the body, and `sn-article-taha.tex` `\input{}`s it instead of hardcoding
   it. This class of bug is now structurally prevented, not just patched.
2. **"No throughput cost" was numerically wrong.** Table 8: LSPM's request
   throughput at c=1 is 8.5-11.9% *lower* than raw's, not unchanged — I'd
   conflated it with completion tokens/sec (which *is* within 0.5%).
   Corrected everywhere: abstract, scope note, Section 5.8, Discussion,
   Limitations, Future Work, Appendix F, `BENCHMARK_INSTRUCTIONS.md`.
3. **Data availability was unsupported.** The Zenodo DOI cited in the
   paper (archived ~Aug 7) predates the GPU sweep (Aug 11) entirely, and
   this session's manuscript/harness changes were sitting locally
   uncommitted — see below.

Also fixed: Table 8's 63.7pt LaTeX overflow (dropped the redundant
Concurrency column since every row is c=1, `\scriptsize`), "Key-Value
(KV)" defined at first use in the abstract, and DOI links added for 11
references that were missing them (verified via web search against
arXiv/ACL Anthology — 2 references, Orca/OSDI and the 2004 ROUGE paper,
genuinely have no DOI in any registry and were left as-is rather than
fabricating one).

## Bottom line — still do not submit yet

**Blocking, needs you (I cannot do this from here):** `.git/index.lock`
in your local repo is still permission-blocked from my side — 21 files
of this session's work (all the fixes above, plus the harness changes)
are sitting locally, uncommitted, unpushed. Run on your Mac:

```bash
cd /Users/akramtaha/Work/Papers/VLLM/vllm-arabic-rag
rm -f .git/index.lock
git add -A
git add -f results/sweep/          # currently gitignored -- this is the
                                     # "no raw GPU files in the repo" gap
git commit -m "Round 7: fold measured GPU benchmark into manuscript, fix abstract/throughput/DOI issues, add v2 harness"
git push origin main
```

Then, since the currently-cited Zenodo DOI predates all of this:
1. On GitHub, go to Releases → Draft a new release → tag `v1.1-submission`.
2. On Zenodo, open your existing record → **New version** (not a new
   record — this keeps one concept DOI with versions underneath it) →
   upload the updated repo archive (or point it at the new GitHub release
   if you have the GitHub-Zenodo integration enabled) → publish.
3. Zenodo will give you a new version-specific DOI. Send it to me and
   I'll update the citation in the paper (currently `10.5281/zenodo.21826992`)
   to the new one.

## Round 8: #5 (human evaluation) — done, folded into the manuscript

Both evaluators finished rating `data/human_eval_sample_blind.csv` (90/90
rows each); `scripts/merge_human_eval_ratings.py` merged the two per-rater
sheets and `scripts/analyze_human_eval.py` produced
`results/human_eval_summary.csv`: mean correctness/faithfulness by method
(raw 1.717/0.933, LSPM 1.800/0.983, naive 1.550/0.800, n=30 each) with
Cohen's kappa 0.794 (correctness) / 0.935 (faithfulness) — substantial to
near-perfect inter-rater agreement. This is folded into the manuscript as
new Section 4.10 (methodology) and Section 5.9 (results, Table 9), plus
updates to the abstract, contributions list, Discussion, Limitations,
Future Work, Conclusion, Ethics declaration, Data Availability, and a new
Appendix H. PDF rebuilt: 40 pages, clean compile, no Overfull/Undefined
warnings. Closes the "automatic metrics only" gap specifically at r = 0.3.

**Still needed before this is genuinely ready:**
- Push this round's changes (see git commands below) and merge with the
  pod's earlier commits (already resolved per your terminal output, minor
  conflict in `metrics_scraper.py` — just needs `git add`/`commit`/`push`
  if not already done).
- Run the v2 corrected GPU re-run (~3 hours) — `benchmark/BENCHMARK_INSTRUCTIONS.md`.
  Send me results and I'll fold in real confidence intervals across the
  full concurrency range, replacing the current single-cell-exploratory
  framing, and do the final rebuild + another Zenodo version archiving
  everything from this round too.

#2/#3/#4 remain deliberately not started per your no-extra-budget call.

## Round 9: preliminary edge-cloud, network-aware extension — small validation grid run, bugs found and fixed, folded into the manuscript honestly

Implemented and ran the edge-cloud extension sketched as future work in
earlier rounds: `benchmark/edge_gateway.py` (FastAPI edge tier: retrieval +
pruning + controller decision + cloud call), two new rule-based
controllers in `benchmark/network_controller.py` (`KVOnlyDecision`
wrapping the existing `DynamicRatioController`; `NetworkAwareController`,
a new 5-input weighted controller), `benchmark/network_profiles.py` (4
network profiles), and `benchmark/run_validation_grid.py` (orchestrator).

`tc netem` and `unshare --net` both failed with "Operation not permitted"
on the rented pod (no `CAP_NET_ADMIN`), so network shaping is
application-level (per-request sleeps/drops inside the gateway), disclosed
as a coarser model than kernel-level shaping throughout the manuscript.

**First run (90 cells, 5 methods x 2 profiles x 3 concurrency x 3 reps):**
0 failures, but post-run analysis found 3 real bugs, none affecting
raw/naive/fixed_lspm: (1) `DynamicRatioController` matched only
`vllm:gpu_cache_usage_perc`, not the deployed vLLM's
`vllm:kv_cache_usage_perc`, so `kv_aware` silently ran as a second
`fixed_lspm` the whole time; (2) blocking synchronous HTTP calls inside
`async def` handlers froze the event loop under concurrency, inflating
TTFT for both adaptive methods at c=50; (3) `network_aware` measured the
real unshaped loopback link instead of the active emulation profile, so
it produced the same ratio regardless of profile.

**Fixed all three** (async HTTP clients, single non-blocking controller
fetch, profile-aware controller inputs), added 2 new regression checks to
`benchmark/smoke_test.py` (now 10/10 passing), and **re-ran the 36
affected cells** (`kv_aware`/`network_aware` x 2 profiles x 3 concurrency
x 3 reps) on a fresh pod: 0 failures. After the fix, `network_aware`
correctly differentiates its ratio by profile (0.700 `edge_lan` vs. 0.500
`constrained_wireless`, continuous means 0.674 vs. 0.530) — it did not do
this before. The fix also unmasked, rather than removed, a real
per-request `/metrics`-polling cost: `network_aware`'s TTFT got *worse* at
every concurrency level after the fix (+24 ms at c=1 to +341 ms at c=50),
because the old blocking-call bug had accidentally serialized concurrent
`/metrics` polling in a way that hid this scaling cost. Presented this
finding and two options (rent more GPU time for a background-polling fix
+ re-run, or accept and disclose); you chose to accept and disclose.

**Folded into the manuscript** as new Sections 4.11 (methodology) and 5.10
(results, Table 10), plus updates to the revision header, contributions
list (#9), Discussion, Limitations, Future Work, Conclusion, Data
Availability, and new Appendix I — all following this paper's existing
honest-disclosure pattern (Appendices F/G/H). Per your standing
instruction, the title and abstract are **not** reframed around
network-awareness: this extension is small-scale (5-query mock corpus, no
correctness/faithfulness data, co-located edge+cloud on one GPU, only 2 of
4 profiles exercised, unfitted controller calibration), and is reported as
exactly that.

**LaTeX/PDF rebuild — done.** Ran `build_latex.py` against the updated
`paper_draft_v2.md`; it regenerated `body_generated.tex` (11 tables, up
from 10) with no code changes needed beyond adding the new TTFT table
(Table 10 in the manuscript / `tab:11` in LaTeX numbering) to the
existing wide-table `\scriptsize` shrink list, since its 9 columns
(`Profile, C, raw, naive, lspm, kv (pre), kv (post), net (pre), net
(post)`) would otherwise overflow the print column the same way Table 8
did in round 6 — same fix, applied proactively this time rather than
caught on review. Shortened the table's column headers and profile-name
cells in the markdown itself (`constrained_wireless` → `constr_wireless`
in the table only; full names stay in the prose and caption legend) to
keep it readable at `\scriptsize`. Copied the regenerated
`body_generated.tex`/`abstract_generated.tex` into
`paper/latex_submission/` (the abstract file is byte-identical since the
abstract itself was deliberately not touched this round) and recompiled
with `pdflatex` (2 passes): **47 pages** (up from 40 in round 8), zero
`!`-errors, zero `Overfull \hbox` warnings, zero undefined references,
`rerunfilecheck` stable on the second pass. Spot-checked via `pdftotext`
that Section 4.11, 5.10, and the new Table 10 all render correctly and
the table's 9 columns fit within the column width without truncation.

**Still needed before this is genuinely ready:**
- Push this round's manuscript + LaTeX/PDF changes (`paper_draft_v2.md`,
  `paper/build_latex.py`, `paper/body_generated.tex`,
  `paper/latex_submission/body_generated.tex`,
  `paper/latex_submission/sn-article-taha.pdf` + `.aux`/`.log`/`.out`,
  this file) from your Mac; the code/results (commits `181ef3d`..`7cb9b7e`)
  are already pushed. Note: `paper/sn-article-taha.tex` and
  `paper/sn-article-taha-merged.tex` (top-level, outside
  `latex_submission/`) are stale leftover files with a long-outdated,
  hand-copied abstract that is **not** part of the actual build pipeline
  — the real, compiled source is `paper/latex_submission/sn-article-taha.tex`,
  which correctly `\input{}`s the generated files. Worth deleting those
  two stale files in a future round to avoid confusion; left untouched
  this round since that's a cleanup, not a content change.
- Fix the per-request `/metrics`-polling scaling cost (periodic background
  polling instead of one live fetch per request) and re-validate on the
  same 36 cells before any larger edge-cloud sweep.
- Tag a release and archive this round's code + both grids' raw data
  (`results/validation_grid/`, `results/validation_grid_rerun2/`,
  `results/edge_gateway_log.jsonl`) as a new Zenodo version — not yet
  done, disclosed as such in the Data Availability statement.

#2/#3/#4 remain deliberately not started per your no-extra-budget call.

## Round 10: methodological review of round 9's edge-cloud extension — descoped, not rebuilt; broader integrity pass; v2 GPU benchmark approved next

A detailed external-style review of round 9's edge-cloud extension raised
10 numbered issues plus scientific-positioning and "minimum work before
submission" concerns. Every checkable claim was independently verified
against the real repository code and logs (not taken on faith) before
acting. Confirmed: the network experiment does not validly isolate a
network effect (co-located edge/cloud, application-level sleeps not
kernel-level shaping, bandwidth throttle applied to response bytes only,
`t_send` recorded after retrieval/pruning so TTFT is not true end-to-end,
placeholder controller calibration, no accuracy/faithfulness measured);
the "zero request failures" framing was misleading (327 `emulated_loss`
events under `constrained_wireless`, 308 of them inside real grid-cell
windows, never registered as failures because the gateway returns HTTP
200 with a JSON error body); the round-9 write-up mislabeled
`ratio_continuous` (0.674/0.530) as `pressure_score` (whose real values
are 0.065/0.424); LSPM overhead was materially underreported (57.8ms from
a 16-call CPU test vs. 506.1ms mean over the real 420-measurement ARCD
data); the naive-truncation baseline is sentence-count-matched, not
length-matched, to LSPM (1,044 vs. 1,184 chars at r = 0.3 on ARCD); three
manuscript cross-references were broken (two "Section 10"s, one
"Section 5.5" that should be "5.6"); "Table 5b" was silently uncaptioned
in the LaTeX build and never consumed a real table-counter slot; the
ethics declaration was internally inconsistent (claimed no human
participants, then described two human annotators); and the Data
Availability statement falsely claimed three human-eval data files were
in the repository when only the protocol doc and analysis scripts are.

Given a choice between (A) rebuilding the edge-cloud contribution properly
(real network separation, request-side bandwidth shaping, true end-to-end
TTFT, corrected failure accounting, calibrated controller, accuracy
results) or (B) descoping it and submitting as a non-network Arabic
RAG/context-compression paper once the GPU benchmark is also fixed, you
chose **(B), descope**. Sections 4.11 and 5.10 were removed from the
manuscript body entirely (not just trimmed), the contributions list
reverted to 8 items, and every Discussion/Limitations/Future
Work/Conclusion/Data-Availability passage that referenced the network
extension was cut or replaced. The historical record is preserved
honestly rather than erased: Appendix I got an "Update, round 10" note
correcting the pressure_score/ratio_continuous mislabeling with real
numbers, and a new **Appendix J** itemizes all five edge-cloud validity
gaps and five further unrelated fixes in table form, each with a
Verification/Resolution column, plus a closing note on what remains
unresolved. Section 8 (Future Work) now specifies exactly what a valid
version of this experiment would require, and states plainly that the
round-9 pilot code is released but its output is not cited as evidence in
this submission. `NETWORK_EXPERIMENT_STATUS.md` was rewritten to point to
Appendix J as the authoritative account rather than describing a
still-pending run.

The five other confirmed bugs were fixed directly in the manuscript: the
LSPM overhead section (4.7/5.6) now reports the real 506.1ms figure with
a per-ratio breakdown and 95% CIs, and states plainly that the earlier
57.8ms figure was materially underreported; "naive length-matched
truncation" was renamed "naive sentence-count-matched truncation"
throughout (~9 locations) and a new disclosure paragraph was added to
Section 4.3/5.4/7 quantifying the real character-count gap; the three
broken cross-references were fixed; every downstream table was
renumbered (+1 from the old "Table 6" onward) to close the "Table 5b"
gap, with the historical Appendix A location column updated to match; the
ethics declaration was rewritten to separate the tokenization-corpus claim
from the human-annotator case and to state explicitly that institutional
ethics-body confirmation has not yet been independently obtained, as an
open action item rather than a settled "not required" claim; and the Data
Availability statement now states plainly that an earlier version of
itself was wrong about which human-eval files are in the repository.
Repo-level cleanup: `README.md`'s `YOUR_USERNAME`/`Your Name` placeholders
were fixed to the real GitHub username and author name.

**Flagged to you, not resolved:** the abstract still says "we release the
implementation, raw data, and a live demo," but `README.md`'s demo link is
a dead placeholder (`[Live demo on Hugging Face Spaces](#)`) — this needs
either a real Hugging Face Space deployment or an abstract edit, both
outside what was done unilaterally this round since the abstract is meant
to stay untouched except when genuinely warranted.

**LaTeX/PDF rebuild — done.** Ran `build_latex.py` against the fully
descoped `paper_draft_v2.md`; it regenerated `body_generated.tex` with 10
tables (down from round 9's 11, since Table 10/TTFT was removed along
with Section 5.10) and no code changes were needed beyond removing the
now-dead TTFT-table `needs_shrink` condition. Copied the regenerated
`body_generated.tex`/`abstract_generated.tex` into
`paper/latex_submission/` and recompiled with `pdflatex` (2 passes):
**43 pages** (down from 47 in round 9), zero `!`-errors, zero
`Overfull \hbox` warnings, zero undefined references, `rerunfilecheck`
stable on the second pass. Spot-checked via `pdftotext` that Tables 1-10
all render with the correct new numbering and no stale "Table 5b" or
"Table 11/12" remnants, that no "Section 10" broken references remain,
that the contributions list is back to 8 items, that no appendices leaked
into the PDF body (per the standing round-2 decision), and that the only
two remaining "edge-cloud" mentions in the compiled text are the intended
Future Work paragraph and the Data Availability transparency note.

**Approved and next: v2 GPU benchmark (Section 5.8).** You separately
approved running the corrected v2 sweep now — `benchmark/BENCHMARK_INSTRUCTIONS.md`
already describes it in full (randomized cell order, `precompute_contexts.py`
to remove Locust-process CPU-scoring confound, `--repeats 3` for 95% CIs);
see that document for the exact commands. Estimated ~3 hours GPU rental.
Once results come back, Section 5.8/Appendix G get rebuilt with real
confidence intervals, the LaTeX/PDF gets rebuilt again, and — per the
chosen descope path — the paper is then ready to submit as a non-network
Arabic RAG/context-compression paper.

**Still needed before this is genuinely ready:**
- Push this round's changes from your Mac (`.git/index.lock` is still
  permission-blocked from my side): `paper_draft_v2.md`, `paper/build_latex.py`,
  `paper/body_generated.tex`, `paper/abstract_generated.tex`,
  `paper/latex_submission/{body_generated.tex,sn-article-taha.pdf,.aux,.log,.out}`,
  `README.md`, `NETWORK_EXPERIMENT_STATUS.md`, this file.
- Run the v2 GPU benchmark sweep (see `benchmark/BENCHMARK_INSTRUCTIONS.md`)
  and send back `results/sweep_summary.csv` + `results/sweep_comparison.png`.
- Decide on the live-demo claim (deploy a real Space, or edit the abstract).
- Tag a release and archive this round + the v2 benchmark results as a
  new Zenodo version once both are done, matching the DOI cited in the
  paper.
