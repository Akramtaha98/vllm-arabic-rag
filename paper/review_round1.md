# Peer Review Package — Round 1
**Manuscript:** Semantic-Driven Context Pruning for Optimizing Arabic RAG Systems in Memory-Constrained vLLM Deployments
**Author:** Akram Taha
**Target venue:** Applied Intelligence (Springer)

---

## Phase 0 — Field Analysis & Reviewer Configuration

**Primary discipline:** Natural Language Processing / Information Retrieval (Arabic NLP subfield)
**Secondary discipline:** Systems for ML (LLM serving infrastructure)
**Methodology type:** Applied systems paper with preliminary empirical validation (not a full controlled experimental study)
**Paper maturity:** Early-stage / preliminary-results paper — explicitly self-labeled as such throughout
**Target journal tier:** Q2 applications-focused AI journal (Applied Intelligence, Springer)

**Reviewer panel:**
- **EIC** — Applied Intelligence handling editor, background in applied NLP systems, evaluates journal fit / originality / whether the paper delivers on its title's promise.
- **R1 (Methodology)** — Empirical NLP methodologist, focuses on experimental design, sample size, statistical validity, baseline comparisons.
- **R2 (Domain)** — Arabic NLP / IR specialist, focuses on literature coverage and domain-specific claims.
- **R3 (Perspective)** — Systems/ML-infra reviewer, focuses on whether the vLLM/PagedAttention/KV-cache claims are actually substantiated.
- **Devil's Advocate** — Challenges the core argument end-to-end.

---

## Reviewer 1 (EIC) — Editorial Assessment

**Recommendation: Major Revision**

**Strengths:** Clear, well-motivated problem (Arabic tokenization inflation → KV-cache pressure); clean system architecture; refreshingly honest framing of what is measured vs. not measured; strong reference base (41 sources); reproducible artifacts released.

**Core concern — title/claim vs. evidence mismatch:** The title promises optimization "in Memory-Constrained vLLM Deployments," and the abstract's motivating claim is a KV-cache/throughput benefit. No experiment in the paper touches vLLM, PagedAttention, or a KV-cache metric — all reported experiments run against a hosted, non-vLLM inference API. This is disclosed honestly in Sections 4.3/6/7, which I credit, but a reader skimming the title and abstract would reasonably expect at least one systems-level number. As written, the paper is a *retrieval-context-compression* paper with a *vLLM-motivated introduction*, not a vLLM systems paper. Either (a) the framing needs to be adjusted so the title/abstract accurately scope the contribution, or (b) the systems experiment needs to materialize before this is publishable as currently framed.

**Secondary concern:** A single compression ratio (r=0.5) and n=8 is thin for a journal-length empirical claim, even a preliminary one.

---

## Reviewer 2 (Methodology) — Peer Review Report

**Recommendation: Major Revision**

1. **[MAJOR] Sample size and single operating point.** n=8 QA items at a single ratio (r=0.5) is a pilot, not a study. No dose-response relationship between compression ratio and fidelity is shown, so a reader cannot tell whether 0.5 was a lucky operating point or representative.
2. **[MAJOR] No baseline comparison.** The paper argues LSPM's cross-encoder sentence scoring is better than naive alternatives but never measures this — there is no length-matched control (e.g., naive truncation, random sentence sampling) run through the same pipeline. Without this, the fidelity numbers show only that *some* 50% reduction preserves answers reasonably well, not that *semantic scoring specifically* is responsible.
3. **[MAJOR] No statistical inference.** Table 2 reports mean/min/max only. With n=8, a min/max range is not informative about reliability; a bootstrap confidence interval, or at minimum a standard error, is expected.
4. **[MINOR] Self-authored dual role.** The same author wrote both the pilot questions and the "ground truth" corpus, and fidelity is scored against the model's own raw-context answer rather than an independent gold answer. This is disclosed as a limitation (7), which is appropriate, but the paper should state explicitly that fidelity ≠ correctness under this design.
5. **[MINOR] Tokenization disparity — good practice, minor gap.** Real measurement, two tokenizers, reasonable n=30 — solid. Missing: a significance test (e.g., one-sample t-test/Wilcoxon against ratio=1) to formally support "consistently present," which is currently only supported descriptively.

---

## Reviewer 3 (Domain — Arabic NLP) — Peer Review Report

**Recommendation: Minor Revision**

Literature coverage is strong and current (AraBERT, AraGPT2, Jais, ARCD, ACVA, ArabicMMLU, CAMeL Tools, MIRACL, multilingual E5 all appropriately cited). The rouge-score Arabic-tokenizer bug fix (Section 4.2) is a genuinely useful, citable methodological contribution in its own right — I'd suggest the authors foreground it more, since it's a concrete, verifiable finding independent of the pruning story.

1. **[MINOR] Single dialect/register.** All pilot text is Modern Standard Arabic, formal register, university-domain. Should be flagged as a further scope limitation (dialectal Arabic tokenization disparity may differ).
2. **[MINOR] ARCD is mentioned as future-work benchmark (Section 8) but never characterized** (size, domain) for readers unfamiliar with it — one sentence would help.
3. **[MINOR] Table 1's aggregate vs. mean/median ratios could be confused by readers;** a one-line explanation of why aggregate (1.76/1.69) differs from mean (1.79/1.72) would preempt reviewer/reader questions.

---

## Reviewer 4 (Perspective — Systems/ML-Infra) — Peer Review Report

**Recommendation: Major Revision**

This is the sharpest gap in the paper from a systems standpoint.

1. **[MAJOR] Zero systems-level evidence.** vLLM, PagedAttention, and the `/metrics`-driven dynamic controller (Section 3.4) are architected but never exercised in any experiment. For a paper whose title and introduction center on "memory-constrained vLLM deployments," this is the load-bearing claim, and it is currently a specification, not a result.
2. **[MAJOR] No quantification of LSPM's own overhead.** The cross-encoder pruning stage adds latency and (for the accurate BGE-reranker-v2-m3 backend) its own memory footprint. The paper never measures this cost, so the net benefit (KV-cache savings minus pruning overhead) is unknown even in principle.
3. **[MINOR, easy fix] An analytical estimate would help even without a GPU.** Given the real, measured token-reduction numbers (Section 5.1/5.2) and Llama-3.1-8B-Instruct's public architecture config, the authors could derive an analytical KV-cache-bytes-saved-per-request estimate. This would not replace the real benchmark but would let the reader translate "50% fewer tokens" into "X MB saved per request," strengthening the systems narrative honestly while the real GPU study is pending.

---

## Devil's Advocate Report

**Strongest counter-argument (against the paper as currently framed):** *This paper measures a compression method's effect on answer similarity using a hosted API with no memory instrumentation, then names that method after and frames it around a GPU serving engine (vLLM) it never touches. The central causal chain the paper needs — "less context → smaller KV cache → higher vLLM throughput" — has exactly zero of its three links empirically shown. Link 1 (less context) is shown. Link 2 (smaller KV cache) is asserted from architecture, not measured. Link 3 (higher throughput) is asserted, not measured. A reviewer who reads only the abstract and the conclusion could reasonably believe vLLM was benchmarked; it was not.*

**Issue list:**
- **[CRITICAL]** Title/abstract scope vs. delivered evidence mismatch (see above). Must be resolved before acceptance at any tier — either narrow the claim or supply the systems evidence.
- **[MAJOR]** No baseline comparison (duplicated from R1 — cross-reviewer convergence strengthens this).
- **[MAJOR]** Fidelity evaluated only at one ratio; the "sweet spot" framing implied by choosing r=0.5 is not earned by the data shown.
- **[MINOR]** The paper's own honesty about limitations (Section 7) partially defuses this critique in spirit, but honesty about a gap is not the same as closing it — a Major Revision should still require concrete narrowing of the claim or additional evidence, not just more caveats.

**Ignored alternative explanation:** High fidelity at r=0.5 could partly reflect that the 10-document mock corpus is small and low-redundancy, so even naive truncation might preserve fidelity about as well as semantic pruning — which is exactly what the missing baseline (R1, R4) would test.

**Missing stakeholder perspective:** A systems/MLOps reader evaluating whether to adopt LSPM in a real vLLM deployment gets no operating data (latency added, memory saved, throughput at concurrency) to make that decision — only an architecture diagram and a promise.

---

## Phase 2 — Editorial Synthesis & Decision

**Consensus across reviewers:** Small n / single ratio (R1, DA), no baseline comparison (R1, R4, DA), no statistical inference on the fidelity numbers (R1). These three converge independently and are the most actionable.

**Disagreement:** EIC and R4 (Perspective) treat the title/scope mismatch as the central blocking issue; R2 (Domain) is close to Minor Revision on the Arabic-NLP content alone and does not weight the systems gap as heavily, since Arabic-NLP contribution stands on its own.

**Devil's Advocate CRITICAL flag:** Present (title/abstract vs. evidence mismatch). Per the IRON RULE, this alone precludes an Accept decision at this round.

### Editorial Decision: **Major Revision**

**Required for the next round (in priority order):**
1. Resolve the title/abstract/evidence mismatch — narrow the framing to what is actually measured, and/or add the analytical KV-cache estimate (R4's suggested easy fix) as a stopgap for the un-run GPU benchmark.
2. Add a real baseline comparison (naive length-matched truncation vs. LSPM) on the same pipeline.
3. Expand the fidelity pilot beyond a single ratio and beyond n=8; report confidence intervals, not just min/max.
4. Add a significance test to the tokenization disparity claim.
5. Minor: dialect-scope caveat, ARCD one-line description, Table 1 aggregate-vs-mean clarification, foreground the rouge-score Arabic-tokenizer bug fix as a standalone contribution.

**Not required, but recommended:** measure LSPM's own scoring-stage latency/overhead even on CPU — this is fully within reach without a GPU and materially strengthens R4's "net benefit" concern.
