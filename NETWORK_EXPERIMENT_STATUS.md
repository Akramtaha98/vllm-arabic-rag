# Network-Aware Edge-Cloud Experiment: Status (Withdrawn from Paper)

**This document is superseded by the manuscript itself and is kept only as
a historical record.** The edge-cloud, network-aware extension described
below was implemented, deployed against a rented GPU, and pilot-tested
(round 9). On methodological review (round 10), the pilot was found to
have validity gaps serious enough that its numbers are **not** reported as
a result in `paper/paper_draft_v2.md`. The extension has been withdrawn
from the paper body and redirected to Section 8 (Future Work). The full,
itemized account of what was found and why is in the paper's **Appendix J
("Round 9 → Round 10: Edge-Cloud Extension Withdrawn")**, with a shorter
pointer added to the end of **Appendix I**. That is now the authoritative
account — read it, not this file, for the current status.

## Why the pilot's numbers are not in the paper

Confirmed on independent verification of the code and raw logs:

- The controller's `pressure_score` and `ratio_continuous` fields were
  mislabeled in early drafts (the reported 0.674/0.530 figures are
  `ratio_continuous`, not `pressure_score`, which was actually 0.065/0.424).
- The gateway returns HTTP 200 with a JSON error body on internal
  failures and on emulated packet loss, so Locust's own stats never
  register these as failures — a "zero request failures" claim built on
  that stat was misleading. 308 of 327 `emulated_loss` events fall inside
  real, reported grid-cell time windows and are genuine, by-design loss
  events under `constrained_wireless`.
- The bandwidth throttle in `network_profiles.py` applies only to
  `response_bytes`, never to `request_bytes` (i.e., the uploaded
  retrieved/pruned context is not throttled), so the emulation does not
  represent a true bidirectional network constraint.
- `t_send` is recorded after retrieval, controller decision, and pruning
  have already happened at the edge, not before — so reported latency is
  not a true end-to-end, client-perceived TTFT.
- Edge and cloud tiers were co-located rather than genuinely separated by
  a real network path; `tc netem` application-level sleeps, not real
  packet shaping, produced the emulated delay/loss.
- Controller weights and normalization ranges are documented placeholders,
  not calibrated against measured traffic.
- No answer correctness or faithfulness was measured under the grid (the
  mock corpus used has no gold answers).

None of this means the code is broken — the underlying pieces (manifest
determinism, controller decision logic, network-profile
apply/verify/failure handling, the gateway's request/response/logging
cycle) were unit-tested and behaved correctly for what they are. It means
the pilot, as run, does not support a network-benefit claim, so it is not
presented as one.

## Current disposition

- Code (`benchmark/edge_gateway.py`, `network_controller.py`,
  `network_profiles.py`, `locustfile_edge.py`, `run_validation_grid.py`)
  remains in the repository, released for transparency, but its pilot
  output is **not** cited as evidence in the submitted paper.
- The paper's Section 8 (Future Work) specifies what a valid version of
  this experiment would require: real network separation, request-side
  bandwidth shaping, true end-to-end TTFT measured from the client,
  corrected failure accounting, a calibrated controller, and an
  accuracy/faithfulness pass — before any network-aware claim is made.
- The manuscript's title, abstract, and contributions list remain
  non-network-aware (Arabic RAG / context-compression), consistent with
  submitting once the corrected GPU benchmark (Section 5.8) is resolved.
- Rerunning this experiment properly is not part of the current
  submission plan; it is left as explicit future work.

## Original implementation notes (round 9, for reference only)

The original "Tasks 1-7" implementation log that used to live in this file
— covering the harness repair, the edge-gateway prototype, the four
network profiles, the five pruning/controller conditions, the
network-aware controller, the validation-grid orchestrator, and the
no-GPU unit verification performed before the pilot run — is preserved in
git history (see the version of this file prior to the round-10 revision)
and is summarized, with corrected numbers, in Appendix J of the paper.
