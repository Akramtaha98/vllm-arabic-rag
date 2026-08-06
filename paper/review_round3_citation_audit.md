# Round 3 — Deep Citation Audit + Honest Re-Assessment
**Manuscript:** Semantic-Driven Context Pruning for Optimizing Arabic RAG Systems in Memory-Constrained vLLM Deployments (Revision 2, v2 docx/md)
**Method:** Automated cross-check of every in-text citation marker against the reference list (not a prose skim) + manual verification of the three flagged references + figure/table cross-reference audit.

---

## 1. Reference sequence and citation-list integrity — real defects found

I ran an automated pass extracting every `[n]` citation marker from the body text, in the order it appears, and diffed it against the reference list. Two genuine problems, both worth fixing before this goes anywhere near a submission portal:

### 1a. Three references are never cited in the body text (orphan references)

**[23] Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)**
**[24] Johnson, Douze, Jégou, "Billion-Scale Similarity Search with GPUs" (FAISS)**
**[27] Brown et al., "Language Models are Few-Shot Learners" (GPT-3)**

I confirmed this two ways: an automated diff of all 65 citation instances in the text against the 42-entry reference list (39 distinct numbers cited, all present in the list; 3 list entries never appear in the diff), and a visual check of the rendered PDF page showing references 20-38 — [23], [24], and [27] sit there with no corresponding `[23]`/`[24]`/`[27]` anywhere in the prose.

This is a real defect, not a style nitpick. Uncited references are something copyeditors and reviewers actively flag — at minimum it looks like padding toward the "40+ references" target you asked for earlier, and at most a careful reviewer reads it as evidence the reference list wasn't checked against the final text. All three are legitimate, on-topic sources that clearly *should* be cited somewhere and were probably dropped when I gathered references early and then never wired citing sentences to them:
- [23] (Transformer) belongs wherever the paper says "transformer" without a citation — Section 3.3 ("jointly encode... through a single transformer") and Section 2.3 are the natural spots.
- [24] (FAISS) belongs in Section 2.1 or 3.1 where the paper discusses the vector index / Chroma-backed retriever, as an example of the similarity-search infrastructure category.
- [27] (GPT-3) belongs in the Introduction's opening sentence about LLMs generally, alongside or instead of the current uncited claim.

**Fix required:** add one real citing clause for each, in context — not just insert the bracket. I have not made this change; per the reviewer role I'm not editing the manuscript, only reporting it.

### 1b. Reference numbering does not follow IEEE's first-citation-order convention

You explicitly chose IEEE numbered citation format. The IEEE convention (and most numbered-citation styles) is that references are numbered in the order they are **first cited in the text**, not grouped by topic. I checked this directly: the first three references cited in the body are [1], [2], then [20] and [21] — StreamingLLM and H2O are cited in the Introduction's second paragraph, as examples inside a parenthetical about the KV-cache-management literature, well before [3]/[4]/[5] (LLMLingua, Selective Context, RECOMP) are cited two paragraphs later. Reference [3] should logically be the third reference cited if the numbering followed IEEE convention; instead it's the seventh.

This happened because I built the reference list in thematic blocks (RAG → compression → reranking → Arabic NLP → serving → evaluation) rather than in strict citation order, then numbered top to bottom. The list itself is internally consistent (1–42, no gaps, no duplicate numbers, every number cited appears once in the list) — this is not a broken bibliography — but it is **not** IEEE-compliant numbering given the citation-format you specified. A meticulous reviewer or a copyeditor at a numbered-citation journal would flag this and require renumbering.

**Fix required:** renumber the reference list to match strict first-citation order in the body text, and update all 65 in-text markers accordingly. This is mechanical but has to touch nearly every citation in the paper, so it's a real (if low-risk) editing pass, not a one-line fix.

### 1c. Minor: reference [8] (BGE-M3) cites a Hugging Face model card rather than the actual paper

I re-verified this while auditing references: BGE-M3 has a real, citable arXiv paper — **Chen, Xiao, Zhang, Luo, Lian, and Liu, "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," arXiv:2402.03216, 2024** — which I did not have fully confirmed when I first wrote this reference in round 1, so I hedged with a weaker HF-model-card citation. This should be replaced with the proper paper citation; it's a stronger, more standard source and I now have the verified author list and arXiv ID.

---

## 2. Everything else I checked in this pass

**Figure/table cross-references:** clean. Figures 1–4 and Tables 1–4 are each defined exactly once and referenced consistently; no orphan or mismatched figure/table numbers.

**Numeric consistency between prose and CSVs:** spot-checked Table 3 (paired comparison) and the pruner-latency figure (57.8ms) against the underlying `results/lspm_vs_naive_paired.csv` and `data/eval_set_expanded.jsonl` in the previous round — still accurate, nothing drifted when I rewrote surrounding prose in round 2.

**Reference-count target:** you asked for 40+ earlier; the list has 42 entries. After fixing 1a (citing all three), it's still 42, all cited — the target is still met and now honestly met.

**What I did not find:** no fabricated citations, no misattributed authors on spot-check, no reference pointing to a source that doesn't exist. The problems above are integrity-of-bibliography-mechanics issues (uncited entries, numbering convention), not fabrication.

---

## 3. Honest overall assessment

Substantively, the paper is in the same good shape the round-2 re-review found: all round-1 major/critical findings were genuinely addressed with real data, including the honest null result on the baseline comparison, and the remaining gap (no GPU benchmark) is disclosed rather than implied away.

What I missed across rounds 1 and 2 was a citation-mechanics audit — I was focused on experimental rigor and scope-of-claims, and didn't run the "does every bracket resolve, and is the list in the right order" check until you asked for it just now. That was a real gap in my own review process, and you were right to ask for it specifically.

**Updated status:** still Minor Revision on substance, but I would not consider this submission-ready until 1a and 1b are fixed — a numbered-citation journal's editorial office or a careful reviewer would send this back for uncited references and out-of-order numbering even if they had no other complaint, and that's an unforced error worth closing before you send it anywhere.

I have not made these fixes yet. Say the word and I'll: (1) write real citing sentences for [23]/[24]/[27], (2) renumber the full reference list and all 65 in-text markers to strict first-citation order, (3) swap [8] to the verified BGE-M3 arXiv citation, and (4) rebuild and re-verify the docx.
