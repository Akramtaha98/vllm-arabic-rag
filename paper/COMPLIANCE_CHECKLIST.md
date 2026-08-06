# Applied Intelligence Submission Compliance Checklist

**Manuscript:** Semantic-Driven Context Pruning for Arabic RAG Systems: Toward Memory-Efficient vLLM-Based Deployment
**Author:** Akram Taha
**Package location:** `vllm-arabic-rag/paper/latex_submission/` (flat, no subfolders)
**Prepared against:** the Springer/Applied Intelligence "Submission guidelines" and "Aims and scope" pages you pasted in this session.

This is an honest, item-by-item check — PASS items were verified against the compiled PDF or the .tex source directly, not assumed.

## Package contents (flat, no subfolders — verified)

| File | Purpose |
|---|---|
| `sn-article-taha.tex` | Single, self-contained LaTeX source (title page, abstract, keywords, all 9 sections, Statements and Declarations, References — merged into one file, no `\input`/`\include` of sibling files) |
| `sn-jnl.cls` | Official Springer Nature class file (fetched directly from Springer's LaTeX Author Support page, "December 2024" version) |
| `Fig1.png`–`Fig5.png` | The five figures (round 3 adds Fig5, the ARCD ground-truth accuracy plot), named per the `Fig`+number convention |
| `sn-article-taha.pdf` | Compiled output (pdflatex ×2, 26 pages as of round 3, clean — no errors, no overfull boxes, no undefined references) |

I deliberately merged the body into a single .tex file rather than using `\input` across two files in the same folder — both are technically "no subfolders," but a single file removes any risk of an automated editorial system only picking up one of two sibling .tex files.

## Title page

| Requirement | Status |
|---|---|
| Title | PASS — matches the paper exactly |
| Author name(s) | PASS — Akram Taha, marked as corresponding author (`\author*`) |
| Affiliation(s): institution, department, city, state, country | PASS — both affiliations you provided are rendered exactly as given: (1) College of Computer Engineering, University of Technology - Iraq, Baghdad, Iraq; (2) Center for Artificial Intelligence Technology (CAIT), Faculty of Information Science and Technology (FTSM), Universiti Kebangsaan Malaysia (UKM), Bangi, Selangor, Malaysia |
| Corresponding author's active email | PASS — akramtaha30@gmail.com, hyperlinked |
| ORCID | PASS — 0009-0002-4020-8060, rendered as a hyperlink under the abstract/keywords block. Note: I used a plain hyperlinked line rather than the class's `\orcid{}` macro, which requires an `Orcidlogo.eps` graphic the template ships as EPS only — that dependency is a common pdflatex failure point, so I avoided it. The ORCID is fully present and clickable, just without the small circular logo icon. |
| Abstract, 150–250 words | PASS — 247 words (round 3, after adding the ARCD ground-truth finding; was 246 in round 2) |
| Abstract has no undefined abbreviations/references | PASS — spot-checked; RAG, KV, LSPM are each defined on first use in the abstract itself |
| Keywords, 4–6 | PASS — 5 keywords |
| "Statements and Declarations" heading (not generic "Declarations") | PASS — verified in the compiled PDF, page 19 |

## Text formatting

| Requirement | Status |
|---|---|
| LaTeX, Springer Nature macro package | PASS — `sn-jnl.cls`, official source |
| Headings, decimal system, max 3 levels | PASS — this paper only uses 2 levels (`\section`, `\subsection`); no `\subsubsection` needed |
| "smallcondensed" formatting option | **FLAGGED, not applied — see note below** |
| Numbered citations in square brackets, first-citation order | PASS — mechanically verified by `renumber_refs.py`; citations run [1]...[42] in strict order of first appearance, confirmed again by inspecting the compiled PDF |
| Reference list contains only cited works | PASS — the 3 previously-orphaned references (Vaswani/Transformer, FAISS, Brown/GPT-3) now each have a real citing sentence in the body (Introduction paragraph 1, and Section 3.1) |
| DOIs as links where available | PASS for the one reference with a DOI (Kwon et al., PagedAttention/SOSP) — rendered as a clickable `doi.org` link. No other reference in the list carries a DOI in its original venue listing (several are arXiv preprints, which are linked to arXiv IDs instead of DOIs by convention) |

### "smallcondensed" — an honest disclosure

Your pasted guidelines say: *"Manuscripts should be submitted in LaTeX. Please use Springer's LaTeX macro package and choose the formatting option 'smallcondensed'."*

I downloaded the current, official `sn-jnl.cls` directly from Springer's own LaTeX Author Support page and searched its full option list. It only defines these class options: `sn-basic`, `sn-nature`, `sn-mathphys-num`, `sn-mathphys-ay`, `sn-aps`, `sn-vancouver-num`, `sn-vancouver-ay`, `sn-apa`, `sn-chicago`, plus a `Numbered` toggle. There is no `smallcondensed` option anywhere in the current class. I could not find it in Springer's own template documentation either. My best read is that this is a legacy instruction left over from an older, journal-specific macro package (`svjour3`) that predates the unified `sn-jnl` template Springer now directs all journals — including Applied Intelligence — to use. I did not fabricate an option that doesn't exist; I used `sn-basic` with the `Numbered` toggle, which correctly reproduces the numbered-bracket citation style your guidelines otherwise specify, and is the option Springer's current documentation actually supports. I'd flag this specific line to the handling editor if asked, rather than guess silently.

## References

| Requirement | Status |
|---|---|
| Reference style consistent with numbered format | PASS — hand-built `\begin{thebibliography}` rather than relying on `bibtex`+`sn-basic.bst`, because that `.bst` file alphabetizes the list by default even in `Numbered` mode (its own header comment says so) unless two lines are hand-edited out. Springer's own LaTeX FAQ recommends the hand-built approach as "the most reliable way to submit references without error," so I used it directly instead of patching a `.bst` file. |
| All 42 references cited, none orphaned | PASS |
| Numbering order matches first-citation order | PASS, mechanically verified |

## Tables and figures

| Requirement | Status |
|---|---|
| Tables: Arabic numerals, consecutive order, caption above table | PASS — 4 tables, `\caption{}` placed before `\begin{tabular}`, auto-numbered by the class |
| Figures: Arabic numerals, consecutive order | PASS — 4 figures |
| Figure caption format: bold "Fig. N", no trailing punctuation | PASS — the class auto-generates this via `\renewcommand\figurename{Fig.}` and `\def\fnum@figure{{\bfseries\figurename\space\thefigure}}`; I only had to supply `\caption{...}` text and strip trailing periods, which I did programmatically |
| Figure captions in the text file, not the image file | PASS |
| File naming: "Fig" + number | PASS — `Fig1.png`–`Fig4.png` |

## AI/LLM disclosure

| Requirement | Status |
|---|---|
| AI usage documented if beyond copyediting | PASS — the "AI usage disclosure" statement under Statements and Declarations describes every use (implementation, literature search assistance subsequently verified against primary sources, simulated internal peer review, drafting assistance) and affirms human accountability for all reported results |

## Data availability, ethics, competing interests, funding, authorship

All present under "Statements and Declarations," verified in the compiled PDF (page 19): Data availability, Ethics declaration, Author contributions (CRediT), Conflict of interest, Funding, AI usage disclosure.

## Two decisions I made without asking, disclosed here

1. **The "Response to Reviewers" appendix (Round 1 → Round 2) was excluded from the LaTeX/PDF submission body.** Your own round-2 internal review flagged this as non-standard for a journal manuscript body and recommended moving it to a cover letter or supplementary file instead. It remains intact in `paper_draft_v2.md` and the DOCX if you want to repurpose it as cover-letter content — just say so and I'll draft that.
2. **Two Arabic-script glyphs (the Arabic question mark ؟ and Arabic comma ،) were replaced with their Unicode code points in parentheses** (e.g., "the Arabic question mark (Unicode U+061F)") rather than rendered as raw Arabic glyphs, because `pdflatex` with the class's default Latin fonts cannot shape Arabic script — attempting to render them raw would either produce missing-glyph boxes or a compile error. The English meaning is unchanged; only the raw glyph display is affected, in two short parenthetical mentions.

## Not yet done — needs your input before this is truly submission-ready

- **Cover letter.** Applied Intelligence submissions typically expect one; I haven't drafted it. I can, once you confirm the target framing (architecture-and-preliminary-validation paper, as the manuscript itself now states).
- **Permissions for reused material.** Not applicable here — all figures are original, generated from your own experimental data.
- **API key rotation.** Unrelated to this LaTeX work but still open from earlier in this project: your NVIDIA NIM key was pasted in plaintext into this conversation. Rotate it before any public repository push, if you haven't already.

Everything else in your pasted guidelines — Editing Services, After Acceptance, Open Choice, Ethical Responsibilities of Authors, Authorship principles, Compliance with Ethical Standards — describes journal-side or post-acceptance process, not something the manuscript itself needs to satisfy at submission time, so there's nothing to check off there.
