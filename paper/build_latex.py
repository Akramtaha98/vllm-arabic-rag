#!/usr/bin/env python3
"""
Converts paper_draft_v2.md (Sections 1-9 + Statements and Declarations +
References; Appendix excluded per round-2 review recommendation) into a
Springer Nature sn-jnl LaTeX body, written to body_generated.tex, AND
generates abstract_generated.tex (abstract + keywords) from the same
source markdown. This file is \input{} by the main sn-article-taha.tex
file, which supplies the documentclass and author/affiliation/ORCID block
directly (those genuinely don't change round to round), and \input{}s
abstract_generated.tex for the abstract/keywords.

IMPORTANT, learned the hard way in round 7 (external review caught it):
the abstract used to be hand-copied into sn-article-taha.tex separately
from paper_draft_v2.md's abstract, and drifted out of sync across several
revision rounds without anyone noticing until a reviewer read the actual
PDF -- the abstract still said "absent GPU access... not yet measured"
while Section 5.8 reported a real GPU benchmark. Generating it from the
same source as the body makes that specific class of bug structurally
impossible going forward. Do not hand-edit the abstract/keywords in
sn-article-taha.tex again; edit paper_draft_v2.md's ## Abstract section
and re-run this script.
"""
import re

SRC = "paper_draft_v2.md"
OUT = "body_generated.tex"
ABSTRACT_OUT = "abstract_generated.tex"

text = open(SRC, encoding="utf-8").read()

refs_marker = "## References"
appendix_marker = "## Appendix"
refs_start = text.find(refs_marker)
appendix_start = text.find(appendix_marker)

# Body = Section 1 (## 1. Introduction) through end of Statements and
# Declarations, i.e. up to refs_start. We locate the start at "## 1. Introduction".
body_start = text.find("## 1. Introduction")
body = text[body_start:refs_start]
refs_block = text[refs_start:appendix_start]

# ---------------------------------------------------------------
# 1. Unicode -> LaTeX-safe replacements (applied before escaping,
#    since these are deliberate substitutions, not raw text that
#    needs backslash-escaping).
# ---------------------------------------------------------------
UNICODE_REPLACEMENTS = [
    ("؟", "(Unicode U+061F)"),   # Arabic question mark glyph -> descriptive, avoids
                                  # requiring Arabic font shaping under pdflatex
    ("،", "(Unicode U+060C)"),   # Arabic comma glyph -> descriptive
    ("—", "TEMPEMDASH"),         # placeholder, replaced post-escape with ---
    ("–", "TEMPENDASH"),         # placeholder, replaced post-escape with --
    ("×", "TEMPTIMES"),
    ("−", "TEMPMINUS"),
    ("→", "TEMPRARROW"),
    ("≤", "TEMPLEQ"),
    ("·", "TEMPCDOT"),
    ("∈", "TEMPIN"),
    ("α", "TEMPALPHA"),
    ("≥", "TEMPGEQ"),
    ("≈", "TEMPAPPROX"),
]

for old, new in UNICODE_REPLACEMENTS:
    body = body.replace(old, new)
    refs_block = refs_block.replace(old, new)


def escape_latex_text(s):
    """Escape LaTeX special characters in plain prose text (not code spans,
    not table pipes, not already-converted markdown markers)."""
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("$", r"\$")
    # underscores handled specially by code-span logic; escape stray ones
    s = s.replace("_", r"\_")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("<", "$<$")
    s = s.replace(">", "$>$")
    return s


def restore_symbols(s):
    s = s.replace("TEMPEMDASH", "---")
    s = s.replace("TEMPENDASH", "--")
    s = s.replace("TEMPTIMES", "$\\times$")
    s = s.replace("TEMPMINUS", "$-$")
    s = s.replace("TEMPRARROW", "$\\rightarrow$")
    s = s.replace("TEMPLEQ", "$\\leq$")
    s = s.replace("TEMPCDOT", "$\\cdot$")
    s = s.replace("TEMPIN", "$\\in$")
    s = s.replace("TEMPALPHA", "$\\alpha$")
    s = s.replace("TEMPGEQ", "$\\geq$")
    s = s.replace("TEMPAPPROX", "$\\approx$")
    return s


CODE_SPANS = []


def stash_code(m):
    code = m.group(1)
    code_escaped = code.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("$", r"\$")
    CODE_SPANS.append(r"\texttt{" + code_escaped + "}")
    return f"@@CODESPAN{len(CODE_SPANS)-1}@@"


def restore_code(s):
    def repl(m):
        idx = int(m.group(1))
        return CODE_SPANS[idx]
    return re.sub(r"@@CODESPAN(\d+)@@", repl, s)


LINK_SPANS = []


def stash_link(m):
    text, url = m.group(1), m.group(2)
    text_escaped = escape_latex_text(text)
    LINK_SPANS.append(r"\href{" + url + "}{" + text_escaped + "}")
    return f"@@LINKSPAN{len(LINK_SPANS)-1}@@"


def restore_links(s):
    def repl(m):
        idx = int(m.group(1))
        return LINK_SPANS[idx]
    return re.sub(r"@@LINKSPAN(\d+)@@", repl, s)


def convert_inline(s):
    """Convert one line/paragraph of markdown inline syntax to LaTeX."""
    s = re.sub(r"`([^`]+)`", stash_code, s)
    s = re.sub(r"\[([^\[\]]+)\]\((https?://[^\s()]+)\)", stash_link, s)
    s = escape_latex_text(s)
    s = restore_links(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\textit{\1}", s)
    s = restore_code(s)
    s = restore_symbols(s)
    return s


FIGURE_FILES = {
    1: "Fig1.png",
    2: "Fig2.png",
    3: "Fig3.png",
    4: "Fig4.png",
}

TABLE_REGISTRY = []
out = []


def build():
    global TABLE_REGISTRY, out
    TABLE_REGISTRY = []
    out = []
    i2 = 0
    lines2 = body.split("\n")
    n2 = len(lines2)
    while i2 < n2:
        line = lines2[i2]
        stripped = line.strip()
        if stripped == "" or stripped == "---":
            i2 += 1
            continue
        m = re.match(r"^##\s+(\d+)\.\s+(.*)$", stripped)
        if m:
            title = convert_inline(m.group(2))
            out.append(f"\\section{{{title}}}")
            i2 += 1
            continue
        m = re.match(r"^###\s+[\d.]+\s+(.*)$", stripped)
        if m:
            title = convert_inline(m.group(1))
            out.append(f"\\subsection{{{title}}}")
            i2 += 1
            continue
        m = re.match(r"^##\s+Statements and Declarations\s*$", stripped)
        if m:
            out.append(r"\section*{Statements and Declarations}")
            i2 += 1
            continue
        m = re.match(r"^\*\*\[Figure (\d+) near here:\s*(.*?)\]\*\*$", stripped)
        if m:
            fignum = int(m.group(1))
            rest = m.group(2)
            fm = re.match(r"^(\S+\.(?:png|jpg|jpeg))\s*:\s*(.*)$", rest)
            caption_text = fm.group(2).strip() if fm else rest.strip()
            caption_text = caption_text.rstrip(".")
            caption_latex = convert_inline(caption_text)
            fname = FIGURE_FILES.get(fignum, f"Fig{fignum}.png")
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(f"\\includegraphics[width=0.85\\textwidth]{{{fname}}}")
            out.append(f"\\caption{{{caption_latex}}}")
            out.append(f"\\label{{fig:{fignum}}}")
            out.append(r"\end{figure}")
            i2 += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i2 < n2 and lines2[i2].strip().startswith("|"):
                table_lines.append(lines2[i2].strip())
                i2 += 1
            header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
            # Shorten header cells that are too wide for a print-column table
            # (full explanatory text already appears in the surrounding prose).
            HEADER_SHORTEN = {
                "Sign test (above/below 1.0)": "Sign test",
                "TTFT p50 / p95 (ms)": "TTFT p50/p95 (ms)",
                "Mean / Peak KV-cache (%)": "Mean/Peak KV (%)",
                "Completion tok/s": "Tok/s",
                "Correctness 95% CI": "Correct. CI",
                "Faithfulness 95% CI": "Faithful. CI",
                "Correctness (0-2)": "Correct. (0-2)",
                "Faithfulness (0-1)": "Faithful. (0-1)",
            }
            header_cells = [HEADER_SHORTEN.get(c, c) for c in header_cells]
            data_rows = []
            for row in table_lines[2:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                data_rows.append(cells)
            ncols = len(header_cells)
            colspec = "l" + "c" * (ncols - 1)
            caption_latex = None
            if out and out[-1].startswith("TABLECAPTION::"):
                caption_latex = out.pop()[len("TABLECAPTION::"):]
            table_num = len(TABLE_REGISTRY) + 1
            TABLE_REGISTRY.append(table_num)
            out.append(r"\begin{table}[htbp]")
            if caption_latex:
                out.append(f"\\caption{{{caption_latex}}}")
            out.append(f"\\label{{tab:{table_num}}}")
            out.append(r"\centering")
            # Known-wide tables (6 numeric columns with long headers, e.g.
            # the GPU benchmark table -- 63.7pt overflow caught on external
            # review) drop to \scriptsize and tighter column padding instead
            # of \footnotesize. (Tried wrapping in \resizebox first, but
            # sn-jnl.cls's internal table wrapper -- threeparttable/tableorg
            # -- doesn't tolerate \resizebox directly around \tabular there:
            # it throws "Division by 0" and visibly corrupts the page. Font
            # size + tabcolsep is the safe fix for this class.)
            needs_shrink = caption_latex is not None and (
                "least time-confounded" in caption_latex
                or "Human-rated correctness and faithfulness" in caption_latex
            )
            out.append(r"\scriptsize" if needs_shrink else r"\footnotesize")
            out.append(r"\setlength{\tabcolsep}{2pt}" if needs_shrink else r"\setlength{\tabcolsep}{3.5pt}")
            out.append(f"\\begin{{tabular}}{{{colspec}}}")
            out.append(r"\toprule")
            out.append(" & ".join(convert_inline(c) for c in header_cells) + r" \\")
            out.append(r"\midrule")
            for row in data_rows:
                out.append(" & ".join(convert_inline(c) for c in row) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{table}")
            continue
        m = re.match(r"^\*\*Table\s+\d+\.\s*(.*)\*\*$", stripped)
        if m:
            cap = convert_inline(m.group(1).rstrip("."))
            out.append("TABLECAPTION::" + cap)
            i2 += 1
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m and m.group(1) == "1":
            items = []
            while i2 < n2:
                mm = re.match(r"^(\d+)\.\s+(.*)$", lines2[i2].strip())
                if not mm:
                    break
                items.append(convert_inline(mm.group(2)))
                i2 += 1
            out.append(r"\begin{enumerate}")
            for it in items:
                out.append(f"\\item {it}")
            out.append(r"\end{enumerate}")
            continue
        if stripped.startswith("- "):
            items = []
            while i2 < n2 and lines2[i2].strip().startswith("- "):
                items.append(convert_inline(lines2[i2].strip()[2:]))
                i2 += 1
            out.append(r"\begin{itemize}")
            for it in items:
                out.append(f"\\item {it}")
            out.append(r"\end{itemize}")
            continue
        para_lines = [stripped]
        i2 += 1
        while i2 < n2 and lines2[i2].strip() != "" and lines2[i2].strip() != "---" \
                and not lines2[i2].strip().startswith("|") \
                and not re.match(r"^##", lines2[i2].strip()) \
                and not re.match(r"^\*\*\[Figure", lines2[i2].strip()) \
                and not re.match(r"^\*\*Table\s+\d+\.", lines2[i2].strip()) \
                and not re.match(r"^(\d+)\.\s+", lines2[i2].strip()) \
                and not lines2[i2].strip().startswith("- "):
            para_lines.append(lines2[i2].strip())
            i2 += 1
        para_text = " ".join(para_lines)
        out.append(convert_inline(para_text))
        out.append("")
    return "\n".join(out)


body_tex = build()

# ---------------------------------------------------------------
# 2b. Generate abstract_generated.tex from the same markdown source
#     (see module docstring -- this replaced a hand-copied abstract that
#     silently drifted out of sync with the body across revision rounds).
# ---------------------------------------------------------------
abstract_marker = "## Abstract"
keywords_marker = "**Keywords:**"
abstract_start = text.find(abstract_marker)
keywords_start = text.find(keywords_marker)
intro_marker = "## 1. Introduction"
intro_start_for_kw = text.find(intro_marker)

assert abstract_start != -1 and keywords_start != -1, "abstract/keywords markers not found"

abstract_raw = text[abstract_start + len(abstract_marker):keywords_start].strip()
# Keywords are a single markdown line right after the "**Keywords:**"
# marker; take only that line, not everything up to Section 1 (which
# would otherwise swallow the "---" separator and blank lines in between).
keywords_line = text[keywords_start + len(keywords_marker):intro_start_for_kw].strip().split("\n")[0]
keywords_list = [k.strip().rstrip(".") for k in keywords_line.split(";") if k.strip()]

for old, new in UNICODE_REPLACEMENTS:
    abstract_raw = abstract_raw.replace(old, new)

abstract_tex = convert_inline(abstract_raw)
keywords_tex = ", ".join(convert_inline(k) for k in keywords_list)

abstract_block = f"\\abstract{{{abstract_tex}}}\n\n\\keywords{{{keywords_tex}}}\n"
open(ABSTRACT_OUT, "w", encoding="utf-8").write(abstract_block)
print(f"Wrote {ABSTRACT_OUT}, {len(keywords_list)} keywords.")

# ---------------------------------------------------------------
# 3. Build References as hand-crafted thebibliography (order already
#    verified correct by renumber_refs.py -- 1..42 first-citation order).
# ---------------------------------------------------------------
entry_pattern = re.compile(r"^\[(\d+)\]\s(.*)$", re.MULTILINE)
ref_entries = {}
for m in entry_pattern.finditer(refs_block):
    ref_entries[int(m.group(1))] = m.group(2).strip()

assert set(ref_entries.keys()) == set(range(1, 44)), "reference parse failed"

def linkify_doi(entry_tex):
    # Turn "doi: 10.xxxx/yyyy" into a clickable hyperref link (sn-jnl.cls
    # already \RequirePackage{hyperref}, so \href is available with no
    # extra \usepackage needed).
    m = re.search(r"doi: (10\.\S+?)\.?$", entry_tex)
    if m:
        doi = m.group(1).rstrip(".")
        link = f"doi: \\href{{https://doi.org/{doi}}}{{{doi}}}"
        entry_tex = entry_tex[:m.start()] + link + entry_tex[m.end():]
    return entry_tex


bib_lines = [r"\begin{thebibliography}{43}"]
for num in range(1, 44):
    entry = ref_entries[num]
    entry_tex = convert_inline(entry)
    entry_tex = re.sub(r"(https://\S+?)(\.?)( |$)", r"\\url{\1}\2\3", entry_tex)
    entry_tex = linkify_doi(entry_tex)
    bib_lines.append(f"\\bibitem{{ref{num}}} {entry_tex}")
bib_lines.append(r"\end{thebibliography}")
bib_tex = "\n\n".join(bib_lines)

full_out = body_tex + "\n\n" + bib_tex + "\n"
open(OUT, "w", encoding="utf-8").write(full_out)
print(f"Wrote {OUT}, {len(full_out)} chars, {len(TABLE_REGISTRY)} tables.")
print("First 800 chars:")
print(full_out[:800])
