"""Summarize /tmp/arcd_prompt_tokens.jsonl into the figures quoted in Section 4.3."""
import json
from pathlib import Path

import numpy as np
from scipy import stats as sstats

rows = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "results" / "arcd_prompt_tokens.jsonl", encoding="utf-8") if l.strip()]
print("cells:", len(rows), "verified:", sum(r["verified_against_live_record"] for r in rows))
T = {(r["method"], r["ratio"], r["id"]): r["prompt_tokens"] for r in rows}
ids = sorted({r["id"] for r in rows})

raw = np.array([T[("raw", 1.0, i)] for i in ids], float)
print(f"raw prompt tokens: mean {raw.mean():.1f}  median {np.median(raw):.1f}  "
      f"min {raw.min():.0f} max {raw.max():.0f}")

rng = np.random.default_rng(42)


def boot_ci(d, n=10000):
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(m, [2.5, 97.5])


KIB_PER_TOKEN = 2 * 32 * 8 * 128 * 2 / 1024  # bytes/1024 = 128 KiB
for ratio in (0.3, 0.5, 0.7):
    l = np.array([T[("lspm", ratio, i)] for i in ids], float)
    n = np.array([T[("naive", ratio, i)] for i in ids], float)
    d = l - n
    lo, hi = boot_ci(d)
    p = sstats.wilcoxon(d).pvalue
    print(f"r={ratio}: lspm mean {l.mean():.1f}  naive mean {n.mean():.1f}  "
          f"diff {d.mean():+.1f} (95% CI {lo:+.1f} to {hi:+.1f}) wilcoxon p={p:.6f}  "
          f"lspm shorter in {(d < 0).sum()}/{len(d)} questions")
    red = raw - l
    rlo, rhi = boot_ci(red)
    print(f"      LSPM token reduction vs raw: {red.mean():.1f} tokens -> "
          f"{red.mean()*KIB_PER_TOKEN/1024:.1f} MiB (95% CI "
          f"{rlo*KIB_PER_TOKEN/1024:.1f}-{rhi*KIB_PER_TOKEN/1024:.1f} MiB)")
