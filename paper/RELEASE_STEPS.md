# Data/Code Release Steps (Do Before Marking Data Availability as Past Tense)

This is the last true blocker for submission. I cannot do this myself — it
needs your GitHub/Zenodo login, which isn't available in this environment.
Two options below; Option A is the more standard, citable form (DOI tied to
the actual git commit); Option B is faster if you'd rather skip the GitHub
release step entirely.

## Option A: GitHub release → Zenodo (recommended, ~15 min)

### 1. Rotate your NVIDIA NIM API key first

Your key was pasted in plaintext earlier in this project's chat history. It is
NOT in the git repo (`.env` is gitignored and was never committed), but rotate
it before making the repo more visible via a tagged release, as a precaution.
Do this at https://build.nvidia.com under your API keys.

### 2. Push the local commit

From your own machine, in the repo folder:

```bash
cd vllm-arabic-rag
git log --oneline -3   # confirm you see: 46037b6 Round 5: expand ARCD re-test...
git push origin main
```

This uploads the commit I already made locally: the n=140 ARCD data, the
end-to-end retrieval check data, all analysis scripts/results, and the
rebuilt paper. Nothing else needs to be staged — `git status` should show a
clean tree before and after.

### 3. Enable Zenodo for this repository (do this BEFORE creating the release)

1. Go to https://zenodo.org and log in with your GitHub account.
2. Go to your GitHub-linked repositories (Settings → GitHub, or zenodo.org/account/settings/github/).
3. Find `Akramtaha98/vllm-arabic-rag` in the list and toggle it **ON**.

Zenodo archives a repo automatically the next time you cut a *new* GitHub
release — it does not retroactively archive past commits. Enable it first,
then create the release in step 4.

### 4. Create a tagged GitHub release

Via the GitHub web UI: your repo → "Releases" → "Draft a new release".

- Tag: `v1.0-submission` (or similar)
- Target: `main`
- Title: "Applied Intelligence submission: LSPM v1.0"
- Description (suggested):
  > Complete implementation, raw experimental data (tokenization corpus,
  > fidelity/baseline pilots, 140-question ARCD ground-truth re-test with
  > Holm-Bonferroni correction and TOST equivalence testing, end-to-end
  > retrieval check), and analysis code accompanying the manuscript
  > "Semantic-Driven Context Pruning for Arabic RAG Systems: Toward
  > Memory-Efficient vLLM-Based Deployment."
- Click "Publish release".

Or via the `gh` CLI if you have it installed:

```bash
gh release create v1.0-submission --title "Applied Intelligence submission: LSPM v1.0" \
  --notes "Complete implementation, raw experimental data, and analysis code accompanying the manuscript."
```

### 5. Verify the Zenodo DOI

Within a few minutes of publishing the release, Zenodo mints a DOI
automatically. Check:

- https://zenodo.org/account/settings/github/ should show the repo with a
  DOI badge next to it.
- Click through to the Zenodo record page, confirm it lists the right files
  (data/, results/, scripts/, paper/), and that the archived snapshot
  actually opens/downloads correctly.
- Copy the DOI (format: `10.5281/zenodo.XXXXXXX`).

## Option B: Direct Zenodo upload (faster, ~5 min, skips GitHub release)

I built `vllm-arabic-rag-v1.0-submission.zip` (4.5 MB, 111 files — the
complete source code plus every raw data/results file the Data Availability
statement promises: tokenization corpus, fidelity/baseline pilot data, the
980-row n=140 ARCD dataset and Holm-Bonferroni/TOST outputs, the end-to-end
retrieval check data, and all figures). It's already checked for secrets
(no `.env`, no API keys — only the placeholder `nvapi-xxxx...` in the
example file). It was delivered to you alongside this file.

1. Rotate your NVIDIA NIM key first (same reasoning as Option A, step 1).
2. Go to https://zenodo.org/deposit/new, log in.
3. Drag in `vllm-arabic-rag-v1.0-submission.zip`.
4. Fill in the metadata form:
   - **Title:** Semantic-Driven Context Pruning for Arabic RAG Systems: Toward Memory-Efficient vLLM-Based Deployment — Code and Data
   - **Authors:** Taha, Akram (University of Technology - Iraq; Universiti Kebangsaan Malaysia)
   - **Description:** Complete implementation, raw experimental data, and analysis code accompanying the manuscript submitted to Applied Intelligence. Includes the tokenization-disparity corpus, fidelity/baseline pilot data, the 140-question ARCD ground-truth re-test (980 generations, Holm-Bonferroni correction, TOST equivalence testing), and the end-to-end retrieval check (real TF-IDF retrieval, no recall guarantee).
   - **Upload type:** Software (or Dataset)
   - **License:** match your repo's `LICENSE` file
   - **Keywords:** retrieval-augmented generation, Arabic NLP, prompt compression, vLLM
5. Click "Publish". Zenodo mints the DOI immediately.
6. Separately, still push the git commit (Option A, step 2) so the public GitHub repo matches what the paper links to — the zip is a snapshot, not a substitute for the live repo.

## Either way: tell me the DOI once you have it

The Data Availability statement (and the cover letter, which I just made
consistent with it) both currently say the release "will be made publicly
available... before final submission." Once you have a real, verified DOI,
that needs exactly one change — from that future-tense wording to a
past-tense statement naming the actual release and DOI. Suggested
replacement text:

> **Data availability.** The complete source code (LSPM middleware,
> evaluation scripts, statistical analysis scripts, benchmarking harness, and
> demonstration interface) and all raw experimental data (the tokenization
> corpus, fidelity/baseline pilot data, the 140-question ARCD ground-truth
> re-test with Holm-Bonferroni and TOST outputs, and the end-to-end retrieval
> check) are publicly available at
> https://github.com/Akramtaha98/vllm-arabic-rag, tagged release
> `v1.0-submission`, archived with DOI
> [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).

Send me the real DOI and I'll drop it into both the manuscript and the cover
letter, rebuild the PDF, and that closes the last blocker. I won't change
either document to past tense before that — it isn't true yet, and this
project's whole discipline has been not saying things are done until they
verifiably are.
