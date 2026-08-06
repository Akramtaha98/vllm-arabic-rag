# Data/Code Release Steps (Do Before Marking Data Availability as Past Tense)

This is the last true blocker for submission. I cannot do this myself (no push
credentials in this environment, and you chose to keep the push under your own
account). Everything is staged and ready — this should take about 15 minutes.

## 1. Rotate your NVIDIA NIM API key first

Your key was pasted in plaintext earlier in this project's chat history. It is
NOT in the git repo (`.env` is gitignored and was never committed), but rotate
it before making the repo more visible via a tagged release, as a precaution.
Do this at https://build.nvidia.com under your API keys.

## 2. Push the local commit

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

## 3. Enable Zenodo for this repository (do this BEFORE creating the release)

1. Go to https://zenodo.org and log in with your GitHub account.
2. Go to your GitHub-linked repositories (Settings → GitHub, or zenodo.org/account/settings/github/).
3. Find `Akramtaha98/vllm-arabic-rag` in the list and toggle it **ON**.

Zenodo archives a repo automatically the next time you cut a *new* GitHub
release — it does not retroactively archive past commits. Enable it first,
then create the release in step 4.

## 4. Create a tagged GitHub release

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

## 5. Verify the Zenodo DOI

Within a few minutes of publishing the release, Zenodo mints a DOI
automatically. Check:

- https://zenodo.org/account/settings/github/ should show the repo with a
  DOI badge next to it.
- Click through to the Zenodo record page, confirm it lists the right files
  (data/, results/, scripts/, paper/), and that the archived snapshot
  actually opens/downloads correctly.
- Copy the DOI (format: `10.5281/zenodo.XXXXXXX`).

## 6. Tell me the DOI (or edit it yourself)

Once you have a working DOI, the Data Availability statement needs exactly
one change — from "will be added... before final submission" to a past-tense
statement naming the actual release and DOI. Suggested replacement text:

> **Data availability.** The complete source code (LSPM middleware,
> evaluation scripts, statistical analysis scripts, benchmarking harness, and
> demonstration interface) and all raw experimental data (the tokenization
> corpus, fidelity/baseline pilot data, the 140-question ARCD ground-truth
> re-test with Holm-Bonferroni and TOST outputs, and the end-to-end retrieval
> check) are publicly available at
> https://github.com/Akramtaha98/vllm-arabic-rag, tagged release
> `v1.0-submission`, archived with DOI
> [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).

Send me the real DOI and I'll drop it in, rebuild the PDF, and that closes
the last blocker.
