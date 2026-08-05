#!/usr/bin/env bash
# One-shot deploy of this project to a Hugging Face Space.
#
# Prereqs:
#   1. Create an empty Space at https://huggingface.co/new-space
#      (SDK: Streamlit, hardware: CPU basic - free)
#   2. Get a HF access token (write scope) from
#      https://huggingface.co/settings/tokens
#
# Usage:
#   HF_TOKEN=hf_xxx HF_USERNAME=yourname HF_SPACE=arabic-rag-optimizer \
#     bash scripts/deploy_hf.sh
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username}"
: "${HF_SPACE:?Set HF_SPACE to your Space slug, e.g. arabic-rag-optimizer}"

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE}"
REMOTE_URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE}"

TMPDIR=$(mktemp -d)
echo "Staging deploy in $TMPDIR"

# Copy only what the Space needs (skip venv, results, data, git metadata).
rsync -a --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'results' --exclude 'data' --exclude '.env' \
  ./ "$TMPDIR/"

# The Space's README needs the YAML frontmatter — swap it in.
cp SPACE_README.md "$TMPDIR/README.md"

cd "$TMPDIR"
git init -q
git checkout -q -b main
git add .
git commit -q -m "Deploy Arabic RAG Optimizer"
git remote add space "$REMOTE_URL"
git push -f space main

echo ""
echo "Deployed. Now set your backend secrets in the Space UI:"
echo "  $SPACE_URL -> Settings -> Variables and secrets"
echo "    VLLM_API_URL, VLLM_MODEL_NAME, VLLM_API_KEY"
echo ""
echo "Space: $SPACE_URL"
