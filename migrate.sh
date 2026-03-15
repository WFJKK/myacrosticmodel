#!/bin/bash
# =============================================================================
# Migrate myacrosticmodel -> stego-internalization
# =============================================================================
# Run this ONCE inside the repo to restructure.
# After running, rename the repo on GitHub.
#
# Usage:
#   cd myacrosticmodel
#   bash migrate.sh
# =============================================================================

set -e

echo "=== Restructuring repo ==="

# Create new directory structure
mkdir -p scripts
mkdir -p data/acrostics/stage1
mkdir -p data/acrostics/v0
mkdir -p data/acrostics/v1a
mkdir -p data/acrostics/v1b
mkdir -p data/acrostics/v2
mkdir -p results/acrostics/qwen-7b
mkdir -p results/acrostics/qwen-14b
mkdir -p adapters/acrostics
mkdir -p analysis

# -- Move scripts --
echo "Moving scripts..."
git mv train_acrostic.py scripts/
git mv generate_v1_data.py scripts/
git mv generate_v2_data.py scripts/
git mv v0-data/generate_v0_poems.py scripts/generate_v0_data.py

# -- Move data --
echo "Moving data..."
# Stage 1
git mv encoder_train.jsonl data/acrostics/stage1/train.jsonl
git mv encoder_val.jsonl data/acrostics/stage1/val.jsonl

# V0
git mv v0-data/v0_train.jsonl data/acrostics/v0/train.jsonl
git mv v0-data/v0_test.jsonl data/acrostics/v0/test.jsonl

# V1a
git mv v1a-data/v1a_train.jsonl data/acrostics/v1a/train.jsonl
git mv v1a-data/v1a_test.jsonl data/acrostics/v1a/test.jsonl

# V1b (files are named v1e_ internally)
git mv v1b-data/v1e_train.jsonl data/acrostics/v1b/train.jsonl
git mv v1b-data/v1e_test.jsonl data/acrostics/v1b/test.jsonl

# V2
git mv v2-data/v2_train.jsonl data/acrostics/v2/train.jsonl
git mv v2-data/v2_test.jsonl data/acrostics/v2/test.jsonl

# -- Move results --
echo "Moving results..."
git mv v0-data/v0_results.json results/acrostics/qwen-7b/v0_results.json
git mv v2-data/v2_results.json results/acrostics/qwen-7b/v2_results.json

# -- Move adapters --
echo "Moving adapters..."
git mv acrostic-lora adapters/acrostics/qwen-7b-stage1
git mv v0-data/v0-lora adapters/acrostics/qwen-7b-v0

# -- Clean up empty directories --
echo "Cleaning up..."
rmdir v0-data 2>/dev/null || rm -rf v0-data
rmdir v1a-data 2>/dev/null || rm -rf v1a-data
rmdir v1b-data 2>/dev/null || rm -rf v1b-data
# v2-data may still have generate_v2_data.py leftover dir
rm -rf v2-data 2>/dev/null || true

# -- Move run scripts --
echo "Moving run scripts..."
# These were created by Claude, move if they exist
[ -f run_14b_experiments.sh ] && git mv run_14b_experiments.sh scripts/
[ -f setup_vastai.sh ] && git mv setup_vastai.sh scripts/

# -- Replace README --
echo "Replacing README..."
git rm README.md
git mv README_new.md README.md

echo ""
echo "=== Migration complete ==="
echo ""
echo "New structure:"
find . -not -path './.git/*' -not -path './.git' -not -name '*.safetensors' -not -name 'tokenizer.json' | head -60
echo ""
echo "Next steps:"
echo "  1. Review with: git status"
echo "  2. Commit: git commit -m 'Restructure repo for multi-scheme multi-model experiments'"
echo "  3. Rename repo on GitHub: Settings -> General -> Repository name -> stego-internalization"
echo "  4. Update local remote: git remote set-url origin https://github.com/WFJKK/stego-internalization.git"
