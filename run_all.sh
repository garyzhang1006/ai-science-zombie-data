#!/usr/bin/env bash
# Full pipeline in order. Every stage checkpoints, so rerunning skips
# finished work. Set SHARD/N_SHARDS to split the snapshot scan across
# sessions or machines (default: one session does all of it).
set -euo pipefail

if [ -z "${OA_MAILTO:-}" ]; then
    echo "set OA_MAILTO=you@example.org first (polite-pool contact)" >&2
    exit 1
fi

SHARD="${SHARD:-0}"
N_SHARDS="${N_SHARDS:-1}"

cd "$(dirname "$0")/src"
python s00_retractions.py
python s01_artifact_harvest.py || echo "stage 1 incomplete (API budget); rerun after reset"
python s02_doi_map.py --shard "$SHARD" --n-shards "$N_SHARDS"
python s03_scan.py    --shard "$SHARD" --n-shards "$N_SHARDS"
python s04_stats.py
python s05_crossref.py
python s06_figures.py
python s07_fill_tk.py
echo "done: see out/tk_report.md"
