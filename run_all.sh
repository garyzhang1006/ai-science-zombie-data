#!/usr/bin/env bash
# Full pipeline, in order. Every stage is resumable; rerunning skips
# checkpointed work. Requires OA_MAILTO to be set.
set -euo pipefail

if [ -z "${OA_MAILTO:-}" ]; then
    echo "set OA_MAILTO=you@example.org first (polite-pool contact)" >&2
    exit 1
fi

cd "$(dirname "$0")/src"
for stage in s00_retraction_watch s01_zombie_harvest s02_population_series \
             s03_artifact_cohort s04_match_controls s05_outcomes \
             s06_bounds s07_figures s08_fill_tk; do
    echo "=== $stage ==="
    python "$stage.py"
done
echo "done: see out/tk_values.json"
