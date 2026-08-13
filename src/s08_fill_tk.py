"""Stage 8: collect every number the paper's [TK] slots need into one JSON
plus a human-readable report. Run last.

Output: out/tk_values.json, out/tk_report.md
"""
import json
import sys

import pandas as pd

import config
import oa


def main():
    retracted = pd.read_csv(config.DATA / "retracted.csv")
    cites = oa.Checkpoint("s01_citations").load_results()
    its = json.load(open(config.OUT / "its_results.json"))
    cands = pd.read_csv(config.OUT / "artifact_candidates.csv")
    cohort = pd.read_csv(config.OUT / "artifact_cohort.csv")
    t1 = json.load(open(config.OUT / "table1.json"))
    bounds = json.load(open(config.OUT / "bounds.json"))

    citing_ids = {r["citing_id"] for r in cites}
    tk = {
        "abstract.N_citing_papers": len(citing_ids),
        "abstract.n_artifact_papers": len(cohort),
        "abstract.zombie_ratio": t1["zombie"]["ratio"],
        "abstract.unresolvable_ratio": t1["unresolvable"]["ratio"],
        "s3.retraction_count": len(retracted),
        "s3.buffer_months_k": config.BUFFER_MONTHS,
        "s3.openalex_snapshot_date": pd.Timestamp.now().date().isoformat(),
        "s3_2.raw_counts_by_phrase": (
            cands.groupby("phrase").size().to_dict()),
        "s3_2.n_after_filter": len(cohort),
        "s3_2.n_controls_per_artifact": config.N_CONTROLS,
        "s3_2.match_tolerance": config.MATCH_TOL,
        "s3_3.fuzzy_threshold": config.FUZZY_THRESHOLD,
        "s4_1.pre_period_mean_rate_per_10k": its["pooled"]["pre_mean_rate"],
        "s4_1.level_change": its["pooled"]["level_change"],
        "s4_1.level_change_p": its["pooled"]["level_change_p"],
        "s4_1.slope_change": its["pooled"]["slope_change"],
        "s4_1.placebo_quantile": its["actual_abs_exceeds_placebo_share"],
        "s4_2.table1": t1,
        "s4_3.bounds": bounds,
        "config.sens_grid": config.SENS_GRID,
        "config.spec_grid": config.SPEC_GRID,
        "config.sample_seed": config.SAMPLE_SEED,
    }
    with open(config.OUT / "tk_values.json", "w") as f:
        json.dump(tk, f, indent=2, default=str)

    lines = ["# TK values for the paper", ""]
    for k, v in tk.items():
        lines.append(f"- `{k}`: {json.dumps(v, default=str)[:400]}")
    (config.OUT / "tk_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {config.OUT / 'tk_values.json'} with {len(tk)} entries")


if __name__ == "__main__":
    sys.exit(main())
