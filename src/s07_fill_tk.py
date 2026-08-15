"""Stage 7: collect every number the paper's [TK] slots need.

Run last. Reports what is missing rather than inventing it, so a partial
pipeline produces a partial but honest list.

Outputs: out/tk_values.json, out/tk_report.md
"""
import json
import sys

import pandas as pd

import config
import snapshot

MISSING = "[NOT YET COMPUTED]"


def read_json(name):
    p = config.OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(name, subdir="out"):
    p = (config.OUT if subdir == "out" else config.DATA) / name
    return pd.read_csv(p) if p.exists() else None


def main():
    series = read_csv("population_series.csv")
    its = read_json("its_results.json") or {}
    t1 = read_json("table1.json") or {}
    bounds = read_json("bounds.json") or {}
    cands = read_csv("artifact_candidates.csv")
    cohort = read_csv("artifact_cohort.csv")
    matches = read_csv("matches.csv")
    retr = read_csv("retracted_dois.csv", "data")
    works = [pd.read_csv(p) for p in config.DATA.glob("retracted_works_*.csv")]
    retr_mapped = (pd.concat(works).drop_duplicates(subset="doi")
                   if works else None)

    pooled = its.get("pooled") or {}
    zom = t1.get("zombie") or {}
    unres = t1.get("unresolvable") or {}
    bz = bounds.get("zombie") or {}

    tk = {
        "data.snapshot_date": snapshot.snapshot_date(),
        "data.retraction_count": len(retr) if retr is not None else MISSING,
        "data.retracted_matched_openalex": (len(retr_mapped)
                                            if retr_mapped is not None
                                            else MISSING),
        "data.buffer_months_k": config.BUFFER_MONTHS,
        "abstract.N_citing_works": (int(series.works.sum())
                                    if series is not None else MISSING),
        "abstract.n_artifact_papers": (len(cohort) if cohort is not None
                                       else MISSING),
        "abstract.zombie_ratio": zom.get("ratio", MISSING),
        "abstract.unresolvable_ratio": unres.get("ratio", MISSING),
        "s3_2.raw_counts_by_phrase": (
            cands.groupby("phrase").size().to_dict()
            if cands is not None else MISSING),
        "s3_2.n_after_filter": len(cohort) if cohort is not None else MISSING,
        "s3_2.filter_drop_reasons": (cands.reason.value_counts().to_dict()
                                     if cands is not None else MISSING),
        "s3_2.n_controls_per_artifact": config.N_CONTROLS,
        "s3_2.n_matched_pairs": (len(matches) if matches is not None
                                 else MISSING),
        "s4_1.pre_mean_rate_per_10k": pooled.get("pre_mean_rate", MISSING),
        "s4_1.level_change": pooled.get("level_change", MISSING),
        "s4_1.level_change_se": pooled.get("level_change_se", MISSING),
        "s4_1.level_change_p": pooled.get("level_change_p", MISSING),
        "s4_1.slope_change": pooled.get("slope_change", MISSING),
        "s4_1.placebo_quantile": its.get(
            "actual_abs_exceeds_placebo_share", MISSING),
        "s4_1.n_placebo_breaks": len(its.get("placebos", [])) or MISSING,
        "s4_2.zombie_artifact_per_1000": zom.get("artifact_rate_per_1000",
                                                 MISSING),
        "s4_2.zombie_control_per_1000": zom.get("control_rate_per_1000",
                                                MISSING),
        "s4_2.zombie_ratio_ci": [zom.get("ratio_ci_lo", MISSING),
                                 zom.get("ratio_ci_hi", MISSING)],
        "s4_2.unresolvable_artifact_per_1000": unres.get(
            "artifact_rate_per_1000", MISSING),
        "s4_2.unresolvable_control_per_1000": unres.get(
            "control_rate_per_1000", MISSING),
        "s4_3.bound_ai": [bz.get("bound_ai_lo", MISSING),
                          bz.get("bound_ai_hi", MISSING)],
        "s4_3.bound_nonai": [bz.get("bound_nonai_lo", MISSING),
                             bz.get("bound_nonai_hi", MISSING)],
        "s4_3.naive_estimate": bz.get("naive_rate_proxy_pos", MISSING),
        "s4_3.attenuation_factor": bz.get("attenuation_factor_max", MISSING),
        "s4_3.lexical_sample_n": bz.get("n", MISSING),
        "config.sens_grid": config.SENS_GRID,
        "config.spec_grid": config.SPEC_GRID,
        "config.seed": config.SAMPLE_SEED,
        "repo.url": "https://github.com/garyzhang1006/ai-science-zombie-data",
    }
    (config.OUT / "tk_values.json").write_text(
        json.dumps(tk, indent=2, default=str))

    missing = [k for k, v in tk.items()
               if v == MISSING or (isinstance(v, list) and MISSING in v)]
    lines = [f"# TK values (snapshot {tk['data.snapshot_date']})", ""]
    for k, v in tk.items():
        lines.append(f"- `{k}`: {json.dumps(v, default=str)[:300]}")
    lines += ["", f"## Still missing: {len(missing)}"]
    lines += [f"- {k}" for k in missing]
    (config.OUT / "tk_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote tk_values.json ({len(tk)} entries, {len(missing)} missing)")
    for k in missing:
        print(f"  MISSING {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
