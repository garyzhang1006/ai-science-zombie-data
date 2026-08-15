"""Stage 4: merge the shard aggregates and compute every estimate in the
paper.

Three prongs, three sections:

Population. Monthly zombie-citation rate per 10,000 references, within
field, fitted with an interrupted time series that allows a level and slope
change at the ChatGPT release. Placebo breaks through 2019-2021 calibrate
the procedure's own false-positive rate, so the headline break is read
against a distribution rather than against zero.

Cohort. Artifact papers matched to controls from the same venue and year,
nearest in reference-list length and citation count, with cluster-bootstrap
intervals on the rate ratio (papers are the resampled unit).

Bounds. The lexical proxy for AI assistance is noisy, and a naive
regression of outcomes on it is attenuated toward zero, which would make a
null uninformative. The misclassification correction is solved over a grid
of sensitivity and specificity values rather than at any single point, so
the reported quantity is an interval that holds for every assumption in the
grid.

Outputs: out/population_series.csv, out/its_results.json, out/matches.csv,
         out/table1.json, out/bounds.json
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


# --------------------------------------------------------------- population

def month_range():
    months, (y, m) = [], (2018, 1)
    end_y, end_m = map(int, config.END_MONTH.split("-"))
    while (y, m) <= (end_y, end_m):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def merge_shards():
    num = [pd.read_csv(p) for p in config.DATA.glob("s03_numer_*.csv")]
    den = [pd.read_csv(p) for p in config.DATA.glob("s03_denom_*.csv")]
    if not den:
        raise SystemExit("run stage 3 first (no s03_denom_*.csv)")
    num = (pd.concat(num).groupby(["month", "field"], as_index=False)
           .zombie_events.sum() if num else
           pd.DataFrame(columns=["month", "field", "zombie_events"]))
    den = (pd.concat(den).groupby(["month", "field"], as_index=False)
           .agg(works=("works", "sum"), refs=("refs", "sum")))
    series = den.merge(num, on=["month", "field"], how="left")
    series["zombie_events"] = series.zombie_events.fillna(0).astype(int)
    series["rate_per_10k"] = np.where(
        series.refs > 0, series.zombie_events / series.refs * 1e4, np.nan)
    return series.sort_values(["month", "field"])


def fit_its(series, break_month, months=None):
    """Level and slope change at break_month, Newey-West SEs for the serial
    correlation a monthly rate series always carries."""
    months = months or month_range()
    if break_month not in months:
        raise ValueError(f"{break_month} outside series window")
    t0 = months.index(break_month)
    pooled = (series.groupby("month")
              .agg(zombie=("zombie_events", "sum"), refs=("refs", "sum"))
              .reindex(months))
    y = (pooled.zombie / pooled.refs * 1e4).values
    keep = ~np.isnan(y)
    if keep.sum() < 24:
        return None
    t = np.arange(len(months), dtype=float)
    post = (t >= t0).astype(float)
    X = sm.add_constant(np.column_stack([t, post, (t - t0) * post]))
    fit = sm.OLS(y[keep], X[keep]).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": 6})
    pre = y[:t0][~np.isnan(y[:t0])]
    return {
        "break_month": break_month,
        "level_change": float(fit.params[2]),
        "level_change_se": float(fit.bse[2]),
        "level_change_p": float(fit.pvalues[2]),
        "slope_change": float(fit.params[3]),
        "slope_change_se": float(fit.bse[3]),
        "slope_change_p": float(fit.pvalues[3]),
        "pre_mean_rate": float(pre.mean()) if len(pre) else None,
        "n_months": int(keep.sum()),
    }


def placebo_distribution(series):
    """Refit on pre-break data only, moving the break through 2019-2021.
    Anything the real break does that these also do is not evidence."""
    months = month_range()
    cutoff = months.index(config.BREAK_MONTH)
    pre_months = months[:cutoff]
    pre = series[series.month.isin(pre_months)]
    out = []
    for pm in config.PLACEBO_MONTHS:
        if pm not in pre_months or pre_months.index(pm) < 6:
            continue
        if len(pre_months) - pre_months.index(pm) < 6:
            continue
        res = fit_its(pre, pm, months=pre_months)
        if res:
            out.append({"month": pm, "level_change": res["level_change"]})
    return out


# ------------------------------------------------------------------- cohort

def match_controls(cohort, candidates):
    """Nearest controls within venue and year on reference-list length and
    citation count, both on a log scale so the metric is relative."""
    rows = []
    cand_by_key = {k: g for k, g in candidates.groupby(["source_id", "year"])}
    for r in cohort.itertuples():
        pool = cand_by_key.get((r.source_id, r.year))
        if pool is None or not len(pool):
            continue
        lo, hi = (r.n_refs * (1 - config.MATCH_TOL),
                  r.n_refs * (1 + config.MATCH_TOL))
        band = pool[(pool.n_refs >= lo) & (pool.n_refs <= hi)]
        use = band if len(band) >= config.N_CONTROLS else pool
        d = (np.abs(np.log1p(use.n_refs) - np.log1p(r.n_refs))
             + 0.5 * np.abs(np.log1p(use.n_cites) - np.log1p(r.n_cites)))
        for c in use.assign(_d=d).nsmallest(config.N_CONTROLS,
                                            "_d").itertuples():
            rows.append({
                "artifact_id": r.openalex_id, "control_id": c.openalex_id,
                "match_basis": "venue_year" if len(band) >= config.N_CONTROLS
                               else "venue_year_relaxed",
                "control_n_refs": c.n_refs, "control_n_cites": c.n_cites,
                "control_zombie": c.zombie_count,
            })
    return pd.DataFrame(rows)


def bootstrap_ratio(art, ctl, count_col, denom_col, n_boot=10_000):
    """Cluster bootstrap over papers of the artifact/control rate ratio,
    rates per 1,000 references."""
    rng = np.random.default_rng(config.SAMPLE_SEED)
    a = art[[count_col, denom_col]].dropna().values.astype(float)
    c = ctl[[count_col, denom_col]].dropna().values.astype(float)
    if not len(a) or not len(c):
        return None

    def rate(m):
        tot = m[:, 1].sum()
        return m[:, 0].sum() / tot * 1000 if tot else np.nan

    obs_a, obs_c = rate(a), rate(c)
    ratios = []
    for _ in range(n_boot):
        ra = rate(a[rng.integers(0, len(a), len(a))])
        rc = rate(c[rng.integers(0, len(c), len(c))])
        if rc and rc > 0 and not np.isnan(ra):
            ratios.append(ra / rc)
    lo, hi = (np.percentile(ratios, [2.5, 97.5]) if ratios
              else (np.nan, np.nan))
    return {
        "artifact_rate_per_1000": float(obs_a),
        "control_rate_per_1000": float(obs_c),
        "ratio": float(obs_a / obs_c) if obs_c else None,
        "ratio_ci_lo": float(lo), "ratio_ci_hi": float(hi),
        "n_artifact": int(len(a)), "n_control": int(len(c)),
    }


# ------------------------------------------------------------------- bounds

def abstract_vocabulary(abstract):
    """Lowercased word set of an abstract.

    OpenAlex stores abstracts as an inverted index serialized to JSON, so
    the words arrive as dictionary keys wrapped in quotes and punctuation.
    Splitting that string naively finds no markers at all, which would
    silently drive the proxy-positive share to zero and leave the bounds
    unidentified. Plain text is also accepted, for tests and for any other
    corpus.
    """
    if not abstract:
        return set()
    text = str(abstract)
    words = None
    if text.lstrip().startswith("{"):
        try:
            words = json.loads(text).keys()
        except (ValueError, AttributeError):
            words = None
    if words is None:
        words = text.split()
    return {w.strip(' ",:{}[]().;').lower() for w in words}


def lexical_score(abstract):
    vocab = abstract_vocabulary(abstract)
    return sum(1 for m in config.LEXICAL_MARKERS if m in vocab)


def corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec):
    """Solve the misclassification system at one (sensitivity, specificity).

    With q = P(AI-assisted) recovered from the proxy-positive share, the
    two observed conditional rates identify the two true rates. Returns
    None where the assumption pair cannot have produced the data.
    """
    denom = sens - (1 - spec)
    if abs(denom) < 1e-9:
        return None
    q = (p_s1 - (1 - spec)) / denom
    if not 0 < q < 1:
        return None
    M = np.array([[sens * q, (1 - spec) * (1 - q)],
                  [(1 - sens) * q, spec * (1 - q)]])
    rhs = np.array([p_y_s1 * p_s1, p_y_s0 * (1 - p_s1)])
    try:
        a, b = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    if not (-0.05 <= a <= 1.05 and -0.05 <= b <= 1.05):
        return None
    return float(np.clip(a, 0, 1)), float(np.clip(b, 0, 1))


def bound_outcome(scored, count_col, denom_col):
    d = scored.dropna(subset=[count_col, denom_col])
    d = d[d[denom_col] > 0]
    if not len(d):
        return None
    y = (d[count_col] > 0).astype(int)
    s = d["proxy"].astype(int)
    p_s1 = float(s.mean())
    if not 0 < p_s1 < 1:
        return None
    p_y_s1, p_y_s0 = float(y[s == 1].mean()), float(y[s == 0].mean())
    sols = [sol for sens in config.SENS_GRID for spec in config.SPEC_GRID
            if (sol := corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec))]
    a_vals = [x[0] for x in sols]
    b_vals = [x[1] for x in sols]
    return {
        "n": int(len(d)),
        "p_proxy_positive": p_s1,
        "naive_rate_proxy_pos": p_y_s1,
        "naive_rate_proxy_neg": p_y_s0,
        "bound_ai_lo": min(a_vals) if a_vals else None,
        "bound_ai_hi": max(a_vals) if a_vals else None,
        "bound_nonai_lo": min(b_vals) if b_vals else None,
        "bound_nonai_hi": max(b_vals) if b_vals else None,
        "n_feasible_grid_points": len(sols),
        "attenuation_factor_max": (max(a_vals) / p_y_s1
                                   if a_vals and p_y_s1 else None),
    }


# --------------------------------------------------------------------- main

def read_shards(name):
    parts = [pd.read_csv(p) for p in config.DATA.glob(f"s03_{name}_*.csv")]
    return (pd.concat(parts).drop_duplicates(subset="openalex_id")
            if parts else pd.DataFrame())


def main():
    series = merge_shards()
    series.to_csv(config.OUT / "population_series.csv", index=False)
    pooled = fit_its(series, config.BREAK_MONTH)
    placebos = placebo_distribution(series)
    pl = [p["level_change"] for p in placebos]
    quantile = (float(np.mean(np.abs(pl) < abs(pooled["level_change"])))
                if pl and pooled else None)
    per_field = {}
    for f in (series.groupby("field").zombie_events.sum()
              .nlargest(8).index):
        res = fit_its(series[series.field == f], config.BREAK_MONTH)
        if res:
            per_field[f] = res
    its = {"pooled": pooled, "placebos": placebos,
           "actual_abs_exceeds_placebo_share": quantile,
           "per_field": per_field}
    (config.OUT / "its_results.json").write_text(json.dumps(its, indent=2))

    cohort_out = read_shards("cohort")
    controls_out = read_shards("control")
    table1 = {}
    if len(cohort_out) and len(controls_out):
        matches = match_controls(cohort_out, controls_out)
        matches.to_csv(config.OUT / "matches.csv", index=False)
        used = controls_out[controls_out.openalex_id.isin(
            matches.control_id)] if len(matches) else controls_out.head(0)
        table1["zombie"] = bootstrap_ratio(cohort_out, used,
                                           "zombie_count", "n_refs")
        table1["mean_refs_artifact"] = float(cohort_out.n_refs.mean())
        table1["mean_refs_control"] = float(used.n_refs.mean()) if len(used) \
            else None
        table1["n_matched_pairs"] = int(len(matches))
    unres = config.OUT / "unresolvable.csv"
    if unres.exists():
        u = pd.read_csv(unres)
        table1["unresolvable"] = bootstrap_ratio(
            u[u.group == "artifact"], u[u.group == "control"],
            "n_unresolvable", "n_refs_crossref")
    (config.OUT / "table1.json").write_text(json.dumps(table1, indent=2))

    lex = read_shards("lexical")
    bounds = {}
    if len(lex):
        lex["score"] = lex.abstract.map(lexical_score)
        lex["proxy"] = lex.score >= config.LEXICAL_THRESHOLD
        lex.drop(columns=["abstract"]).to_csv(
            config.OUT / "lexical_sample.csv", index=False)
        bounds["zombie"] = bound_outcome(lex, "zombie_count", "n_refs")
        bounds["n_sample"] = int(len(lex))
        bounds["proxy_positive_share"] = float(lex.proxy.mean())
    bounds.update({"sens_grid": config.SENS_GRID,
                   "spec_grid": config.SPEC_GRID,
                   "lexical_threshold": config.LEXICAL_THRESHOLD})
    (config.OUT / "bounds.json").write_text(json.dumps(bounds, indent=2))

    print(json.dumps({"its": pooled, "placebo_quantile": quantile,
                      "table1": table1,
                      "bounds_zombie": bounds.get("zombie")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
