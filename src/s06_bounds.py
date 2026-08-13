"""Stage 6: partial-identification bounds on citation-integrity outcomes
under lexical-proxy misclassification (Manski 2003; matrix-method
correction swept over a sensitivity/specificity grid).

Design: draw a seeded random sample of 2023-2026 works with abstracts,
score each abstract with the excess-vocabulary marker list, and set the
proxy S = 1 when the paper carries >= LEXICAL_THRESHOLD distinct markers.
For each sampled paper compute the zombie-citation outcome (cheap, via
OpenAlex referenced_works); the unresolvable-reference outcome runs on a
random subsample because Crossref lookups dominate the runtime.

Identification: with sensitivity s = P(S=1|A=1) and specificity
c = P(S=0|A=0) known, and misclassification nondifferential in Y,
    P(Y,S=1) = a*s*q + b*(1-c)*(1-q)
    P(Y,S=0) = a*(1-s)*q + b*c*(1-q)
where a = P(Y|A=1), b = P(Y|A=0), q = P(A=1) is identified from
P(S=1) = s*q + (1-c)*(1-q). At a single (s,c) the system point-identifies
(a, b); we do not trust any single (s, c), so the reported interval is the
range of solutions over the whole SENS_GRID x SPEC_GRID, intersected with
[0, 1]. The naive point estimate P(Y|S=1) is reported alongside so the
attenuation factor is visible.

Outputs: out/lexical_sample.csv, out/bounds.json
"""
import json
import sys

import numpy as np
import pandas as pd

import config
import oa
from s05_outcomes import load_retracted_map, zombie_count, ref_unresolvable


def harvest_sample():
    ckpt = oa.Checkpoint("s06_sample")
    for year in (2023, 2024, 2025, 2026):
        key = str(year)
        if ckpt.is_done(key):
            continue
        rows = []
        for rec in oa.oa_page_all(
                "/works",
                f"publication_year:{year},has_abstract:true,type:article",
                "id,doi,publication_date,abstract_inverted_index,"
                "referenced_works_count",
                extra={"sample": config.SAMPLE_PER_YEAR,
                       "seed": config.SAMPLE_SEED},
                max_records=config.SAMPLE_PER_YEAR):
            rows.append({
                "openalex_id": oa.short_id(rec["id"]),
                "doi": oa.norm_doi(rec.get("doi")),
                "date": rec.get("publication_date"),
                "n_refs": rec.get("referenced_works_count", 0),
                "abstract": oa.invert_abstract(
                    rec.get("abstract_inverted_index"))[:3000],
            })
        ckpt.write(rows, key=key)
        print(f"  sampled {len(rows)} works from {year}")
    return pd.DataFrame(ckpt.load_results())


def lexical_score(abstract):
    words = set(str(abstract).lower().split())
    return sum(1 for m in config.LEXICAL_MARKERS if m in words)


def zombie_outcomes(df, retracted_map):
    ckpt = oa.Checkpoint("s06_zombie")
    todo = df[df.n_refs > 0]
    for i, row in enumerate(todo.itertuples()):
        if ckpt.is_done(row.openalex_id):
            continue
        rec = oa.oa_get(f"/works/{row.openalex_id}",
                        select="id,publication_date,referenced_works")
        refs = [oa.short_id(r) for r in (rec or {}).get(
            "referenced_works", [])]
        pub = pd.to_datetime((rec or {}).get("publication_date"),
                             errors="coerce")
        ckpt.write([{
            "openalex_id": row.openalex_id,
            "n_refs_resolved": len(refs),
            "zombie_count": zombie_count(refs, pub, retracted_map),
        }], key=row.openalex_id)
        if i % 200 == 0:
            print(f"  zombie outcomes: {i}/{len(todo)}")
    return pd.DataFrame(ckpt.load_results())


def unresolvable_outcomes(df):
    sub = df[df.doi.notna()].sample(
        min(config.UNRESOLVABLE_SUBSAMPLE, int(df.doi.notna().sum())),
        random_state=config.SAMPLE_SEED)
    ckpt = oa.Checkpoint("s06_unresolvable")
    for i, row in enumerate(sub.itertuples()):
        if ckpt.is_done(row.openalex_id):
            continue
        cr = oa.cr_get(f"/works/{row.doi}")
        cr_refs = (cr or {}).get("message", {}).get("reference") or []
        out = []
        if cr_refs:
            out = [{
                "openalex_id": row.openalex_id,
                "n_refs_crossref": len(cr_refs),
                "n_unresolvable": sum(
                    1 for ref in cr_refs if ref_unresolvable(ref)),
            }]
        ckpt.write(out, key=row.openalex_id)
        if i % 100 == 0:
            print(f"  unresolvable outcomes: {i}/{len(sub)}")
    return pd.DataFrame(ckpt.load_results())


def corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec):
    """Solve the 2x2 misclassification system at one (sens, spec).
    Returns (a, b) = (P(Y|A=1), P(Y|A=0)) or None if infeasible."""
    q = (p_s1 - (1 - spec)) / (sens - (1 - spec))
    if not 0 < q < 1:
        return None
    M = np.array([
        [sens * q, (1 - spec) * (1 - q)],
        [(1 - sens) * q, spec * (1 - q)],
    ])
    rhs = np.array([p_y_s1 * p_s1, p_y_s0 * (1 - p_s1)])
    try:
        a, b = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    if not (-0.05 <= a <= 1.05 and -0.05 <= b <= 1.05):
        return None
    return float(np.clip(a, 0, 1)), float(np.clip(b, 0, 1))


def bound_outcome(scored, count_col, denom_col):
    """Per-paper binary outcome: has at least one event. Bounds on
    P(outcome | AI-assisted) over the (sens, spec) grid."""
    d = scored.dropna(subset=[count_col, denom_col])
    d = d[d[denom_col] > 0]
    y = (d[count_col] > 0).astype(int)
    s = d["proxy"].astype(int)
    p_s1 = s.mean()
    p_y_s1 = y[s == 1].mean() if (s == 1).any() else np.nan
    p_y_s0 = y[s == 0].mean() if (s == 0).any() else np.nan
    solutions = []
    for sens in config.SENS_GRID:
        for spec in config.SPEC_GRID:
            sol = corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec)
            if sol:
                solutions.append(sol)
    a_vals = [sol[0] for sol in solutions]
    b_vals = [sol[1] for sol in solutions]
    return {
        "n": int(len(d)),
        "p_proxy_positive": float(p_s1),
        "naive_rate_proxy_pos": float(p_y_s1),
        "naive_rate_proxy_neg": float(p_y_s0),
        "bound_ai_lo": min(a_vals) if a_vals else None,
        "bound_ai_hi": max(a_vals) if a_vals else None,
        "bound_nonai_lo": min(b_vals) if b_vals else None,
        "bound_nonai_hi": max(b_vals) if b_vals else None,
        "n_feasible_grid_points": len(solutions),
        "attenuation_factor_max": (max(a_vals) / p_y_s1
                                   if a_vals and p_y_s1 else None),
    }


def main():
    df = harvest_sample()
    df = df.drop_duplicates(subset="openalex_id")
    df["score"] = df.abstract.map(lexical_score)
    df["proxy"] = df.score >= config.LEXICAL_THRESHOLD
    print(f"sample {len(df)}, proxy-positive share {df.proxy.mean():.3f}")

    retracted_map = load_retracted_map()
    zdf = zombie_outcomes(df, retracted_map)
    udf = unresolvable_outcomes(df)
    scored = df.merge(zdf, on="openalex_id", how="left").merge(
        udf, on="openalex_id", how="left")
    scored.drop(columns=["abstract"]).to_csv(
        config.OUT / "lexical_sample.csv", index=False)

    results = {
        "zombie": bound_outcome(scored, "zombie_count", "n_refs_resolved"),
        "unresolvable": bound_outcome(scored, "n_unresolvable",
                                      "n_refs_crossref"),
        "sens_grid": config.SENS_GRID,
        "spec_grid": config.SPEC_GRID,
        "lexical_threshold": config.LEXICAL_THRESHOLD,
    }
    with open(config.OUT / "bounds.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
