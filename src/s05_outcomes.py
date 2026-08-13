"""Stage 5: compute both outcomes for every artifact paper and control:

1. zombie citations: references to works that were already retracted
   (retraction date + buffer before the citing paper's publication date),
   from OpenAlex referenced_works.
2. unresolvable references: raw Crossref reference entries whose DOI does
   not resolve in Crossref AND whose bibliographic string matches no work
   under fuzzy search. Papers without a DOI or without Crossref reference
   data are excluded from this outcome's denominator (flagged, not dropped).

Then builds the cohort comparison table with cluster-bootstrap CIs
(resampling papers, 10,000 reps).

Outputs: out/outcomes.csv, out/table1.json
"""
import json
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import config
import oa


def load_paper_sets():
    cohort = pd.read_csv(config.OUT / "artifact_cohort.csv")
    matches = pd.read_csv(config.OUT / "matches.csv")
    papers = [{"id": r.openalex_id, "group": "artifact", "doi": r.doi}
              for r in cohort.itertuples()]
    control_ids = matches.control_id.unique()
    papers += [{"id": cid, "group": "control", "doi": None}
               for cid in control_ids]
    return papers


def load_retracted_map():
    df = pd.read_csv(config.DATA / "retracted.csv")
    df["retraction_date"] = pd.to_datetime(df.retraction_date)
    return dict(zip(df.openalex_id, df.retraction_date))


def zombie_count(refs, pub_date, retracted_map):
    if pd.isna(pub_date):
        return 0
    n = 0
    for ref in refs:
        rdate = retracted_map.get(ref)
        if rdate is not None and pub_date >= rdate + pd.DateOffset(
                months=config.BUFFER_MONTHS):
            n += 1
    return n


_doi_cache = {}


def doi_resolves(doi):
    if doi in _doi_cache:
        return _doi_cache[doi]
    ok = oa.cr_get(f"/works/{doi}") is not None
    _doi_cache[doi] = ok
    return ok


def ref_unresolvable(ref):
    """Crossref reference entry -> True if it points nowhere."""
    doi = oa.norm_doi(ref.get("DOI"))
    if doi:
        return not doi_resolves(doi)
    text = (ref.get("article-title") or ref.get("unstructured") or "").strip()
    if len(text) < 15:
        return False  # too little information to judge; count as resolvable
    hit = oa.cr_get("/works", **{"query.bibliographic": text[:250], "rows": 1})
    items = (hit or {}).get("message", {}).get("items", [])
    if not items:
        return True
    cand = items[0].get("title", [""])
    cand_title = cand[0] if cand else ""
    score = fuzz.token_sort_ratio(text.lower(), cand_title.lower())
    partial = fuzz.partial_ratio(cand_title.lower(), text.lower())
    return max(score, partial) < config.FUZZY_THRESHOLD


def process_paper(paper, retracted_map):
    rec = oa.oa_get(f"/works/{paper['id']}",
                    select="id,doi,publication_date,referenced_works")
    if rec is None:
        return None
    refs = [oa.short_id(r) for r in rec.get("referenced_works", [])]
    pub_date = pd.to_datetime(rec.get("publication_date"), errors="coerce")
    row = {
        "openalex_id": paper["id"],
        "group": paper["group"],
        "n_refs_openalex": len(refs),
        "zombie_count": zombie_count(refs, pub_date, retracted_map),
        "n_refs_crossref": np.nan,
        "n_unresolvable": np.nan,
    }
    doi = paper.get("doi") or oa.norm_doi(rec.get("doi"))
    if doi:
        cr = oa.cr_get(f"/works/{doi}")
        cr_refs = (cr or {}).get("message", {}).get("reference") or []
        if cr_refs:
            row["n_refs_crossref"] = len(cr_refs)
            row["n_unresolvable"] = sum(
                1 for ref in cr_refs if ref_unresolvable(ref))
    return row


def bootstrap_ratio(df, count_col, denom_col, n_boot=10_000):
    """Cluster bootstrap (papers are the clusters) of the artifact/control
    rate ratio, rates per 1,000 references."""
    rng = np.random.default_rng(config.SAMPLE_SEED)
    art = df[df.group == "artifact"][[count_col, denom_col]].dropna().values
    ctl = df[df.group == "control"][[count_col, denom_col]].dropna().values

    def rate(mat):
        tot = mat[:, 1].sum()
        return mat[:, 0].sum() / tot * 1000 if tot else np.nan

    obs_a, obs_c = rate(art), rate(ctl)
    ratios = []
    for _ in range(n_boot):
        a = art[rng.integers(0, len(art), len(art))]
        c = ctl[rng.integers(0, len(ctl), len(ctl))]
        ra, rc = rate(a), rate(c)
        if rc and not np.isnan(rc) and rc > 0:
            ratios.append(ra / rc)
    lo, hi = (np.percentile(ratios, [2.5, 97.5])
              if ratios else (np.nan, np.nan))
    return {
        "artifact_rate_per_1000": obs_a,
        "control_rate_per_1000": obs_c,
        "ratio": obs_a / obs_c if obs_c else np.nan,
        "ratio_ci_lo": lo, "ratio_ci_hi": hi,
        "n_artifact": int(len(art)), "n_control": int(len(ctl)),
    }


def main():
    papers = load_paper_sets()
    retracted_map = load_retracted_map()
    ckpt = oa.Checkpoint("s05_outcomes")
    for i, paper in enumerate(papers):
        if ckpt.is_done(paper["id"]):
            continue
        row = process_paper(paper, retracted_map)
        ckpt.write([row] if row else [], key=paper["id"])
        if i % 50 == 0:
            print(f"  outcomes: {i}/{len(papers)}")
    df = pd.DataFrame(ckpt.load_results())
    df.to_csv(config.OUT / "outcomes.csv", index=False)

    table1 = {
        "zombie": bootstrap_ratio(df, "zombie_count", "n_refs_openalex"),
        "unresolvable": bootstrap_ratio(df, "n_unresolvable",
                                        "n_refs_crossref"),
        "mean_refs_artifact": float(
            df[df.group == "artifact"].n_refs_openalex.mean()),
        "mean_refs_control": float(
            df[df.group == "control"].n_refs_openalex.mean()),
    }
    with open(config.OUT / "table1.json", "w") as f:
        json.dump(table1, f, indent=2, default=float)
    print(json.dumps(table1, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
