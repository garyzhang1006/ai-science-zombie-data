"""Stage 5: the unresolvable-reference outcome, from Crossref.

A reference is unresolvable when its DOI does not resolve and its
bibliographic string matches no indexed work. Prior work links these to
model fabrication at rates far above the human baseline, and they are far
denser than zombie citations, so this endpoint carries most of the
statistical weight in the cohort comparison.

Crossref's public API is free and unmetered under the polite pool, which is
why this outcome survives the OpenAlex budget limits that forced the rest
of the pipeline onto the snapshot.

Input: out/matches.csv plus the cohort/control rows from stage 3.
Output: out/unresolvable.csv (openalex_id, group, n_refs_crossref,
        n_unresolvable)
"""
import argparse
import sys
import time

import pandas as pd
from rapidfuzz import fuzz

import config
import oa

_doi_cache = {}


def doi_resolves(doi):
    if doi not in _doi_cache:
        _doi_cache[doi] = oa.cr_get(f"/works/{doi}") is not None
    return _doi_cache[doi]


def ref_unresolvable(ref):
    """True when a Crossref reference entry points at nothing.

    Entries too short to identify a work are treated as resolvable: the
    conservative direction, since counting them as fabricated would inflate
    the very quantity the study is testing.
    """
    doi = oa.norm_doi(ref.get("DOI"))
    if doi:
        return not doi_resolves(doi)
    text = (ref.get("article-title") or ref.get("unstructured") or "").strip()
    if len(text) < 15:
        return False
    hit = oa.cr_get("/works", **{"query.bibliographic": text[:250],
                                 "rows": 1})
    items = (hit or {}).get("message", {}).get("items", [])
    if not items:
        return True
    titles = items[0].get("title") or [""]
    cand = titles[0] if titles else ""
    if not cand:
        return True
    score = max(fuzz.token_sort_ratio(text.lower(), cand.lower()),
                fuzz.partial_ratio(cand.lower(), text.lower()))
    return score < config.FUZZY_THRESHOLD


def ckpt_done(work_id):
    """Cheap membership test used before the checkpoint object exists, so a
    shard can report how much of its slice is already covered."""
    path = config.DATA / "s05_unresolvable.done"
    if not hasattr(ckpt_done, "_cache"):
        ckpt_done._cache = (set(path.read_text().split())
                            if path.exists() else set())
    return work_id in ckpt_done._cache


def papers_to_check():
    def read(name):
        parts = [pd.read_csv(p) for p in config.DATA.glob(f"s03_{name}_*.csv")]
        return (pd.concat(parts).drop_duplicates(subset="openalex_id")
                if parts else pd.DataFrame())

    cohort, controls = read("cohort"), read("control")
    matches_path = config.OUT / "matches.csv"
    if matches_path.exists() and len(controls):
        used = set(pd.read_csv(matches_path).control_id)
        controls = controls[controls.openalex_id.isin(used)]
    rows = []
    for df, group in ((cohort, "artifact"), (controls, "control")):
        if not len(df):
            continue
        for r in df.itertuples():
            doi = getattr(r, "doi", None)
            if isinstance(doi, str) and doi:
                rows.append({"openalex_id": r.openalex_id, "doi": doi,
                             "group": group})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--time-budget-hours", type=float, default=10.5)
    args = ap.parse_args()
    deadline = time.time() + args.time_budget_hours * 3600

    papers = papers_to_check()
    if not papers:
        print("nothing to check; run stages 1-4 first")
        return 1
    # Each paper costs one lookup plus one check per reference, so a full
    # pass runs far past a single session. Sorting first makes the split
    # deterministic; the checkpoint is keyed by work id, so shards never
    # duplicate each other's work even when rerun.
    papers.sort(key=lambda p: p["openalex_id"])
    mine = [p for i, p in enumerate(papers)
            if i % args.n_shards == args.shard]
    todo = [p for p in mine if not ckpt_done(p["openalex_id"])]
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} assigned, "
          f"{len(todo)} still to do (of {len(papers)} total)", flush=True)

    ckpt = oa.Checkpoint("s05_unresolvable")
    stopped = False
    for i, p in enumerate(todo):
        if ckpt.is_done(p["openalex_id"]):
            continue
        if time.time() > deadline:
            stopped = True
            print(">>> time budget reached; checkpoint kept", flush=True)
            break
        cr = oa.cr_get(f"/works/{p['doi']}")
        refs = (cr or {}).get("message", {}).get("reference") or []
        out = []
        if refs:
            out = [{
                "openalex_id": p["openalex_id"], "group": p["group"],
                "n_refs_crossref": len(refs),
                "n_unresolvable": sum(1 for r in refs
                                      if ref_unresolvable(r)),
            }]
        ckpt.write(out, key=p["openalex_id"])
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("COMPLETE" if not stopped else "INCOMPLETE_RESUMABLE", flush=True)
    df = pd.DataFrame(ckpt.load_results())
    df.to_csv(config.OUT / "unresolvable.csv", index=False)
    if len(df):
        by = df.groupby("group").agg(
            papers=("openalex_id", "size"),
            refs=("n_refs_crossref", "sum"),
            unresolvable=("n_unresolvable", "sum"))
        by["per_1000"] = by.unresolvable / by.refs * 1000
        print(by.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
