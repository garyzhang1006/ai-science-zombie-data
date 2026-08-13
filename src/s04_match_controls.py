"""Stage 4: match each artifact paper to controls from the same venue and
year, nearest in reference-list length and citation count. Falls back to
same primary topic + year when the venue has too few candidates.

Output: out/matches.csv with columns
    artifact_id, control_id, match_basis, control_n_refs, control_n_cites
"""
import sys

import numpy as np
import pandas as pd

import config
import oa

SELECT = "id,referenced_works_count,cited_by_count,publication_year"


def fetch_pool(filt):
    pool = []
    for rec in oa.oa_page_all("/works", filt, SELECT,
                              max_records=config.CONTROL_POOL_SIZE,
                              extra={"sample": config.CONTROL_POOL_SIZE,
                                     "seed": config.SAMPLE_SEED}):
        pool.append({
            "id": oa.short_id(rec["id"]),
            "n_refs": rec.get("referenced_works_count", 0),
            "n_cites": rec.get("cited_by_count", 0),
        })
    return pool


def pick_controls(row, pool, artifact_ids):
    lo = row.n_refs * (1 - config.MATCH_TOL)
    hi = row.n_refs * (1 + config.MATCH_TOL)
    cands = [c for c in pool
             if c["id"] not in artifact_ids and c["id"] != row.openalex_id
             and c["n_refs"] > 0 and lo <= c["n_refs"] <= hi]
    if len(cands) < config.N_CONTROLS:  # relax the band rather than drop
        cands = [c for c in pool
                 if c["id"] not in artifact_ids
                 and c["id"] != row.openalex_id and c["n_refs"] > 0]
    def dist(c):
        return (abs(np.log1p(c["n_refs"]) - np.log1p(row.n_refs))
                + 0.5 * abs(np.log1p(c["n_cites"]) - np.log1p(row.n_cites)))
    return sorted(cands, key=dist)[:config.N_CONTROLS]


def main():
    cohort = pd.read_csv(config.OUT / "artifact_cohort.csv")
    artifact_ids = set(cohort.openalex_id)
    ckpt = oa.Checkpoint("s04_matches")
    for i, row in enumerate(cohort.itertuples()):
        if ckpt.is_done(row.openalex_id):
            continue
        basis, pool = "venue_year", []
        if isinstance(row.source_id, str) and row.source_id:
            pool = fetch_pool(
                f"primary_location.source.id:{row.source_id},"
                f"publication_year:{row.year}")
        if len(pool) < config.N_CONTROLS and getattr(row, "topic_id", ""):
            basis = "topic_year"
            pool = fetch_pool(
                f"primary_topic.id:{row.topic_id},"
                f"publication_year:{row.year}")
        controls = pick_controls(row, pool, artifact_ids)
        ckpt.write([{
            "artifact_id": row.openalex_id,
            "control_id": c["id"],
            "match_basis": basis,
            "control_n_refs": c["n_refs"],
            "control_n_cites": c["n_cites"],
        } for c in controls], key=row.openalex_id)
        if i % 50 == 0:
            print(f"  matched {i}/{len(cohort)}")
    matches = pd.DataFrame(ckpt.load_results())
    matches.to_csv(config.OUT / "matches.csv", index=False)
    per = matches.groupby("artifact_id").size()
    print(f"{len(matches)} controls for {per.count()} artifact papers "
          f"(median {per.median():.0f} each)")


if __name__ == "__main__":
    sys.exit(main())
