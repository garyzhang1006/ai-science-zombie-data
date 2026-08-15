"""Stage 3: the main snapshot pass. One sweep over the works corpus
produces every population-level quantity the study needs.

Four things come out of the same sweep, which is why they share a pass:

1. Population numerator. References pointing at a paper that was already
   retracted when the citing work appeared, by month and field.
2. Population denominator. Works published per month per field and the
   references they carry. The API design had to estimate this from a
   sample; here it is exact.
3. Cohort outcomes. Zombie counts for the artifact papers and for control
   candidates drawn from the same venue and year.
4. Lexical-proxy sample. Abstracts for a seeded subset of 2023-onward
   works, feeding the misclassification bounds.

The corpus holds 510M works carrying billions of references, so the inner
work is vectorized: Arrow's is_in and index_in resolve every reference in a
batch against the retracted set in C++, and the per-work tallies come from
a bincount over reference owners. A Python loop over references would take
hours per shard.

Sharding: --shard i --n-shards k. Each shard writes partial aggregates and
stage 4 merges them.
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

import config
import oa
import snapshot
from s04_stats import lexical_score

BASE_COLUMNS = ["id", "doi", "publication_date", "referenced_works",
                "referenced_works_count", "primary_topic", "type",
                "primary_location", "cited_by_count"]
ABSTRACT_COLUMN = "abstract_inverted_index"
EPOCH = np.datetime64("1970-01-01", "D")


def stable_fraction(text):
    """Deterministic [0,1) hash. Python's hash() is salted per process, so
    using it here would resample different files on every run."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def load_targets():
    frames = [pd.read_csv(p) for p in
              config.DATA.glob("retracted_works_*.csv")]
    if not frames:
        raise SystemExit("run stage 2 first (no retracted_works_*.csv)")
    retr = (pd.concat(frames).dropna(subset=["openalex_id"])
            .drop_duplicates(subset="openalex_id"))
    eligible = (pd.to_datetime(retr.retraction_date)
                + pd.DateOffset(months=config.BUFFER_MONTHS))
    # Days since epoch: integer comparison keeps the inner loop vectorized.
    elig_days = (eligible.values.astype("datetime64[D]")
                 - EPOCH).astype(np.int64)
    ids = pa.array(retr.openalex_id.astype(str).tolist(), type=pa.string())

    cohort_path = config.OUT / "artifact_cohort.csv"
    cohort = (pd.read_csv(cohort_path) if cohort_path.exists()
              else pd.DataFrame(columns=["openalex_id", "source_id", "year"]))
    cohort_ids = set(cohort.openalex_id.dropna().astype(str))
    venue_years = {(str(s), int(y)) for s, y in
                   zip(cohort.get("source_id", pd.Series(dtype=str)).fillna(""),
                       cohort.get("year", pd.Series(dtype=float)).fillna(0))
                   if s and y}
    print(f"{len(retr)} retracted works, {len(cohort_ids)} artifact papers, "
          f"{len(venue_years)} venue-years for controls", flush=True)
    return ids, elig_days, cohort_ids, venue_years


def strip_prefix(arr):
    """https://openalex.org/W123 -> W123, vectorized."""
    return pc.utf8_slice_codeunits(arr, 21)


def zombie_counts(refs_col, pub_days, retracted_ids, elig_days):
    """Per-work count of references that were already retracted.

    Resolves every reference in the batch at once, then folds the flat
    per-reference verdicts back onto their owning works.
    """
    n_rows = len(refs_col)
    if refs_col.null_count == n_rows:
        return np.zeros(n_rows, dtype=np.int64)
    flat = refs_col.flatten()
    if len(flat) == 0:
        return np.zeros(n_rows, dtype=np.int64)

    lengths = np.asarray(pc.list_value_length(
        pc.fill_null(refs_col, [])).to_numpy(zero_copy_only=False),
        dtype=np.int64)
    idx = pc.index_in(strip_prefix(flat), value_set=retracted_ids)
    idx = idx.to_numpy(zero_copy_only=False).astype(np.float64)
    hit = ~np.isnan(idx)
    if not hit.any():
        return np.zeros(n_rows, dtype=np.int64)

    owner = np.repeat(np.arange(n_rows, dtype=np.int64), lengths)
    owner_pub = pub_days[owner]
    is_zombie = np.zeros(len(flat), dtype=bool)
    hit_idx = idx[hit].astype(np.int64)
    # Zombie only if the citing work appeared at or after the retraction
    # date plus the indexing buffer.
    is_zombie[hit] = owner_pub[hit] >= elig_days[hit_idx]
    return np.bincount(owner[is_zombie], minlength=n_rows).astype(np.int64)


class Reservoir:
    """Uniform sample of a stream, bounded in memory (Algorithm R).

    A venue-year can hold hundreds of thousands of papers and the corpus
    holds half a billion, so keeping every candidate is not an option.
    Taking the first N instead of a random N would bias the sample toward
    whichever update partition happened to be scanned first, which
    correlates with publication date.
    """

    def __init__(self, cap, seed):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.items = []

    def offer(self, item):
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, self.seen))
            if j < self.cap:
                self.items[j] = item

    def __len__(self):
        return len(self.items)


def struct_path(col, path, default=""):
    """Pull a nested struct field as a python list, tolerating nulls."""
    cur = col
    for part in path:
        try:
            cur = pc.struct_field(cur, part)
        except (pa.ArrowInvalid, KeyError, TypeError):
            return [default] * len(col)
    return [default if v is None else v
            for v in cur.to_pylist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--abstract-sample-rate", type=float, default=0.02)
    ap.add_argument("--time-budget-hours", type=float, default=10.5)
    args = ap.parse_args()

    import time
    deadline = time.time() + args.time_budget_hours * 3600

    retracted_ids, elig_days, cohort_ids, venue_years = load_targets()
    cohort_arrow = pa.array(sorted(cohort_ids), type=pa.string())
    # is_in needs the full OpenAlex source URL, matching the struct field.
    venue_src_arrow = pa.array(
        sorted({f"https://openalex.org/{s}" if not s.startswith("http")
                else s for s, _ in venue_years}), type=pa.string())
    paths = snapshot.shard(snapshot.file_paths(), args.shard, args.n_shards)
    tag = f"{args.shard}of{args.n_shards}"
    fs = snapshot.filesystem()

    abs_files = {p for p in paths
                 if stable_fraction(p) < args.abstract_sample_rate}
    print(f"shard {tag}: {len(paths)} files, {len(abs_files)} with abstracts",
          flush=True)

    done_path = config.DATA / f"s03_done_{tag}.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    numer, denom = {}, {}
    cohort_rows = []
    # Bounded samples: controls per venue-year, lexical rows per shard.
    control_pools = {}
    lex_reservoir = Reservoir(config.LEXICAL_SAMPLE_CAP,
                              config.SAMPLE_SEED + args.shard)

    def flush():
        pd.DataFrame([{"month": m, "field": f, "zombie_events": v}
                      for (m, f), v in numer.items()]).to_csv(
            config.DATA / f"s03_numer_{tag}.csv", index=False)
        pd.DataFrame([{"month": m, "field": f, "works": w, "refs": r}
                      for (m, f), (w, r) in denom.items()]).to_csv(
            config.DATA / f"s03_denom_{tag}.csv", index=False)
        control_rows = [r for pool in control_pools.values()
                        for r in pool.items]
        for name, rows in (("cohort", cohort_rows),
                           ("control", control_rows),
                           ("lexical", lex_reservoir.items)):
            if rows:
                pd.DataFrame(rows).to_csv(
                    config.DATA / f"s03_{name}_{tag}.csv", index=False)
        done_path.write_text("\n".join(sorted(done)))

    stopped_early = False
    for i, path in enumerate(paths):
        key = path.rsplit("/", 1)[-1] + "@" + \
            path.split("updated_date=")[-1][:10]
        if key in done:
            continue
        if time.time() > deadline:
            stopped_early = True
            print(">>> time budget reached; checkpointing", flush=True)
            break
        want_abs = path in abs_files
        cols = BASE_COLUMNS + ([ABSTRACT_COLUMN] if want_abs else [])

        for batch in snapshot.iter_batches(path, cols, fs=fs):
            n = batch.num_rows
            pub = batch.column("publication_date").to_numpy(
                zero_copy_only=False)
            valid = ~pd.isna(pub)
            if not valid.any():
                continue
            pub_days = np.zeros(n, dtype=np.int64)
            pub_days[valid] = (pub[valid].astype("datetime64[D]")
                               - EPOCH).astype(np.int64)

            zc = zombie_counts(batch.column("referenced_works"), pub_days,
                               retracted_ids, elig_days)
            n_refs = np.asarray(pc.fill_null(
                batch.column("referenced_works_count"), 0).to_numpy(
                    zero_copy_only=False), dtype=np.int64)
            months = np.where(
                valid, pub.astype("datetime64[M]").astype(str), "")
            fields = struct_path(batch.column("primary_topic"),
                                 ["field", "display_name"], "Unknown")
            fields = np.array([f or "Unknown" for f in fields], dtype=object)

            in_window = valid & (months >= "2018-01") & \
                (months <= config.END_MONTH)
            if in_window.any():
                df = pd.DataFrame({
                    "month": months[in_window], "field": fields[in_window],
                    "z": zc[in_window], "r": n_refs[in_window],
                })
                agg = df.groupby(["month", "field"], sort=False).agg(
                    z=("z", "sum"), r=("r", "sum"), w=("z", "size"))
                for (m, f), row in agg.iterrows():
                    if row.z:
                        numer[(m, f)] = numer.get((m, f), 0) + int(row.z)
                    prev = denom.get((m, f), (0, 0))
                    denom[(m, f)] = (prev[0] + int(row.w),
                                     prev[1] + int(row.r))

            # Only a handful of rows in any batch are cohort members,
            # control candidates, or abstract samples. Select them with
            # vectorized masks and let Python touch just those, instead of
            # materializing every one of the corpus's 510M rows.
            wid_arr = strip_prefix(batch.column("id"))
            years = np.zeros(n, dtype=np.int64)
            years[valid] = (pub[valid].astype("datetime64[Y]").astype(int)
                            + 1970)

            is_cohort = (np.asarray(pc.is_in(wid_arr, value_set=cohort_arrow)
                                    .to_numpy(zero_copy_only=False), dtype=bool)
                         if len(cohort_arrow) else np.zeros(n, dtype=bool))
            src_arr = pc.struct_field(batch.column("primary_location"),
                                      ["source", "id"])
            maybe_control = (
                np.asarray(pc.is_in(src_arr, value_set=venue_src_arrow)
                           .to_numpy(zero_copy_only=False), dtype=bool)
                if len(venue_src_arrow) else np.zeros(n, dtype=bool))
            maybe_control &= valid & (n_refs > 0) & ~is_cohort
            want_lex = (want_abs & valid & (years >= 2023) & (n_refs > 0)
                        if want_abs else np.zeros(n, dtype=bool))

            picked = np.flatnonzero(is_cohort | maybe_control | want_lex)
            if len(picked):
                wids_sel = wid_arr.take(pa.array(picked)).to_pylist()
                dois_sel = batch.column("doi").take(
                    pa.array(picked)).to_pylist()
                srcs_sel = src_arr.take(pa.array(picked)).to_pylist()
                cites = np.asarray(pc.fill_null(
                    batch.column("cited_by_count"), 0).to_numpy(
                        zero_copy_only=False), dtype=np.int64)
                abs_sel = (batch.column(ABSTRACT_COLUMN)
                           .take(pa.array(picked)).to_pylist()
                           if want_abs else None)
                for k, j in enumerate(picked):
                    wid = wids_sel[k]
                    sid = (srcs_sel[k] or "").rsplit("/", 1)[-1]
                    year = int(years[j])
                    if is_cohort[j] or maybe_control[j]:
                        rec = {"openalex_id": wid,
                               "doi": oa.norm_doi(dois_sel[k]),
                               "pub_date": str(pub[j])[:10],
                               "n_refs": int(n_refs[j]),
                               "zombie_count": int(zc[j]),
                               "source_id": sid, "year": year,
                               "n_cites": int(cites[j])}
                        if is_cohort[j]:
                            cohort_rows.append(rec)
                        elif (sid, year) in venue_years:
                            pool = control_pools.get((sid, year))
                            if pool is None:
                                pool = control_pools[(sid, year)] = Reservoir(
                                    config.CONTROL_POOL_PER_VENUE_YEAR,
                                    config.SAMPLE_SEED + int(
                                        stable_fraction(f"{sid}:{year}")
                                        * 1e6))
                            pool.offer(rec)
                    if want_lex[j] and abs_sel and abs_sel[k]:
                        # Score inline and keep the score, not the text:
                        # abstracts are a third of the snapshot and only
                        # the marker count feeds the bounds.
                        lex_reservoir.offer({
                            "openalex_id": wid, "year": year,
                            "n_refs": int(n_refs[j]),
                            "zombie_count": int(zc[j]),
                            "score": lexical_score(abs_sel[k]),
                        })
        done.add(key)
        if i % 10 == 0:
            flush()
            print(f"  {i}/{len(paths)} files | zombie events "
                  f"{sum(numer.values())} | cohort {len(cohort_rows)} | "
                  f"control pools {len(control_pools)} | lexical "
                  f"{len(lex_reservoir)}/{lex_reservoir.seen}", flush=True)

    flush()
    summary = {
        "shard": tag,
        "snapshot_date": snapshot.snapshot_date(),
        "files_done": len(done),
        "files_assigned": len(paths),
        "complete": not stopped_early and len(done) >= len(paths),
        "zombie_events": int(sum(numer.values())),
        "works_counted": int(sum(w for w, _ in denom.values())),
        "cohort_rows": len(cohort_rows),
        "control_venue_years": len(control_pools),
        "control_rows": sum(len(p) for p in control_pools.values()),
        "lexical_rows": len(lex_reservoir),
        "lexical_seen": lex_reservoir.seen,
    }
    (config.DATA / f"s03_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
