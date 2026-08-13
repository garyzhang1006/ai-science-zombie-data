"""Stage 2: build the monthly population series and fit the interrupted
time series with placebo breaks.

Numerator: zombie citation events (citing paper published >= retraction
date + buffer, not engaging the retraction) by citing month and field.
Denominator: total references indexed that month = works-per-month (from
OpenAlex group-by, per field) x mean reference-list length (sampled per
year). The rate is zombie events per 10,000 references.

ITS: rate_t = b0 + b1*t + b2*post_t + b3*(t - t0)*post_t, Newey-West SEs.
Placebo: the same model refit on pre-break data only, with the break moved
to each month of 2019-2021; the actual level-change estimate is compared
against the placebo distribution.

Outputs: out/population_series.csv, out/its_results.json
"""
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
import oa


def month_range():
    months = []
    y, m = 2018, 1
    end_y, end_m = map(int, config.END_MONTH.split("-"))
    while (y, m) <= (end_y, end_m):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def add_months(d, k):
    y, m = d.year, d.month + k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return date(y, m, min(d.day, 28))


def zombie_events():
    cites = pd.DataFrame(oa.Checkpoint("s01_citations").load_results())
    engaged = pd.DataFrame(oa.Checkpoint("s01_engaged").load_results())
    engaged_keys = set()
    if len(engaged):
        engaged_keys = set(zip(engaged.retracted_id, engaged.citing_id))
    cites["citing_date"] = pd.to_datetime(cites.citing_date, errors="coerce")
    cites["retraction_date"] = pd.to_datetime(cites.retraction_date)
    cites = cites.dropna(subset=["citing_date"])

    cutoff = cites.retraction_date + pd.DateOffset(months=config.BUFFER_MONTHS)
    zombie = cites[cites.citing_date >= cutoff].copy()
    zombie["engaged"] = [
        (r, c) in engaged_keys
        for r, c in zip(zombie.retracted_id, zombie.citing_id)
    ]
    title_engaged = zombie.title.str.contains("retract", case=False, na=False)
    zombie = zombie[~zombie.engaged & ~title_engaged]
    zombie["month"] = zombie.citing_date.dt.strftime("%Y-%m")
    print(f"{len(zombie)} zombie citation events after exclusions")
    return zombie


def denominator():
    """works per month per field, and mean refs per year (sampled)."""
    ckpt = oa.Checkpoint("s02_denominator")
    for month in month_range():
        if ckpt.is_done(month):
            continue
        y, m = map(int, month.split("-"))
        last_day = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).day
        filt = (f"from_publication_date:{month}-01,"
                f"to_publication_date:{month}-{last_day:02d}")
        page = oa.oa_get("/works", filter=filt,
                         group_by="primary_topic.field.id",
                         **{"per-page": 200})
        rows = [{"month": month,
                 "field": g.get("key_display_name", ""),
                 "works": g["count"]}
                for g in (page or {}).get("group_by", [])]
        ckpt.write(rows, key=month)
    works = pd.DataFrame(ckpt.load_results())

    ckpt2 = oa.Checkpoint("s02_meanrefs")
    for year in range(2018, int(config.END_MONTH[:4]) + 1):
        key = str(year)
        if ckpt2.is_done(key):
            continue
        counts = [rec.get("referenced_works_count", 0)
                  for rec in oa.oa_page_all(
                      "/works", f"publication_year:{year}",
                      "id,referenced_works_count",
                      extra={"sample": 2000, "seed": config.SAMPLE_SEED},
                      max_records=2000)]
        ckpt2.write([{"year": year, "mean_refs": float(np.mean(counts))}],
                    key=key)
    meanrefs = pd.DataFrame(ckpt2.load_results())
    return works, meanrefs


def build_series(zombie, works, meanrefs):
    num = (zombie.groupby(["month", "field"]).size()
           .rename("zombie_events").reset_index())
    df = works.merge(num, on=["month", "field"], how="left")
    df["zombie_events"] = df.zombie_events.fillna(0).astype(int)
    df["year"] = df.month.str[:4].astype(int)
    df = df.merge(meanrefs, on="year")
    df["total_refs"] = df.works * df.mean_refs
    df["rate_per_10k"] = df.zombie_events / df.total_refs * 1e4
    return df


def fit_its(series, break_month):
    months = month_range()
    t0 = months.index(break_month)
    pooled = (series.groupby("month")
              .agg(zombie=("zombie_events", "sum"),
                   refs=("total_refs", "sum")).reindex(months).fillna(0))
    y = (pooled.zombie / pooled.refs * 1e4).values
    t = np.arange(len(months))
    post = (t >= t0).astype(float)
    X = sm.add_constant(np.column_stack([t, post, (t - t0) * post]))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {
        "break_month": break_month,
        "level_change": fit.params[2],
        "level_change_se": fit.bse[2],
        "level_change_p": fit.pvalues[2],
        "slope_change": fit.params[3],
        "slope_change_se": fit.bse[3],
        "pre_mean_rate": float(y[:t0].mean()),
        "n_months": len(months),
    }


def placebo_distribution(series):
    """Refit with fake breaks on PRE-actual-break data only."""
    months = month_range()
    cutoff = months.index(config.BREAK_MONTH)
    pre = series[series.month.isin(months[:cutoff])]
    out = []
    for pm in config.PLACEBO_MONTHS:
        pooled = (pre.groupby("month")
                  .agg(zombie=("zombie_events", "sum"),
                       refs=("total_refs", "sum"))
                  .reindex(months[:cutoff]).fillna(0))
        y = (pooled.zombie / pooled.refs * 1e4).values
        t = np.arange(len(y))
        p0 = months.index(pm)
        post = (t >= p0).astype(float)
        X = sm.add_constant(np.column_stack([t, post, (t - p0) * post]))
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        out.append(fit.params[2])
    return out


def main():
    zombie = zombie_events()
    works, meanrefs = denominator()
    series = build_series(zombie, works, meanrefs)
    series.to_csv(config.OUT / "population_series.csv", index=False)

    actual = fit_its(series, config.BREAK_MONTH)
    placebos = placebo_distribution(series)
    abs_pl = np.abs(placebos)
    quantile = float((abs_pl < abs(actual["level_change"])).mean())
    per_field = {
        f: fit_its(series[series.field == f], config.BREAK_MONTH)
        for f in series.field.value_counts().head(8).index if f
    }
    results = {"pooled": actual,
               "placebo_level_changes": placebos,
               "actual_abs_exceeds_placebo_share": quantile,
               "per_field": per_field}
    with open(config.OUT / "its_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(actual, indent=2, default=float))
    print(f"actual |level change| exceeds {quantile:.0%} of placebo breaks")


if __name__ == "__main__":
    sys.exit(main())
