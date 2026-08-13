"""Fast correctness checks: pure functions plus one tiny live OpenAlex call.
Run from repo root: python tests/smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import config
import oa


def test_norm_doi():
    assert oa.norm_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert oa.norm_doi("doi:10.1/x") == "10.1/x"
    assert oa.norm_doi(None) is None
    assert oa.norm_doi("") is None


def test_invert_abstract():
    inv = {"world": [1], "hello": [0], "again": [2]}
    assert oa.invert_abstract(inv) == "hello world again"
    assert oa.invert_abstract(None) == ""


def test_month_range():
    from s02_population_series import month_range
    months = month_range()
    assert months[0] == "2018-01"
    assert months[-1] == config.END_MONTH
    assert len(months) == len(set(months))
    # Jan 2018 .. Jun 2026 inclusive = 102 months for the default config
    if config.END_MONTH == "2026-06":
        assert len(months) == 102


def test_zombie_count():
    from s05_outcomes import zombie_count
    rmap = {"W1": pd.Timestamp("2020-01-01"), "W2": pd.Timestamp("2024-01-01")}
    pub = pd.Timestamp("2021-06-01")
    # W1 retracted 2020-01 + 6mo buffer = 2020-07 <= 2021-06 -> zombie.
    # W2 retracted after publication -> not zombie. W9 not retracted.
    assert zombie_count(["W1", "W2", "W9"], pub, rmap) == 1
    # inside the buffer window -> not zombie
    assert zombie_count(["W1"], pd.Timestamp("2020-03-01"), rmap) == 0


def test_corrected_rates_recovers_truth():
    """Forward-simulate observables from known (a, b, q, sens, spec); the
    solver must recover a and b."""
    from s06_bounds import corrected_rates
    a, b, q, sens, spec = 0.30, 0.05, 0.10, 0.65, 0.95
    p_s1 = sens * q + (1 - spec) * (1 - q)
    p_y_s1 = (a * sens * q + b * (1 - spec) * (1 - q)) / p_s1
    p_y_s0 = (a * (1 - sens) * q + b * spec * (1 - q)) / (1 - p_s1)
    sol = corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec)
    assert sol is not None
    assert abs(sol[0] - a) < 1e-9, sol
    assert abs(sol[1] - b) < 1e-9, sol
    # infeasible q must return None
    assert corrected_rates(0.5, 0.5, 0.01, 0.65, 0.95) is None


def test_its_recovers_break():
    """Synthetic series with a known level jump: ITS must find it."""
    from s02_population_series import fit_its, month_range
    months = month_range()
    t0 = months.index(config.BREAK_MONTH)
    rng = np.random.default_rng(0)
    rate = 2.0 + 0.01 * np.arange(len(months)) + rng.normal(0, 0.02,
                                                            len(months))
    rate[t0:] += 1.5  # injected jump
    series = pd.DataFrame({
        "month": months, "field": "Synthetic",
        "zombie_events": rate * 1000, "total_refs": 1e7,
    })
    # rate_per_10k = events/refs*1e4 = rate*1000/1e7*1e4 = rate exactly
    res = fit_its(series, config.BREAK_MONTH)
    assert abs(res["level_change"] - 1.5) < 0.1, res["level_change"]
    assert res["level_change_p"] < 1e-6


def test_classify():
    from s03_artifact_cohort import classify
    meta = pd.Series({"title": "Detecting ChatGPT text in journals",
                      "abstract": "", "subfield_id": "", "type": "article",
                      "n_refs": 30})
    keep = pd.Series({"title": "Effects of irrigation on wheat yield",
                      "abstract": "field experiment in northern plains",
                      "subfield_id": "", "type": "article", "n_refs": 30})
    assert classify(meta)[0] is False
    assert classify(keep)[0] is True


def test_live_openalex_one_page():
    recs = list(oa.oa_page_all(
        "/works", "publication_year:2024", "id,publication_date",
        per_page=5, max_records=5))
    assert len(recs) == 5
    assert all(r["id"].startswith("https://openalex.org/W") for r in recs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
