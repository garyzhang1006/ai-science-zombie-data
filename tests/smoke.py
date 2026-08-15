"""Correctness checks for the pipeline's computational core.

Pure functions are checked against hand-computed or forward-simulated
ground truth; two live checks confirm the external contracts (OpenAlex
credit headers, the S3 snapshot manifest) still hold. Run from the repo
root: python tests/smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa

import config
import oa
import snapshot
from s01_artifact_harvest import classify
from s03_scan import EPOCH, stable_fraction, zombie_counts
from s04_stats import (bootstrap_ratio, bound_outcome, corrected_rates,
                       fit_its, lexical_score, match_controls, month_range)


def test_norm_doi():
    assert oa.norm_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert oa.norm_doi("doi:10.1/x") == "10.1/x"
    assert oa.norm_doi(None) is None and oa.norm_doi("") is None


def test_invert_abstract():
    assert oa.invert_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert oa.invert_abstract(None) == ""


def test_month_range():
    months = month_range()
    assert months[0] == "2018-01" and months[-1] == config.END_MONTH
    assert len(months) == len(set(months))


def test_stable_fraction_is_process_stable():
    # Python's hash() is salted per process; this must not be.
    assert stable_fraction("a/b.parquet") == stable_fraction("a/b.parquet")
    assert 0.0 <= stable_fraction("x") < 1.0
    # Regression guard: if this constant moves, the abstract sample silently
    # changes membership and prior results stop being reproducible.
    known = stable_fraction("openalex/data/parquet/works/part_0000.parquet")
    assert abs(known - 0.16529035) < 1e-6, known


def test_zombie_counts_matches_naive():
    """The vectorized counter must agree exactly with a literal reading of
    the definition, including the buffer boundary."""
    refs = pa.array([
        ["https://openalex.org/W1", "https://openalex.org/W2"],
        ["https://openalex.org/W3"],
        [],
        None,
        ["https://openalex.org/W1", "https://openalex.org/W1"],
    ])
    retracted = pa.array(["W1", "W2"], type=pa.string())
    elig = np.array([1000, 5000], dtype=np.int64)
    pub_days = np.array([2000, 2000, 2000, 2000, 900], dtype=np.int64)
    got = zombie_counts(refs, pub_days, retracted, elig)
    # row 0: W1 eligible (2000>=1000) counts, W2 not yet (2000<5000) -> 1
    # row 1: W3 not retracted -> 0; rows 2,3: no refs -> 0
    # row 4: cites W1 twice but before eligibility (900<1000) -> 0
    assert list(got) == [1, 0, 0, 0, 0], list(got)

    # Same work citing an eligible retraction twice counts twice.
    pub2 = np.array([2000, 2000, 2000, 2000, 2000], dtype=np.int64)
    assert list(zombie_counts(refs, pub2, retracted, elig))[4] == 2


def test_zombie_counts_no_matches():
    refs = pa.array([["https://openalex.org/W9"], ["https://openalex.org/W8"]])
    got = zombie_counts(refs, np.array([5000, 5000]),
                        pa.array(["W1"], type=pa.string()),
                        np.array([10], dtype=np.int64))
    assert list(got) == [0, 0]


def test_its_recovers_injected_break():
    months = month_range()
    t0 = months.index(config.BREAK_MONTH)
    rng = np.random.default_rng(0)
    rate = 2.0 + 0.01 * np.arange(len(months)) + rng.normal(0, 0.02,
                                                            len(months))
    rate[t0:] += 1.5
    series = pd.DataFrame({"month": months, "field": "Synthetic",
                           "zombie_events": rate * 1000, "refs": 1e7})
    res = fit_its(series, config.BREAK_MONTH)
    assert abs(res["level_change"] - 1.5) < 0.1, res["level_change"]
    assert res["level_change_p"] < 1e-6


def test_its_finds_nothing_when_nothing_happened():
    months = month_range()
    rng = np.random.default_rng(1)
    rate = 2.0 + rng.normal(0, 0.05, len(months))
    series = pd.DataFrame({"month": months, "field": "Flat",
                           "zombie_events": rate * 1000, "refs": 1e7})
    res = fit_its(series, config.BREAK_MONTH)
    assert abs(res["level_change"]) < 0.15, res["level_change"]
    assert res["level_change_p"] > 0.05, res["level_change_p"]


def test_corrected_rates_recovers_truth():
    a, b, q, sens, spec = 0.30, 0.05, 0.10, 0.65, 0.95
    p_s1 = sens * q + (1 - spec) * (1 - q)
    p_y_s1 = (a * sens * q + b * (1 - spec) * (1 - q)) / p_s1
    p_y_s0 = (a * (1 - sens) * q + b * spec * (1 - q)) / (1 - p_s1)
    sol = corrected_rates(p_y_s1, p_y_s0, p_s1, sens, spec)
    assert sol and abs(sol[0] - a) < 1e-9 and abs(sol[1] - b) < 1e-9
    assert corrected_rates(0.5, 0.5, 0.01, 0.65, 0.95) is None


def test_bounds_contain_truth_and_beat_naive():
    """The whole point of the bounds: the naive proxy estimate is attenuated
    toward the population rate, and the interval must still cover truth."""
    rng = np.random.default_rng(7)
    n, q, sens, spec = 200_000, 0.10, 0.65, 0.95
    ai = rng.random(n) < q
    proxy = np.where(ai, rng.random(n) < sens, rng.random(n) > spec)
    p_event = np.where(ai, 0.30, 0.05)
    event = rng.random(n) < p_event
    scored = pd.DataFrame({"proxy": proxy, "count": event.astype(int),
                           "denom": 1})
    res = bound_outcome(scored, "count", "denom")
    assert res["bound_ai_lo"] <= 0.30 <= res["bound_ai_hi"], res
    assert res["naive_rate_proxy_pos"] < 0.30, res["naive_rate_proxy_pos"]


def test_bootstrap_ratio():
    art = pd.DataFrame({"c": [10] * 50, "d": [100] * 50})
    ctl = pd.DataFrame({"c": [5] * 50, "d": [100] * 50})
    res = bootstrap_ratio(art, ctl, "c", "d", n_boot=500)
    assert abs(res["ratio"] - 2.0) < 1e-9
    assert res["ratio_ci_lo"] <= res["ratio"] <= res["ratio_ci_hi"]


def test_match_controls_respects_venue_and_year():
    cohort = pd.DataFrame([{"openalex_id": "W1", "source_id": "S1",
                            "year": 2024, "n_refs": 40, "n_cites": 3}])
    cands = pd.DataFrame([
        {"openalex_id": "C1", "source_id": "S1", "year": 2024,
         "n_refs": 41, "n_cites": 3, "zombie_count": 0},
        {"openalex_id": "C2", "source_id": "S1", "year": 2024,
         "n_refs": 39, "n_cites": 2, "zombie_count": 1},
        {"openalex_id": "X1", "source_id": "S2", "year": 2024,
         "n_refs": 40, "n_cites": 3, "zombie_count": 0},
        {"openalex_id": "X2", "source_id": "S1", "year": 2019,
         "n_refs": 40, "n_cites": 3, "zombie_count": 0},
    ])
    m = match_controls(cohort, cands)
    assert set(m.control_id) == {"C1", "C2"}, list(m.control_id)


def test_lexical_score():
    assert lexical_score("we delve into the intricate realm") >= 3
    assert lexical_score("we measured the wheat yield") == 0


def test_lexical_score_reads_inverted_index():
    """OpenAlex stores abstracts as inverted-index JSON. Scoring the raw
    string finds no markers, which would silently zero out the proxy and
    leave the bounds unidentified."""
    inv = ('{"We": [0], "delve": [1], "into": [2], "intricate": [3], '
           '"realm": [4], "of": [5], "wheat": [6]}')
    assert lexical_score(inv) == 3, lexical_score(inv)
    plain = '{"Wheat": [0], "yield": [1], "rose": [2]}'
    assert lexical_score(plain) == 0


def test_reservoir_is_uniform_and_bounded():
    from s03_scan import Reservoir
    r = Reservoir(cap=50, seed=1)
    for i in range(10_000):
        r.offer(i)
    assert len(r) == 50 and r.seen == 10_000
    # A uniform sample of 0..9999 has mean near 5000; taking the first 50
    # would give ~25, which is the bias this guards against.
    assert 3000 < np.mean(r.items) < 7000, np.mean(r.items)


def test_classify_drops_meta_discussion():
    base = {"phrase": "as an AI language model", "year": 2024,
            "subfield_id": "", "type": "article", "n_refs": 30}
    meta = dict(base, title="Detecting ChatGPT text in journals",
                abstract="")
    keep = dict(base, title="Irrigation effects on wheat yield",
                abstract="field experiment in northern plains")
    assert classify(meta)[0] is False
    assert classify(keep)[0] is True


def test_classify_excludes_impossible_and_retired():
    """A phrase in a pre-ChatGPT paper cannot be model output, and the
    stem-colliding phrase must stay out of the cohort."""
    base = {"title": "Irrigation effects on wheat yield", "abstract": "",
            "subfield_id": "", "type": "article", "n_refs": 30}
    old = dict(base, phrase="as an AI language model", year=2019)
    retired = dict(base, phrase="Regenerate response", year=2024)
    good = dict(base, phrase="as an AI language model", year=2024)
    assert classify(old) == (False, "pre_chatgpt")
    assert classify(retired) == (False, "phrase_retired")
    assert classify(good)[0] is True
    assert "Regenerate response" not in config.ARTIFACT_PHRASES


def test_live_openalex_reports_quota():
    """The free tier meters by call and by dollar; the client must see it."""
    oa.oa_get("/works", filter="publication_year:2024", select="id",
              **{"per-page": 1})
    assert oa.quota["limit"] is not None, oa.quota
    assert oa.quota["remaining"] is not None, oa.quota
    print(f"    (quota: {oa.quota['remaining']}/{oa.quota['limit']} calls, "
          f"${oa.quota['usd_remaining']} left)")


def test_live_snapshot_manifest():
    man = snapshot.manifest()
    assert man["record_count"] > 400_000_000, man["record_count"]
    paths = snapshot.file_paths()
    assert len(paths) == len(man["files"])
    # Sharding must partition exactly: no work lost, none done twice.
    shards = [snapshot.shard(paths, i, 8) for i in range(8)]
    flat = [p for s in shards for p in s]
    assert len(flat) == len(paths) and len(set(flat)) == len(paths)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
