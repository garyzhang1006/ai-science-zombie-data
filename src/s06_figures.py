"""Stage 6: the paper's three figures, built from stage outputs only.

Figure 1: monthly zombie-citation rate with the ChatGPT break and the
          placebo distribution inset.
Figure 2: artifact-versus-control rate ratios for both endpoints.
Figure 3: set-identified outcome rates with the attenuated naive estimates
          overlaid, which is the point of the bounds.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from s04_stats import month_range


def fig1():
    series = pd.read_csv(config.OUT / "population_series.csv")
    its = json.loads((config.OUT / "its_results.json").read_text())
    months = month_range()
    pooled = (series.groupby("month")
              .agg(zombie=("zombie_events", "sum"), refs=("refs", "sum"))
              .reindex(months))
    rate = (pooled.zombie / pooled.refs * 1e4).values

    fig, (ax, iax) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [3, 1]})

    top = series.groupby("field").zombie_events.sum().nlargest(5).index
    for f in top:
        sub = series[series.field == f].set_index("month").reindex(months)
        ax.plot(range(len(months)), sub.zombie_events / sub.refs * 1e4,
                lw=0.7, alpha=0.4, label=f if len(f) < 30 else f[:27] + "...")
    ax.plot(range(len(months)), rate, "k-", lw=2.1, label="All fields",
            zorder=5)
    if config.BREAK_MONTH in months:
        ax.axvline(months.index(config.BREAK_MONTH), color="crimson",
                   ls="--", lw=1.2, zorder=4)
        ax.annotate("ChatGPT", xy=(months.index(config.BREAK_MONTH), 0.2),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color="crimson")
    ax.set_xticks(range(0, len(months), 12))
    ax.set_xticklabels([m[:4] for m in months[::12]], fontsize=8)
    ax.set_ylabel("Zombie citations per 10,000 references", fontsize=9)
    ax.set_ylim(0, None)
    # Legend below the axes: an inset legend collided with the panel.
    ax.legend(fontsize=6.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=False)

    pl = [p["level_change"] for p in its.get("placebos", [])]
    actual = (its.get("pooled") or {}).get("level_change")
    if pl:
        iax.hist(pl, bins=12, color="0.6", edgecolor="white", linewidth=0.4)
        if actual is not None:
            iax.axvline(actual, color="crimson", lw=1.8)
            iax.annotate("observed", xy=(actual, 0), xytext=(-3, 4),
                         textcoords="offset points", fontsize=6.5,
                         color="crimson", rotation=90, va="bottom", ha="right")
        # Pad the axis so the observed line is not flush against the frame.
        lo, hi = min(pl + [actual]), max(pl + [actual])
        pad = 0.12 * (hi - lo)
        iax.set_xlim(lo - pad, hi + pad)
        iax.set_title("36 placebo breaks", fontsize=8)
        iax.set_xlabel("Level change per 10,000", fontsize=7.5)
        iax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(config.OUT / "fig1_population_series.pdf",
                bbox_inches="tight")
    print("wrote fig1")


def fig2():
    t1 = json.loads((config.OUT / "table1.json").read_text())
    items = [(n, t1[k]) for n, k in (("Zombie citations", "zombie"),
                                     ("Unresolvable refs", "unresolvable"))
             if isinstance(t1.get(k), dict)]
    if not items:
        print("skip fig2: no cohort results yet")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [n for n, _ in items]
    ratios = [d["ratio"] for _, d in items]
    lo = [d["ratio"] - d["ratio_ci_lo"] for _, d in items]
    hi = [d["ratio_ci_hi"] - d["ratio"] for _, d in items]
    ax.errorbar(ratios, range(len(labels)), xerr=[lo, hi], fmt="o",
                color="k", capsize=4)
    ax.axvline(1, ls="--", color="grey", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Artifact / control rate ratio (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(config.OUT / "fig2_cohort.pdf")
    print("wrote fig2")


def fig3():
    """Bounds by reference-count stratum.

    The unrestricted comparison is dominated by article type: flagged
    papers average 44.7 references against 9.2 for the rest, and a
    per-paper "at least one event" outcome rises with reference count.
    Holding reference count fixed is what makes the contrast interpretable,
    so the strata are the figure rather than a footnote.
    """
    b = json.loads((config.OUT / "bounds.json").read_text())
    strata = b.get("by_ref_stratum", {})
    usable = [(k, v) for k, v in strata.items()
              if v and v.get("bound_ai_lo") is not None]
    if not usable:
        print("skip fig3: no feasible strata")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels, xs = [], np.arange(len(usable))
    for i, (k, v) in enumerate(usable):
        lo, hi = k.split("-")
        labels.append(f"{lo}\u2013{hi}" if int(hi) < 10**5 else f"{lo}+")
        ax.vlines(i - 0.11, v["bound_ai_lo"] * 100, v["bound_ai_hi"] * 100,
                  color="crimson", lw=7, alpha=0.85,
                  label="AI-assisted" if i == 0 else None)
        ax.vlines(i + 0.11, v["bound_nonai_lo"] * 100,
                  v["bound_nonai_hi"] * 100, color="steelblue", lw=7,
                  alpha=0.85, label="Not AI-assisted" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("References in the citing paper")
    ax.set_ylabel("P(at least one zombie citation), %")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.OUT / "fig3_bounds.pdf")
    print("wrote fig3")


def main():
    for fn in (fig1, fig2, fig3):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: missing {e.filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
