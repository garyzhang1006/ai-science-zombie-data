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
    rate = pooled.zombie / pooled.refs * 1e4

    fig, ax = plt.subplots(figsize=(7, 4.2))
    top = series.groupby("field").zombie_events.sum().nlargest(5).index
    for f in top:
        sub = series[series.field == f].set_index("month").reindex(months)
        ax.plot(range(len(months)), sub.zombie_events / sub.refs * 1e4,
                lw=0.8, alpha=0.45, label=f)
    ax.plot(range(len(months)), rate.values, "k-", lw=2, label="All fields")
    if config.BREAK_MONTH in months:
        ax.axvline(months.index(config.BREAK_MONTH), color="crimson",
                   ls="--", lw=1)
    ax.set_xticks(range(0, len(months), 12))
    ax.set_xticklabels([m[:4] for m in months[::12]])
    ax.set_ylabel("Zombie citations per 10,000 references")
    ax.legend(fontsize=7, ncol=2)

    pl = [p["level_change"] for p in its.get("placebos", [])]
    if pl:
        inset = fig.add_axes((0.62, 0.55, 0.25, 0.28))
        inset.hist(pl, bins=12, color="grey")
        if its.get("pooled"):
            inset.axvline(its["pooled"]["level_change"], color="crimson",
                          lw=1.5)
        inset.set_title("Placebo breaks", fontsize=7)
        inset.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(config.OUT / "fig1_population_series.pdf")
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
    b = json.loads((config.OUT / "bounds.json").read_text())
    panels = [(n, b[k]) for n, k in (("Zombie", "zombie"),
                                     ("Unresolvable", "unresolvable"))
              if isinstance(b.get(k), dict)]
    if not panels:
        print("skip fig3: no bounds yet")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3),
                             squeeze=False)
    for ax, (name, d) in zip(axes[0], panels):
        if d.get("bound_ai_lo") is None:
            ax.set_title(f"{name}: no feasible grid points", fontsize=8)
            continue
        ax.vlines([0, 1],
                  [d["bound_ai_lo"], d["bound_nonai_lo"]],
                  [d["bound_ai_hi"], d["bound_nonai_hi"]],
                  color="steelblue", lw=6, alpha=0.7, label="Set-identified")
        ax.plot([0, 1], [d["naive_rate_proxy_pos"],
                         d["naive_rate_proxy_neg"]], "ko",
                label="Naive (proxy)")
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["AI-assisted", "Not AI-assisted"], fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("P(at least one event)")
    axes[0][0].legend(fontsize=7)
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
