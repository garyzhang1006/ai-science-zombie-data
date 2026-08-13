"""Stage 7: the paper's three figures, from stage outputs only (no API).

Figure 1: monthly zombie-citation rate by field, ITS break line, placebo
          inset. -> out/fig1_population_series.pdf
Figure 2: artifact-vs-control ratios for both endpoints. -> out/fig2_cohort.pdf
Figure 3: set-identified outcome rates by proxy bin with naive points
          overlaid. -> out/fig3_bounds.pdf
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def fig1():
    series = pd.read_csv(config.OUT / "population_series.csv")
    its = json.load(open(config.OUT / "its_results.json"))
    pooled = (series.groupby("month")
              .agg(zombie=("zombie_events", "sum"),
                   refs=("total_refs", "sum")).reset_index())
    pooled["rate"] = pooled.zombie / pooled.refs * 1e4
    top_fields = series.groupby("field").zombie_events.sum().nlargest(5).index

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for f in top_fields:
        sub = series[series.field == f]
        sub = (sub.assign(rate=sub.zombie_events / sub.total_refs * 1e4)
               .sort_values("month"))
        ax.plot(sub.month, sub.rate, lw=0.8, alpha=0.5, label=f)
    ax.plot(pooled.month, pooled.rate, "k-", lw=2, label="All fields")
    bi = list(pooled.month).index(config.BREAK_MONTH) \
        if config.BREAK_MONTH in list(pooled.month) else None
    if bi is not None:
        ax.axvline(bi, color="crimson", ls="--", lw=1)
    ax.set_xticks(range(0, len(pooled), 12))
    ax.set_xticklabels([m[:4] for m in pooled.month[::12]])
    ax.set_ylabel("Zombie citations per 10,000 references")
    ax.legend(fontsize=7, ncol=2)

    inset = fig.add_axes([0.62, 0.55, 0.25, 0.28])
    inset.hist(its["placebo_level_changes"], bins=15, color="grey")
    inset.axvline(its["pooled"]["level_change"], color="crimson", lw=1.5)
    inset.set_title("Placebo breaks", fontsize=7)
    inset.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(config.OUT / "fig1_population_series.pdf")
    print("wrote fig1")


def fig2():
    t1 = json.load(open(config.OUT / "table1.json"))
    fig, ax = plt.subplots(figsize=(5, 3))
    labels, ratios, los, his = [], [], [], []
    for name, key in [("Zombie citations", "zombie"),
                      ("Unresolvable refs", "unresolvable")]:
        d = t1[key]
        labels.append(name)
        ratios.append(d["ratio"])
        los.append(d["ratio"] - d["ratio_ci_lo"])
        his.append(d["ratio_ci_hi"] - d["ratio"])
    ax.errorbar(ratios, range(len(labels)), xerr=[los, his],
                fmt="o", color="k", capsize=4)
    ax.axvline(1, ls="--", color="grey", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Artifact / control rate ratio (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(config.OUT / "fig2_cohort.pdf")
    print("wrote fig2")


def fig3():
    b = json.load(open(config.OUT / "bounds.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)
    for ax, (name, key) in zip(axes, [("Zombie", "zombie"),
                                      ("Unresolvable", "unresolvable")]):
        d = b[key]
        if d["bound_ai_lo"] is None:
            ax.set_title(f"{name}: no feasible grid points")
            continue
        ax.vlines([0, 1],
                  [d["bound_ai_lo"], d["bound_nonai_lo"]],
                  [d["bound_ai_hi"], d["bound_nonai_hi"]],
                  color="steelblue", lw=6, alpha=0.7,
                  label="Set-identified")
        ax.plot([0, 1], [d["naive_rate_proxy_pos"],
                         d["naive_rate_proxy_neg"]], "ko",
                label="Naive (proxy)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["AI-assisted", "Not AI-assisted"])
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("P(>=1 event)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.OUT / "fig3_bounds.pdf")
    print("wrote fig3")


def main():
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    sys.exit(main())
