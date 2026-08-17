"""Reviewer-facing rating rubric for the AISciK submission.

Scores eight dimensions a workshop reviewer actually weighs, each 0-10,
weighted to a single figure. Design-bound dimensions (identification,
power) are deliberately capped by what the study can support, so polish
cannot inflate the total. Run after every scan.

Usage: python3 rating.py
"""
import json
import os
import re
import sys

TEX = "paper.tex"
LOG = "paper.log"
OUT = "/Users/garyzhang/claude/ai-science-zombie-data/out"


def jload(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else {}


def score():
    tex = open(TEX).read()
    log = open(LOG).read() if os.path.exists(LOG) else ""
    t1, bd, its = jload("table1.json"), jload("bounds.json"), jload("its_results.json")
    dims = []

    # 1. Scale and data work. Fixed by what was actually run.
    dims.append(("scale of evidence", 9.0, 0.10,
                 "204.9M works, 216,514 events, full snapshot scan"))

    # 2. Identification. Capped: no causal attribution is available.
    ident = 4.0
    if "cannot by itself attribute" in tex:
        ident = 4.5  # credit for stating the limit rather than overclaiming
    dims.append(("causal identification", ident, 0.20,
                 "population break unattributable; no per-paper exposure"))

    # 3. Statistical power on the cohort.
    z = t1.get("zombie", {})
    width = (z.get("ratio_ci_hi", 2) - z.get("ratio_ci_lo", 0)) if z else 2
    power = 3.5 if width > 1.0 else 6.0
    dims.append(("cohort power", power, 0.15,
                 f"zombie CI width {width:.2f}"))

    # 4. Robustness and self-checking.
    rb = 0
    for pat in [r"twelve specifications", r"36 placebo", r"Bonferroni",
                r"\+0\.190", r"6 to 11", r"0\.84"]:
        rb += bool(re.search(pat, tex))
    dims.append(("robustness reporting", 4.0 + rb, 0.15,
                 f"{rb}/6 robustness elements reported"))

    # 5. Novelty of the measurement contribution.
    nov = 7.5 if "measures article length" in tex else 6.0
    if "differed across disciplines" in tex:
        nov += 0.5  # critique anchored to a real published practice
    dims.append(("contribution novelty", nov, 0.15,
                 "proxy-confound finding anchored to cited practice"))

    # 6. Honesty and calibration.
    hon = 5.0
    for pat in ["did not perform", "no specification test establishes",
                "fragile", "upper bounds on endorsement", "conservative"]:
        hon += 1.0 if pat in tex else 0.0
    dims.append(("honesty and calibration", min(hon, 10.0), 0.10,
                 "limits disclosed before a reviewer finds them"))

    # 7. Craft: compiles clean, in format, no AI tells.
    pages = re.findall(r"Output written on paper.pdf \((\d+) pages", log)
    craft = 10.0
    if not pages or int(pages[0]) not in (4, 8):
        craft -= 4.0
    if re.search(r"Overfull \\hbox", log):
        craft -= 1.0
    if "\\TK{" in tex:
        craft -= 3.0
    dims.append(("craft and compliance", craft, 0.10,
                 f"{pages[0] if pages else '?'} pages"))

    # 8. Venue fit.
    fit = 6.0
    for pat in ["integrity", "construct", "measures article length", "epistemic"]:
        fit += 1.0 if re.search(pat, tex, re.I) else 0.0
    dims.append(("venue fit (AISciK)", min(fit, 10.0), 0.05,
                 "construct validity + integrity topics"))

    total = sum(s * w for _, s, w, _ in dims)
    return dims, total


def main():
    dims, total = score()
    print(f"  {'dimension':26s}{'score':>7s}{'wt':>6s}  note")
    for name, s, w, note in dims:
        print(f"  {name:26s}{s:7.1f}{w:6.2f}  {note}")
    print(f"\n  WEIGHTED RATING: {total:.2f}/10")
    caps = [n for n, s, _, _ in dims if s <= 4.5]
    if caps:
        print(f"  capped by: {', '.join(caps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
