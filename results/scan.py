"""Pre-submission scan battery for the AISciK zombie-citation paper.

Each run executes 22 named sub-checks. A parent scan is one invocation
after a round of edits, so consecutive runs also verify that the previous
round did not break anything upstream of it.

Usage: python3 scan.py
Exit code is the number of failing checks.
"""
import json
import os
import re
import statistics as st
import sys

TEX = "paper.tex"
LOG = "paper.log"
OUT = "/Users/garyzhang/claude/ai-science-zombie-data/out"
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def ctx():
    s = open(TEX).read()
    log = open(LOG).read() if os.path.exists(LOG) else ""
    plain = re.sub(r"\\begin\{thebibliography\}.*", "", s, flags=re.S)
    plain = re.sub(r"\\begin\{(figure|table|tabular)\}.*?\\end\{\1\}", "",
                   plain, flags=re.S)
    plain = re.sub(r"\\(section|subsection)\{[^}]*\}", " . ", plain)
    plain = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", plain)
    plain = re.sub(r"\\[a-zA-Z]+", " ", plain)
    plain = re.sub(r"[{}$\\]", " ", plain)
    plain = re.sub(r"\s+", " ", plain)
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", plain)
             if len(x.split()) > 4]
    return dict(s=s, log=log, plain=plain, sents=sents,
                lens=[len(x.split()) for x in sents])


# 1-4: compliance -----------------------------------------------------------

@check("01 page count is exactly 4 or 8")
def c01(c):
    m = re.findall(r"Output written on paper.pdf \((\d+) pages", c["log"])
    n = int(m[0]) if m else 0
    return n in (4, 8), f"{n} pages"


@check("02 no TK placeholders remain")
def c02(c):
    n = c["s"].count("\\TK{")
    return n == 0, f"{n} markers"


@check("03 no undefined citations or references")
def c03(c):
    n = len(re.findall(r"(Citation|Reference) .* undefined", c["log"]))
    return n == 0, f"{n} undefined"


@check("04 no overfull hboxes")
def c04(c):
    n = len(re.findall(r"Overfull \\hbox", c["log"]))
    return n == 0, f"{n} overfull"


# 5-9: citation integrity (workshop desk-rejects hallucinated refs) ---------

@check("05 every bibitem is cited")
def c05(c):
    keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", c["s"]))
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", c["s"]):
        cited |= {k.strip() for k in m.group(1).split(",")}
    return keys <= cited, f"uncited: {sorted(keys - cited) or 'none'}"


@check("06 every citation has a bibitem")
def c06(c):
    keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", c["s"]))
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", c["s"]):
        cited |= {k.strip() for k in m.group(1).split(",")}
    return cited <= keys, f"dangling: {sorted(cited - keys) or 'none'}"


@check("07 bibliography carries resolvable identifiers")
def c07(c):
    n_items = len(re.findall(r"\\bibitem", c["s"]))
    n_ids = len(re.findall(r"doi:", c["s"])) + len(re.findall(r"arXiv:", c["s"]))
    return n_ids >= n_items - 1, f"{n_ids} ids for {n_items} items"


@check("08 no fabricated-citation key from the original draft")
def c08(c):
    return "orduna" not in c["s"].lower(), "orduna key absent"


@check("09 repo URL present for data availability")
def c09(c):
    return "github.com/garyzhang1006/ai-science-zombie-data" in c["s"], "url"


# 10-14: numeric provenance -------------------------------------------------

@check("10 population numbers match its_results.json")
def c10(c):
    its = jload("its_results.json").get("pooled", {})
    if not its:
        return False, "no its_results.json"
    ok = (f"{its['pre_mean_rate']:.3f}" in c["s"]
          and f"{its['level_change']:.3f}" in c["s"])
    return ok, f"pre={its['pre_mean_rate']:.3f} lvl={its['level_change']:.3f}"


@check("11 cohort ratios match table1.json")
def c11(c):
    t = jload("table1.json")
    if not t:
        return False, "no table1.json"
    ok = (f"{t['zombie']['ratio']:.3f}" in c["s"]
          and f"{t['unresolvable']['ratio']:.3f}" in c["s"])
    return ok, f"z={t['zombie']['ratio']:.3f} u={t['unresolvable']['ratio']:.3f}"


@check("12 adjusted odds ratio matches bounds.json")
def c12(c):
    a = jload("bounds.json").get("adjusted", {})
    if not a:
        return False, "no adjusted fit"
    ok = f"{a['proxy_odds_ratio']:.2f}" in c["s"]
    return ok, f"OR={a['proxy_odds_ratio']:.2f} p={a['proxy_p']:.3f}"


@check("13 reference-count means match bounds.json")
def c13(c):
    b = jload("bounds.json")
    u = b.get("zombie_unrestricted", {})
    if not u:
        return False, "no unrestricted fit"
    ok = (f"{u['mean_refs_proxy_pos']:.1f}" in c["s"]
          and f"{u['mean_refs_proxy_neg']:.1f}" in c["s"])
    return ok, f"{u['mean_refs_proxy_pos']:.1f} vs {u['mean_refs_proxy_neg']:.1f}"


@check("14 no claim of a clean null the sweep contradicts")
def c14(c):
    bad = ["no detectable information", "carries no detectable signal",
           "entirely explained by reference count"]
    hits = [b for b in bad if b in c["s"]]
    return not hits, f"stale claims: {hits or 'none'}"


# 15-19: prose quality ------------------------------------------------------

@check("15 abstract length inside genre band (150-260 words)")
def c15(c):
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", c["s"], re.S)
    if not a:
        return False, "no abstract"
    w = len(re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", " ", a.group(1)).split())
    return 150 <= w <= 260, f"{w} words"


@check("16 median body sentence 18-28 words")
def c16(c):
    m = st.median(c["lens"])
    return 18 <= m <= 28, f"median {m:.1f}"


@check("17 longest body sentence under 60 words")
def c17(c):
    return max(c["lens"]) < 60, f"max {max(c['lens'])}"


@check("18 no banned AI-tell vocabulary")
def c18(c):
    pat = (r"\b(delv\w*|leverag\w*|crucial|vital|pivotal|seamless\w*|holistic|"
           r"underscor\w*|showcas\w*|realm|tapestry|comprehensive|"
           r"groundbreaking|cutting-edge|it is worth noting)\b")
    hits = re.findall(pat, c["plain"], re.I)
    return not hits, f"{hits[:4] or 'clean'}"


@check("19 no em dashes or not-only-but-also")
def c19(c):
    hits = c["plain"].count("\u2014")
    hits += len(re.findall(r"\bnot only\b.{0,40}\bbut also\b", c["plain"], re.I))
    return hits == 0, f"{hits} hits"


# 20-22: structure ----------------------------------------------------------

@check("20 every figure and table is referenced in text")
def c20(c):
    bad = []
    for lab in re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", c["s"]):
        if not re.search(r"\\ref\{" + re.escape(lab) + r"\}", c["s"]):
            bad.append(lab)
    return not bad, f"unreferenced: {bad or 'none'}"


@check("21 results subsections lead with claims, not labels")
def c21(c):
    res = c["s"][c["s"].index(r"\section{Results}"):] if r"\section{Results}" in c["s"] else ""
    subs = re.findall(r"\\subsection\{([^}]+)\}", res)
    verbs = [x for x in subs if re.search(r"\b(shift|predict|measure|survive|explain|rise|fail)", x, re.I)]
    return len(verbs) == len(subs) and subs, f"{len(verbs)}/{len(subs)} claim-shaped"


@check("22 limitations names the unvalidated filter precision")
def c22(c):
    return "did not perform" in c["s"], "manual adjudication disclosed"


def main():
    c = ctx()
    fails = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(c)
        except Exception as e:
            ok, detail = False, f"ERROR {type(e).__name__}: {e}"
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")
    print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} sub-checks pass")
    return fails


if __name__ == "__main__":
    sys.exit(main())
