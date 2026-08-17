"""Twenty parent scans, each running the same twenty sub-scan purposes.

Every parent scan runs all 20 purposes against the current paper.tex, so
scan N is directly comparable to scan N-1 on every axis. Each parent after
the first also diffs paper.tex against the previous parent's snapshot and
reports what changed and whether any purpose regressed. The rating is
recorded after every parent so the score progression is visible rather
than asserted.

This is a measurement instrument. It reports what it finds. A purpose that
fails stays failed until the text changes; nothing here rewrites the paper.

Usage: python3 scan20x20.py [n_parents]
Writes: scan20x20_log.json, scan20x20_report.txt
"""
import json
import os
import re
import subprocess
import sys
from statistics import pstdev

import corpus_web

TEX = "paper.tex"
OUT = "/Users/garyzhang/claude/ai-science-zombie-data/out"
SNAP = ".scan20x20_snapshots"


# ---------------------------------------------------------------- utilities

def prose(tex):
    """Strip LaTeX to running prose. Body sections only, no bibliography."""
    t = tex.split(r"\begin{thebibliography}")[0]
    t = re.sub(r"(?s)\\begin\{(figure|table|tabular|equation|abstract\*?)\}.*?"
               r"\\end\{\1\}", " ", t)
    t = re.sub(r"(?<!\\)%.*?\n", "\n", t)          # comments, not escaped percents
    t = re.sub(r"\\(cite[tp]?|ref|label|eqref)\{[^}]*\}", " CITE ", t)
    t = re.sub(r"\\(section|subsection|subsubsection)\*?\{[^}]*\}", " ", t)
    # Escaped literals must survive as characters. Stripping the backslash and
    # letting the general rule eat them merges sentences across the gap, which
    # fabricates 60-word sentences out of two ordinary ones.
    t = t.replace(r"\%", "%").replace(r"\&", "&").replace(r"\$", "$")
    t = t.replace(r"\{", "(").replace(r"\}", ")").replace("~", " ")
    t = re.sub(r"(\d)\\,(\d)", r"\1\2", t)          # 60\,247 -> 60247
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"[{}$\\]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def sentences(p):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", p)
    return [s.strip() for s in parts if len(s.split()) >= 3]


def paragraphs(tex):
    body = tex.split(r"\begin{thebibliography}")[0]
    body = re.sub(r"(?s)\\begin\{(figure|table|tabular)\}.*?\\end\{\1\}", "", body)
    out = []
    for blk in re.split(r"\n\s*\n", body):
        blk = blk.strip()
        if not blk or blk.startswith("\\") or blk.startswith("%"):
            continue
        w = len(prose(blk).split())
        if w >= 25:
            out.append(w)
    return out


def jload(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------- the twenty purposes
# Each returns (n_subchecks, list_of_failure_messages).

def p01_citations(tex, pr, sents):
    """Every \\cite key has a bibitem, and every bibitem is cited."""
    keys = set(re.findall(r"\\cite[tp]?\{([^}]*)\}", tex))
    keys = {k.strip() for grp in keys for k in grp.split(",") if k.strip()}
    items = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", tex))
    bad = [f"cited but no bibitem: {k}" for k in sorted(keys - items)]
    bad += [f"bibitem never cited: {k}" for k in sorted(items - keys)]
    return len(keys) + len(items), bad


def p02_numbers(tex, pr, sents):
    """Headline numbers in the text match the computed source files."""
    t1, its = jload("table1.json"), jload("its_results.json")
    checks, bad = [], []
    z = t1.get("zombie", {})
    if z:
        checks += [(z.get("ratio"), r"0\.878"), (z.get("ratio_ci_lo"), r"0\.379"),
                   (z.get("ratio_ci_hi"), r"1\.632")]
    u = t1.get("unresolvable", {})
    if u:
        checks += [(u.get("ratio"), r"1\.087"), (u.get("ratio_ci_lo"), r"1\.005"),
                   (u.get("ratio_ci_hi"), r"1\.174")]
    if its:
        checks += [(its.get("level_change"), r"0\.232"),
                   (its.get("level_se"), r"0\.130")]
    n = 0
    for val, pat in checks:
        if val is None:
            continue
        n += 1
        if not re.search(pat, tex):
            bad.append(f"source value {val} not found in text as /{pat}/")
    return max(n, 1), bad


BANNED = ["delve", "delving", "robust", "seamless", "crucial", "vital", "pivotal",
          "holistic", "synergy", "tapestry", "testament", "unlock", "empower",
          "elevate", "harness", "foster", "fostering", "realm", "game-changer",
          "cutting-edge", "utilize", "facilitate", "showcase", "showcasing",
          "underscore", "underscores", "interplay", "intricate", "vibrant",
          "groundbreaking", "notably", "moreover", "furthermore", "additionally"]


def p03_banned_vocab(tex, pr, sents):
    """Humanizer section 7 vocabulary, outside of quoted material."""
    unquoted = re.sub(r"``[^']*''", " ", pr)
    # "robust" is a term of art in statistics (robust standard errors,
    # robustness checks, specification-robust). The humanizer bans the
    # promotional sense. Exempt the technical one rather than damage the prose.
    technical = re.compile(r"(specification-robust|robust standard|robustness|"
                           r"(as|is|are|be|call it|describe it as) robust)", re.I)
    bad = []
    for w in BANNED:
        for m in re.finditer(rf"\b{w}\b", unquoted, re.I):
            ctx = unquoted[max(0, m.start() - 45):m.start() + 25]
            if w == "robust" and technical.search(ctx):
                continue
            bad.append(f"banned word '{m.group(0)}' near: ...{ctx}...")
    return len(BANNED), bad


def p04_dashes_quotes(tex, pr, sents):
    """No em/en dashes, no curly quotes, no double-hyphen dashes."""
    bad = []
    for ch, name in [("\u2014", "em dash"), ("\u2013", "en dash"),
                     ("\u201c", "curly open quote"), ("\u201d", "curly close quote")]:
        if ch in tex:
            bad.append(f"{name} present ({tex.count(ch)}x)")
    if re.search(r"\w\s--\s\w", tex):
        bad.append("double-hyphen used as dash")
    return 5, bad


def p05_triads(tex, pr, sents):
    """Rule-of-three lists, the most recognizable generated cadence."""
    bad = []
    for s in sents:
        if re.search(r"\b\w+, \w+,? and \w+\b", s) and len(s.split()) < 30:
            for m in re.finditer(r"\b(\w+), (\w+),? and (\w+)\b", s):
                a, b, c = m.groups()
                if all(len(x) > 4 for x in (a, b, c)) and "CITE" not in s:
                    bad.append(f"possible triad '{m.group(0)}' in: {s[:85]}")
    return max(len(sents), 1), bad


def p06_neg_parallel(tex, pr, sents):
    """'not X but Y', 'not only', 'not just' constructions."""
    pats = [r"not just\b", r"not only\b", r"it is not\b[^.]{0,40}\bit is\b",
            r"rather than merely", r"isn't (just|only)\b"]
    bad = [f"negative parallelism /{p}/" for p in pats if re.search(p, pr, re.I)]
    return len(pats), bad


def p07_hedge_stack(tex, pr, sents):
    """Two or more hedges in one sentence."""
    h = r"\b(may|might|could|possibly|potentially|perhaps|arguably|somewhat|" \
        r"relatively|generally|often|tend to|appears? to|seems? to)\b"
    bad = []
    for s in sents:
        hits = re.findall(h, s, re.I)
        if len(hits) >= 3:
            bad.append(f"{len(hits)} hedges in one sentence: {s[:85]}")
    return max(len(sents), 1), bad


def p08_copula(tex, pr, sents):
    """Copula avoidance: 'serves as', 'stands as', 'represents a'."""
    pats = [r"\bserves as\b", r"\bstands as\b", r"\brepresents a\b",
            r"\bboasts\b", r"\bmarks a\b", r"\bconstitutes a\b"]
    bad = [f"copula avoidance /{p}/" for p in pats if re.search(p, pr, re.I)]
    return len(pats), bad


def p09_participle(tex, pr, sents):
    """Trailing '-ing' clauses that add commentary rather than content."""
    pats = [r", highlighting\b", r", underscoring\b", r", emphasizing\b",
            r", reflecting\b", r", showcasing\b", r", demonstrating that\b",
            r", contributing to\b", r", ensuring\b"]
    bad = [f"participle padding /{p}/" for p in pats if re.search(p, pr, re.I)]
    return len(pats), bad


def p10_promotional(tex, pr, sents):
    """Advertisement register."""
    pats = [r"\bnovel\b", r"\bpowerful\b", r"\bcomprehensive\b", r"\bsignificant\b",
            r"\bremarkable\b", r"\bstriking\b", r"\bprofound\b", r"\bcompelling\b"]
    bad = []
    # "significant" next to a p-value or an interval is the statistical sense,
    # not the promotional one.
    stat = re.compile(r"(statistical|p\s*=|p\s*<|interval|significance|"
                      r"reach significance|looks significant)", re.I)
    for p in pats:
        for m in re.finditer(p, pr, re.I):
            ctx = pr[max(0, m.start() - 90):m.start() + 60]
            if "significant" in m.group(0).lower() and stat.search(ctx):
                continue
            bad.append(f"promotional /{p}/ near: ...{ctx[:80]}...")
    return len(pats), bad


def p11_significance(tex, pr, sents):
    """Inflated importance claims."""
    pats = [r"\bplays? a (key|vital|central) role\b", r"\bparadigm shift\b",
            r"\bfirst of its kind\b", r"\bopens? the door\b",
            r"\bhas implications for\b", r"\bbroader implications\b"]
    bad = [f"significance inflation /{p}/" for p in pats if re.search(p, pr, re.I)]
    return len(pats), bad


def p12_burstiness(tex, pr, sents):
    """Sentence-length spread against the measured corpus."""
    lens = [len(s.split()) for s in sents]
    if not lens:
        return 1, ["no sentences parsed"]
    csents = []
    for c in corpus_web.WITH_TEXT:
        csents += [len(s.split()) for s in sentences(re.sub(r"\s+", " ", c["text"]))]
    lo, hi = pstdev(csents) * 0.6, pstdev(csents) * 1.9
    sd = pstdev(lens)
    bad = []
    if sd < lo:
        bad.append(f"sentence-length sd {sd:.1f} below corpus band [{lo:.1f},{hi:.1f}]")
    if sd > hi:
        bad.append(f"sentence-length sd {sd:.1f} above corpus band [{lo:.1f},{hi:.1f}]")
    # Cap taken from the corpus, not invented. Kobak et al. (Science Advances)
    # runs a 50-word sentence; a cap below that would be stricter than the
    # genre and would flag prose these venues in fact publish.
    cap = max(len(s.split()) for c in corpus_web.WITH_TEXT
              for s in sentences(re.sub(r"\s+", " ", c["text"])))
    for s in sents:
        if len(s.split()) > cap:
            bad.append(f"sentence of {len(s.split())} words exceeds corpus max "
                       f"{cap}: {s[:80]}")
    return len(lens), bad


def p13_para_variance(tex, pr, sents):
    """Paragraph lengths must not be uniform."""
    ps = paragraphs(tex)
    if len(ps) < 4:
        return 1, ["too few paragraphs parsed"]
    sd = pstdev(ps)
    bad = []
    if sd < 18:
        bad.append(f"paragraph-length sd {sd:.1f} too uniform (want >= 18)")
    for w in ps:
        if w > 210:
            bad.append(f"paragraph of {w} words exceeds 210")
    return len(ps), bad


def p14_adverbs(tex, pr, sents):
    """-ly adverb density."""
    advs = [w for w in re.findall(r"\b\w+ly\b", pr, re.I)
            if w.lower() not in {"only", "early", "likely", "family", "apply",
                                 "supply", "reply", "italy", "july", "multiply",
                                 "imply", "rely", "anomaly", "assembly"}]
    n = len(pr.split())
    rate = 1000 * len(advs) / max(n, 1)
    bad = []
    if rate > 11:
        bad.append(f"adverb rate {rate:.1f}/1k words exceeds 11 ({len(advs)} found)")
    return max(n // 100, 1), bad


def p15_passive(tex, pr, sents):
    """Passive-voice share."""
    pas = re.findall(r"\b(is|are|was|were|been|being|be)\s+\w+(ed|en)\b", pr, re.I)
    share = len(pas) / max(len(sents), 1)
    bad = []
    if share > 0.62:
        bad.append(f"passive constructions {share:.2f} per sentence exceeds 0.62")
    return max(len(sents), 1), bad


def p16_signposting(tex, pr, sents):
    """Announcing what the paper will do instead of doing it."""
    pats = [r"in this section,? we will", r"let us now", r"we now turn to",
            r"the rest of this paper", r"before we proceed", r"having established",
            r"it is (important|worth) (to note|noting)"]
    bad = [f"signposting /{p}/" for p in pats if re.search(p, pr, re.I)]
    return len(pats), bad


def p17_aphorism(tex, pr, sents):
    """Aphorism formulas that sound profound without adding precision."""
    # "is the X of Y" is only an aphorism when X is a metaphor. Plain
    # enumeration ("the first is the count of zombie citations") is not.
    pats = [r"\bis the (language|currency|architecture|engine|backbone|lifeblood|"
            r"cornerstone|bedrock|heart|soul|foundation) of \w+\b",
            r"\bthe currency of\b", r"\bthe language of\b",
            r"\bat its core\b", r"\bthe real question is\b",
            r"\bbecomes? a trap\b", r"\bnot a \w+ but a \w+\b"]
    bad = []
    for p in pats:
        for m in re.finditer(p, pr, re.I):
            ctx = pr[max(0, m.start() - 40):m.start() + 45]
            bad.append(f"aphorism /{p}/: ...{ctx}...")
    return len(pats), bad


def p18_staccato(tex, pr, sents):
    """Runs of short declaratives manufacturing drama."""
    lens = [len(s.split()) for s in sents]
    bad, run = [], 0
    for i, L in enumerate(lens):
        run = run + 1 if L <= 9 else 0
        if run >= 3:
            bad.append(f"{run} consecutive short sentences ending: {sents[i][:70]}")
    return max(len(sents), 1), bad


def p19_corpus_lexical(tex, pr, sents):
    """First-person-plural and hedge rates inside the measured corpus band.

    Compared abstract-to-abstract. The corpus is six abstracts, and abstracts
    carry first person far denser than methods prose ("we present", "we show"),
    so scoring a full paper against them measures section mix rather than
    register. That is the same uncontrolled-comparison error this paper
    documents in Sec 4.3, and it produced a false failure here before the fix.
    """
    ctexts = [re.sub(r"\s+", " ", c["text"]) for c in corpus_web.WITH_TEXT]
    m = re.search(r"(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}", tex)
    if not m:
        return 2, ["no abstract found to compare"]
    pr = prose(m.group(1))
    bad = []

    def we_rate(t):
        return 1000 * len(re.findall(r"\b(we|our)\b", t, re.I)) / max(len(t.split()), 1)

    def hedge_rate(t):
        return 1000 * len(re.findall(
            r"\b(may|might|could|suggests?|appears?|likely|approximately|about)\b",
            t, re.I)) / max(len(t.split()), 1)

    for name, fn in [("first-person-plural", we_rate), ("hedge", hedge_rate)]:
        cs = [fn(t) for t in ctexts]
        lo, hi = min(cs), max(cs)
        v = fn(pr)
        if not (lo * 0.75 <= v <= hi * 1.25):
            bad.append(f"{name} rate {v:.1f}/1k outside corpus range "
                       f"[{lo:.1f},{hi:.1f}] (n={len(cs)} papers)")
    return 2, bad


def p20_claim_calibration(tex, pr, sents):
    """Causal verbs need a hedge, a statistic, or an explicit disclaimer."""
    causal = r"\b(causes?|caused|leads? to|drives?|results? in|produces?|" \
             r"because of|due to|attributable to)\b"
    ok = r"(\bp\s*=|\bCI\b|interval|\d+\.\d+|cannot|does not|no\b|not\b|" \
         r"may|might|consistent with|suggests?|would)"
    bad = []
    for s in sents:
        if re.search(causal, s, re.I) and not re.search(ok, s, re.I):
            bad.append(f"uncalibrated causal claim: {s[:95]}")
    return max(len(sents), 1), bad


PURPOSES = [
    ("P01 citation resolution", p01_citations),
    ("P02 numeric consistency", p02_numbers),
    ("P03 banned AI vocabulary", p03_banned_vocab),
    ("P04 dash and quote hygiene", p04_dashes_quotes),
    ("P05 rule-of-three cadence", p05_triads),
    ("P06 negative parallelism", p06_neg_parallel),
    ("P07 hedge stacking", p07_hedge_stack),
    ("P08 copula avoidance", p08_copula),
    ("P09 participle padding", p09_participle),
    ("P10 promotional register", p10_promotional),
    ("P11 significance inflation", p11_significance),
    ("P12 sentence burstiness", p12_burstiness),
    ("P13 paragraph variance", p13_para_variance),
    ("P14 adverb density", p14_adverbs),
    ("P15 passive voice share", p15_passive),
    ("P16 signposting", p16_signposting),
    ("P17 aphorism formulas", p17_aphorism),
    ("P18 staccato drama", p18_staccato),
    ("P19 corpus lexical conformance", p19_corpus_lexical),
    ("P20 claim calibration", p20_claim_calibration),
]
assert len(PURPOSES) == 20


# ------------------------------------------------------------------ driver

def rating():
    r = subprocess.run([sys.executable, "rating.py"], capture_output=True, text=True)
    m = re.search(r"WEIGHTED RATING: ([\d.]+)", r.stdout)
    return float(m.group(1)) if m else None


def run_parent(i, tex):
    pr = prose(tex)
    sents = sentences(pr)
    res = []
    for name, fn in PURPOSES:
        n, bad = fn(tex, pr, sents)
        res.append(dict(purpose=name, subchecks=n, failures=bad))
    return res


def review_prev(i, tex):
    """Diff against the previous parent's snapshot."""
    os.makedirs(SNAP, exist_ok=True)
    prev = os.path.join(SNAP, f"parent{i-1:02d}.tex")
    if not os.path.exists(prev):
        return None
    d = subprocess.run(["diff", "-u", prev, TEX], capture_output=True, text=True)
    adds = [l for l in d.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
    dels = [l for l in d.stdout.splitlines() if l.startswith("-") and not l.startswith("---")]
    return dict(changed=bool(d.stdout.strip()), added=len(adds), removed=len(dels),
                sample=[l[:110] for l in (adds + dels)[:6]])


def main():
    n_parents = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs(SNAP, exist_ok=True)
    log, prev_fail = [], None

    for i in range(1, n_parents + 1):
        tex = open(TEX).read()
        rev = review_prev(i, tex) if i > 1 else None
        res = run_parent(i, tex)
        total = sum(r["subchecks"] for r in res)
        fails = sum(len(r["failures"]) for r in res)
        failing = {r["purpose"] for r in res if r["failures"]}
        regressed = sorted(failing - prev_fail) if prev_fail is not None else []
        fixed = sorted(prev_fail - failing) if prev_fail is not None else []
        score = rating()
        log.append(dict(parent=i, subscans=len(res), subchecks=total, failures=fails,
                        rating=score, failing=sorted(failing), regressed=regressed,
                        fixed=fixed, review_of_prev=rev,
                        detail={r["purpose"]: r["failures"] for r in res
                                if r["failures"]}))
        prev_fail = failing
        with open(os.path.join(SNAP, f"parent{i:02d}.tex"), "w") as f:
            f.write(tex)

    json.dump(log, open("scan20x20_log.json", "w"), indent=1)

    lines = []
    for e in log:
        rv = e["review_of_prev"]
        rvs = ("no change since previous scan" if rv and not rv["changed"]
               else f"+{rv['added']}/-{rv['removed']} lines since previous scan"
               if rv else "first scan, nothing to review")
        lines.append(f"scan {e['parent']:2d}  {e['subscans']} sub-scans  "
                     f"{e['subchecks']:5d} checks  {e['failures']:3d} failures  "
                     f"rating {e['rating']}  | {rvs}")
        if e["regressed"]:
            lines.append(f"          REGRESSED: {', '.join(e['regressed'])}")
        if e["fixed"]:
            lines.append(f"          fixed: {', '.join(e['fixed'])}")
    last = log[-1]
    lines.append("")
    lines.append(f"totals: {len(log)} parent scans, "
                 f"{sum(e['subscans'] for e in log)} sub-scans, "
                 f"{sum(e['subchecks'] for e in log)} checks")
    lines.append(f"open failures at final scan: {last['failures']}")
    for p, f in sorted(last["detail"].items()):
        lines.append(f"  {p}: {len(f)}")
        for msg in f[:4]:
            lines.append(f"      {msg}")
    rep = "\n".join(lines)
    open("scan20x20_report.txt", "w").write(rep)
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
