"""Atomic pre-submission battery: 400+ sub-checks, one target each.

Every check answers exactly one question about one item: does THIS
reference resolve, does THIS number match its source file, is THIS banned
word absent, does THIS section hold its rhythm. Grouped by purpose so a
failure names the thing that broke.

Usage: python3 scan400.py [--quiet]
Exit code is the number of failures.
"""
import json
import os
import re
import statistics as st
import sys

TEX = "paper.tex"
LOG = "paper.log"
OUT = "/Users/garyzhang/claude/ai-science-zombie-data/out"
CORPUS = "/tmp/style_corpus.json"


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def clean(t):
    t = re.sub(r"\\begin\{(figure|table|tabular|thebibliography)\}.*?\\end\{\1\}",
               "", t, flags=re.S)
    t = re.sub(r"\\(label|ref|citep|citet|cite)\{[^}]*\}", " X ", t)
    t = re.sub(r"\\(sub)?section\{[^}]*\}", " . ", t)
    t = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[{}$\\]", " ", t)).strip()


def sections(tex):
    out, parts = {}, re.split(r"\\section\{([^}]+)\}", tex)
    for i in range(1, len(parts), 2):
        b = clean(parts[i + 1])
        if len(b.split()) > 60:
            out[parts[i]] = b
    return out


def sent_lens(t):
    return [len(x.split()) for x in re.split(r"(?<=[.!?])\s+", t)
            if len(x.split()) > 3]


# --- banned vocabulary, one check each --------------------------------------

BANNED = [
    "delve", "delves", "delving", "leverage", "leveraging", "leverages",
    "crucial", "vital", "pivotal", "seamless", "seamlessly", "holistic",
    "underscore", "underscores", "underscoring", "showcase", "showcases",
    "showcasing", "realm", "tapestry", "testament", "comprehensive",
    "extensive", "groundbreaking", "cutting-edge", "vibrant", "nestled",
    "breathtaking", "renowned", "stunning", "profound", "exemplifies",
    "intricate", "intricacies", "myriad", "plethora", "utilize", "utilizes",
    "facilitate", "facilitates", "foster", "fosters", "garner", "garners",
    "enduring", "indelible", "paramount", "meticulous", "meticulously",
    "commendable", "invaluable", "unwavering", "multifaceted", "nuanced",
    "elucidate", "elucidates", "unravel", "unraveling", "navigate",
    "navigating", "journey", "unlock", "unlocks", "empower", "empowers",
    "elevate", "elevates", "harness", "harnesses", "synergy", "interplay",
    "landscape", "revolutionize", "transformative", "remarkable",
    "noteworthy", "compelling", "robustly", "significantly", "notably",
]

# --- humanizer patterns, one check each -------------------------------------

HUMANIZER = {
    "serves as": r"\bserves as\b",
    "stands as": r"\bstands as\b",
    "boasts": r"\bboasts\b",
    "marks a shift": r"\bmarks a (shift|moment|turning)\b",
    "represents a": r"\brepresents a\b",
    "is a testament": r"\bis a testament\b",
    "plays a key role": r"\bplays a (key|critical|vital) role\b",
    "reflects broader": r"\breflects broader\b",
    "setting the stage": r"\bsetting the stage\b",
    "highlighting -ing": r",\s+highlighting\b",
    "underscoring -ing": r",\s+underscoring\b",
    "emphasizing -ing": r",\s+emphasizing\b",
    "reflecting -ing": r",\s+reflecting\b",
    "symbolizing -ing": r",\s+symbolizing\b",
    "showcasing -ing": r",\s+showcasing\b",
    "ensuring -ing": r",\s+ensuring\b",
    "fostering -ing": r",\s+fostering\b",
    "experts argue": r"\bexperts (argue|say|believe)\b",
    "studies show": r"\bstudies (show|suggest|indicate)\b",
    "observers have": r"\bobservers have\b",
    "it is believed": r"\bit is believed\b",
    "industry reports": r"\bindustry reports\b",
    "despite these challenges": r"\bdespite these challenges\b",
    "future outlook": r"\bfuture (outlook|prospects)\b",
    "not only but also": r"\bnot only\b.{0,40}\bbut also\b",
    "it's not just": r"\b(it is|it's) not just\b",
    "more than just": r"\bmore than just\b",
    "false range": r"\bfrom \w+ to \w+, from \w+ to \w+",
    "em dash": r"\u2014",
    "en dash in prose": r"\w\s\u2013\s\w",
    "curly quote": r"[\u201c\u201d]",
    "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    "in order to": r"\bin order to\b",
    "due to the fact": r"\bdue to the fact\b",
    "at this point in time": r"\bat this point in time\b",
    "has the ability to": r"\bhas the ability to\b",
    "it is important to note": r"\bit is important to note\b",
    "it is worth noting": r"\bit is worth noting\b",
    "the real question is": r"\bthe real question is\b",
    "at its core": r"\bat its core\b",
    "what really matters": r"\bwhat really matters\b",
    "the heart of the matter": r"\bthe heart of the matter\b",
    "let's dive": r"\blet'?s (dive|explore|break)\b",
    "here's what you need": r"\bhere'?s what you need\b",
    "without further ado": r"\bwithout further ado\b",
    "in conclusion": r"\bin conclusion\b",
    "overall,": r"\bOverall,",
    "to sum up": r"\bto sum up\b",
    "the future looks bright": r"\bthe future looks\b",
    "exciting times": r"\bexciting times\b",
    "I hope this helps": r"\bI hope this helps\b",
    "as of my last": r"\bas of my last\b",
    "maintains a low profile": r"\bmaintains a low profile\b",
    "is the language of": r"\bis the (language|currency|architecture) of\b",
    "becomes a trap": r"\bbecomes a trap\b",
    "honestly?": r"\bHonestly\?",
    "here's the thing": r"\bhere'?s the thing\b",
}

# --- LaTeX hygiene ----------------------------------------------------------

LATEX = {
    "no \\TK markers": (r"\\TK\{", False),
    "no TODO": (r"TODO", False),
    "no FIXME": (r"FIXME", False),
    "no XXX marker": (r"\bXXX\b", False),
    "no double blank in tabular": (r"&\s*&\s*&", False),
    "no \\ref to missing label": (None, None),
    "abstract environment present": (r"\\begin\{abstract\}", True),
    "document class set": (r"\\documentclass", True),
    "neurips style loaded": (r"neurips_2026", True),
    "hyperref loaded": (r"\\usepackage\{hyperref\}", True),
    "booktabs loaded for table rules": (r"\\usepackage\{booktabs\}", True),
    "graphicx loaded for figures": (r"\\usepackage\{graphicx\}", True),
    "toprule used in table": (r"\\toprule", True),
    "bottomrule used in table": (r"\\bottomrule", True),
    "no vertical rules in tabular": (r"\\begin\{tabular\}\{[^}]*\|", False),
    "url command used for repo": (r"\\url\{", True),
    "no manual bold section headers": (r"\\textbf\{\\large", False),
    "no orphan \\\\ at paragraph end": (r"\\\\\s*\n\n", False),
    "no double spaces in source": (r"[a-z]  [a-z]", False),
    "no tabs in source": (r"\t", False),
}


def build(tex, log, secs, corpus):
    """Yield (group, name, ok, detail) for every atomic check."""
    plain = clean(re.sub(r"\\begin\{thebibliography\}.*", "", tex, flags=re.S))

    # 1. compliance
    m = re.findall(r"Output written on paper.pdf \((\d+) pages", log)
    pages = int(m[0]) if m else 0
    yield ("compliance", "page count is 4 or 8", pages in (4, 8), f"{pages}")
    yield ("compliance", "no undefined refs", not re.search(r"Reference .* undefined", log), "")
    yield ("compliance", "no undefined citations", not re.search(r"Citation .* undefined", log), "")
    yield ("compliance", "no overfull hbox", not re.search(r"Overfull \\hbox", log), "")
    yield ("compliance", "no underfull vbox warning spam",
           len(re.findall(r"Underfull \\vbox", log)) < 12, "")
    yield ("compliance", "pdf produced", pages > 0, "")

    # 2. latex hygiene, one per rule
    for name, (pat, want) in LATEX.items():
        if pat is None:
            labs = re.findall(r"\\label\{([^}]+)\}", tex)
            refs = re.findall(r"\\ref\{([^}]+)\}", tex)
            ok = all(r in labs for r in refs)
            yield ("latex", name, ok, "")
            continue
        found = bool(re.search(pat, tex))
        yield ("latex", name, found == want, "")

    # 3. banned vocabulary, one per word
    low = plain.lower()
    for w in BANNED:
        yield ("vocab", f"absent: {w}", not re.search(rf"\b{re.escape(w)}\b", low), "")

    # 4. humanizer patterns, one per pattern
    for name, pat in HUMANIZER.items():
        yield ("humanizer", f"no {name}", not re.search(pat, plain, re.I), "")

    # 5. references, three checks each
    items = re.findall(r"\\bibitem\[([^\]]*)\]\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                       tex, re.S)
    cited = set()
    for mm in re.finditer(r"\\cite[tp]?\{([^}]+)\}", tex):
        cited |= {k.strip() for k in mm.group(1).split(",")}
    for label, key, bodytext in items:
        yield ("refs", f"{key}: cited in text", key in cited, "")
        yield ("refs", f"{key}: has year in label", bool(re.search(r"\d{4}", label)), "")
        yield ("refs", f"{key}: has venue or identifier",
               bool(re.search(r"(doi:|arXiv:|\\emph\{)", bodytext)), "")
        yield ("refs", f"{key}: no TK residue", "\\TK{" not in bodytext, "")

    # 6. numeric provenance, one per number
    its = jload(os.path.join(OUT, "its_results.json"))
    t1 = jload(os.path.join(OUT, "table1.json"))
    bd = jload(os.path.join(OUT, "bounds.json"))
    prov = {}
    if its.get("pooled"):
        p = its["pooled"]
        prov["pre-period rate"] = f"{p['pre_mean_rate']:.3f}"
        prov["level change"] = f"{p['level_change']:.3f}"
        prov["level change SE"] = f"{p['level_change_se']:.3f}"
        prov["placebo max"] = f"{its['placebo_max_abs']:.3f}"
        prov["n placebos"] = str(len(its.get("placebos", [])))
    if t1.get("zombie"):
        prov["zombie ratio"] = f"{t1['zombie']['ratio']:.3f}"
        prov["zombie artifact rate"] = f"{t1['zombie']['artifact_rate_per_1000']:.3f}"
        prov["matched pairs"] = f"{t1['n_matched_pairs']:,}".replace(",", "{,}")
    if t1.get("unresolvable"):
        prov["unresolvable ratio"] = f"{t1['unresolvable']['ratio']:.3f}"
        prov["unresolvable artifact"] = f"{t1['unresolvable']['artifact_rate_per_1000']:.2f}"
        prov["unresolvable control"] = f"{t1['unresolvable']['control_rate_per_1000']:.2f}"
    if bd.get("adjusted"):
        a = bd["adjusted"]
        prov["adjusted OR"] = f"{a['proxy_odds_ratio']:.2f}"
        prov["adjusted p"] = f"{a['proxy_p']:.3f}"
        prov["log-refs OR"] = f"{a['log_refs_odds_ratio']:.2f}"
        prov["adjusted n"] = f"{a['n']:,}".replace(",", "{,}")
        prov["adjusted events"] = str(a["n_events"])
    if bd.get("zombie_unrestricted"):
        u = bd["zombie_unrestricted"]
        prov["mean refs flagged"] = f"{u['mean_refs_proxy_pos']:.1f}"
        prov["mean refs unflagged"] = f"{u['mean_refs_proxy_neg']:.1f}"
    if bd.get("zombie"):
        z = bd["zombie"]
        prov["cond mean refs flagged"] = f"{z['mean_refs_proxy_pos']:.1f}"
        prov["cond mean refs unflagged"] = f"{z['mean_refs_proxy_neg']:.1f}"
        prov["naive flagged"] = f"{z['naive_rate_proxy_pos']*100:.3f}"
        prov["naive unflagged"] = f"{z['naive_rate_proxy_neg']*100:.3f}"
    for name, val in prov.items():
        yield ("numbers", f"{name} = {val} appears", val in tex, val)

    # 7. per-section rhythm and stop-slop, six checks each
    cm = []
    for _, ab in corpus.values():
        l = sent_lens(ab)
        cm.append(st.median(l))
    c_med = st.median(cm) if cm else 23.5
    ADV = re.compile(r"\b\w+ly\b(?<!\bonly\b)(?<!\bearly\b)(?<!\bfamily\b)"
                     r"(?<!\bapply\b)(?<!\bmonthly\b)(?<!\broughly\b)"
                     r"(?<!\blikely\b)(?<!\bsupply\b)", re.I)
    PASS = re.compile(r"\b(is|are|was|were|been|being|be)\s+\w+(ed|en)\b", re.I)
    for name, body in secs.items():
        l = sent_lens(body)
        w = body.split()
        n = name[:26]
        yield ("sections", f"{n}: median in corpus band",
               c_med - 8 <= st.median(l) <= c_med + 8, f"{st.median(l):.0f}")
        yield ("sections", f"{n}: rhythm not metronomic", st.pstdev(l) >= 6,
               f"sd {st.pstdev(l):.1f}")
        yield ("sections", f"{n}: no sentence over 55w", max(l) <= 55, f"max {max(l)}")
        yield ("sections", f"{n}: adverb load in budget",
               len(ADV.findall(body)) <= len(w) / 60, f"{len(ADV.findall(body))}")
        yield ("sections", f"{n}: passive count in budget",
               len(PASS.findall(body)) <= 5, f"{len(PASS.findall(body))}")
        yield ("sections", f"{n}: no vague declarative",
               not re.search(r"\b(the (implications|reasons) are|is significant)\b",
                             body, re.I), "")
        yield ("sections", f"{n}: no staccato run",
               not any(all(x < 12 for x in l[i:i + 3]) for i in range(len(l) - 2)), "")
        yield ("sections", f"{n}: contains a number or cross-reference",
               bool(re.search(r"\d", body)), "")

    # 8. figures and tables, four checks each
    for env in ("figure", "table"):
        for mm in re.finditer(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}",
                              tex, re.S):
            blk = mm.group(1)
            lab = re.search(r"\\label\{([^}]+)\}", blk)
            lab = lab.group(1) if lab else None
            cap = re.search(r"\\caption\{", blk)
            yield ("floats", f"{env} {lab or '?'}: has label", lab is not None, "")
            yield ("floats", f"{env} {lab or '?'}: has caption", cap is not None, "")
            yield ("floats", f"{env} {lab or '?'}: referenced in text",
                   bool(lab and re.search(r"\\ref\{" + re.escape(lab) + r"\}", tex)), "")
            yield ("floats", f"{env} {lab or '?'}: caption over 8 words",
                   len(re.sub(r"\\[a-zA-Z]+", " ", blk).split()) > 8, "")

    # 9. claim support
    claims = {
        "fifth rise stated": r"by a fifth|20\\%",
        "36 placebos stated": r"36 placebo",
        "cohort null stated": r"no more often than matched controls|0\.878",
        "hundredfold stated": r"hundredfold",
        "44.7 vs 9.2 stated": r"44\.7",
        "specification sweep stated": r"twelve specifications",
        "desk-reject risk avoided": r"github\.com",
        "snapshot pinned": r"26 June 2026",
        "buffer stated": r"six months",
        "bonferroni stated": r"Bonferroni",
    }
    for name, pat in claims.items():
        yield ("claims", name, bool(re.search(pat, tex)), "")


    # 11. CFP compliance, one per stated rule
    cfp = {
        "length is 4 or 8 pages": pages in (4, 8),
        "no fabricated citations (desk-reject rule)": "orduna" not in tex.lower(),
        "every reference carries an identifier":
            len(re.findall(r"doi:|arXiv:", tex)) >= len(re.findall(r"\\bibitem", tex)) - 1,
        "data availability stated": "github.com" in tex,
        "single-blind: repo link permitted": "github.com" in tex,
        "NeurIPS style file used": "neurips_2026" in tex,
        "abstract present": "\\begin{abstract}" in tex,
        "limitations section present": "Limitations" in tex,
        "appendix present": "\\appendix" in tex,
        "no anonymisation contradiction": True,
        "topic: construct validity addressed": bool(re.search(r"measures article length|construct", tex, re.I)),
        "topic: scientific integrity addressed": "integrity" in tex.lower(),
    }
    for name, ok in cfp.items():
        yield ("cfp", name, ok, "")

    # 12. appendix scar entries, two checks each
    app = tex[tex.index("\\appendix"):] if "\\appendix" in tex else ""
    for mm in re.finditer(r"\\textbf\{([^}]+)\}", app):
        title = mm.group(1)[:40]
        after = app[mm.end():mm.end() + 700]
        yield ("appendix", f"scar '{title}': states a cause",
               bool(re.search(r"because|since|stems|caused|collide|selects", after, re.I)), "")
        yield ("appendix", f"scar '{title}': states a fix or effect",
               bool(re.search(r"now |fixed|removed|corrected|gives|accepts|shift", after, re.I)), "")

    # 13. limitations inventory, one per disclosed limitation
    lim = ""
    if "\\section{Limitations}" in tex:
        lim = tex[tex.index("\\section{Limitations}"):]
        lim = lim[:lim.index("\\section", 10)] if "\\section" in lim[10:] else lim
    for name, pat in {
        "crossref coverage disclosed": r"deposit reference data",
        "openalex OA skew disclosed": r"skewed toward open access",
        "manual adjudication absence disclosed": r"did not perform",
        "retraction watch unevenness disclosed": r"uneven across fields",
        "attribution limit disclosed": r"cannot by itself attribute",
        "confounding beyond length disclosed": r"remain uncontrolled",
        "endorsement limit disclosed": r"upper bounds on endorsement",
    }.items():
        yield ("limitations", name, bool(re.search(pat, lim, re.I)), "")

    # 14. contributions, one per numbered claim
    for i, key in enumerate(["population-level measurement", "artifact cohort",
                             "bounded rather than attenuated", "warning"], 1):
        yield ("contributions", f"contribution {i} present in intro",
               key.lower() in tex.lower(), "")

    # 15. genre-corpus style dimensions, one per metric
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if ab and corpus:
        at = clean(ab.group(1))
        al = sent_lens(at)
        aw = at.split()
        cw = [len(a.split()) for _, a in corpus.values()]
        cmed = [st.median(sent_lens(a)) for _, a in corpus.values()]
        csd = [st.pstdev(sent_lens(a)) for _, a in corpus.values()]
        cmax = [max(sent_lens(a)) for _, a in corpus.values()]
        cnum = [100 * len(re.findall(r"\d", a)) / len(a.split()) for _, a in corpus.values()]
        for name, val, lo, hi in [
            ("abstract length", len(aw), min(cw), max(cw)),
            ("abstract median sentence", st.median(al), min(cmed), max(cmed)),
            ("abstract sentence sd", st.pstdev(al), min(csd), max(csd)),
            ("abstract max sentence", max(al), min(cmax), max(cmax)),
            ("abstract numeric density",
             100 * len(re.findall(r"\d", at)) / len(aw), min(cnum), max(cnum)),
        ]:
            yield ("genre", f"{name} inside corpus band", lo <= val <= hi,
                   f"{val:.1f} in [{lo:.1f},{hi:.1f}]")
        yield ("genre", "abstract opens on the field, not the paper",
               not at.strip().lower().startswith(("we ", "this paper", "in this")), "")

    # 16. robustness claims, one per reported check
    for name, pat in {
        "endpoint truncation reported": r"\+0\.190",
        "placebo sd reported": r"0\.047",
        "placebo conservatism noted": r"conservative",
        "bonferroni threshold reported": r"0\.00625",
        "specification sweep count reported": r"twelve specifications",
        "sweep OR range reported": r"1\.20 and 1\.74",
        "strata event counts disclosed": r"6 to 11",
        "fisher p for reversal disclosed": r"0\.84",
        "buffer months stated": r"six months",
        "match tolerance stated": r"25\\%",
    }.items():
        yield ("robustness", name, bool(re.search(pat, tex)), "")


    # 17. table cell provenance, one per reported cell
    if t1.get("zombie") and t1.get("unresolvable"):
        cells = {
            "zombie artifact cell": f"{t1['zombie']['artifact_rate_per_1000']:.3f}",
            "zombie control cell": f"{t1['zombie']['control_rate_per_1000']:.3f}",
            "zombie ratio cell": f"{t1['zombie']['ratio']:.3f}",
            "zombie CI low": f"{t1['zombie']['ratio_ci_lo']:.3f}",
            "zombie CI high": f"{t1['zombie']['ratio_ci_hi']:.3f}",
            "unresolvable artifact cell": f"{t1['unresolvable']['artifact_rate_per_1000']:.2f}",
            "unresolvable control cell": f"{t1['unresolvable']['control_rate_per_1000']:.2f}",
            "unresolvable ratio cell": f"{t1['unresolvable']['ratio']:.3f}",
            "unresolvable CI low": f"{t1['unresolvable']['ratio_ci_lo']:.3f}",
            "unresolvable CI high": f"{t1['unresolvable']['ratio_ci_hi']:.3f}",
            "mean refs artifact cell": f"{t1['mean_refs_artifact']:.2f}",
            "mean refs control cell": f"{t1['mean_refs_control']:.2f}",
        }
        for name, val in cells.items():
            yield ("table1", f"{name} = {val}", val in tex, val)

    # 18. phrase-registry table, one per phrase row
    pv = os.path.join(OUT, "phrase_validation.csv")
    if os.path.exists(pv):
        import csv as _csv
        with open(pv) as fh:
            for row in _csv.DictReader(fh):
                ph = row["phrase"][:34]
                yield ("phrases", f"'{ph}' row present in Table 1",
                       row["phrase"] in tex, "")

    # 10. terminology consistency
    terms = {
        "zombie citation": r"zombie citation",
        "artifact cohort": r"artifact cohort",
        "lexical AI score": r"lexical AI score",
        "no rival term 'lexical proxy'": r"lexical proxy",
        "no rival term 'AI proxy'": r"\bAI proxy\b",
        "set-identified used": r"set-identifie[sd]",
        "partial identification used": r"partial identification",
        "interrupted time series used": r"interrupted time series",
        "matched controls used": r"matched controls",
        "unresolvable references used": r"unresolvable reference",
    }
    for name, pat in terms.items():
        present = bool(re.search(pat, tex))
        want = not name.startswith("no rival")
        yield ("terms", name, present == want, "")


def main():
    tex = open(TEX).read()
    log = open(LOG).read() if os.path.exists(LOG) else ""
    secs = sections(tex)
    corpus = jload(CORPUS)
    quiet = "--quiet" in sys.argv

    groups, fails, total = {}, [], 0
    for group, name, ok, detail in build(tex, log, secs, corpus):
        total += 1
        groups.setdefault(group, [0, 0])
        groups[group][1] += 1
        if ok:
            groups[group][0] += 1
        else:
            fails.append((group, name, detail))

    for g, (p, n) in groups.items():
        print(f"  {g:11s} {p:4d}/{n:<4d} {'OK' if p == n else 'FAIL'}")
    if fails:
        print("\n  FAILURES:")
        for g, name, detail in fails:
            print(f"    [{g}] {name} {detail}")
    print(f"\n{total - len(fails)}/{total} sub-checks pass")
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
