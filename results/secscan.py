"""Per-section prose scan: genre-corpus rhythm plus stop-slop rules."""
import json, re, statistics as st, sys

CORP = json.load(open('/tmp/style_corpus.json'))
# -ly words that are adjectives or precision markers, not filler adverbs.
_KEEP = r'(?<!\bonly\b)(?<!\bearly\b)(?<!\bfamily\b)(?<!\bapply\b)(?<!\bmonthly\b)(?<!\byearly\b)(?<!\broughly\b)(?<!\blikely\b)(?<!\bsupply\b)'
ADV = re.compile(r'\b\w+ly\b' + _KEEP, re.I)
PASS = re.compile(r'\b(is|are|was|were|been|being|be)\s+\w+(ed|en)\b', re.I)
VAGUE = re.compile(r'\b(the (implications|reasons|results) are|is significant|are significant)\b', re.I)

def clean(t):
    t = re.sub(r'\\begin\{(figure|table|tabular|thebibliography)\}.*?\\end\{\1\}', '', t, flags=re.S)
    t = re.sub(r'\\(label|ref|citep|citet|cite)\{[^}]*\}', ' X ', t)
    # Subsection headers are not sentences; merging them inflates length.
    t = re.sub(r'\\subsection\{[^}]*\}', ' . ', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\[a-zA-Z]+', ' ', t)
    return re.sub(r'\s+', ' ', re.sub(r'[{}$\\]', ' ', t)).strip()

def sections(tex):
    out, parts = {}, re.split(r'\\section\{([^}]+)\}', tex)
    for i in range(1, len(parts), 2):
        b = clean(parts[i + 1])
        if len(b.split()) > 60:
            out[parts[i]] = b
    return out

def main():
    tex = open('paper.tex').read()
    cm = []
    for _, ab in CORP.values():
        l = [len(x.split()) for x in re.split(r'(?<=[.!?])\s+', ab) if len(x.split()) > 3]
        cm.append((st.median(l), st.pstdev(l)))
    c_med = st.median([x[0] for x in cm])
    fails = 0
    for name, body in sections(tex).items():
        l = [len(x.split()) for x in re.split(r'(?<=[.!?])\s+', body) if len(x.split()) > 3]
        w = body.split()
        issues = []
        if not (c_med - 8 <= st.median(l) <= c_med + 8):
            issues.append(f"median {st.median(l):.0f} vs corpus {c_med:.0f}")
        if st.pstdev(l) < 6:
            issues.append(f"sd {st.pstdev(l):.1f} metronomic")
        if max(l) > 55:
            issues.append(f"max {max(l)}w")
        if len(ADV.findall(body)) > len(w) / 60:
            issues.append(f"{len(ADV.findall(body))} adverbs")
        if len(PASS.findall(body)) > 5:
            issues.append(f"{len(PASS.findall(body))} passives")
        if VAGUE.findall(body):
            issues.append("vague declarative")
        fails += bool(issues)
        print(f"  {'PASS' if not issues else 'FAIL'}  {name[:42]:42s} {'; '.join(issues) or 'clean'}")
    print(f"\n{len(sections(tex)) - fails}/{len(sections(tex))} sections pass")
    return fails

if __name__ == '__main__':
    sys.exit(main())
