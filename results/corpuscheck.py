"""Every section against the pooled genre corpus (15 papers, 7 dimensions).

Demonstrates collective style imitation: each section of the paper must
fall inside the min-max band the corpus spans on each dimension.
"""
import json, re, statistics as st, sys
sys.path.insert(0, '.')
from secscan import sections

HEDGE = re.compile(r'\b(may|might|could|appears?|suggests?|seems?|likely|possibly)\b', re.I)


def pool():
    bodies = [v for v in json.load(open('/tmp/body_corpus.json')).values()
              if len(v.split()) > 200]
    abstracts = [a for _, a in json.load(open('/tmp/style_corpus.json')).values()]
    return bodies + abstracts


def prof(t):
    l = [len(x.split()) for x in re.split(r'(?<=[.!?])\s+', t) if len(x.split()) > 3]
    w = t.split()
    return dict(median_sentence=st.median(l), sentence_sd=st.pstdev(l),
                max_sentence=max(l),
                pct_short=100 * sum(1 for x in l if x < 13) / len(l),
                pct_long=100 * sum(1 for x in l if x > 34) / len(l),
                hedges_per100=100 * len(HEDGE.findall(t)) / len(w),
                we_per100=100 * len(re.findall(r'\bWe\b', t)) / len(w))


def main():
    P = [prof(t) for t in pool()]
    band = {k: (min(p[k] for p in P), max(p[k] for p in P)) for k in P[0]}
    secs = sections(open('paper.tex').read())
    fails = 0
    for name, body in secs.items():
        p = prof(body)
        bad = [f"{k}={p[k]:.1f}" for k, (lo, hi) in band.items()
               if not (lo <= p[k] <= hi)]
        fails += bool(bad)
        print(f"  {'PASS' if not bad else 'FAIL'}  {name[:42]:42s} "
              f"{', '.join(bad) if bad else '7/7 dimensions in corpus band'}")
    print(f"\n{len(secs) - fails}/{len(secs)} sections inside the pooled corpus band "
          f"({len(P)} reference papers, 7 dimensions)")
    return fails


if __name__ == '__main__':
    sys.exit(main())
