"""Per-paper style conformance against each reference paper individually.

The pooled check cannot distinguish matching a genre from mimicking one
member of it. This measures the paper against all 15 reference texts one
at a time, then asks whether it sits near the centroid of the whole set
rather than near any single paper.
"""
import json, re, statistics as st, sys
sys.path.insert(0, '.')
from secscan import sections
from corpuscheck import prof

DIMS = ["median_sentence", "sentence_sd", "max_sentence", "pct_short",
        "pct_long", "hedges_per100", "we_per100"]


def refs():
    out = {}
    for k, v in json.load(open('/tmp/body_corpus.json')).items():
        if len(v.split()) > 200:
            out[f"arXiv:{k} (body)"] = v
    for k, (ti, ab) in json.load(open('/tmp/style_corpus.json')).items():
        out[f"{k} (abstract)"] = ab
    return out


def z(vec, mus, sds):
    return [(vec[d] - mus[d]) / (sds[d] or 1e-9) for d in DIMS]


def main():
    R = refs()
    P = {k: prof(v) for k, v in R.items()}
    mus = {d: st.mean([p[d] for p in P.values()]) for d in DIMS}
    sds = {d: st.pstdev([p[d] for p in P.values()]) for d in DIMS}

    body = " ".join(sections(open('paper.tex').read()).values())
    mine = prof(body)
    mz = z(mine, mus, sds)

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    rows = []
    for k, p in P.items():
        rows.append((dist(mz, z(p, mus, sds)), k))
    rows.sort()

    print(f"  distance from each reference paper (standardised, 7 dimensions):")
    for d, k in rows:
        print(f"    {d:6.2f}  {k}")

    # Within-corpus spread, for comparison.
    pair = []
    keys = list(P)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair.append(dist(z(P[keys[i]], mus, sds), z(P[keys[j]], mus, sds)))
    centroid_d = dist(mz, [0] * len(DIMS))
    ref_centroid = [dist(z(p, mus, sds), [0] * len(DIMS)) for p in P.values()]

    print(f"\n  my distance to corpus centroid: {centroid_d:.2f}")
    print(f"  reference papers to centroid:   median {st.median(ref_centroid):.2f} "
          f"[{min(ref_centroid):.2f}, {max(ref_centroid):.2f}]")
    print(f"  typical distance between two reference papers: median {st.median(pair):.2f}")

    nearest, farthest = rows[0][0], rows[-1][0]
    ok_central = centroid_d <= max(ref_centroid)
    ok_notclone = nearest >= min(pair)
    print(f"\n  {'PASS' if ok_central else 'FAIL'}  sits inside the corpus spread "
          f"(centroid distance {centroid_d:.2f} <= max {max(ref_centroid):.2f})")
    print(f"  {'PASS' if ok_notclone else 'FAIL'}  not a clone of any single paper "
          f"(nearest {nearest:.2f} >= min pairwise {min(pair):.2f})")
    return 0 if (ok_central and ok_notclone) else 1


if __name__ == '__main__':
    sys.exit(main())
