"""Stage 2: join Retraction Watch DOIs to OpenAlex work IDs by scanning the
snapshot's id and doi columns.

Those two columns are a small share of the snapshot, so this pass is cheap
compared with stage 3. It exists because references inside works are stored
as OpenAlex IDs, not DOIs: without this join there is no way to recognize a
reference as pointing at a retracted paper.

Sharding: pass --shard i --n-shards k to split across kernel sessions.

Output: data/retracted_works.csv (doi, openalex_id, retraction_date, ...)
"""
import argparse
import sys

import pandas as pd

import config
import oa
import snapshot

COLUMNS = ["id", "doi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    retracted = pd.read_csv(config.DATA / "retracted_dois.csv")
    want = set(retracted.doi.dropna())
    print(f"looking for {len(want)} retracted DOIs in snapshot "
          f"{snapshot.snapshot_date()}", flush=True)

    paths = snapshot.shard(snapshot.file_paths(), args.shard, args.n_shards)
    ckpt = oa.Checkpoint(f"s02_doimap_{args.shard}of{args.n_shards}")
    fs = snapshot.filesystem()

    for i, path in enumerate(paths):
        key = path.rsplit("/", 2)[-2] + "/" + path.rsplit("/", 1)[-1]
        if ckpt.is_done(key):
            continue
        found = []
        for batch in snapshot.iter_batches(path, COLUMNS, fs=fs):
            ids = batch.column("id").to_pylist()
            dois = batch.column("doi").to_pylist()
            for wid, doi in zip(ids, dois):
                d = oa.norm_doi(doi)
                if d and d in want:
                    found.append({"doi": d,
                                  "openalex_id": oa.short_id(wid)})
        ckpt.write(found, key=key)
        if i % 25 == 0:
            print(f"  {i}/{len(paths)} files, "
                  f"{len(ckpt.load_results())} matches so far", flush=True)

    mapped = pd.DataFrame(ckpt.load_results())
    if not len(mapped):
        print("no matches; check that stage 0 ran")
        return 1
    mapped = mapped.drop_duplicates(subset="doi")
    out = retracted.merge(mapped, on="doi", how="inner")
    dest = config.DATA / f"retracted_works_{args.shard}of{args.n_shards}.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(out)} of {len(retracted)} retracted papers "
          f"found in OpenAlex ({len(out)/len(retracted):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
