"""Stage 1: for every retracted work with >=1 citation, harvest all citing
works from 2018 on. A separate pass marks citing works that engage the
retraction (title/abstract mentions retract*), which are excluded downstream:
citing a retraction to discuss it is the system working as intended.

Output: data/s01_citations.jsonl rows
    {retracted_id, retraction_date, field, citing_id, citing_date, title}
        data/s01_engaged.jsonl rows
    {retracted_id, citing_id}

Resumable; safe to interrupt. Budget note: one OpenAlex query per retracted
work plus one per work for the engagement pass; the polite pool allows
100k requests/day, so a full run fits in a day but the checkpoint lets you
split it.
"""
import sys

import pandas as pd

import config
import oa


def harvest_citations(retracted):
    ckpt = oa.Checkpoint("s01_citations")
    todo = retracted[retracted.cited_by_count > 0]
    for i, row in enumerate(todo.itertuples()):
        wid = row.openalex_id
        if ckpt.is_done(wid):
            continue
        rows = []
        filt = f"cites:{wid},from_publication_date:{config.START_DATE}"
        for rec in oa.oa_page_all("/works", filt,
                                  "id,publication_date,title"):
            rows.append({
                "retracted_id": wid,
                "retraction_date": row.retraction_date,
                "field": row.field,
                "citing_id": oa.short_id(rec["id"]),
                "citing_date": rec.get("publication_date"),
                "title": rec.get("title") or "",
            })
        ckpt.write(rows, key=wid)
        if i % 200 == 0:
            print(f"  citations: {i}/{len(todo)} retracted works done")
    return ckpt


def harvest_engaged(retracted):
    """Citing works whose title/abstract mentions retract* -> excluded."""
    ckpt = oa.Checkpoint("s01_engaged")
    todo = retracted[retracted.cited_by_count > 0]
    for i, row in enumerate(todo.itertuples()):
        wid = row.openalex_id
        if ckpt.is_done(wid):
            continue
        rows = []
        filt = (f"cites:{wid},from_publication_date:{config.START_DATE},"
                f"title_and_abstract.search:retract")
        for rec in oa.oa_page_all("/works", filt, "id"):
            rows.append({"retracted_id": wid,
                         "citing_id": oa.short_id(rec["id"])})
        ckpt.write(rows, key=wid)
        if i % 200 == 0:
            print(f"  engaged: {i}/{len(todo)} retracted works done")
    return ckpt


def main():
    retracted = pd.read_csv(config.DATA / "retracted.csv")
    print(f"{len(retracted)} retracted works, "
          f"{(retracted.cited_by_count > 0).sum()} with citations")
    harvest_citations(retracted)
    harvest_engaged(retracted)
    print("stage 1 complete")


if __name__ == "__main__":
    sys.exit(main())
