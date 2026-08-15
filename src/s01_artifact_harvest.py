"""Stage 1: harvest the artifact cohort through the OpenAlex REST API.

This is the one prong the parquet snapshot cannot serve, because the
snapshot carries no full text and the cohort is defined by verbatim
phrases inside papers. Full-text queries cost ten times a metadata call
against the free tier's daily budget, so the stage harvests until the
budget is spent, checkpoints, and exits telling the caller when to resume.
Each phrase is paged independently, so an interrupted phrase resumes from
its own cursor rather than from the beginning.

Filtering removes meta-discussion: papers ABOUT language models quote
these phrases deliberately, and counting them as unedited model output
would inflate the cohort with exactly the careful authors it means to
exclude. Every candidate keeps its keep/drop decision and reason.

Outputs: out/artifact_candidates.csv, out/artifact_cohort.csv,
         out/hand_validation_sample.csv
"""
import json
import re
import sys

import pandas as pd

import config
import oa

SELECT = ("id,doi,title,publication_year,publication_date,primary_topic,"
          "primary_location,referenced_works_count,cited_by_count,"
          "abstract_inverted_index,type")


def _row(rec, phrase):
    topic = rec.get("primary_topic") or {}
    loc = rec.get("primary_location") or {}
    src = loc.get("source") or {}
    return {
        "phrase": phrase,
        "openalex_id": oa.short_id(rec["id"]),
        "doi": oa.norm_doi(rec.get("doi")),
        "title": rec.get("title") or "",
        "year": rec.get("publication_year"),
        "date": rec.get("publication_date"),
        "type": rec.get("type"),
        "topic": topic.get("display_name", ""),
        "topic_id": oa.short_id(topic.get("id")) or "",
        "subfield_id": (topic.get("subfield") or {}).get("id", ""),
        "field": (topic.get("field") or {}).get("display_name", ""),
        "source_id": oa.short_id(src.get("id")) or "",
        "source_name": src.get("display_name", ""),
        "n_refs": rec.get("referenced_works_count", 0),
        "n_cites": rec.get("cited_by_count", 0),
        "abstract": oa.invert_abstract(
            rec.get("abstract_inverted_index"))[:2000],
    }


def harvest():
    """Page every phrase, saving a cursor per phrase so a budget stop costs
    at most one page of re-work."""
    ckpt = oa.Checkpoint("s01_candidates")
    cursor_path = config.DATA / "s01_cursors.json"
    cursors = (json.loads(cursor_path.read_text())
               if cursor_path.exists() else {})
    exhausted = False

    for phrase in config.ARTIFACT_PHRASES:
        if cursors.get(phrase) == "DONE":
            continue
        cursor = cursors.get(phrase, "*")
        page_no = 0
        while cursor and not exhausted:
            try:
                page = oa.oa_get("/works",
                                 filter=f'fulltext.search:"{phrase}"',
                                 select=SELECT, cursor=cursor,
                                 **{"per-page": 200})
            except oa.QuotaExhausted as e:
                print(f"  budget spent during {phrase!r}: {e}", flush=True)
                exhausted = True
                break
            if page is None:
                break
            rows = [_row(rec, phrase) for rec in page.get("results", [])]
            ckpt.write(rows)
            page_no += 1
            total = page["meta"]["count"]
            cursor = page.get("meta", {}).get("next_cursor")
            cursors[phrase] = cursor or "DONE"
            cursor_path.write_text(json.dumps(cursors))
            print(f"  {phrase!r}: page {page_no}, {len(rows)} rows "
                  f"(total {total}), credits left "
                  f"{oa.quota.get('remaining')}", flush=True)
        if exhausted:
            break

    df = pd.DataFrame(ckpt.load_results())
    if len(df):
        df = df.drop_duplicates(subset="openalex_id")
    done = sum(1 for p in config.ARTIFACT_PHRASES
               if cursors.get(p) == "DONE")
    print(f"{len(df)} unique candidates; "
          f"{done}/{len(config.ARTIFACT_PHRASES)} phrases complete")
    return df, done == len(config.ARTIFACT_PHRASES)


def classify(row):
    """Return (keep, reason). Drop anything whose own subject is language
    models, since those papers quote the artifact phrases on purpose."""
    title = (row.get("title") or "").lower()
    abstract = (row.get("abstract") or "").lower()
    if row.get("subfield_id") in config.META_SUBFIELD_IDS:
        return False, "ai_subfield"
    if re.search(config.META_TITLE_RE, title, re.I):
        return False, "meta_title"
    hits = [t for t in config.META_ABSTRACT_TERMS if t in abstract]
    if len(hits) >= 2:
        return False, f"meta_abstract:{'|'.join(hits[:3])}"
    if row.get("type") not in ("article", "review", "book-chapter",
                               "preprint", "dissertation"):
        return False, f"type:{row.get('type')}"
    if not row.get("n_refs"):
        return False, "no_references"
    return True, "kept"


def main():
    df, complete = harvest()
    if not len(df):
        print("no candidates harvested yet; rerun after the budget resets")
        return 1
    decisions = df.apply(lambda r: classify(r.to_dict()), axis=1,
                         result_type="expand")
    df["keep"], df["reason"] = decisions[0], decisions[1]
    df.to_csv(config.OUT / "artifact_candidates.csv", index=False)
    cohort = df[df.keep].copy()
    cohort.to_csv(config.OUT / "artifact_cohort.csv", index=False)
    sample = df.sample(min(100, len(df)), random_state=config.SAMPLE_SEED)
    sample[["openalex_id", "doi", "title", "phrase", "keep",
            "reason"]].to_csv(config.OUT / "hand_validation_sample.csv",
                              index=False)
    print(f"kept {len(cohort)} / {len(df)}")
    print(df.reason.value_counts().head(10).to_string())
    if not complete:
        print("\nINCOMPLETE: budget spent. Rerun after reset to continue.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
