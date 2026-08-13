"""Stage 3: harvest the artifact cohort — papers containing verbatim LLM
artifact phrases — and filter out meta-discussion (papers ABOUT LLMs that
quote the phrases deliberately).

Every candidate is kept in the audit file with its keep/drop decision and
reason, so the filter is fully inspectable. A seeded random sample of 100
candidates is written out for hand validation of filter precision.

Outputs: out/artifact_candidates.csv (all, with decisions)
         out/artifact_cohort.csv     (kept)
         out/hand_validation_sample.csv
"""
import re
import sys

import pandas as pd

import config
import oa


def harvest():
    ckpt = oa.Checkpoint("s03_candidates")
    for phrase in config.ARTIFACT_PHRASES:
        key = phrase.replace(" ", "_")
        if ckpt.is_done(key):
            continue
        filt = f'fulltext.search:"{phrase}"'
        rows = []
        for rec in oa.oa_page_all(
                "/works", filt,
                "id,doi,title,publication_year,publication_date,"
                "primary_topic,primary_location,referenced_works_count,"
                "cited_by_count,abstract_inverted_index,type"):
            topic = rec.get("primary_topic") or {}
            loc = rec.get("primary_location") or {}
            src = loc.get("source") or {}
            rows.append({
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
            })
        ckpt.write(rows, key=key)
        print(f"  {phrase!r}: {len(rows)} raw hits")
    df = pd.DataFrame(ckpt.load_results())
    df = df.drop_duplicates(subset="openalex_id")
    print(f"{len(df)} unique candidate works")
    return df


def classify(row):
    """Return (keep: bool, reason: str)."""
    title = (row.title or "").lower()
    abstract = (row.abstract or "").lower()
    if row.subfield_id in config.META_SUBFIELD_IDS:
        return False, "ai_subfield"
    if re.search(config.META_TITLE_RE, title, re.I):
        return False, "meta_title"
    hits = [t for t in config.META_ABSTRACT_TERMS if t in abstract]
    if len(hits) >= 2:
        return False, f"meta_abstract:{'|'.join(hits[:3])}"
    if row.type not in ("article", "review", "book-chapter",
                        "preprint", "dissertation"):
        return False, f"type:{row.type}"
    if not row.n_refs:
        return False, "no_references"
    return True, "kept"


def main():
    df = harvest()
    decisions = df.apply(classify, axis=1, result_type="expand")
    df["keep"], df["reason"] = decisions[0], decisions[1]
    df.to_csv(config.OUT / "artifact_candidates.csv", index=False)
    cohort = df[df.keep].copy()
    cohort.to_csv(config.OUT / "artifact_cohort.csv", index=False)
    sample = df.sample(min(100, len(df)), random_state=config.SAMPLE_SEED)
    sample[["openalex_id", "doi", "title", "phrase", "keep", "reason"]].to_csv(
        config.OUT / "hand_validation_sample.csv", index=False)
    print(f"kept {len(cohort)} / {len(df)} candidates")
    print(df.reason.value_counts().head(10).to_string())


if __name__ == "__main__":
    sys.exit(main())
