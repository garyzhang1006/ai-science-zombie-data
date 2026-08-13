"""Stage 0: download Retraction Watch, keep true retractions with usable
dates and DOIs, map them to OpenAlex works.

Output: data/retracted.csv with columns
    rw_record_id, doi, retraction_date, original_date, reason,
    openalex_id, cited_by_count, field
"""
import csv
import sys

import pandas as pd
import requests
from dateutil import parser as dateparser

import config
import oa


def download_rw():
    dest = config.DATA / "retraction_watch.csv"
    if dest.exists():
        print(f"already have {dest}")
        return dest
    print("downloading Retraction Watch CSV (~65 MB)...")
    r = requests.get(config.RW_CSV_URL, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def parse_date(s):
    if not s or not s.strip():
        return None
    try:
        return dateparser.parse(s.split()[0], dayfirst=False).date()
    except (ValueError, OverflowError):
        return None


def load_retractions(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for rec in csv.DictReader(f):
            if rec.get("RetractionNature", "").strip() != "Retraction":
                continue
            doi = oa.norm_doi(rec.get("OriginalPaperDOI"))
            rdate = parse_date(rec.get("RetractionDate"))
            if not doi or not rdate or "unavailable" in doi:
                continue
            rows.append({
                "rw_record_id": rec.get("Record ID"),
                "doi": doi,
                "retraction_date": rdate.isoformat(),
                "original_date": (parse_date(rec.get("OriginalPaperDate"))
                                  or "").__str__(),
                "reason": (rec.get("Reason") or "").strip(";+ "),
            })
    df = pd.DataFrame(rows).drop_duplicates(subset="doi", keep="first")
    print(f"{len(df)} retracted papers with DOI + retraction date")
    return df


def map_to_openalex(df):
    ckpt = oa.Checkpoint("s00_oa_map")
    dois = df["doi"].tolist()
    batches = [dois[i:i + 40] for i in range(0, len(dois), 40)]
    for bi, batch in enumerate(batches):
        key = f"batch{bi}"
        if ckpt.is_done(key):
            continue
        filt = "doi:" + "|".join(batch)
        found = []
        for rec in oa.oa_page_all(
                "/works", filt,
                "id,doi,cited_by_count,primary_topic"):
            topic = rec.get("primary_topic") or {}
            found.append({
                "doi": oa.norm_doi(rec.get("doi")),
                "openalex_id": oa.short_id(rec["id"]),
                "cited_by_count": rec.get("cited_by_count", 0),
                "field": (topic.get("field") or {}).get("display_name", ""),
            })
        ckpt.write(found, key=key)
        if bi % 50 == 0:
            print(f"  mapped batch {bi}/{len(batches)}")
    mapped = pd.DataFrame(ckpt.load_results()).drop_duplicates(subset="doi")
    print(f"{len(mapped)} retracted papers matched in OpenAlex")
    return df.merge(mapped, on="doi", how="inner")


def main():
    path = download_rw()
    df = load_retractions(path)
    df = map_to_openalex(df)
    out = config.DATA / "retracted.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, "
          f"{(df.cited_by_count > 0).sum()} with >=1 citation")


if __name__ == "__main__":
    sys.exit(main())
