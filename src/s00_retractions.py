"""Stage 0: parse the Retraction Watch database into the retracted-work
table the rest of the pipeline joins against.

Retraction Watch is the source of truth for retraction DATES, which the
zombie definition needs and which OpenAlex's is_retracted flag does not
carry. No API calls; the DOI to OpenAlex-ID join happens in stage 2 against
the snapshot.

Output: data/retracted_dois.csv (doi, retraction_date, original_date, reason)
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
    if dest.exists() and dest.stat().st_size > 10_000_000:
        print(f"already have {dest}")
        return dest
    print("downloading Retraction Watch CSV (~63 MB)...")
    r = requests.get(config.RW_CSV_URL, timeout=600)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def parse_date(s):
    if not s or not s.strip():
        return None
    try:
        return dateparser.parse(s.split()[0], dayfirst=False).date()
    except (ValueError, OverflowError, TypeError):
        return None


def load_retractions(path):
    """Keep true retractions only. Expressions of concern, corrections and
    reinstatements are different events and would corrupt the outcome."""
    rows, skipped = [], {"nature": 0, "no_doi": 0, "no_date": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for rec in csv.DictReader(f):
            if rec.get("RetractionNature", "").strip() != "Retraction":
                skipped["nature"] += 1
                continue
            doi = oa.norm_doi(rec.get("OriginalPaperDOI"))
            if not doi or "unavailable" in doi:
                skipped["no_doi"] += 1
                continue
            rdate = parse_date(rec.get("RetractionDate"))
            if not rdate:
                skipped["no_date"] += 1
                continue
            rows.append({
                "rw_record_id": rec.get("Record ID"),
                "doi": doi,
                "retraction_date": rdate.isoformat(),
                "original_date": (parse_date(rec.get("OriginalPaperDate"))
                                  or ""),
                "reason": (rec.get("Reason") or "").strip(";+ "),
                "journal": (rec.get("Journal") or "").strip(),
            })
    df = pd.DataFrame(rows)
    # Same paper can carry several RW records; keep the earliest retraction.
    df = (df.sort_values("retraction_date")
            .drop_duplicates(subset="doi", keep="first"))
    print(f"{len(df)} retracted papers with DOI + retraction date")
    print(f"  skipped: {skipped}")
    return df


def main():
    path = download_rw()
    df = load_retractions(path)
    out = config.DATA / "retracted_dois.csv"
    df.to_csv(out, index=False)
    years = pd.to_datetime(df.retraction_date).dt.year
    print(f"wrote {out}")
    print(f"retraction years {years.min()}-{years.max()}, "
          f"{(years >= 2018).sum()} retracted in 2018 or later")


if __name__ == "__main__":
    sys.exit(main())
