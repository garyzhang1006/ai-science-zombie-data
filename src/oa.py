"""Minimal OpenAlex + Crossref client: rate limiting, retries, cursor paging,
JSONL checkpointing. No API key needed; set OA_MAILTO for the polite pool.
"""
import json
import time

import requests

import config

OA_BASE = "https://api.openalex.org"
CR_BASE = "https://api.crossref.org"

_session = requests.Session()
_session.headers["User-Agent"] = (
    f"zombie-citation-pipeline (mailto:{config.MAILTO})"
    if config.MAILTO else "zombie-citation-pipeline"
)

_last_call = {"oa": 0.0, "cr": 0.0}
_MIN_INTERVAL = {"oa": 0.15, "cr": 0.05}  # ~6.5/s OpenAlex, ~20/s Crossref

# Live view of the OpenAlex credit budget, refreshed from response headers.
quota = {"limit": None, "remaining": None, "usd_remaining": None,
         "reset_seconds": None}


class QuotaExhausted(RuntimeError):
    """OpenAlex free-tier budget for the day is spent.

    The free tier allows 1,000 calls or $0.10 per day, whichever binds
    first, and a full-text search costs ten times a metadata call. Callers
    catch this, checkpoint, and resume after the reset rather than
    hammering a closed door.
    """


def _note_quota(headers):
    def _num(key, cast):
        v = headers.get(key)
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None
    quota["limit"] = _num("x-ratelimit-limit", int)
    quota["remaining"] = _num("x-ratelimit-remaining", int)
    quota["usd_remaining"] = _num("x-ratelimit-remaining-usd", float)
    quota["reset_seconds"] = _num("x-ratelimit-reset", int)


def _throttled_get(kind, url, params=None, max_tries=12):
    """Long harvests must survive sustained 429 spells (shared datacenter
    IPs get throttled hard); tolerate ~20 min of failure before giving up."""
    for attempt in range(max_tries):
        wait = _MIN_INTERVAL[kind] - (time.time() - _last_call[kind])
        if wait > 0:
            time.sleep(wait)
        _last_call[kind] = time.time()
        try:
            r = _session.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 60))
            continue
        if kind == "oa":
            _note_quota(r.headers)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (402, 429) and kind == "oa":
            # Distinguish a spent daily budget from ordinary throttling:
            # waiting out the former takes hours, so callers must be told.
            spent = (quota["remaining"] is not None
                     and quota["remaining"] <= 0)
            spent = spent or (quota["usd_remaining"] is not None
                              and quota["usd_remaining"] <= 0)
            if spent or r.status_code == 402:
                raise QuotaExhausted(
                    f"OpenAlex daily budget spent; resets in "
                    f"{(quota['reset_seconds'] or 0) / 3600:.1f} h")
        # Retry every transient class rather than an enumerated list:
        # timeouts, rate limits, and any server-side error. Listing codes
        # one at a time has cost three long runs already (503, then 504).
        if r.status_code in (408, 429) or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(min(int(retry_after) + 1, 300))
            else:
                time.sleep(min(2 ** attempt * 2, 120))
            continue
        if 400 <= r.status_code < 500:
            # A single unusable identifier must not abort a harvest that
            # has already spent hours on the other thousands of records.
            print(f"  skipping {r.status_code} on {url}", flush=True)
            return None
        r.raise_for_status()
    raise RuntimeError(f"gave up after {max_tries} tries: {url}")


def oa_get(path, **params):
    if config.MAILTO:
        params.setdefault("mailto", config.MAILTO)
    return _throttled_get("oa", f"{OA_BASE}{path}", params)


def oa_page_all(path, filter_str, select, per_page=200, max_records=None,
                extra=None):
    """Cursor-paginate an OpenAlex works query. Yields result dicts."""
    cursor = "*"
    seen = 0
    while cursor:
        params = {"filter": filter_str, "select": select,
                  "per-page": per_page, "cursor": cursor}
        if extra:
            params.update(extra)
        page = oa_get(path, **params)
        if page is None:
            return
        for rec in page.get("results", []):
            yield rec
            seen += 1
            if max_records and seen >= max_records:
                return
        cursor = page.get("meta", {}).get("next_cursor")


def oa_count(filter_str):
    page = oa_get("/works", filter=filter_str, **{"per-page": 1},
                  select="id")
    return page["meta"]["count"] if page else 0


def cr_get(path, **params):
    if config.MAILTO:
        params.setdefault("mailto", config.MAILTO)
    return _throttled_get("cr", f"{CR_BASE}{path}", params)


def short_id(openalex_url):
    """https://openalex.org/W123 -> W123 (also handles bare IDs)."""
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else None


def norm_doi(doi):
    """Normalize a DOI for lookup.

    Publisher-scraped DOIs often carry tracking query strings, e.g.
    "10.1021/acs.jafc.2c04673?ref=article_openpdf". Crossref rejects those
    with a 400, so the suffix has to go before the DOI is used as a key or
    a URL path.
    """
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("?")[0].split("#")[0].rstrip(".,;)")
    return d or None


def invert_abstract(inv):
    """Reconstruct plain text from OpenAlex abstract_inverted_index."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


class Checkpoint:
    """Append-only JSONL results plus a done-set, so any stage can resume."""

    def __init__(self, name):
        self.results_path = config.DATA / f"{name}.jsonl"
        self.done_path = config.DATA / f"{name}.done"
        self.done = set()
        if self.done_path.exists():
            self.done = set(self.done_path.read_text().split())

    def is_done(self, key):
        return key in self.done

    def write(self, rows, key=None):
        with open(self.results_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        if key:
            with open(self.done_path, "a") as f:
                f.write(key + "\n")
            self.done.add(key)

    def load_results(self):
        if not self.results_path.exists():
            return []
        with open(self.results_path) as f:
            return [json.loads(line) for line in f if line.strip()]
