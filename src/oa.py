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
_MIN_INTERVAL = {"oa": 0.11, "cr": 0.05}  # ~9/s OpenAlex, ~20/s Crossref


def _throttled_get(kind, url, params=None, max_tries=6):
    for attempt in range(max_tries):
        wait = _MIN_INTERVAL[kind] - (time.time() - _last_call[kind])
        if wait > 0:
            time.sleep(wait)
        _last_call[kind] = time.time()
        try:
            r = _session.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(min(2 ** attempt * 2, 60))
            continue
        if r.status_code == 404:
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
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
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
