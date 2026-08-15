"""Access to the free OpenAlex parquet snapshot on S3.

The REST API meters usage (1,000 calls or $0.10 per day on the free tier),
which cannot cover a 510M-work scan. The snapshot is unmetered, and pinning
one release date also makes the whole study reproducible: every number in
the paper comes from a single frozen corpus rather than a moving API.

Column pruning is what makes this affordable. The columns this study needs
are about 12% of the snapshot's 725 GB; parquet readers fetch only those
column chunks over HTTP range requests.
"""
import json
import urllib.request

import pyarrow.parquet as pq

import config

MANIFEST_URL = ("https://openalex.s3.amazonaws.com/data/parquet/works/"
                "manifest.json")
_manifest = None


def manifest():
    """Cached works manifest: {'date', 'record_count', 'files': [...]}."""
    global _manifest
    if _manifest is None:
        cache = config.DATA / "works_manifest.json"
        if cache.exists():
            _manifest = json.loads(cache.read_text())
        else:
            with urllib.request.urlopen(MANIFEST_URL, timeout=120) as r:
                _manifest = json.load(r)
            cache.write_text(json.dumps(_manifest))
    return _manifest


def snapshot_date():
    return manifest()["date"]


def file_paths():
    """S3 paths ('openalex/data/...') for every works part file, sorted so
    shard assignment is stable across runs."""
    return sorted(e["url"].replace("s3://", "") for e in manifest()["files"])


def shard(paths, shard_index, n_shards):
    """Deterministic round-robin split. Round-robin rather than contiguous
    blocks so each shard sees a mix of old and new update partitions, which
    keeps per-shard runtime and publication-year coverage comparable."""
    return [p for i, p in enumerate(paths) if i % n_shards == shard_index]


def filesystem():
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def read_columns(path, columns, fs=None):
    """Read selected columns of one part file. Returns a pyarrow Table.

    Reads through an open file handle rather than passing the path to
    pyarrow's dataset layer: the snapshot is hive-partitioned on
    updated_date, and the dataset reader turns that directory name into a
    dictionary column that collides with the file's own timestamp column
    ("Unable to merge: Field updated_date has incompatible types").
    """
    fs = fs or filesystem()
    with fs.open(path, "rb") as fh:
        return pq.ParquetFile(fh).read(columns=columns)


def iter_batches(path, columns, fs=None, batch_size=50_000):
    """Stream one part file in record batches to bound peak memory: the
    largest parts hold 400k works, and reference lists make a full
    materialization expensive."""
    fs = fs or filesystem()
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield batch
