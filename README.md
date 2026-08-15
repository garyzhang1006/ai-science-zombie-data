# Zombie citations: does AI-assisted writing resurrect retracted science?

Pipeline for a population-level study of citations to already-retracted
papers ("zombie citations") across the LLM transition, 2018-2026. Three
prongs, chosen so their failure modes do not overlap: an interrupted time
series of the population rate around the ChatGPT release, a matched cohort
of papers carrying verbatim LLM artifacts ("as an AI language model"), and
partial-identification bounds that turn a noisy lexical AI proxy into an
honest interval rather than an attenuated point estimate.

## Where the data comes from, and why

The study runs on the **free OpenAlex parquet snapshot** (725 GB, 510M
works, pinned to one release date), not on the REST API. The API meters
usage: the free tier allows 1,000 calls or $0.10 per day, and a full-text
search costs ten times a metadata call. A scan of half a billion works is
simply not reachable through it. The snapshot is unmetered, and pinning one
release also makes the study reproducible, since every number comes from a
frozen corpus instead of a moving index.

Column pruning is what makes the scan affordable: the columns this study
needs are about 12% of the snapshot, and parquet readers fetch only those
column chunks.

Two things still need live services. The artifact cohort is defined by
phrases inside full text, which the snapshot does not carry, so stage 1
pages the OpenAlex API within the daily budget and resumes across days. The
unresolvable-reference outcome uses Crossref, which is free and unmetered.

Retraction Watch (CC BY 4.0, via Crossref) supplies retraction dates, which
OpenAlex's `is_retracted` flag does not carry and which the zombie
definition needs.

## Setup

```
pip install -r requirements.txt
export OA_MAILTO=you@example.org
```

## Run

```
cd src
python s00_retractions.py                      # Retraction Watch parse
python s01_artifact_harvest.py                 # artifact cohort (API budget)
python s02_doi_map.py    --shard 0 --n-shards 8   # DOI -> OpenAlex id
python s03_scan.py       --shard 0 --n-shards 8   # the main snapshot pass
python s04_stats.py                            # ITS, cohort, bounds
python s05_crossref.py                         # unresolvable references
python s06_figures.py                          # the three figures
python s07_fill_tk.py                          # every paper number
```

Stages 2 and 3 shard: run `--shard 0..7` (in parallel sessions or in
sequence), then stage 4 merges the partial aggregates. Every stage
checkpoints and resumes, so an interrupted run costs at most one file.

`python tests/smoke.py` verifies the computational core: the vectorized
zombie counter against a literal reading of the definition, the interrupted
time series against an injected break and against a flat series, and the
bounds against forward-simulated data where the truth is known.

## Outputs (out/)

| File | Feeds |
|---|---|
| `population_series.csv`, `its_results.json` | Sec 4.1, Fig 1 |
| `artifact_candidates.csv`, `artifact_cohort.csv`, `hand_validation_sample.csv` | Sec 3.2, Appendix A |
| `matches.csv`, `unresolvable.csv`, `table1.json` | Sec 4.2, Table 1, Fig 2 |
| `lexical_sample.csv`, `bounds.json` | Sec 4.3, Fig 3 |
| `tk_values.json`, `tk_report.md` | every [TK] slot, plus what is still missing |

## Method notes

- A citation counts as zombie only if the citing paper appeared at least
  `BUFFER_MONTHS` (default 6) after the retraction date, absorbing indexing
  lag. Citing works that discuss retraction in their title or abstract are
  excluded: citing a retracted paper to discuss its retraction is the
  correction system working.
- The population denominator is exact rather than estimated. The scan
  counts works and references per month per field directly.
- Every artifact-filter decision is logged with its reason in
  `artifact_candidates.csv`, and `hand_validation_sample.csv` holds a
  seeded 100-paper sample for manual precision checking.
- Sensitivity and specificity of the lexical proxy are never trusted at a
  point. Bounds are reported across the whole grid in `config.py`, and the
  attenuated naive estimate is reported beside them.

## License

MIT (code). Retraction Watch data is CC BY 4.0, attribution: Retraction
Watch / Crossref. OpenAlex data is CC0.
