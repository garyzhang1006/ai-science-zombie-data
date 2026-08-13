# Zombie citations: does AI-assisted writing resurrect retracted science?

Pipeline for a population-level study of citations to already-retracted
papers ("zombie citations") across the LLM transition, 2018-2026. Three
prongs: an interrupted time series around the ChatGPT release, a matched
cohort of papers with verbatim LLM artifacts ("as an AI language model"),
and partial-identification bounds that turn a noisy lexical AI proxy into
an honest interval.

Data sources: the [Retraction Watch database](https://gitlab.com/crossref/retraction-watch-data)
(CC BY 4.0, via Crossref), [OpenAlex](https://openalex.org), and
[Crossref](https://api.crossref.org). Everything is public and free; no
API keys.

## Setup

```
pip install -r requirements.txt
export OA_MAILTO=you@example.org   # polite-pool contact, required
```

## Run

```
bash run_all.sh          # or stage by stage:
cd src
python s00_retraction_watch.py   # download RW, map DOIs to OpenAlex  (~1 h)
python s01_zombie_harvest.py     # citing works of retracted papers   (~4-8 h)
python s02_population_series.py  # monthly series + ITS + placebos    (~30 min)
python s03_artifact_cohort.py    # artifact-phrase harvest + filter   (~1-2 h)
python s04_match_controls.py     # venue/year matched controls        (~1 h)
python s05_outcomes.py           # zombie + unresolvable outcomes     (~2-6 h)
python s06_bounds.py             # lexical proxy + Manski bounds      (~2-4 h)
python s07_figures.py            # the three paper figures            (seconds)
python s08_fill_tk.py            # every paper number in one JSON     (seconds)
```

Every harvest stage checkpoints to `data/*.jsonl` + `data/*.done` and is
safe to interrupt and resume, so the pipeline also fits inside a 12-hour
Kaggle session or an overnight laptop run. OpenAlex's polite pool allows
100k requests/day; a full run uses roughly that, so a cold start may need
to straddle two days. Reruns cost only what is not yet checkpointed.

## Outputs (out/)

| File | Feeds |
|---|---|
| `population_series.csv`, `its_results.json` | Sec 4.1, Fig 1 |
| `artifact_candidates.csv`, `artifact_cohort.csv`, `hand_validation_sample.csv` | Sec 3.2, Appendix A |
| `matches.csv`, `outcomes.csv`, `table1.json` | Sec 4.2, Table 1, Fig 2 |
| `lexical_sample.csv`, `bounds.json` | Sec 4.3, Fig 3 |
| `tk_values.json`, `tk_report.md` | every [TK] slot in the paper |

## Method notes

- A citation is *zombie* only if the citing paper appears at least
  `BUFFER_MONTHS` (default 6) after the retraction date, and citing works
  that mention retraction in title/abstract are excluded: citing a
  retracted paper to discuss its retraction is the system working.
- The population denominator is works-per-month (OpenAlex group-by, per
  field) times mean reference-list length from a seeded 2,000-work sample
  per year. This is an approximation; it cancels in the ITS break
  estimate unless reference-list norms shifted at exactly the break.
- The artifact filter's decisions are all logged with reasons in
  `artifact_candidates.csv`; `hand_validation_sample.csv` is a seeded
  100-paper sample for manual precision checks.
- Sensitivity/specificity of the lexical proxy are never trusted at a
  point: bounds are reported over the whole grid in `config.py`.

## License

MIT (code). Retraction Watch data is CC BY 4.0, attribution: Retraction
Watch / Crossref. OpenAlex data is CC0.
