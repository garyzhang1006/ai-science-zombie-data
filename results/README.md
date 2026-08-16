# Results archive

Every derived output behind the paper, from the run on the OpenAlex
snapshot of 2026-06-26. Large tables are gzipped.

| File | What it is |
|---|---|
| `population_series.csv.gz` | Monthly zombie citations and references, by field, 2018-2026 |
| `its_results.json` | Interrupted time series, 36 placebo breaks, per-field fits, end-month sensitivity |
| `table1.json` | Cohort rates and cluster-bootstrap intervals for both endpoints |
| `bounds.json` | Partial-identification bounds over the sensitivity/specificity grid |
| `phrase_validation.csv.gz` | Per-phrase false-positive rate against the pre-ChatGPT era |
| `artifact_candidates.csv.gz` | Every candidate with its keep/drop decision and reason |
| `artifact_cohort.csv.gz` | The 1,731 retained artifact papers |
| `hand_validation_sample.csv.gz` | Seeded 100-paper sample for manual precision checking |
| `matches.csv.gz` | Artifact-to-control matching |
| `unresolvable.csv.gz`, `s05_unresolvable.jsonl.gz` | Crossref reference-resolution outcomes |
| `lexical_sample.csv.gz`, `s03_lexical_merged.csv.gz` | Lexical-proxy sample with scores and outcomes |
| `s03_numer_*.csv.gz`, `s03_denom_*.csv.gz` | Per-shard population aggregates |
| `s03_cohort_merged.csv.gz`, `s03_control_merged.csv.gz` | Per-paper cohort and control outcomes |
| `retracted_dois.csv.gz` | Retraction Watch entries with dates, after parsing |
| `tk_values.json`, `tk_report.md` | Every number cited in the paper, with its source key |
| `figures/` | The three paper figures as PDFs |
| `logs/` | Kaggle kernel logs for each stage, as run provenance |

Note on shard tags: files are named `<i>of<n>`. Stage 4 merges only the
largest complete sharding and reports any stray tag, since a debug run over
a single file leaves aggregates that would otherwise be double-counted.
