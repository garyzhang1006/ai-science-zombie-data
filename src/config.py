"""Shared configuration for the zombie-citation pipeline.

All stages read from here. Override via environment variables where noted.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Polite-pool contact for OpenAlex / Crossref. REQUIRED for sane rate limits.
# Never hardcode an address here; the repo is public.
MAILTO = os.environ.get("OA_MAILTO", "")

# Retraction Watch public CSV (Crossref-hosted, CC BY 4.0).
RW_CSV_URL = (
    "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/"
    "retraction_watch.csv"
)

# Study window (publication dates of CITING papers).
START_DATE = "2018-01-01"
END_MONTH = "2026-06"  # last complete-ish indexed month; OpenAlex lags ~6-8 wk

# ChatGPT release = ITS break point.
BREAK_MONTH = "2022-12"
PLACEBO_MONTHS = [f"{y}-{m:02d}" for y in (2019, 2020, 2021) for m in range(1, 13)]

# A citation is "zombie" only if the citing paper appears at least
# BUFFER_MONTHS after the cited paper's retraction date (indexing lag buffer).
BUFFER_MONTHS = 6

# Artifact phrase registry (prong 2). Assembled from Strzelecki (2025) and
# our own screening.
#
# OpenAlex full-text search is STEMMED, so a phrase whose words are common
# scientific vocabulary collides badly. "Regenerate response" was in this
# list and matched "regenerative response": 89.4% of its hits predate
# ChatGPT, and they are zebrafish and stem-cell regeneration papers. It is
# excluded, and the pre-ChatGPT hit rate of every surviving phrase is
# reported in out/phrase_validation.csv as a measured false-positive rate
# rather than an assumption. See PHRASE_ERA_CUTOFF.
ARTIFACT_PHRASES = [
    "as an AI language model",
    "as a large language model, I",
    "as of my last knowledge update",
    "I cannot browse the internet",
    "I don't have access to real-time",
    "as an AI developed by OpenAI",
    "my knowledge cutoff",
]

# ChatGPT was released 2022-11-30, so an artifact phrase in a paper
# published before 2023 cannot be unedited model output. Those hits both
# get excluded from the cohort and serve as the filter's negative control.
PHRASE_ERA_CUTOFF = 2023

# Meta-discussion filter (papers ABOUT LLMs quote the phrases deliberately).
META_TITLE_RE = (
    r"(chatgpt|gpt-?[345o]|large language|language model|\bllm\b|chatbot|"
    r"generative a\.?i\.?|artificial intelligence|ai-generated|ai text|"
    r"detect|plagiar|academic integrity)"
)
META_ABSTRACT_TERMS = [
    "language model", "chatgpt", "detect", "plagiar", "integrity",
    "artifact", "prevalence", "ai-generated", "generative ai", "chatbot",
]
# OpenAlex subfields treated as meta-discussion venues for these phrases.
META_SUBFIELD_IDS = {
    "https://openalex.org/subfields/1702",  # Artificial Intelligence
    "https://openalex.org/subfields/1703",  # Computational Theory and Mathematics
}

# Control matching (prong 2).
N_CONTROLS = 5
MATCH_TOL = 0.25          # relative tolerance on reference-list length
CONTROL_POOL_SIZE = 40    # candidates fetched per artifact paper before ranking
# A single venue-year can hold hundreds of thousands of papers (PLOS ONE
# alone), so the scan reservoir-samples candidates instead of keeping all
# of them; matching only ever needs a few dozen per artifact paper.
CONTROL_POOL_PER_VENUE_YEAR = 60
LEXICAL_SAMPLE_CAP = 150_000   # reservoir size per shard

# Unresolvable-reference check (Crossref).
FUZZY_THRESHOLD = 85      # rapidfuzz token_sort_ratio, 0-100

# Prong 3: lexical AI score.
# Excess-vocabulary markers, from Kobak et al. (2024) arXiv:2406.07016 and
# Liang et al. (2025). A paper's score = number of DISTINCT markers in its
# abstract; S = 1 (proxy says AI-assisted) if score >= LEXICAL_THRESHOLD.
LEXICAL_MARKERS = [
    "delve", "delves", "delving", "intricate", "intricacies", "intricately",
    "showcasing", "showcases", "showcased", "underscore", "underscores",
    "underscoring", "pivotal", "realm", "crucial", "crucially", "notably",
    "boasts", "meticulous", "meticulously", "commendable", "invaluable",
    "unwavering", "multifaceted", "nuanced", "paramount", "leverage",
    "leveraging", "seamless", "seamlessly", "holistic", "fostering",
    "elucidate", "elucidates", "elucidating", "unravel", "unraveling",
]
LEXICAL_THRESHOLD = 2
SAMPLE_PER_YEAR = 5000    # sampled works per year, 2023-2026
SAMPLE_SEED = 20260813    # fixed for reproducibility
UNRESOLVABLE_SUBSAMPLE = 2000  # Crossref checks are slow; cap the subsample

# Sensitivity/specificity grid for the misclassification bounds.
#
# Requiring at least LEXICAL_THRESHOLD distinct markers from a 37-term list
# flags 0.86% of 2023-onward papers, so the rule is high-specificity and
# low-sensitivity by construction, and the grid has to sit in that regime.
# The data also imposes a hard identification constraint: the prevalence
# solves q = (P(S=1) - (1-spec)) / (sens - (1-spec)), so any specificity
# below 1 - P(S=1) = 0.991 implies more false positives than observed
# positives and admits no solution at all. An earlier grid centered on
# 0.90-0.99 specificity was infeasible at every point for exactly that
# reason.
SENS_GRID = [0.03, 0.05, 0.10, 0.20]
SPEC_GRID = [0.992, 0.995, 0.998, 0.999]
