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
# our own screening. Order matters only for reporting.
ARTIFACT_PHRASES = [
    "as an AI language model",
    "as a large language model, I",
    "as of my last knowledge update",
    "I cannot browse the internet",
    "I don't have access to real-time",
    "as an AI developed by OpenAI",
    "my knowledge cutoff",
    "Regenerate response",
]

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

# Sensitivity/specificity grid for the misclassification bounds. The point
# values bracket what the validation experiments in Kobak et al. (2024) and
# Liang et al. (2025) support for marker-based detection at our threshold;
# the bounds are reported over the whole grid, never at a single point.
SENS_GRID = [0.45, 0.55, 0.65, 0.75]
SPEC_GRID = [0.90, 0.93, 0.96, 0.99]
