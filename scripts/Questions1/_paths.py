"""
_paths.py
---------
Shared folder locations for the whole pipeline, so every script writes to
(and reads from) the same folders regardless of where it is run from.

Not a standalone analysis step -- imported by the numbered scripts.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Make sure every folder exists no matter which script runs first
for _dir in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# File names used throughout the pipeline
RAW_TEAM_MATCH_CSV = RAW_DIR / "team_match_possession_raw.csv"
WRANGLED_MATCH_CSV = PROCESSED_DIR / "match_level_possession_wrangled.csv"
ANALYSIS_SAMPLE_CSV = PROCESSED_DIR / "analysis_sample.csv"

SAMPLING_SUMMARY_TXT = RESULTS_DIR / "sampling_summary.txt"
DESCRIPTIVE_STATS_CSV = RESULTS_DIR / "descriptive_statistics.csv"
CONFIDENCE_INTERVAL_TXT = RESULTS_DIR / "confidence_interval.txt"
HYPOTHESIS_TEST_TXT = RESULTS_DIR / "hypothesis_test_results.txt"
FIGURE_PNG = FIGURES_DIR / "possession_diff_plot.png"
