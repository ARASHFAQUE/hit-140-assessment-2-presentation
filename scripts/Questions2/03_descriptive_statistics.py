"""
03_descriptive_statistics.py
==============================
SKILL: Descriptive statistics

Reads the analysis-ready sample from data/processed/analysis_sample.csv
and computes mean, median, sample standard deviation, sample variance,
min, max, range and IQR for:
    - winner_poss
    - loser_poss
    - diff (winner_poss - loser_poss)

Saves the results table to results/descriptive_statistics.csv
"""

import statistics as stats_lib
import numpy as np
import pandas as pd
from Questions1._paths import ANALYSIS_SAMPLE_CSV, DESCRIPTIVE_STATS_CSV

print("=" * 78)
print("STEP 3: DESCRIPTIVE STATISTICS")
print("=" * 78)

df = pd.read_csv(ANALYSIS_SAMPLE_CSV)
print(f"Loaded analysis sample: {ANALYSIS_SAMPLE_CSV}  (n={len(df)})")


def describe(series: pd.Series) -> dict:
    s = series.astype(float)
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    return {
        "n": len(s),
        "mean": stats_lib.mean(s),
        "median": stats_lib.median(s),
        "sample_std": s.std(ddof=1),
        "sample_var": s.var(ddof=1),
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
        "IQR": q3 - q1,
    }


desc_table = pd.DataFrame({
    "winner_poss": describe(df["winner_poss"]),
    "loser_poss": describe(df["loser_poss"]),
    "diff": describe(df["diff"]),
}).T
desc_table.index.name = "variable"

print("\n" + desc_table.round(2).to_string())

n_more = (df["diff"] > 0).sum()
n_less = (df["diff"] < 0).sum()
print(f"\nMatches where winner had MORE possession : {n_more} / {len(df)}")
print(f"Matches where winner had LESS possession : {n_less} / {len(df)}")

desc_table.round(2).to_csv(DESCRIPTIVE_STATS_CSV)
print(f"\nSaved descriptive statistics to: {DESCRIPTIVE_STATS_CSV}")
