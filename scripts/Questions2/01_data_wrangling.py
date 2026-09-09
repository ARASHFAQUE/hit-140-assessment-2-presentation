"""
01_data_wrangling.py
=====================
SKILL: Data wrangling

Reads the RAW data (data/raw/team_match_possession_raw.csv), which is in
"long" format: one row per TEAM per match (so every team that appears in
the sample has its own row -- 22 rows for 11 matches). This mirrors how
match statistics are actually published (e.g. FBref/Opta team logs).

This script:
    1. Loads the raw file and checks it for problems (missing values,
       each match having exactly 2 rows, possession summing to 100%).
    2. Wrangles/reshapes it from long (team-per-row) to wide (match-per-row),
       identifying which of the two teams won and which lost, so the
       possession of the winner and loser sit side by side in one row.
    3. Saves the cleaned, match-level table to
       data/processed/match_level_possession_wrangled.csv

Run this first -- every later script depends on its output.
"""

import pandas as pd
from Questions1._paths import RAW_TEAM_MATCH_CSV, WRANGLED_MATCH_CSV

print("=" * 78)
print("STEP 1: DATA WRANGLING")
print("=" * 78)

# 1a. Load raw data

raw = pd.read_csv(RAW_TEAM_MATCH_CSV)
print(f"Loaded raw file: {RAW_TEAM_MATCH_CSV}")
print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns "
      f"({raw['match_id'].nunique()} unique matches, "
      f"{raw.shape[0]} team-match rows)")


# 1b. Clean / validate

issues = []

# No missing values in the columns we need
required_cols = ["match_id", "round", "team", "opponent", "result",
                  "decided_by", "possession_pct"]
missing = raw[required_cols].isna().sum()
if missing.sum() > 0:
    issues.append(f"Missing values found:\n{missing[missing > 0]}")

# Every match must have exactly 2 rows (one per team)
counts = raw.groupby("match_id").size()
bad_matches = counts[counts != 2]
if len(bad_matches) > 0:
    issues.append(f"Matches without exactly 2 team-rows: {bad_matches.to_dict()}")

# Possession for the two teams in a match should sum to (approximately) 100
poss_sums = raw.groupby("match_id")["possession_pct"].sum().round(1)
bad_sums = poss_sums[(poss_sums < 99.0) | (poss_sums > 101.0)]
if len(bad_sums) > 0:
    issues.append(f"Match(es) where possession does not sum to ~100%: "
                   f"{bad_sums.to_dict()}")

# Each match should have exactly one W and one L result
result_check = raw.groupby("match_id")["result"].apply(lambda s: sorted(s.tolist()))
bad_results = result_check[result_check.apply(lambda x: x != ["L", "W"])]
if len(bad_results) > 0:
    issues.append(f"Match(es) without exactly one W and one L: "
                   f"{bad_results.to_dict()}")

if issues:
    print("\nDATA QUALITY ISSUES FOUND:")
    for i in issues:
        print(" -", i)
    raise ValueError("Fix data quality issues in the raw CSV before proceeding.")
else:
    print("Data quality checks passed: no missing values, every match has "
          "exactly one winner + one loser row, and possession sums to 100%.")


# 1c. Reshape: long (team-per-row) -> wide (match-per-row)

winners = raw[raw["result"] == "W"].rename(columns={
    "team": "winner", "possession_pct": "winner_poss"
})[["match_id", "round", "winner", "winner_poss", "decided_by"]]

losers = raw[raw["result"] == "L"].rename(columns={
    "team": "loser", "possession_pct": "loser_poss"
})[["match_id", "loser", "loser_poss"]]

wrangled = winners.merge(losers, on="match_id", how="inner")

# A readable score column, e.g. "Spain 2-1 Belgium"
scores = raw[raw["result"] == "W"][["match_id", "team", "team_score",
                                     "opponent", "opponent_score"]]
scores["score_display"] = (scores["team"] + " " + scores["team_score"].astype(str)
                            + "-" + scores["opponent_score"].astype(str) + " "
                            + scores["opponent"])
wrangled = wrangled.merge(scores[["match_id", "score_display"]], on="match_id")

wrangled = wrangled[["match_id", "round", "score_display", "winner", "loser",
                      "winner_poss", "loser_poss", "decided_by"]]
wrangled = wrangled.sort_values("match_id").reset_index(drop=True)

print(f"\nWrangled (match-level) shape: {wrangled.shape[0]} rows x "
      f"{wrangled.shape[1]} columns")
print(wrangled.to_string(index=False))


# 1d. Save

wrangled.to_csv(WRANGLED_MATCH_CSV, index=False)
print(f"\nSaved wrangled match-level data to: {WRANGLED_MATCH_CSV}")
