"""
02_data_preparation_sampling.py
=================================
SKILL: Data preparation and sampling

Reads the WRANGLED match-level data produced by 01_data_wrangling.py, then:
    1. Engineers the analysis variable: diff = winner_poss - loser_poss
    2. Defines and documents the POPULATION vs the SAMPLE, and the
       sampling technique used (see the printed / saved summary).
    3. Demonstrates a simple-random-sampling function (the "textbook"
       technique for drawing a sample from a sampling frame), even
       though the main analysis uses the full purposive sample -- see
       the explanation printed below for why.
    4. Saves the final analysis-ready table to
       data/processed/analysis_sample.csv

Run this after 01_data_wrangling.py.
"""

import numpy as np
import pandas as pd
from Questions1._paths import WRANGLED_MATCH_CSV, ANALYSIS_SAMPLE_CSV, SAMPLING_SUMMARY_TXT

print("=" * 78)
print("STEP 2: DATA PREPARATION AND SAMPLING")
print("=" * 78)

df = pd.read_csv(WRANGLED_MATCH_CSV)
print(f"Loaded wrangled data: {WRANGLED_MATCH_CSV}  ({len(df)} matches)")

 
# 2a. Feature engineering
 
df["diff"] = df["winner_poss"] - df["loser_poss"]


# 2b. Population vs sample, and sampling technique

TOTAL_MATCHES_PLAYED = 104          # 72 group + 32 knockout, FIFA World Cup 2026
APPROX_DECISIVE_POPULATION = 90     # knockout matches always decisive; most,
                                     # but not all, group matches are too

summary = f"""
POPULATION
----------
All *decisive* matches of the FIFA World Cup 2026 -- i.e. every knockout
match (which always has a winner) plus every group-stage match that did
NOT end in a draw. "Winner possession" and "loser possession" are
undefined for a drawn group match, so those are excluded from the
population for this question.
    Total matches played at the tournament : {TOTAL_MATCHES_PLAYED}
    Approx. size of the decisive population : ~{APPROX_DECISIVE_POPULATION}
      (72 group matches, most but not all decisive, + 32 knockout matches,
      all decisive by definition)

SAMPLE
------
    Sample size (n)      : {len(df)}
    Sampling technique   : Purposive (judgement) sampling, NOT simple
                            random sampling.
    Why not SRS          : Ball-possession-by-match is not centrally
                            published for all ~90 decisive matches in one
                            place (unlike goals or results). Each figure
                            in this sample had to be individually verified
                            against an official match-centre provider
                            (Sky Sports/Opta, TNT Sports/Eurosport) or a
                            wire-service recap. A true SRS would require a
                            complete, verifiable possession figure for
                            every match in the population, which was not
                            available.
    How matches were chosen : Deliberately spread across every stage of
                            the draw (group stage, Round of 32, Round of
                            16, quarter-final, semi-final, 3rd-place
                            play-off, Final) and deliberately including
                            BOTH comfortable possession-dominant wins
                            (e.g. Spain 67.9% vs Belgium) AND shock,
                            low-possession wins (e.g. Paraguay won on
                            penalties with only 24.4% possession), so the
                            sample is not cherry-picked to only support
                            the hypothesis.
    Unit of analysis     : one MATCH (not one team). winner_poss and
                            loser_poss come from the same match and sum to
                            100%, so they are not independent -- this is
                            why the inferential tests later treat "diff"
                            as a single paired sample per match.

VARIABLES CARRIED FORWARD
--------------------------
    winner_poss  : winning team's share of possession (%)
    loser_poss   : losing team's share of possession (%)
    diff         : winner_poss - loser_poss (percentage points)
"""
print(summary)

with open(SAMPLING_SUMMARY_TXT, "w") as f:
    f.write(summary)
print(f"Saved sampling documentation to: {SAMPLING_SUMMARY_TXT}")

# 2c. Demonstration: simple random sampling technique (for reference)

def simple_random_sample(data: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Draws a simple random sample of n rows WITHOUT replacement from `data`.
    This is the standard technique taught for drawing a sample from a
    known sampling frame, shown here for completeness/reproducibility.
    Not used for the main analysis below (see summary above for why),
    but easy to switch on if a stricter random-sampling exercise is later
    required, e.g.:  df_srs = simple_random_sample(df, n=8)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(data.index, size=n, replace=False)
    return data.loc[sorted(idx)].reset_index(drop=True)

demo_n = max(1, len(df) - 5)
demo_srs = simple_random_sample(df, n=demo_n, seed=42)
print(f"Demonstration only (not used further) -- a {demo_n}-match simple "
      f"random sub-sample drawn from the {len(df)} curated matches, seed=42:")
print(demo_srs[["match_id", "score_display"]].to_string(index=False))

# 2d. Save the analysis-ready dataset (full n=11 purposive sample)

analysis_cols = ["match_id", "round", "score_display", "winner", "loser",
                  "winner_poss", "loser_poss", "diff", "decided_by"]
df[analysis_cols].to_csv(ANALYSIS_SAMPLE_CSV, index=False)
print(f"\nSaved analysis-ready data ({len(df)} rows) to: {ANALYSIS_SAMPLE_CSV}")
