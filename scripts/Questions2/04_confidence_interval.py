"""
04_confidence_interval.py

SKILL: Inferential statistics - Confidence Interval

Reads the analysis-ready sample and computes a 95% confidence interval,
using the t-distribution (appropriate because n is small and the
population standard deviation is unknown), for the population mean
possession difference between winning and losing teams.

Saves the result to results/confidence_interval.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
from Questions2._paths import ANALYSIS_SAMPLE_CSV, CONFIDENCE_INTERVAL_TXT

print("=" * 78)
print("STEP 4: 95% CONFIDENCE INTERVAL FOR THE MEAN POSSESSION DIFFERENCE")
print("=" * 78)

df = pd.read_csv(ANALYSIS_SAMPLE_CSV)
n = len(df)
mean_diff = df["diff"].mean()
sd_diff = df["diff"].std(ddof=1)
se_diff = sd_diff / np.sqrt(n)
degf = n - 1

confidence_level = 0.95
t_crit = stats.t.ppf(1 - (1 - confidence_level) / 2, degf)
ci_low = mean_diff - t_crit * se_diff
ci_high = mean_diff + t_crit * se_diff

report = f"""
95% CONFIDENCE INTERVAL FOR THE MEAN POSSESSION DIFFERENCE
(winner_poss - loser_poss), FIFA World Cup 2026 sample (n={n})

Sample mean difference       : {mean_diff:.2f} percentage points
Sample std dev of difference : {sd_diff:.2f} percentage points
Standard error                : {se_diff:.2f}
Degrees of freedom (n-1)      : {degf}
t critical value (95%, two-tailed) : {t_crit:.3f}

95% CI for the population mean difference : [{ci_low:.2f}, {ci_high:.2f}] pp

Interpretation: we are 95% confident that, across the population of
decisive FIFA World Cup 2026 matches, the winning team's average
possession share exceeds the losing team's by somewhere between
{ci_low:.2f} and {ci_high:.2f} percentage points. Because this interval
{"does" if ci_low <= 0 <= ci_high else "does not"} contain 0, the data
{"are" if ci_low <= 0 <= ci_high else "are not"} consistent with there
being no true difference on average.
"""
print(report)

with open(CONFIDENCE_INTERVAL_TXT, "w") as f:
    f.write(report)
print(f"Saved confidence interval report to: {CONFIDENCE_INTERVAL_TXT}")
