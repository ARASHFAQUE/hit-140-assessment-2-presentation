"""
05_hypothesis_testing.py

SKILL: Inferential statistics - One-Sample t-Test

H0: mu_diff = 0   (winning teams have, on average, the SAME possession as
                    losing teams)
H1: mu_diff > 0   (winning teams have, on average, HIGHER possession)

winner_poss and loser_poss come from the SAME match and sum to 100%, so
they are NOT independent samples. Treating them as two separate,
unrelated groups of teams would violate the independence assumption of an
independent two-sample t-test. The correct test collapses each match to a
single paired difference (diff = winner_poss - loser_poss) and runs a
ONE-SAMPLE t-test on that difference against a hypothesised value of 0 --
algebraically identical to a paired-samples t-test.

For teaching purposes, this script ALSO runs the (incorrect) independent
two-sample t-test on the same data, to show numerically how ignoring the
pairing changes -- and inflates -- the apparent significance.

Saves the full report to results/hypothesis_test_results.txt
"""

import numpy as np
import pandas as pd
import math
from Questions2._paths import ANALYSIS_SAMPLE_CSV, HYPOTHESIS_TEST_TXT


def _betacf(x, a, b):
    qab, qap, qam = a + b, a + b + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    h = 1.0 / d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            break
    return h


def _t_two_sided(t, df):
    x = df / (df + t * t)
    a, b = df / 2, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return 2 * front * _betacf(x, a, b) / a
    return 2 * (1 - front * _betacf(1 - x, b, a) / b)


def _one_sample_t(values):
    values = np.asarray(values, dtype=float)
    t = values.mean() / (values.std(ddof=1) / math.sqrt(len(values)))
    return t, _t_two_sided(t, len(values) - 1)


def _independent_t(x, y):
    nx, ny = len(x), len(y)
    pooled = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    t = (np.mean(x) - np.mean(y)) / math.sqrt(pooled * (1 / nx + 1 / ny))
    return t, _t_two_sided(t, nx + ny - 2)

print("=" * 78)
print("STEP 5: ONE-SAMPLE t-TEST ON PAIRED POSSESSION DIFFERENCES")
print("=" * 78)

df = pd.read_csv(ANALYSIS_SAMPLE_CSV)
n = len(df)
alpha = 0.05

# 5a. Correct test: one-sample t-test on the paired differences

t_correct, p_two_correct = _one_sample_t(df["diff"])
p_one_correct = p_two_correct / 2 if t_correct > 0 else 1 - p_two_correct / 2
verdict_correct = "REJECT H0" if p_one_correct < alpha else "FAIL TO REJECT H0"

# 5b. For comparison only: independent two-sample t-test (INCORRECT here)

t_wrong, p_two_wrong = _independent_t(df["winner_poss"], df["loser_poss"])
p_one_wrong = p_two_wrong / 2 if t_wrong > 0 else 1 - p_two_wrong / 2
verdict_wrong = "REJECT H0" if p_one_wrong < alpha else "FAIL TO REJECT H0"

r_corr = np.corrcoef(df["winner_poss"], df["loser_poss"])[0, 1]

report = f"""
ONE-SAMPLE t-TEST ON PAIRED POSSESSION DIFFERENCES (n={n} matches)

H0: mu_diff = 0      H1: mu_diff > 0      alpha = {alpha}

CORRECT APPROACH -- one-sample t-test on diff = winner_poss - loser_poss
(equivalent to a paired-samples t-test):
    t statistic                         : {t_correct:.3f}
    degrees of freedom                  : {n - 1}
    p-value (two-tailed)                : {p_two_correct:.4f}
    p-value (one-tailed, H1: mean>0)    : {p_one_correct:.4f}
    Decision at alpha = {alpha}                : {verdict_correct}


FOR COMPARISON ONLY -- (incorrect) independent two-sample t-test,
treating winner_poss and loser_poss as if they were two unrelated
groups of teams:
    t statistic                         : {t_wrong:.3f}
    p-value (two-tailed)                : {p_two_wrong:.4f}
    p-value (one-tailed, H1: mean>0)    : {p_one_wrong:.4f}
    Decision at alpha = {alpha}                : {verdict_wrong}

Correlation between winner_poss and loser_poss in this sample: r = {r_corr:.3f}

WHY THEY DIFFER

Because winner_poss + loser_poss = 100 in every match, the two columns
are perfectly negatively correlated (r = -1), not independent. The
independent-samples formula estimates the standard error of the
difference as sqrt(Var(winner)/n + Var(loser)/n), which completely
ignores that negative correlation. The true variance of the paired
difference is Var(winner) + Var(loser) + 2*|Cov| -- roughly double what
the independent formula assumes here. The independent test therefore
UNDERSTATES the true standard error, producing an artificially larger
|t| and a smaller ("more significant") p-value than is actually
justified. This is exactly why match-level possession data must be
analysed with a one-sample test on the PAIRED differences, not an
independent two-sample test.
"""
print(report)

with open(HYPOTHESIS_TEST_TXT, "w") as f:
    f.write(report)
print(f"Saved hypothesis test report to: {HYPOTHESIS_TEST_TXT}")
