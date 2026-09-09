"""
06_visualization.py

Builds three charts from the analysis-ready sample:
    (a) A boxplot comparing the distribution of winner_poss vs loser_poss
        side by side (the classic way to compare two groups -- see the
        course's own boxplot material).
    (b) A paired slope plot: possession of the losing team -> winning team
        for every match (red line = winner had LESS possession).
    (c) A histogram of the paired possession differences with the sample
        mean and 95% CI overlaid.

The figure is BOTH:
    - displayed in the output window (plt.show()) -- when this script is
      run in a normal Python environment (terminal with a GUI backend,
      Jupyter, Spyder, VS Code interactive, etc.) a window/cell pops up
      with the chart, exactly like the boxplot/histogram examples in the
      course material.
    - saved to disk at results/figures/possession_diff_plot.png, so it is
      still available even after the window is closed, or if you run this
      in a headless/no-display environment (e.g. a plain server or CI
      job), where plt.show() is a harmless no-op.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from Questions2._paths import ANALYSIS_SAMPLE_CSV, FIGURE_PNG

print("=" * 78)
print("STEP 6: VISUALISATION")
print("=" * 78)

df = pd.read_csv(ANALYSIS_SAMPLE_CSV)
n = len(df)
mean_diff = df["diff"].mean()
sd_diff = df["diff"].std(ddof=1)
se_diff = sd_diff / np.sqrt(n)
t_crit = stats.t.ppf(0.975, n - 1)
ci_low, ci_high = mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))


# (a) Boxplot: winner_poss vs loser_poss

ax0 = axes[0]
box_data = pd.DataFrame({
    "Winning team": df["winner_poss"],
    "Losing team": df["loser_poss"],
})
sns.boxplot(data=box_data, ax=ax0, palette=["tab:blue", "tab:red"])
sns.stripplot(data=box_data, ax=ax0, color="black", alpha=0.6, size=5, jitter=0.08)
ax0.set_ylabel("Ball possession (%)")
ax0.set_title(f"Possession: winner vs loser\n(n={n} matches)")


# (b) Paired slope plot

ax1 = axes[1]
for _, row in df.iterrows():
    colour = "tab:red" if row["diff"] < 0 else "tab:blue"
    ax1.plot([0, 1], [row["loser_poss"], row["winner_poss"]],
             marker="o", color=colour, alpha=0.8)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Losing team", "Winning team"])
ax1.set_ylabel("Ball possession (%)")
ax1.set_title("Possession per match: loser -> winner\n(red = winner had LESS possession)")
ax1.set_xlim(-0.2, 1.2)


# (c) Histogram of paired differences

ax2 = axes[2]
ax2.hist(df["diff"], bins=8, color="tab:blue", edgecolor="white")
ax2.axvline(0, color="black", linestyle="--", linewidth=1, label="No difference (H0)")
ax2.axvline(mean_diff, color="tab:orange", linewidth=2,
            label=f"Sample mean = {mean_diff:.1f}pp")
ax2.axvspan(ci_low, ci_high, color="tab:orange", alpha=0.15, label="95% CI")
ax2.set_xlabel("Possession difference, winner - loser (percentage points)")
ax2.set_ylabel("Number of matches")
ax2.set_title("Distribution of paired possession differences")
ax2.legend(fontsize=8)

plt.tight_layout()

# Save to file first (so the file exists even if plt.show() blocks/closes
# the figure), then display in the output window.
plt.savefig(FIGURE_PNG, dpi=150)
print(f"Saved figure to: {FIGURE_PNG}")

print("Displaying figure in the output window (close the window / cell to "
      "continue if running interactively)...")
plt.show()
