"""
run_all.py

Runs the whole pipeline in the correct order, from raw data through to the
final figure. Equivalent to running each numbered script one at a time.

Usage:
    cd scripts
    python run_all.py
"""

import runpy
from pathlib import Path

STEPS = [
    "01_data_wrangling.py",
    "02_data_preparation_sampling.py",
    "03_descriptive_statistics.py",
    "04_confidence_interval.py",
    "05_hypothesis_testing.py",
    "06_visualization.py",
]

SCRIPTS_DIR = Path(__file__).resolve().parent / "Questions2"

for step in STEPS:
    print("\n" + "#" * 78)
    print(f"# RUNNING {step}")
    print("#" * 78)
    runpy.run_path(str(SCRIPTS_DIR / step), run_name="__main__")

print("\nPipeline complete. See the data/processed/ and results/ folders "
      "for every generated file.")
