
# ============================================================
# MDAS — NEW COLAB NOTEBOOK INITIALIZATION
# Continue from Step 40 without repeating previous experiments
# ============================================================

from google.colab import drive
from pathlib import Path
import pandas as pd
import numpy as np
import json

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive")

PROJECT = ROOT / "MDAS_Registration_Aware_Revision_2026"

STEP39 = PROJECT / "Step39_Original_Estimator_Numerical_Replay"
STEP40 = PROJECT / "Step40_Reproduction_Discrepancy_Audit"
STEP30 = PROJECT / "Step30_Frozen_Potsdam_Benchmark"

required = {
    "Original NCC replay":
        STEP39 / "Step39_NCC_Replay.csv",
    "Original phase replay":
        STEP39 / "Step39_Phase_Replay.csv",
    "Step 39 report":
        STEP39 / "Step39_Final_Report.json",
    "Potsdam benchmark":
        STEP30 / "Step30_Final_Report.json"
}

print("=" * 70)
print("MDAS — NEW NOTEBOOK INITIALIZATION")
print("=" * 70)

for name, path in required.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    print(f"VERIFIED: {name}")

report = json.loads(
    (STEP39 / "Step39_Final_Report.json").read_text()
)

print("\nPrevious experimental results:")
print("NCC reproduced:", report["ncc_matches"], "/ 680")
print("Phase reproduced:", report["phase_matches"], "/ 680")

step40_report = STEP40 / "Step40_Final_Report.json"

if step40_report.exists():
    print("\nStep 40 already completed:")
    print(json.loads(step40_report.read_text())["status"])
else:
    print("\nNEXT: Run Step 40 in this notebook.")

print("\nREADY — Previous results preserved.")
