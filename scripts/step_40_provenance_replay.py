
# =============================================================================
# STEP 40 — HISTORICAL REPRODUCTION DISCREPANCY RESOLUTION
#
# Goal:
#   Resolve the two Step 39 phase-correlation discrepancies.
#
# Input:
#   Step39_NCC_Replay.csv
#   Step39_Phase_Replay.csv
#   Step39_Final_Report.json
#   Step20_Test_Registration.csv
#   Step20_Frozen_Method.json
#   Step13_Gated_Test_Registration.csv
#
# Checks:
#   - Exact saved-record consistency
#   - Errors at 1e-5 and 3e-5 pixel tolerances
#   - Detailed inspection of all mismatched cases
#   - Frozen phase bias and snapping decisions
#   - Whether small raw differences change EPE or identity decisions
#   - Original NCC source-replay completeness
#
# No estimator reruns.
# No parameter tuning or corrections.
# Original historical files remain unchanged.
# =============================================================================

from google.colab import drive
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

drive.mount("/content/drive")

PROJECT = Path(
    "/content/drive/MyDrive/"
    "MDAS_Registration_Aware_Revision_2026"
)

STEP13 = PROJECT / "Step13_Identity_Preserving_Gate"
STEP20 = PROJECT / "Step20_Bounded_Phase_Correlation"
STEP39 = PROJECT / "Step39_Original_Estimator_Numerical_Replay"

OUT = PROJECT / "Step40_Reproduction_Discrepancy_Audit"
OUT.mkdir(parents=True, exist_ok=True)

NCC_REPLAY = STEP39 / "Step39_NCC_Replay.csv"
PHASE_REPLAY = STEP39 / "Step39_Phase_Replay.csv"
STEP39_REPORT = STEP39 / "Step39_Final_Report.json"

PHASE_HISTORY = STEP20 / "Step20_Test_Registration.csv"
PHASE_CONFIG = STEP20 / "Step20_Frozen_Method.json"

GATE_HISTORY = STEP13 / "Step13_Gated_Test_Registration.csv"

FILES = {
    "ncc_replay": NCC_REPLAY,
    "phase_replay": PHASE_REPLAY,
    "step39_report": STEP39_REPORT,
    "phase_history": PHASE_HISTORY,
    "phase_config": PHASE_CONFIG,
    "gate_history": GATE_HISTORY,
}

OUTPUTS = [
    "Step40_All_Phase_Replay_Differences.csv",
    "Step40_Phase_Mismatch_Details.csv",
    "Step40_Tolerance_Sensitivity.csv",
    "Step40_Replayed_Phase_Decisions.csv",
    "Step40_Reproduction_Summary.csv",
    "Step40_Final_Report.json",
]

if any((OUT / filename).exists() for filename in OUTPUTS):
    raise FileExistsError(
        "Existing Step 40 results found. "
        "Preserve the previous output and choose a new directory."
    )

for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def key_frame(frame):
    result = frame.copy()

    result["image_key"] = (
        result["image"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )

    for name in ["true_dy", "true_dx"]:
        result[name] = pd.to_numeric(
            result[name], errors="raise"
        ).round(6)

    return result

KEY = ["image_key", "true_dy", "true_dx"]

print("=" * 88)
print("STEP 40 — MDAS ESTIMATOR REPRODUCTION AUDIT")
print("=" * 88)

ncc = key_frame(pd.read_csv(NCC_REPLAY))
phase = key_frame(pd.read_csv(PHASE_REPLAY))
history = key_frame(pd.read_csv(PHASE_HISTORY))
gate = key_frame(pd.read_csv(GATE_HISTORY))

config = json.loads(
    PHASE_CONFIG.read_text(encoding="utf-8")
)

prior = json.loads(
    STEP39_REPORT.read_text(encoding="utf-8")
)

assert len(ncc) == 680
assert len(phase) == 680
assert len(history) == 680
assert len(gate) == 680

for frame, label in [
    (ncc, "NCC"),
    (phase, "phase"),
    (history, "historical phase"),
    (gate, "Step13 gate")
]:
    if frame.duplicated(KEY).any():
        raise ValueError(
            f"Duplicate image/shift keys in {label}"
        )

assert prior["ncc_cases"] == 680
assert prior["phase_cases"] == 680

print("\n40A — VERIFIED INPUTS")
print("NCC replay cases:", len(ncc))
print("Phase replay cases:", len(phase))
print("Historical phase records:", len(history))
print("Saved NCC matches at 1e-5:", prior["ncc_matches"])
print("Saved phase matches at 1e-5:", prior["phase_matches"])

# -------------------------------------------------------------------------
# 40B — RECOMPUTE NUMERICAL DIFFERENCES
# -------------------------------------------------------------------------

ncc["recomputed_difference"] = np.hypot(
    ncc["replayed_dy"] - ncc["recorded_dy"],
    ncc["replayed_dx"] - ncc["recorded_dx"]
)

phase["recomputed_difference"] = np.hypot(
    phase["replayed_raw_dy"] - phase["recorded_raw_dy"],
    phase["replayed_raw_dx"] - phase["recorded_raw_dx"]
)

assert np.allclose(
    ncc["recomputed_difference"],
    ncc["difference_epe"],
    atol=1e-10,
    rtol=1e-5
)

assert np.allclose(
    phase["recomputed_difference"],
    phase["difference_epe"],
    atol=1e-10,
    rtol=1e-5
)

assert np.isfinite(
    ncc["recomputed_difference"]
).all()

assert np.isfinite(
    phase["recomputed_difference"]
).all()

strict_tol = 1e-5
review_tol = 3e-5

strict_ncc = (
    ncc["recomputed_difference"] <= strict_tol
)

strict_phase = (
    phase["recomputed_difference"] <= strict_tol
)

review_ncc = (
    ncc["recomputed_difference"] <= review_tol
)

review_phase = (
    phase["recomputed_difference"] <= review_tol
)

print("\n40B — REPRODUCTION COUNTS")
print(
    "NCC at 1e-5:",
    int(strict_ncc.sum()),
    "/ 680"
)
print(
    "Phase at 1e-5:",
    int(strict_phase.sum()),
    "/ 680"
)
print(
    "Phase at 3e-5:",
    int(review_phase.sum()),
    "/ 680"
)

# -------------------------------------------------------------------------
# 40C — RECONCILE AGAINST ACTUAL HISTORICAL PHASE CSV
# -------------------------------------------------------------------------

historical_fields = [
    "raw_dy", "raw_dx",
    "estimated_dy", "estimated_dx",
    "snapped", "epe"
]

merged = phase.merge(
    history[KEY + historical_fields],
    on=KEY,
    how="left",
    validate="one_to_one",
    indicator=True
)

if not merged["_merge"].eq("both").all():
    raise RuntimeError(
        "Step39 replay does not match all historical phase keys."
    )

assert np.allclose(
    merged["recorded_raw_dy"],
    merged["raw_dy"],
    rtol=0,
    atol=1e-9
)

assert np.allclose(
    merged["recorded_raw_dx"],
    merged["raw_dx"],
    rtol=0,
    atol=1e-9
)

# -------------------------------------------------------------------------
# 40D — APPLY FROZEN PHASE POSTPROCESSING TO BOTH
#        RECORDED AND REPLAYED RAW ESTIMATES
# -------------------------------------------------------------------------

bias_y = float(config["bias_dy"])
bias_x = float(config["bias_dx"])
tau = float(config["tau"])

assert int(config["radius"]) == 5
assert np.isclose(tau, 0.25)

def apply_phase_postprocessing(raw_y, raw_x):

    corrected_y = raw_y - bias_y
    corrected_x = raw_x - bias_x

    magnitude = np.hypot(
        corrected_y,
        corrected_x
    )

    snapped = magnitude < tau

    final_y = np.where(
        snapped, 0.0, corrected_y
    )

    final_x = np.where(
        snapped, 0.0, corrected_x
    )

    return final_y, final_x, snapped, magnitude

recorded_final_y, recorded_final_x, recorded_snap, recorded_mag = (
    apply_phase_postprocessing(
        merged["raw_dy"].to_numpy(float),
        merged["raw_dx"].to_numpy(float)
    )
)

replayed_final_y, replayed_final_x, replayed_snap, replayed_mag = (
    apply_phase_postprocessing(
        merged["replayed_raw_dy"].to_numpy(float),
        merged["replayed_raw_dx"].to_numpy(float)
    )
)

# Confirm previously established Step38 postprocessing.
assert np.allclose(
    recorded_final_y,
    merged["estimated_dy"],
    rtol=0,
    atol=1e-8
)

assert np.allclose(
    recorded_final_x,
    merged["estimated_dx"],
    rtol=0,
    atol=1e-8
)

historical_snap = (
    merged["snapped"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
    .to_numpy(bool)
)

assert np.array_equal(
    recorded_snap,
    historical_snap
)

raw_epe_difference = merged[
    "recomputed_difference"
].to_numpy(float)

final_displacement_difference = np.hypot(
    replayed_final_y - recorded_final_y,
    replayed_final_x - recorded_final_x
)

recorded_epe = np.hypot(
    recorded_final_y - merged["true_dy"].to_numpy(float),
    recorded_final_x - merged["true_dx"].to_numpy(float)
)

replayed_epe = np.hypot(
    replayed_final_y - merged["true_dy"].to_numpy(float),
    replayed_final_x - merged["true_dx"].to_numpy(float)
)

epe_change = np.abs(
    replayed_epe - recorded_epe
)

snap_changed = (
    replayed_snap != recorded_snap
)

decision_table = merged[
    KEY + [
        "recorded_raw_dy",
        "recorded_raw_dx",
        "replayed_raw_dy",
        "replayed_raw_dx"
    ]
].copy()

decision_table["raw_difference_px"] = raw_epe_difference
decision_table["recorded_corrected_magnitude"] = recorded_mag
decision_table["replayed_corrected_magnitude"] = replayed_mag
decision_table["recorded_snapped"] = recorded_snap
decision_table["replayed_snapped"] = replayed_snap
decision_table["snap_decision_changed"] = snap_changed
decision_table["final_displacement_difference_px"] = (
    final_displacement_difference
)
decision_table["recorded_epe"] = recorded_epe
decision_table["replayed_epe"] = replayed_epe
decision_table["absolute_epe_change"] = epe_change

decision_table.to_csv(
    OUT / "Step40_Replayed_Phase_Decisions.csv",
    index=False
)

phase.to_csv(
    OUT / "Step40_All_Phase_Replay_Differences.csv",
    index=False
)

# -------------------------------------------------------------------------
# 40E — INSPECT THE TWO STRICT-TOLERANCE FAILURES
# -------------------------------------------------------------------------

mismatch_mask = (
    raw_epe_difference > strict_tol
)

mismatches = decision_table.loc[
    mismatch_mask
].copy()

mismatches = mismatches.sort_values(
    "raw_difference_px",
    ascending=False
)

mismatches.to_csv(
    OUT / "Step40_Phase_Mismatch_Details.csv",
    index=False
)

print("\n40C — STRICT-TOLERANCE PHASE MISMATCHES")
print("Number:", len(mismatches))

if len(mismatches):
    print(
        mismatches[
            [
                "image_key",
                "true_dy",
                "true_dx",
                "raw_difference_px",
                "snap_decision_changed",
                "final_displacement_difference_px",
                "absolute_epe_change"
            ]
        ].to_string(index=False)
    )

print("\n40D — DECISION STABILITY")
print("Changed identity decisions:", int(snap_changed.sum()))
print(
    "Maximum final displacement difference:",
    float(final_displacement_difference.max())
)
print(
    "Maximum EPE difference:",
    float(epe_change.max())
)

# -------------------------------------------------------------------------
# 40F — TOLERANCE SENSITIVITY
# -------------------------------------------------------------------------

tolerance_values = [
    1e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5
]

tolerance_rows = []

for tolerance in tolerance_values:
    tolerance_rows.append({
        "tolerance_pixels": tolerance,
        "ncc_matches": int(
            (ncc["recomputed_difference"] <= tolerance).sum()
        ),
        "phase_matches": int(
            (phase["recomputed_difference"] <= tolerance).sum()
        ),
        "ncc_max_difference_px": float(
            ncc["recomputed_difference"].max()
        ),
        "phase_max_difference_px": float(
            phase["recomputed_difference"].max()
        )
    })

tolerance_df = pd.DataFrame(tolerance_rows)

tolerance_df.to_csv(
    OUT / "Step40_Tolerance_Sensitivity.csv",
    index=False
)

print("\n40E — TOLERANCE SENSITIVITY")
print(tolerance_df.to_string(index=False))

# -------------------------------------------------------------------------
# 40G — FINAL AUDIT
# -------------------------------------------------------------------------

ncc_full_at_strict = bool(strict_ncc.all())
phase_full_at_review = bool(review_phase.all())
phase_decisions_stable = bool(
    not snap_changed.any()
)

# Pass criterion for agreement at the documented
# review tolerance, not bitwise reproduction.
reproduction_consistent = bool(
    ncc_full_at_strict
    and phase_full_at_review
    and phase_decisions_stable
)

if reproduction_consistent:
    status = "NUMERICAL_REPRODUCTION_CONSISTENT"
else:
    status = "REPRODUCTION_DISCREPANCIES_REMAIN"

summary = pd.DataFrame([
    {
        "method": "original_ncc",
        "cases": 680,
        "matches_at_1e5": int(strict_ncc.sum()),
        "matches_at_3e5": int(review_ncc.sum()),
        "mean_difference_px": float(
            ncc["recomputed_difference"].mean()
        ),
        "max_difference_px": float(
            ncc["recomputed_difference"].max()
        )
    },
    {
        "method": "original_phase_raw",
        "cases": 680,
        "matches_at_1e5": int(strict_phase.sum()),
        "matches_at_3e5": int(review_phase.sum()),
        "mean_difference_px": float(
            phase["recomputed_difference"].mean()
        ),
        "max_difference_px": float(
            phase["recomputed_difference"].max()
        )
    }
])

summary.to_csv(
    OUT / "Step40_Reproduction_Summary.csv",
    index=False
)

report = {
    "experiment": "Historical estimator numerical discrepancy audit",
    "status": status,
    "ncc_cases": 680,
    "phase_cases": 680,
    "ncc_matches_at_1e_minus_5": int(strict_ncc.sum()),
    "phase_matches_at_1e_minus_5": int(strict_phase.sum()),
    "ncc_matches_at_3e_minus_5": int(review_ncc.sum()),
    "phase_matches_at_3e_minus_5": int(review_phase.sum()),
    "ncc_max_difference_px": float(
        ncc["recomputed_difference"].max()
    ),
    "phase_max_difference_px": float(
        phase["recomputed_difference"].max()
    ),
    "phase_strict_mismatch_cases": int(
        mismatch_mask.sum()
    ),
    "phase_identity_snap_decisions_changed": int(
        snap_changed.sum()
    ),
    "phase_max_final_displacement_difference_px": float(
        final_displacement_difference.max()
    ),
    "phase_max_absolute_epe_change": float(
        epe_change.max()
    ),
    "original_parameters_changed": False,
    "original_estimators_rerun": False,
    "external_potsdam_evaluated": False,
    "source_step39_report_sha256": sha256(STEP39_REPORT),
    "source_ncc_replay_sha256": sha256(NCC_REPLAY),
    "source_phase_replay_sha256": sha256(PHASE_REPLAY),
    "limitations": [
        "The 3e-5 tolerance is a stated numerical agreement "
        "criterion, not proof of bitwise identical computation.",
        "Replays use the previously explored MDAS test mosaic.",
        "Stable snap decisions do not establish the same "
        "FFT surface in every numerical detail.",
        "Potsdam remains a single independent acquisition.",
        "No external transfer or fusion checkpoint testing "
        "is performed here."
    ],
    "output_directory": str(OUT)
}

(OUT / "Step40_Final_Report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 88)
print("STEP 40 — FINAL REPORT")
print("=" * 88)
print(json.dumps(report, indent=2))

print("\nSTEP 40 COMPLETE")
print("No estimator reruns or parameter changes.")
print("Historical and external data unchanged.")
