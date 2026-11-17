
# =============================================================================
# STEP 42 — EXTERNAL CALIBRATION TRANSFER FAILURE AUDIT
#
# Research objective:
#   Attribute the approximately (+1,+1) pixel external
#   residual to the original MDAS bias correction,
#   while distinguishing that effect from the underlying
#   registration estimator.
#
# Inputs:
#   Step41C_Registration_Results.csv
#   Step41C_Final_Report.json
#   Step13_Frozen_Gate.json
#   Step20_Frozen_Method.json
#
# Methods:
#   A. Frozen original NCC + gate (unchanged)
#   B. Frozen original bounded phase (unchanged)
#   C. No correction baseline
#   D. NCC pre-bias estimate + ORIGINAL identity gate
#   E. Phase pre-bias estimate + ORIGINAL identity gate
#
# D and E are POST HOC DIAGNOSTIC COUNTERFACTUALS,
# not externally validated competing methods.
#
# No registration functions rerun.
# No model parameters optimized.
# No historical or external files overwritten.
#
# Outputs:
#   Step42_Bias_Attribution_Cases.csv
#   Step42_Descriptive_Method_Summary.csv
#   Step42_Aligned_Residuals.csv
#   Step42_Patch_Level_Comparison.csv
#   Step42_Shift_Magnitude_Comparison.csv
#   Step42_Frozen_vs_Counterfactual.png/.pdf
#   Step42_Aligned_Residual_Vectors.png/.pdf
#   Step42_Final_Report.json
# =============================================================================

from google.colab import drive
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

drive.mount("/content/drive")

PROJECT = Path(
    "/content/drive/MyDrive/MDAS_Registration_Aware_Revision_2026"
)

S13 = PROJECT / "Step13_Identity_Preserving_Gate"
S20 = PROJECT / "Step20_Bounded_Phase_Correlation"
S41 = PROJECT / "Step41C_Frozen_Potsdam_Transfer"

OUT = PROJECT / "Step42_External_Calibration_Transfer_Audit"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_RESULTS = S41 / "Step41C_Registration_Results.csv"
INPUT_REPORT = S41 / "Step41C_Final_Report.json"
NCC_CONFIG = S13 / "Step13_Frozen_Gate.json"
PHASE_CONFIG = S20 / "Step20_Frozen_Method.json"

OUTPUT_NAMES = [
    "Step42_Bias_Attribution_Cases.csv",
    "Step42_Descriptive_Method_Summary.csv",
    "Step42_Aligned_Residuals.csv",
    "Step42_Patch_Level_Comparison.csv",
    "Step42_Shift_Magnitude_Comparison.csv",
    "Step42_Frozen_vs_Counterfactual.png",
    "Step42_Frozen_vs_Counterfactual.pdf",
    "Step42_Aligned_Residual_Vectors.png",
    "Step42_Aligned_Residual_Vectors.pdf",
    "Step42_Final_Report.json",
]

existing = [
    x for x in OUTPUT_NAMES if (OUT / x).exists()
]
if existing:
    raise FileExistsError(
        "Step42 outputs already exist. Preserve them "
        "and use a different output directory."
    )

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path, value):
    path.write_text(
        json.dumps(
            value, indent=2, allow_nan=False, default=str
        ),
        encoding="utf-8"
    )

print("=" * 88)
print("STEP 42 — EXTERNAL CALIBRATION TRANSFER FAILURE AUDIT")
print("=" * 88)

for path in [
    INPUT_RESULTS, INPUT_REPORT, NCC_CONFIG, PHASE_CONFIG
]:
    if not path.is_file():
        raise FileNotFoundError(path)

df = pd.read_csv(INPUT_RESULTS)
previous = load_json(INPUT_REPORT)
ncfg = load_json(NCC_CONFIG)
pcfg = load_json(PHASE_CONFIG)

assert previous["status"] == (
    "EXTERNAL_FROZEN_REGISTRATION_EVALUATION_COMPLETE"
)
assert len(df) == 255

NCC_METHOD = "frozen_ncc_step13"
PHASE_METHOD = "frozen_phase_step20_size_adapted"
IDENTITY = "identity"

assert set(df["method"]) == {
    NCC_METHOD, PHASE_METHOD, IDENTITY
}

KEYS = ["patch_id", "condition_index"]

assert not df.duplicated(KEYS + ["method"]).any()
assert df["patch_id"].nunique() == 5

for method in [NCC_METHOD, PHASE_METHOD, IDENTITY]:
    assert int((df["method"] == method).sum()) == 85

NCC_GATE = float(ncfg["threshold_pixels"])
PHASE_GATE = float(pcfg["tau"])

NCC_BIAS_DY = -1.0
NCC_BIAS_DX = -1.0

PHASE_BIAS_DY = float(pcfg["bias_dy"])
PHASE_BIAS_DX = float(pcfg["bias_dx"])

assert NCC_GATE == 0.225
assert PHASE_GATE == 0.25
assert int(pcfg["radius"]) == 5

numeric_cols = [
    "true_dy", "true_dx",
    "raw_dy", "raw_dx",
    "estimated_dy", "estimated_dx", "epe"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="raise")

assert np.isfinite(df[numeric_cols].to_numpy(float)).all()

print("\n42A — VERIFIED FROZEN INPUTS")
print("Records:", len(df))
print("Source patches:", df["patch_id"].nunique())
print("NCC historical bias:", NCC_BIAS_DY, NCC_BIAS_DX)
print("Phase historical bias:", PHASE_BIAS_DY, PHASE_BIAS_DX)

# -------------------------------------------------------------------------
# B — RECONSTRUCT PRE-BIAS DISPLACEMENTS
#
# Original NCC saved 'raw_dy/raw_dx' in Step41C
# actually contain the refined, BIAS-CORRECTED
# estimator outputs from estimate_both().
#
# Therefore:
#   NCC pre-bias = saved refined + historical bias
#
# Original phase saved 'raw_dy/raw_dx' are the genuine
# raw FFT displacements, before the bias correction.
# -------------------------------------------------------------------------

counterfactual_rows = []

for method, sub in df.groupby("method", sort=False):

    sub = sub.copy()

    if method == NCC_METHOD:

        pre_y = sub["raw_dy"].to_numpy(float) + NCC_BIAS_DY
        pre_x = sub["raw_dx"].to_numpy(float) + NCC_BIAS_DX

        gate = NCC_GATE
        name = "ncc_prebias_same_gate_diagnostic"

    elif method == PHASE_METHOD:

        pre_y = sub["raw_dy"].to_numpy(float)
        pre_x = sub["raw_dx"].to_numpy(float)

        gate = PHASE_GATE
        name = "phase_prebias_same_gate_diagnostic"

    else:
        continue

    mag = np.hypot(pre_y, pre_x)
    snap = mag < gate

    diagnostic_y = np.where(snap, 0.0, pre_y)
    diagnostic_x = np.where(snap, 0.0, pre_x)

    true_y = sub["true_dy"].to_numpy(float)
    true_x = sub["true_dx"].to_numpy(float)

    diagnostic_epe = np.hypot(
        diagnostic_y - true_y,
        diagnostic_x - true_x
    )

    for i, row in enumerate(sub.itertuples(index=False)):
        counterfactual_rows.append({
            "patch_id": row.patch_id,
            "condition_index": int(row.condition_index),
            "true_dy": float(true_y[i]),
            "true_dx": float(true_x[i]),
            "shift_magnitude": float(np.hypot(
                true_y[i], true_x[i]
            )),
            "method": name,
            "estimated_dy": float(diagnostic_y[i]),
            "estimated_dx": float(diagnostic_x[i]),
            "epe": float(diagnostic_epe[i]),
            "snapped": bool(snap[i]),
            "is_diagnostic_counterfactual": True,
            "prebias_dy": float(pre_y[i]),
            "prebias_dx": float(pre_x[i])
        })

diagnostics = pd.DataFrame(counterfactual_rows)

# Historical frozen results are retained directly from Step41C.
frozen = df[
    KEYS + [
        "true_dy", "true_dx", "method",
        "estimated_dy", "estimated_dx", "epe"
    ]
].copy()

frozen["shift_magnitude"] = np.hypot(
    frozen["true_dy"], frozen["true_dx"]
)
frozen["is_diagnostic_counterfactual"] = False
frozen["snapped"] = (
    df["snapped"].astype(str).str.lower().isin(["true", "1"])
)
frozen["prebias_dy"] = np.nan
frozen["prebias_dx"] = np.nan

all_cases = pd.concat(
    [frozen, diagnostics],
    ignore_index=True
)

assert len(all_cases) == 425
assert (all_cases.groupby("method").size() == 85).all()
assert not all_cases.duplicated(KEYS + ["method"]).any()

all_cases.to_csv(
    OUT / "Step42_Bias_Attribution_Cases.csv",
    index=False
)

# -------------------------------------------------------------------------
# C — VERIFY THAT FROZEN ORIGINAL RESULTS WERE NOT ALTERED
# -------------------------------------------------------------------------

for method in [NCC_METHOD, PHASE_METHOD, IDENTITY]:

    a = df[df["method"] == method].sort_values(KEYS)
    b = all_cases[
        all_cases["method"] == method
    ].sort_values(KEYS)

    assert np.array_equal(
        a["epe"].to_numpy(float),
        b["epe"].to_numpy(float)
    )

    assert np.array_equal(
        a["estimated_dy"].to_numpy(float),
        b["estimated_dy"].to_numpy(float)
    )

    assert np.array_equal(
        a["estimated_dx"].to_numpy(float),
        b["estimated_dx"].to_numpy(float)
    )

print("\n42B — ORIGINAL STEP41C RESULTS VERIFIED UNCHANGED")

# -------------------------------------------------------------------------
# D — METHOD SUMMARIES
# -------------------------------------------------------------------------

summary_rows = []

for method, group in all_cases.groupby("method"):

    aligned = group[
        group["shift_magnitude"] == 0.0
    ]

    summary_rows.append({
        "method": method,
        "status": (
            "POST_HOC_DIAGNOSTIC"
            if group["is_diagnostic_counterfactual"].iloc[0]
            else "FROZEN_ORIGINAL_OR_BASELINE"
        ),
        "cases": len(group),
        "mean_epe": float(group["epe"].mean()),
        "median_epe": float(group["epe"].median()),
        "max_epe": float(group["epe"].max()),
        "within_025": float((group["epe"] <= 0.25).mean()),
        "within_050": float((group["epe"] <= 0.50).mean()),
        "aligned_cases": len(aligned),
        "aligned_preserved": int((
            (aligned["estimated_dy"] == 0)
            & (aligned["estimated_dx"] == 0)
        ).sum()),
        "aligned_mean_epe": float(aligned["epe"].mean()),
        "signed_dy_bias": float(
            (group["estimated_dy"] - group["true_dy"]).mean()
        ),
        "signed_dx_bias": float(
            (group["estimated_dx"] - group["true_dx"]).mean()
        )
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "Step42_Descriptive_Method_Summary.csv",
    index=False
)

print("\n42C — FROZEN VERSUS POST HOC DIAGNOSTIC")
print(summary.round(6).to_string(index=False))

# -------------------------------------------------------------------------
# E — PAIRED ATTRIBUTION AT THE SAME 85 CONDITIONS
# -------------------------------------------------------------------------

paired_rows = []

pairs = [
    (NCC_METHOD, "ncc_prebias_same_gate_diagnostic"),
    (
        PHASE_METHOD,
        "phase_prebias_same_gate_diagnostic"
    )
]

for frozen_name, diagnostic_name in pairs:

    f = all_cases[
        all_cases["method"] == frozen_name
    ][KEYS + ["epe", "estimated_dy", "estimated_dx"]].copy()

    d = all_cases[
        all_cases["method"] == diagnostic_name
    ][KEYS + ["epe", "estimated_dy", "estimated_dx"]].copy()

    combined = f.merge(
        d, on=KEYS,
        suffixes=("_frozen", "_prebias"),
        validate="one_to_one"
    )

    for row in combined.itertuples(index=False):
        paired_rows.append({
            "patch_id": row.patch_id,
            "condition_index": int(row.condition_index),
            "method_family": frozen_name,
            "frozen_epe": float(row.epe_frozen),
            "prebias_epe": float(row.epe_prebias),
            "diagnostic_epe_reduction": float(
                row.epe_frozen - row.epe_prebias
            )
        })

attribution = pd.DataFrame(paired_rows)

print("\n42D — POST HOC BIAS ATTRIBUTION")

for family, group in attribution.groupby("method_family"):
    print(
        family,
        "| frozen mean:",
        round(float(group["frozen_epe"].mean()), 6),
        "| prebias diagnostic mean:",
        round(float(group["prebias_epe"].mean()), 6),
        "| mean difference:",
        round(float(
            group["diagnostic_epe_reduction"].mean()
        ), 6)
    )

# -------------------------------------------------------------------------
# F — PATCH AND SHIFT GROUP SUMMARY
# -------------------------------------------------------------------------

patch_summary = (
    all_cases.groupby(
        ["patch_id", "method"],
        as_index=False
    )
    .agg(
        cases=("epe", "size"),
        mean_epe=("epe", "mean"),
        max_epe=("epe", "max")
    )
)

patch_summary.to_csv(
    OUT / "Step42_Patch_Level_Comparison.csv",
    index=False
)

shift_group = (
    all_cases.groupby(
        ["method", "shift_magnitude"],
        as_index=False
    )
    .agg(
        cases=("epe", "size"),
        mean_epe=("epe", "mean"),
        median_epe=("epe", "median"),
        max_epe=("epe", "max")
    )
)

shift_group.to_csv(
    OUT / "Step42_Shift_Magnitude_Comparison.csv",
    index=False
)

aligned = all_cases[
    all_cases["shift_magnitude"] == 0.0
].copy()

aligned["signed_error_dy"] = aligned["estimated_dy"]
aligned["signed_error_dx"] = aligned["estimated_dx"]

aligned.to_csv(
    OUT / "Step42_Aligned_Residuals.csv",
    index=False
)

# -------------------------------------------------------------------------
# G — FIGURE 1: MEAN EPE BY DISPLACEMENT
# -------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

styles = {
    NCC_METHOD: ("Frozen NCC", "#205781", "o", "-"),
    PHASE_METHOD: ("Frozen phase", "#D1762A", "s", "-"),
    "ncc_prebias_same_gate_diagnostic": (
        "NCC pre-bias diagnostic", "#4382AF", "o", "--"
    ),
    "phase_prebias_same_gate_diagnostic": (
        "Phase pre-bias diagnostic", "#E7A268", "s", "--"
    ),
    IDENTITY: ("No correction", "#666666", "^", ":")
}

fig, ax = plt.subplots(figsize=(8.5, 5.2))

for method, (label, color, marker, linestyle) in styles.items():

    sub = shift_group[
        shift_group["method"] == method
    ].sort_values("shift_magnitude")

    ax.plot(
        sub["shift_magnitude"],
        sub["mean_epe"],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.9,
        markersize=5.5
    )

ax.set_xlabel("Applied displacement magnitude (pixels)")
ax.set_ylabel("Mean endpoint error (pixels)")

ax.set_xticks(
    sorted(shift_group["shift_magnitude"].unique())
)

ax.grid(axis="y", alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=9, ncol=2)

fig.tight_layout()

fig.savefig(
    OUT / "Step42_Frozen_vs_Counterfactual.png",
    dpi=900, bbox_inches="tight", facecolor="white"
)

fig.savefig(
    OUT / "Step42_Frozen_vs_Counterfactual.pdf",
    bbox_inches="tight", facecolor="white"
)

plt.show()
plt.close(fig)

# -------------------------------------------------------------------------
# H — FIGURE 2: ALIGNED RESIDUAL VECTOR DISTRIBUTION
# -------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 6))

vector_styles = [
    (NCC_METHOD, "#205781", "o", "Frozen NCC"),
    (PHASE_METHOD, "#D1762A", "s", "Frozen phase"),
    (
        "ncc_prebias_same_gate_diagnostic",
        "#4382AF", "o", "NCC pre-bias diagnostic"
    ),
    (
        "phase_prebias_same_gate_diagnostic",
        "#E7A268", "s", "Phase pre-bias diagnostic"
    )
]

for method, color, marker, label in vector_styles:

    sub = aligned[
        aligned["method"] == method
    ]

    ax.scatter(
        sub["signed_error_dx"],
        sub["signed_error_dy"],
        s=64, color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.83,
        label=label,
        zorder=3
    )

ax.scatter(
    [0], [0],
    marker="+",
    color="black",
    s=120,
    linewidth=1.7,
    label="Zero displacement",
    zorder=5
)

ax.axhline(0, color="#888888", linewidth=0.8)
ax.axvline(0, color="#888888", linewidth=0.8)

ax.set_xlabel("Horizontal residual, dx (pixels)")
ax.set_ylabel("Vertical residual, dy (pixels)")

ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.13)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=8,
    loc="best"
)

fig.tight_layout()

fig.savefig(
    OUT / "Step42_Aligned_Residual_Vectors.png",
    dpi=900, bbox_inches="tight", facecolor="white"
)

fig.savefig(
    OUT / "Step42_Aligned_Residual_Vectors.pdf",
    bbox_inches="tight", facecolor="white"
)

plt.show()
plt.close(fig)

# -------------------------------------------------------------------------
# I — FINAL REPORT
# -------------------------------------------------------------------------

report = {
    "experiment": (
        "External MDAS frozen-bias transfer attribution"
    ),
    "status": "POST_HOC_CALIBRATION_TRANSFER_AUDIT_COMPLETED",
    "source_step41c_results_sha256": sha256(INPUT_RESULTS),
    "source_step41c_report_sha256": sha256(INPUT_REPORT),
    "source_acquisitions": 1,
    "patches": 5,
    "conditions_per_patch": 17,
    "frozen_method_records": 255,
    "diagnostic_counterfactual_records": 170,
    "historical_ncc_bias": [
        NCC_BIAS_DY, NCC_BIAS_DX
    ],
    "historical_phase_bias": [
        PHASE_BIAS_DY, PHASE_BIAS_DX
    ],
    "original_ncc_gate": NCC_GATE,
    "original_phase_gate": PHASE_GATE,
    "summary": summary.to_dict(orient="records"),
    "paired_mean_diagnostic_epe_reductions": {
        family: float(
            group["diagnostic_epe_reduction"].mean()
        )
        for family, group in attribution.groupby(
            "method_family"
        )
    },
    "registration_estimators_rerun": False,
    "frozen_registration_parameters_modified": False,
    "independent_external_improvement_claim_supported": False,
    "external_fusion_models_evaluated": False,
    "interpretation": [
        "The frozen methods are the primary external results.",
        "Bias-neutral estimates are post hoc counterfactuals.",
        "Any apparent improvement cannot be reported as "
        "independent validation of a new method.",
        "A small number of patches from one acquisition "
        "does not establish broad generalization.",
        "The external sensor interface and synthetic MSI "
        "differ from the original MDAS training domain."
    ],
    "output_directory": str(OUT)
}

save_json(
    OUT / "Step42_Final_Report.json",
    report
)

print("\n" + "=" * 88)
print("STEP 42 — FINAL REPORT")
print("=" * 88)
print(json.dumps(report, indent=2))

print("\nSTEP 42 COMPLETE")
print("All Step 41C results and frozen configurations preserved.")
