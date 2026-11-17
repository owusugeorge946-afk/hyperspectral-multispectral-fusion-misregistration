
# =============================================================================
# STEP 43 — MDAS FINAL EXTERNAL VALIDATION RESULTS PACKAGE
#
# Purpose:
#   Consolidate frozen external transfer and exploratory bias attribution
#   without mixing confirmatory and post hoc results.
#
# Sources:
#   Step41C original frozen Potsdam transfer
#   Step42 external calibration-transfer diagnostic
#   Step40 historical reproduction verification
#
# Outputs:
#   Step43_Table_External_Performance.csv
#   Step43_Table_Patch_Level.csv
#   Step43_Table_Bias_Attribution.csv
#   Step43_Figure_External_EPE.png/.pdf
#   Step43_Figure_Aligned_Residuals.png/.pdf
#   Step43_Figure_Captions.txt
#   Step43_Final_Report.json
#
# Figure requirements:
#   - White background, no decorative titles
#   - Print-readable fonts and aligned axes
#   - PNG at 900 dpi and vector PDF
#   - No confidence intervals treating 85 shifts as independent scenes
#
# Does not retrain, rerun registration, or modify source outputs.
# =============================================================================

from google.colab import drive
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

drive.mount("/content/drive")

P = Path(
    "/content/drive/MyDrive/"
    "MDAS_Registration_Aware_Revision_2026"
)

S40 = P / "Step40_Reproduction_Discrepancy_Audit"
S41 = P / "Step41C_Frozen_Potsdam_Transfer"
S42 = P / "Step42_External_Calibration_Transfer_Audit"

OUT = P / "Step43_External_Manuscript_Results"
OUT.mkdir(parents=True, exist_ok=True)

outputs = [
    "Step43_Table_External_Performance.csv",
    "Step43_Table_Patch_Level.csv",
    "Step43_Table_Bias_Attribution.csv",
    "Step43_Figure_External_EPE.png",
    "Step43_Figure_External_EPE.pdf",
    "Step43_Figure_Aligned_Residuals.png",
    "Step43_Figure_Aligned_Residuals.pdf",
    "Step43_Figure_Captions.txt",
    "Step43_Final_Report.json"
]

existing = [x for x in outputs if (OUT / x).exists()]
if existing:
    raise FileExistsError(
        "Previous Step43 outputs exist; preserving them:\n"
        + "\n".join(existing)
    )

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

sources = {
    "historical_reproduction":
        S40 / "Step40_Final_Report.json",
    "frozen_results":
        S41 / "Step41C_Registration_Results.csv",
    "frozen_report":
        S41 / "Step41C_Final_Report.json",
    "diagnostic_results":
        S42 / "Step42_Bias_Attribution_Cases.csv",
    "diagnostic_report":
        S42 / "Step42_Final_Report.json"
}

for name, path in sources.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")

r40 = read_json(sources["historical_reproduction"])
r41 = read_json(sources["frozen_report"])
r42 = read_json(sources["diagnostic_report"])

assert r40["status"] == "NUMERICAL_REPRODUCTION_CONSISTENT"
assert r41["status"] == (
    "EXTERNAL_FROZEN_REGISTRATION_EVALUATION_COMPLETE"
)
assert r42["status"] == (
    "POST_HOC_CALIBRATION_TRANSFER_AUDIT_COMPLETED"
)

# Confirm Step42 is tied to the exact same original Step41C CSV.
assert r42["source_step41c_results_sha256"] == digest(
    sources["frozen_results"]
)

frozen = pd.read_csv(sources["frozen_results"])
combined = pd.read_csv(sources["diagnostic_results"])

assert len(frozen) == 255
assert len(combined) == 425

KEYS = ["patch_id", "condition_index"]

METHODS = {
    "frozen_ncc_step13": {
        "label": "NCC, frozen",
        "status": "Primary frozen transfer",
        "color": "#205781"
    },
    "frozen_phase_step20_size_adapted": {
        "label": "Phase, frozen",
        "status": "Primary frozen transfer",
        "color": "#C16C25"
    },
    "identity": {
        "label": "No correction",
        "status": "Baseline",
        "color": "#707070"
    },
    "ncc_prebias_same_gate_diagnostic": {
        "label": "NCC, bias omitted",
        "status": "Post hoc diagnostic",
        "color": "#6199C3"
    },
    "phase_prebias_same_gate_diagnostic": {
        "label": "Phase, bias omitted",
        "status": "Post hoc diagnostic",
        "color": "#E0A371"
    }
}

if set(combined["method"]) != set(METHODS):
    raise ValueError(
        "Unexpected Step42 methods: "
        + str(sorted(combined["method"].unique()))
    )

if combined.duplicated(KEYS + ["method"]).any():
    raise RuntimeError("Duplicate experimental condition keys")

assert combined["patch_id"].nunique() == 5
assert (combined.groupby("method").size() == 85).all()

for column in [
    "true_dy", "true_dx",
    "estimated_dy", "estimated_dx", "epe"
]:
    combined[column] = pd.to_numeric(
        combined[column], errors="raise"
    )

assert np.isfinite(
    combined[
        ["true_dy", "true_dx",
         "estimated_dy", "estimated_dx", "epe"]
    ].to_numpy(float)
).all()

# Independently verify the EPE calculations.
calculated_epe = np.hypot(
    combined["estimated_dy"] - combined["true_dy"],
    combined["estimated_dx"] - combined["true_dx"]
)

if not np.allclose(
    calculated_epe, combined["epe"], rtol=0, atol=1e-10
):
    raise RuntimeError("Stored endpoint error mismatch")

# Check original frozen results are identical across steps.
for method in [
    "frozen_ncc_step13",
    "frozen_phase_step20_size_adapted",
    "identity"
]:
    a = frozen[frozen["method"] == method].sort_values(KEYS)
    b = combined[combined["method"] == method].sort_values(KEYS)

    if not np.allclose(
        a["epe"].to_numpy(float),
        b["epe"].to_numpy(float),
        atol=0,
        rtol=0
    ):
        raise RuntimeError(
            f"Original frozen EPE changed: {method}"
        )

print("=" * 88)
print("STEP 43 — VERIFIED MANUSCRIPT RESULTS PACKAGE")
print("=" * 88)
print("Historical reproduction: PASSED")
print("Frozen experiment: 255 records")
print("Post hoc diagnostics: 170 records")
print("Independent external acquisitions: 1")
print("Spatially distinct patches: 5")

# -------------------------------------------------------------------------
# 43A — MAIN EXTERNAL PERFORMANCE TABLE
# -------------------------------------------------------------------------

table_rows = []

for method, values in METHODS.items():
    group = combined[combined["method"] == method]
    aligned = group[
        (group["true_dy"] == 0)
        & (group["true_dx"] == 0)
    ]

    table_rows.append({
        "method": values["label"],
        "method_id": method,
        "evidence_class": values["status"],
        "source_scenes": 1,
        "patches": int(group["patch_id"].nunique()),
        "conditions": int(len(group)),
        "mean_epe_px": float(group["epe"].mean()),
        "median_epe_px": float(group["epe"].median()),
        "maximum_epe_px": float(group["epe"].max()),
        "within_025_pct": float(
            100 * (group["epe"] <= 0.25).mean()
        ),
        "within_050_pct": float(
            100 * (group["epe"] <= 0.5).mean()
        ),
        "aligned_preserved": int((
            (aligned["estimated_dy"] == 0)
            & (aligned["estimated_dx"] == 0)
        ).sum()),
        "aligned_cases": int(len(aligned)),
        "aligned_mean_epe_px": float(
            aligned["epe"].mean()
        )
    })

table = pd.DataFrame(table_rows)
table.to_csv(
    OUT / "Step43_Table_External_Performance.csv",
    index=False
)

print("\n43A — EXTERNAL REGISTRATION TABLE")
print(
    table[
        [
            "method", "evidence_class", "mean_epe_px",
            "within_025_pct", "aligned_preserved"
        ]
    ].round(4).to_string(index=False)
)

# -------------------------------------------------------------------------
# 43B — PATCH-LEVEL RESULTS
# -------------------------------------------------------------------------

patch_table = (
    combined
    .groupby(["patch_id", "method"], as_index=False)
    .agg(
        conditions=("epe", "size"),
        mean_epe_px=("epe", "mean"),
        maximum_epe_px=("epe", "max")
    )
)

patch_table["evidence_class"] = (
    patch_table["method"]
    .map(lambda m: METHODS[m]["status"])
)

patch_table.to_csv(
    OUT / "Step43_Table_Patch_Level.csv",
    index=False
)

# -------------------------------------------------------------------------
# 43C — PAIRED FROZEN VERSUS BIAS-OMITTED DIFFERENCES
# -------------------------------------------------------------------------

pairs = [
    (
        "frozen_ncc_step13",
        "ncc_prebias_same_gate_diagnostic",
        "NCC"
    ),
    (
        "frozen_phase_step20_size_adapted",
        "phase_prebias_same_gate_diagnostic",
        "Phase correlation"
    )
]

attribution_rows = []

for frozen_id, diagnostic_id, label in pairs:
    a = combined[
        combined["method"] == frozen_id
    ][KEYS + ["epe"]].rename(
        columns={"epe": "frozen_epe"}
    )

    b = combined[
        combined["method"] == diagnostic_id
    ][KEYS + ["epe"]].rename(
        columns={"epe": "diagnostic_epe"}
    )

    comparison = a.merge(
        b, on=KEYS, validate="one_to_one"
    )

    comparison["difference"] = (
        comparison["frozen_epe"]
        - comparison["diagnostic_epe"]
    )

    patch_differences = (
        comparison
        .groupby("patch_id")["difference"]
        .mean()
    )

    attribution_rows.append({
        "family": label,
        "paired_conditions": len(comparison),
        "independent_source_scenes": 1,
        "patches": len(patch_differences),
        "frozen_mean_epe_px": float(
            comparison["frozen_epe"].mean()
        ),
        "diagnostic_mean_epe_px": float(
            comparison["diagnostic_epe"].mean()
        ),
        "posthoc_mean_reduction_px": float(
            comparison["difference"].mean()
        ),
        "patches_with_positive_reduction": int(
            (patch_differences > 0).sum()
        ),
        "interpretation": "Exploratory, not independent validation"
    })

attribution_table = pd.DataFrame(attribution_rows)

attribution_table.to_csv(
    OUT / "Step43_Table_Bias_Attribution.csv",
    index=False
)

print("\n43B — BIAS ATTRIBUTION")
print(
    attribution_table[
        [
            "family",
            "frozen_mean_epe_px",
            "diagnostic_mean_epe_px",
            "posthoc_mean_reduction_px",
            "patches_with_positive_reduction"
        ]
    ].round(5).to_string(index=False)
)

# -------------------------------------------------------------------------
# 43D — FIGURE 1: MEAN EPE BY METHOD
# -------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white"
})

fig, ax = plt.subplots(figsize=(9.3, 4.6))

order = [
    "frozen_ncc_step13",
    "frozen_phase_step20_size_adapted",
    "identity",
    "ncc_prebias_same_gate_diagnostic",
    "phase_prebias_same_gate_diagnostic"
]

plot_table = table.set_index("method_id").loc[order]

x = np.arange(len(plot_table))

bars = ax.bar(
    x,
    plot_table["mean_epe_px"],
    width=0.62,
    color=[METHODS[m]["color"] for m in order],
    edgecolor="#353535",
    linewidth=0.5
)

for i, bar in enumerate(bars):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.035,
        f"{plot_table['mean_epe_px'].iloc[i]:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [METHODS[m]["label"] for m in order],
    rotation=18,
    ha="right"
)

ax.set_ylabel("Mean endpoint error (pixels)")
ax.set_ylim(
    0,
    float(plot_table["mean_epe_px"].max()) * 1.22
)

ax.axvline(
    2.5, color="#777777",
    linewidth=0.9,
    linestyle="--"
)

ax.text(
    1.0, 0.97,
    "Frozen transfer and baseline",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=9
)

ax.text(
    3.5, 0.97,
    "Post hoc diagnostics",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=9
)

ax.grid(
    axis="y", alpha=0.18,
    linewidth=0.7
)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=1.1)

fig.savefig(
    OUT / "Step43_Figure_External_EPE.png",
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "Step43_Figure_External_EPE.pdf",
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(fig)

# -------------------------------------------------------------------------
# 43E — FIGURE 2: ALIGNED RESIDUALS
# -------------------------------------------------------------------------

aligned = combined[
    (combined["true_dy"] == 0)
    & (combined["true_dx"] == 0)
].copy()

fig, ax = plt.subplots(figsize=(6.7, 6.0))

for method in [
    "frozen_ncc_step13",
    "frozen_phase_step20_size_adapted",
    "ncc_prebias_same_gate_diagnostic",
    "phase_prebias_same_gate_diagnostic"
]:
    group = aligned[
        aligned["method"] == method
    ]

    ax.scatter(
        group["estimated_dx"],
        group["estimated_dy"],
        s=60,
        color=METHODS[method]["color"],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        label=METHODS[method]["label"]
    )

ax.scatter(
    [0], [0], marker="+",
    color="black", s=130,
    linewidth=1.7, zorder=6,
    label="Correct alignment"
)

ax.axhline(0, color="#777777", lw=0.8)
ax.axvline(0, color="#777777", lw=0.8)

ax.set_xlabel("Horizontal residual, dx (pixels)")
ax.set_ylabel("Vertical residual, dy (pixels)")
ax.set_aspect("equal", adjustable="box")

# Fixed descriptive limits derived from all aligned residuals.
low = float(min(
    aligned["estimated_dx"].min(),
    aligned["estimated_dy"].min(),
    0
))
high = float(max(
    aligned["estimated_dx"].max(),
    aligned["estimated_dy"].max(),
    0
))

pad = max(0.12, (high - low) * 0.15)
ax.set_xlim(low - pad, high + pad)
ax.set_ylim(low - pad, high + pad)

ax.grid(alpha=0.16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=8,
    loc="upper left"
)

fig.tight_layout()

fig.savefig(
    OUT / "Step43_Figure_Aligned_Residuals.png",
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "Step43_Figure_Aligned_Residuals.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

# -------------------------------------------------------------------------
# 43F — FIGURE CAPTIONS
# -------------------------------------------------------------------------

captions = """Figure 1. Registration endpoint error on the Potsdam
synthetic HSI–MSI benchmark for the frozen MDAS methods,
the no-correction baseline, and post hoc bias-omission
diagnostics. Each method was evaluated under 85 synthetic
translation conditions across five non-overlapping patches
from one EnMAP acquisition. The frozen methods represent
the primary transfer experiment. The bias-omission results
are exploratory counterfactuals examined after the external
transfer outcomes were known.

Figure 2. Estimated residual displacement vectors for the
five truly aligned Potsdam patch observations. Residuals
are shown for both frozen MDAS registration methods and
their post hoc bias-omission diagnostics. The origin denotes
correctly preserved alignment. All five patches originate
from a single EnMAP acquisition; the plotted observations
must not be interpreted as independent geographic scenes.
"""

(OUT / "Step43_Figure_Captions.txt").write_text(
    captions,
    encoding="utf-8"
)

# -------------------------------------------------------------------------
# 43G — FINAL REPORT
# -------------------------------------------------------------------------

report = {
    "status": "EXTERNAL_MANUSCRIPT_RESULTS_PACKAGE_COMPLETE",
    "historical_registration_reproduced": True,
    "external_acquisitions": 1,
    "external_patches": 5,
    "synthetic_conditions_per_patch": 17,
    "frozen_method_case_records": 255,
    "posthoc_diagnostic_case_records": 170,
    "primary_analysis_methods": [
        "frozen_ncc_step13",
        "frozen_phase_step20_size_adapted",
        "identity"
    ],
    "exploratory_methods": [
        "ncc_prebias_same_gate_diagnostic",
        "phase_prebias_same_gate_diagnostic"
    ],
    "source_sha256": {
        name: digest(path)
        for name, path in sources.items()
    },
    "external_result_table": table.to_dict("records"),
    "bias_attribution": attribution_table.to_dict("records"),
    "inferential_statistics_performed": False,
    "source_scene_replicates": 1,
    "fusion_checkpoints_evaluated_on_potsdam": False,
    "scientific_conclusion": (
        "The original registration estimators reproduce historical "
        "MDAS records. Transferring their frozen calibration offsets "
        "to the Potsdam synthetic benchmark introduces systematic "
        "alignment errors. Bias-omission counterfactuals suggest "
        "calibration portability is a major cause; their favorable "
        "performance is exploratory rather than independently validated."
    ),
    "limitations": [
        "One independent EnMAP acquisition",
        "Dependent synthetic perturbations of five spatial patches",
        "Synthetic rather than real multispectral sensor observations",
        "Necessary spectral-interface and phase-size adaptations",
        "Post hoc bias-neutral diagnostics",
        "No external fusion-model quality measurements"
    ],
    "output_directory": str(OUT)
}

(OUT / "Step43_Final_Report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False),
    encoding="utf-8"
)

print("\n" + "=" * 88)
print("STEP 43 — FINAL REPORT")
print("=" * 88)

print(json.dumps({
    "status": report["status"],
    "source_scene_replicates": 1,
    "primary_methods": 3,
    "posthoc_diagnostic_methods": 2,
    "tables": 3,
    "figures": 2,
    "output_directory": str(OUT)
}, indent=2))

print("\nSTEP 43 COMPLETE")
print("All historical and external results unchanged.")
