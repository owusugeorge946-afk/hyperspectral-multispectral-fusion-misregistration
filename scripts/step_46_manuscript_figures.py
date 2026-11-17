
# =============================================================================
# STEP 46 — FINAL MDAS MANUSCRIPT FIGURE AND TABLE PACKAGE
# =============================================================================
#
# PURPOSE
#   Consolidate all verified NEW manuscript figures and tables.
#   Correct the Step45 F02 layout by removing overlapping footer text.
#   Separate frozen original registration from post hoc diagnostics.
#   Prepare a single ZIP package for manuscript revision in ChatGPT.
#
# FIGURES
#   N01  External mean endpoint error
#   N02  Aligned displacement diagnostics
#   N03  Actual Potsdam EnMAP scene and sampling patches
#   N04  Registration images + edge overlays, patch P1
#   N05  Registration images + edge overlays, patch P5
#   N06  Four-band MSI difference maps
#   N07  Spatial patch-level endpoint-error maps
#
# TABLES
#   External performance
#   Patch-level results
#   Post hoc bias attribution
#   Historical reproduction
#   Registration comparison
#   Additional saved numerical summaries
#
# OUTPUT
#   Manuscript_Figures/          PNG + vector PDF
#   Manuscript_Tables/           CSV + readable Markdown tables
#   Figure_Captions.md
#   Figure_and_Table_Inventory.csv
#   Manuscript_Insertion_Guide.md
#   Source_Provenance.json
#   Complete ZIP package
#
# Original experiments and figures are never overwritten.
# =============================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
from scipy.ndimage import shift as ndi_shift, sobel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import json
import shutil
import zipfile
import re
import time

START = time.monotonic()

# =============================================================================
# 46A — PROJECT AND SOURCE PATHS
# =============================================================================

PROJECT = Path(
    "/content/drive/MyDrive/"
    "MDAS_Registration_Aware_Revision_2026"
)

S30 = PROJECT / "Step30_Frozen_Potsdam_Benchmark"
S40 = PROJECT / "Step40_Reproduction_Discrepancy_Audit"
S41 = PROJECT / "Step41C_Frozen_Potsdam_Transfer"
S42 = PROJECT / "Step42_External_Calibration_Transfer_Audit"
S43 = PROJECT / "Step43_External_Manuscript_Results"
S44 = PROJECT / "Step44C_Final_Verified_Spatial_Figures"
S45 = PROJECT / "Step45_Refined_Spatial_Manuscript_Figures"

# Each execution receives a new output folder.
# No previous research products are overwritten.

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

OUT = (
    PROJECT
    / f"Step46_Final_Manuscript_Package_{RUN_ID}"
)

FIG_DIR = OUT / "Manuscript_Figures"
TABLE_DIR = OUT / "Manuscript_Tables"

FIG_DIR.mkdir(parents=True, exist_ok=False)
TABLE_DIR.mkdir(parents=True, exist_ok=False)

print("=" * 90)
print("STEP 46 — FINAL MANUSCRIPT FIGURE AND TABLE PACKAGE")
print("=" * 90)
print("Output:", OUT)

# =============================================================================
# 46B — HELPERS
# =============================================================================

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=str
        ),
        encoding="utf-8"
    )


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_verified(source, destination):
    require_file(source)

    if destination.exists():
        raise FileExistsError(destination)

    shutil.copy2(source, destination)

    if sha256(source) != sha256(destination):
        raise RuntimeError(
            "File copy checksum mismatch: "
            + str(source)
        )

    return destination


def save_figure(fig, stem):

    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"

    if png.exists() or pdf.exists():
        raise FileExistsError(stem)

    fig.savefig(
        png,
        dpi=900,
        bbox_inches="tight",
        pad_inches=0.10,
        facecolor="white"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.10,
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

    print("Generated:", stem)


def save_table(df, stem):
    path = TABLE_DIR / f"{stem}.csv"
    if path.exists():
        raise FileExistsError(path)
    df.to_csv(path, index=False)
    return path


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# =============================================================================
# 46C — VERIFY THE EXPERIMENTS
# =============================================================================

report_paths = {
    "step40": S40 / "Step40_Final_Report.json",
    "step41c": S41 / "Step41C_Final_Report.json",
    "step42": S42 / "Step42_Final_Report.json",
    "step43": S43 / "Step43_Final_Report.json",
    "step44c": S44 / "Step44C_Final_Report.json",
    "step45": S45 / "Step45_Final_Report.json",
}

for path in report_paths.values():
    require_file(path)

reports = {
    name: load_json(path)
    for name, path in report_paths.items()
}

expected_status = {
    "step40": "NUMERICAL_REPRODUCTION_CONSISTENT",
    "step41c": "EXTERNAL_FROZEN_REGISTRATION_EVALUATION_COMPLETE",
    "step42": "POST_HOC_CALIBRATION_TRANSFER_AUDIT_COMPLETED",
    "step43": "EXTERNAL_MANUSCRIPT_RESULTS_PACKAGE_COMPLETE",
    "step44c": "SPATIAL_FIGURES_COMPLETE",
    "step45": "REFINED_SPATIAL_FIGURES_GENERATED",
}

for step, status in expected_status.items():

    actual = reports[step].get("status")

    if actual != status:
        raise RuntimeError(
            f"{step} status mismatch: {actual}"
        )

FROZEN_CSV = require_file(
    S41 / "Step41C_Registration_Results.csv"
)

DIAGNOSTIC_CSV = require_file(
    S42 / "Step42_Bias_Attribution_Cases.csv"
)

if reports["step42"]["source_step41c_results_sha256"] != sha256(
    FROZEN_CSV
):
    raise RuntimeError(
        "Step42 provenance does not match Step41C."
    )

frozen = pd.read_csv(FROZEN_CSV)
combined = pd.read_csv(DIAGNOSTIC_CSV)

assert len(frozen) == 255
assert len(combined) == 425

NCC = "frozen_ncc_step13"
PHASE = "frozen_phase_step20_size_adapted"
IDENTITY = "identity"

NCC_DIAG = "ncc_prebias_same_gate_diagnostic"
PHASE_DIAG = "phase_prebias_same_gate_diagnostic"

METHODS = [
    NCC, PHASE, IDENTITY,
    NCC_DIAG, PHASE_DIAG
]

assert set(combined["method"]) == set(METHODS)

KEY = ["patch_id", "condition_index"]

for frame in [frozen, combined]:

    if frame.duplicated(KEY + ["method"]).any():
        raise RuntimeError(
            "Duplicated registration case keys"
        )

    for col in [
        "true_dy",
        "true_dx",
        "estimated_dy",
        "estimated_dx",
        "epe"
    ]:
        frame[col] = pd.to_numeric(
            frame[col],
            errors="raise"
        )

    if not np.isfinite(
        frame[
            [
                "true_dy",
                "true_dx",
                "estimated_dy",
                "estimated_dx",
                "epe"
            ]
        ].to_numpy(float)
    ).all():
        raise RuntimeError(
            "Nonfinite experimental results"
        )

if not (combined.groupby("method").size() == 85).all():
    raise RuntimeError(
        "Expected 85 cases per method."
    )

# Reconcile frozen records against Step42 copies,
# using matching case keys and CSV-safe precision.

comparison_tolerance = 1e-9
maximum_cross_step_difference = 0.0

for method in [NCC, PHASE, IDENTITY]:

    a = frozen[
        frozen["method"] == method
    ][KEY + [
        "true_dy", "true_dx",
        "estimated_dy", "estimated_dx", "epe"
    ]]

    b = combined[
        combined["method"] == method
    ][KEY + [
        "true_dy", "true_dx",
        "estimated_dy", "estimated_dx", "epe"
    ]]

    joined = a.merge(
        b,
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_41c", "_42")
    )

    if (
        len(joined) != 85
        or not joined["_merge"].eq("both").all()
    ):
        raise RuntimeError(
            f"Case reconciliation failed: {method}"
        )

    for field in [
        "true_dy", "true_dx",
        "estimated_dy", "estimated_dx", "epe"
    ]:

        a_values = joined[
            field + "_41c"
        ].to_numpy(float)

        b_values = joined[
            field + "_42"
        ].to_numpy(float)

        diff = float(
            np.max(np.abs(a_values - b_values))
        )

        maximum_cross_step_difference = max(
            maximum_cross_step_difference,
            diff
        )

        if diff > comparison_tolerance:
            raise RuntimeError(
                f"Frozen data discrepancy: "
                f"{method}/{field}: {diff}"
            )

print("\n46C — ALL SOURCE RESULTS VERIFIED")
print("Frozen records:", len(frozen))
print("Post hoc records:", len(combined)-len(frozen))
print(
    "Maximum cross-step numerical difference:",
    maximum_cross_step_difference
)

# =============================================================================
# 46D — COLLECT THE BEST PREVIOUS FIGURES
# =============================================================================

# We retain existing VERIFIED images without rerendering them.
#
# N01/N02: Step43 statistical figures
# N03/N06: Step44C real EnMAP figures
# N07: Step45 refined spatial EPE figure
#
# N04/N05 are rendered afresh below.

existing_figure_sources = {
    "N01_External_Mean_EPE": (
        S43 / "Step43_Figure_External_EPE"
    ),
    "N02_Aligned_Residuals": (
        S43 / "Step43_Figure_Aligned_Residuals"
    ),
    "N03_EnMAP_Scene_and_Patches": (
        S44 / "F01_EnMAP_Scene_Patches"
    ),
    "N06_Four_Band_Difference_Maps": (
        S44 / "F03_Four_Band_Difference_Maps"
    ),
    "N07_Spatial_Patch_Mean_EPE": (
        S45 / "F04_Refined_Compact_Spatial_EPE"
    ),
}

figure_inventory = []

for new_stem, original_stem in existing_figure_sources.items():

    source_png = original_stem.with_suffix(".png")
    source_pdf = original_stem.with_suffix(".pdf")

    copied_png = copy_verified(
        source_png,
        FIG_DIR / f"{new_stem}.png"
    )

    copied_pdf = copy_verified(
        source_pdf,
        FIG_DIR / f"{new_stem}.pdf"
    )

    with Image.open(copied_png) as im:
        size = im.size

    figure_inventory.append({
        "figure_id": new_stem.split("_")[0],
        "filename_stem": new_stem,
        "source": str(original_stem),
        "png_width_px": int(size[0]),
        "png_height_px": int(size[1]),
        "png_sha256": sha256(copied_png),
        "pdf_sha256": sha256(copied_pdf),
        "evidence_status": (
            "PRIMARY"
            if new_stem.startswith(("N01", "N03", "N06"))
            else "MIXED_PRIMARY_AND_EXPLORATORY"
        )
    })

    print("Copied:", new_stem)

# =============================================================================
# 46E — GENERATE CORRECTED REGISTRATION FIGURES N04 / N05
# =============================================================================

# This replaces Step45 F02 with two readable figures.
# No overlapping footer text.
# One spatial patch per figure.
#
# Rows:
#   1. Full 96x96 image
#   2. Enlarged 40x40 detail
#   3. Reference-versus-correction edge overlay
#
# Columns:
#   Reference, displaced, frozen NCC, NCC diagnostic,
#   frozen phase, phase diagnostic.

PROTOCOL_PATH = require_file(
    S30 / "Step30_Frozen_Forward_Model.json"
)

protocol = load_json(PROTOCOL_PATH)

assert protocol["source_scene_count"] == 1
assert protocol["retained_band_count"] == 218

def patch_coordinates(pid):
    m = re.fullmatch(
        r"potsdam_r(\d+)_c(\d+)",
        pid
    )
    if m is None:
        raise ValueError(
            f"Unexpected patch ID: {pid}"
        )
    return int(m.group(1)), int(m.group(2))

PATCH_IDS = sorted(
    protocol["selected_patch_ids"],
    key=patch_coordinates
)

assert len(PATCH_IDS) == 5

retained_band_ids = np.asarray(
    protocol["retained_band_ids_1_based"],
    dtype=int
)

def load_patch(pid):

    path = require_file(
        S30 / "Paired_Observations"
        / f"{pid}.npz"
    )

    with np.load(
        path, allow_pickle=False
    ) as z:

        reference = z["hrmsi"].copy()
        shifted = z["shifted_msi"].copy()
        offsets = z[
            "shift_offsets_dy_dx"
        ].copy()

        band_ids = z[
            "retained_band_ids"
        ].copy()

    if reference.shape != (4,96,96):
        raise RuntimeError("Unexpected reference MSI shape")

    if shifted.shape != (17,4,96,96):
        raise RuntimeError("Unexpected displaced MSI shape")

    if offsets.shape != (17,2):
        raise RuntimeError("Unexpected shift-array shape")

    if not np.array_equal(
        band_ids.astype(int),
        retained_band_ids
    ):
        raise RuntimeError("Frozen spectral-band mismatch")

    zero = np.flatnonzero(
        np.all(
            np.isclose(
                offsets, [0,0],
                atol=1e-10
            ),
            axis=1
        )
    )

    if (
        len(zero) != 1
        or not np.array_equal(
            shifted[zero[0]], reference
        )
    ):
        raise RuntimeError(
            "Aligned observation verification failed"
        )

    return reference, shifted, offsets


def find_condition(offsets, dy=1.5, dx=0.0):

    matches = np.flatnonzero(
        np.all(
            np.isclose(
                offsets,
                [dy,dx],
                atol=1e-8
            ),
            axis=1
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Chosen synthetic shift is missing"
        )

    return int(matches[0])


def get_estimate(pid, index, method):

    sub = combined[
        (combined["patch_id"] == pid)
        & (combined["condition_index"] == index)
        & (combined["method"] == method)
    ]

    if len(sub) != 1:
        raise RuntimeError(
            "Missing saved registration estimate"
        )

    row = sub.iloc[0]

    return (
        float(row["estimated_dy"]),
        float(row["estimated_dx"]),
        float(row["epe"])
    )


def visual_warp(msi, dy, dx):

    return ndi_shift(
        msi.astype(np.float32),
        shift=(0,-dy,-dx),
        order=1,
        mode="nearest",
        prefilter=False
    )


def rgb_limits(reference):

    rgb = np.moveaxis(
        reference[[2,1,0]],
        0,-1
    )

    return (
        np.percentile(
            rgb,2,axis=(0,1)
        ),
        np.percentile(
            rgb,98,axis=(0,1)
        )
    )


def render_rgb(msi, limits):

    rgb = np.moveaxis(
        msi[[2,1,0]],0,-1
    ).astype(np.float64)

    low, high = limits

    return np.clip(
        (rgb-low)
        /np.maximum(high-low,1e-12),
        0,1
    )**0.85


def edge_magnitude(msi):

    visible = np.mean(
        msi[[0,1,2]].astype(np.float64),
        axis=0
    )

    return np.hypot(
        sobel(visible,axis=0),
        sobel(visible,axis=1)
    )


def edge_overlay(reference, comparison, threshold):

    ref = edge_magnitude(reference) >= threshold
    obs = edge_magnitude(comparison) >= threshold

    result = np.zeros(
        (96,96,3),
        dtype=np.float32
    ) + 0.055

    # Reference edges: magenta.
    result[ref] = [1.0,0.24,0.64]

    # Comparison edges: cyan.
    result[obs] = [0.12,0.84,0.92]

    # Common edges: white.
    result[ref & obs] = [1,1,1]

    return result


PLOT_COLUMNS = [
    (None,"Aligned reference"),
    ("shifted","Displaced MSI"),
    (NCC,"Frozen NCC"),
    (NCC_DIAG,"NCC, bias omitted*"),
    (PHASE,"Frozen phase"),
    (PHASE_DIAG,"Phase, bias omitted*")
]

CROP = (25,65,25,65)

def generate_registration_figure(pid, figure_id):

    reference, shifted, offsets = load_patch(pid)

    condition = find_condition(
        offsets,dy=1.5,dx=0.0
    )

    observed = shifted[condition]

    limits = rgb_limits(reference)

    threshold = float(
        np.percentile(
            edge_magnitude(reference)[8:-8,8:-8],
            85
        )
    )

    images = [reference, observed]
    errors = [None, 1.5]

    audit_records = []

    for method,label in PLOT_COLUMNS[2:]:

        dy,dx,epe = get_estimate(
            pid,condition,method
        )

        corrected = visual_warp(
            observed,dy,dx
        )

        images.append(corrected)
        errors.append(epe)

        audit_records.append({
            "figure":figure_id,
            "patch_id":pid,
            "method":method,
            "condition_index":condition,
            "true_dy":1.5,
            "true_dx":0.0,
            "estimated_dy":dy,
            "estimated_dx":dx,
            "saved_epe":epe,
            "edge_threshold":threshold
        })

    fig, axes = plt.subplots(
        3,6,
        figsize=(15.3,7.5),
        gridspec_kw={
            "height_ratios":[1,1,1],
            "hspace":0.12,
            "wspace":0.06
        }
    )

    for col, (
        (method,label), image
    ) in enumerate(
        zip(PLOT_COLUMNS,images)
    ):

        rgb = render_rgb(image,limits)

        y0,y1,x0,x1 = CROP

        for row in range(3):

            ax = axes[row,col]

            if row == 0:

                ax.imshow(
                    rgb,
                    interpolation="nearest"
                )

                ax.add_patch(
                    plt.Rectangle(
                        (x0-0.5,y0-0.5),
                        x1-x0,
                        y1-y0,
                        edgecolor="#F8DA64",
                        facecolor="none",
                        lw=1.2
                    )
                )

                if errors[col] is not None:

                    ax.text(
                        0.035,0.05,
                        f"EPE {errors[col]:.2f} px",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="white",
                        bbox=dict(
                            facecolor="black",
                            alpha=0.75,
                            edgecolor="none",
                            pad=2
                        )
                    )

                ax.set_title(
                    f"({chr(97+col)}) {label}",
                    fontsize=9.2,
                    pad=8
                )

            elif row == 1:

                ax.imshow(
                    rgb[y0:y1,x0:x1],
                    interpolation="nearest"
                )

            else:

                if col == 0:

                    ref_edge = (
                        edge_magnitude(reference)
                        >=threshold
                    )

                    ax.imshow(
                        ref_edge[
                            y0:y1,x0:x1
                        ],
                        cmap="gray",
                        vmin=0,vmax=1,
                        interpolation="nearest"
                    )

                else:

                    overlay = edge_overlay(
                        reference,
                        image,
                        threshold
                    )

                    ax.imshow(
                        overlay[
                            y0:y1,x0:x1
                        ],
                        interpolation="nearest"
                    )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.6)

    axes[0,0].set_ylabel(
        "Full image\n96 × 96",
        fontsize=9
    )

    axes[1,0].set_ylabel(
        "Magnified detail\n40 × 40",
        fontsize=9
    )

    axes[2,0].set_ylabel(
        "Edge overlap\n40 × 40",
        fontsize=9
    )

    # No bottom footer is placed in the figure.
    # All explanations belong in the manuscript caption.

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.93,
        bottom=0.045,
        hspace=0.16,
        wspace=0.065
    )

    stem = (
        f"{figure_id}_Registration_"
        + pid
    )

    save_figure(fig,stem)

    return stem, audit_records


# The two patches are selected by spatial position,
# not by registration accuracy.

NEW_FIGURES = []
example_audit = []

for pid, figure_id in [
    (PATCH_IDS[0],"N04"),
    (PATCH_IDS[-1],"N05")
]:

    stem, rows = generate_registration_figure(
        pid,figure_id
    )

    NEW_FIGURES.append(stem)
    example_audit.extend(rows)

    png = FIG_DIR/f"{stem}.png"
    pdf = FIG_DIR/f"{stem}.pdf"

    with Image.open(png) as im:
        w,h = im.size

    figure_inventory.append({
        "figure_id":figure_id,
        "filename_stem":stem,
        "source":(
            "Step30 imagery + "
            "Step41C/42 saved estimates"
        ),
        "png_width_px":w,
        "png_height_px":h,
        "png_sha256":sha256(png),
        "pdf_sha256":sha256(pdf),
        "evidence_status":(
            "MIXED_PRIMARY_AND_EXPLORATORY"
        )
    })

save_table(
    pd.DataFrame(example_audit),
    "T_Registration_Example_Audit"
)

# =============================================================================
# 46F — COLLECT THE VERIFIED NUMERICAL TABLES
# =============================================================================

table_sources = {
    "T01_External_Performance":
        S43 / "Step43_Table_External_Performance.csv",

    "T02_Patch_Level":
        S43 / "Step43_Table_Patch_Level.csv",

    "T03_Bias_Attribution":
        S43 / "Step43_Table_Bias_Attribution.csv",

    "T04_Historical_Reproduction":
        S40 / "Step40_Reproduction_Summary.csv",

    "T05_Phase_Tolerance_Sensitivity":
        S40 / "Step40_Tolerance_Sensitivity.csv",

    "T06_Frozen_External_Registration":
        S41 / "Step41C_Method_Summary.csv",

    "T07_Frozen_Patch_Results":
        S41 / "Step41C_Patch_Summary.csv",

    "T08_Frozen_Paired_Comparison":
        S41 / "Step41C_Paired_Comparison.csv",

    "T09_Posthoc_Diagnostic_Summary":
        S42 / "Step42_Descriptive_Method_Summary.csv",

    "T10_Registration_Example_Values":
        S44 / "Step44C_Example_Values.csv",

    "T11_Patch_Footprints":
        S44 / "Step44C_Patch_Footprints.csv",
}

table_inventory = []

for stem, source in table_sources.items():

    require_file(source)

    target = copy_verified(
        source,
        TABLE_DIR / f"{stem}.csv"
    )

    frame = pd.read_csv(target)

    table_inventory.append({
        "table_id":stem.split("_")[0],
        "filename":target.name,
        "rows":len(frame),
        "columns":len(frame.columns),
        "source":str(source),
        "sha256":sha256(target)
    })

    print(
        "Table:",
        stem,
        f"({len(frame)} rows)"
    )

# =============================================================================
# 46G — VERIFIED MANUSCRIPT CAPTIONS
# =============================================================================

captions = """
# New MDAS manuscript figure captions

## N01 — External registration performance

Mean registration endpoint error for frozen MDAS NCC,
frozen bounded phase correlation, the no-correction
baseline, and exploratory post hoc bias-omission diagnostics
on the Potsdam synthetic HSI–MSI benchmark. Each method
was assessed under 85 synthetic translation conditions
across five nonoverlapping patches from one EnMAP
acquisition. The frozen methods constitute the primary
external transfer experiment. Bias-omission results were
examined after the frozen-transfer outcomes and therefore
do not represent independent validation.

## N02 — Aligned displacement diagnostics

Residual displacement estimates for truly aligned Potsdam
observations under the frozen MDAS registrations and the
post hoc bias-omission diagnostics. The correct displacement
is zero. All five aligned observations originate from one
external EnMAP acquisition. This figure is descriptive and
should not be interpreted as five independent external
validation scenes.

## N03 — Potsdam EnMAP scene and selected patches

Georeferenced representation of the Potsdam EnMAP scene
with five nonoverlapping 96 × 96-pixel evaluation windows.
Patch locations were derived from their saved raster
row-column identifiers and the source image geotransform.
The displayed composite uses the frozen four-channel
synthetic spectral-response weights. The source raster
uses EPSG:32633 projected coordinates.

## N04 — Registration examples, patch P1

Aligned reference, synthetically displaced MSI, frozen
MDAS NCC and bounded-phase corrections, and exploratory
bias-omission diagnostics for the first selected Potsdam
patch under a 1.5-pixel vertical translation. Full
96 × 96-pixel images, common 40 × 40-pixel details and
edge-overlap diagnostics are shown. Magenta denotes
reference edges, cyan denotes comparison edges and white
denotes overlapping edge detections. Registration
endpoint errors are taken from the saved numerical
evaluation; inverse resampling was performed solely for
visualization. Bias-omission corrections are post hoc
diagnostics.

## N05 — Registration examples, patch P5

Registration comparison for the fifth selected Potsdam
patch using the same observation protocol, display
conventions, 1.5-pixel translation, magnification and
edge-overlay method as N04. The example was selected
by spatial position rather than registration performance.
The bias-omission variants are exploratory.

## N06 — Four-band MSI difference maps

Pixelwise mean absolute difference between an aligned
synthetic four-channel MSI reference and the displaced
or visually corrected MSI in one Potsdam patch. All
panels share a common colour scale. A six-pixel image
border is excluded to reduce edge-resampling effects.
The differences describe synthetic multispectral image
signals, not fused hyperspectral reconstruction errors.
Bias-omission results are exploratory.

## N07 — Spatial patch-level registration errors

Spatial pattern of patch-level mean endpoint error for
frozen original NCC, frozen bounded phase correlation and
their exploratory bias-omission diagnostics. Each patch
value averages 17 synthetic translation conditions. The
four panels use the same projected scene extent and
colour scale. All five patches belong to a single EnMAP
acquisition, and the coloured regions are patch-level
summaries rather than pixelwise displacement fields.

# Notes for manuscript preparation

- Rename N01–N07 to the final sequential figure numbers only
  after reviewing the full manuscript.
- Distinguish the frozen external transfer from post hoc
  diagnostics in all captions, legends and discussions.
- Do not describe the Potsdam benchmark as five independent
  source scenes.
- Do not describe four-band signal-difference maps as
  hyperspectral fusion-error maps.
- Preserve the important original Augsburg fusion
  architecture and spatial reconstruction figures.
"""

(OUT/"Figure_Captions.md").write_text(
    captions.strip(),
    encoding="utf-8"
)

# =============================================================================
# 46H — FIGURE INVENTORY AND TABLE INVENTORY
# =============================================================================

figure_inventory_df = pd.DataFrame(
    figure_inventory
).sort_values("figure_id")

assert len(figure_inventory_df) == 7
assert figure_inventory_df["figure_id"].nunique() == 7

figure_inventory_df.to_csv(
    OUT/"Figure_and_Table_Inventory.csv",
    index=False
)

pd.DataFrame(table_inventory).to_csv(
    OUT/"Table_Inventory.csv",
    index=False
)

# Generate a single readable Markdown document
# with the core tables.

CORE_TABLES = [
    "T01_External_Performance",
    "T03_Bias_Attribution",
    "T04_Historical_Reproduction",
    "T06_Frozen_External_Registration"
]

sections = [
    "# New MDAS results tables",
    "",
    "All tables are derived from previously saved experiments.",
    "Repeated shift conditions are not independent source scenes.",
    ""
]

for stem in CORE_TABLES:

    table = pd.read_csv(
        TABLE_DIR/f"{stem}.csv"
    )

    sections.extend([
        f"## {stem}",
        "",
        table.round(5).to_markdown(index=False),
        ""
    ])

(OUT/"Core_Tables_For_Manuscript.md").write_text(
    "\n".join(sections),
    encoding="utf-8"
)

# =============================================================================
# 46I — MANUSCRIPT INSERTION GUIDE
# =============================================================================

guide = """
# MDAS manuscript insertion guide

This package contains new external registration and
reproducibility figures only. It does not replace the
original Augsburg manuscript figures.

## Suggested main manuscript figures

N03 — Place in Study Area / External Dataset subsection.
N04 — Place in external registration visual analysis.
N06 — Place with the synthetic MSI image-difference results.
N07 — Place with external calibration-transfer analysis.
N01 — Use if numerical performance comparisons need a
standalone figure rather than a table.

## Suggested supplementary figures

N02 — Aligned residual diagnostic plot.
N05 — Additional image registration example from a second patch.

## Suggested main manuscript tables

T01 — External registration performance (frozen methods and
post hoc diagnostics explicitly separated).
T03 — Post hoc calibration-bias attribution, labelled exploratory.
T04 — Historical estimator numerical reproducibility.

## Suggested supplementary tables

T02, T05, T07, T08, T09, T10, T11 — detailed patch,
method, tolerance and example information.

## Important limitations

- One independent external EnMAP acquisition.
- Five nonoverlapping source patches.
- Seventeen repeated synthetic shifts per patch.
- Synthetic MSI rather than independently acquired multispectral data.
- Frozen MDAS biases applied without external recalibration.
- Post hoc bias omission is descriptive, not independently validated.
- No Potsdam fusion checkpoint inference was performed.

## Rules for revising the existing manuscript

1. Review every original figure and table before removal.
2. Preserve the original architecture diagram unless it is
   genuinely redundant or scientifically incorrect.
3. Keep the strongest Augsburg fusion reconstruction panels.
4. Avoid presenting repeated plots of the same mean EPE values.
5. Cross-reference every new table and figure in the text.
6. Keep figure and table numbering sequential.
7. Separate Results from exploratory diagnostic interpretation.
8. Include relevant reproducibility and validation limitations.

Final decisions about figure removal, table replacement and
placement must be made after reading the current full manuscript.
"""

(OUT/"Manuscript_Insertion_Guide.md").write_text(
    guide.strip(),
    encoding="utf-8"
)

# =============================================================================
# 46J — GENERATE A LIGHTWEIGHT VISUAL CONTACT SHEET
# =============================================================================

# Downsample only for the contact sheet.
# Original 900-dpi images remain unchanged.

ordered_stems = (
    figure_inventory_df["filename_stem"].tolist()
)

cols = 2
rows = int(np.ceil(len(ordered_stems)/cols))

thumb_w = 850
thumb_h = 520

sheet = Image.new(
    "RGB",
    (cols*thumb_w, rows*thumb_h),
    color="white"
)

from PIL import ImageDraw

draw = ImageDraw.Draw(sheet)

for idx, stem in enumerate(ordered_stems):

    source = FIG_DIR/f"{stem}.png"

    with Image.open(source) as im:
        thumb = im.convert("RGB")
        thumb.thumbnail(
            (thumb_w-40,thumb_h-60)
        )

    x0 = (idx%cols)*thumb_w
    y0 = (idx//cols)*thumb_h

    x = x0+(thumb_w-thumb.width)//2
    y = y0+38

    sheet.paste(thumb,(x,y))

    draw.text(
        (x0+20,y0+12),
        stem,
        fill="black"
    )

contact = OUT/"All_New_Figures_Contact_Sheet.jpg"

sheet.save(
    contact,
    quality=90,
    optimize=True
)

print("\nContact sheet generated:", contact.name)

# =============================================================================
# 46K — FINAL MANIFEST AND ZIP FILE
# =============================================================================

manifest = {
    "experiment": (
        "MDAS final manuscript new-figure and "
        "table consolidation"
    ),
    "status": "MANUSCRIPT_FIGURE_TABLE_PACKAGE_COMPLETE",
    "figures":len(figure_inventory_df),
    "table_csv_files":len(list(TABLE_DIR.glob("*.csv"))),
    "png_dpi_for_newly_rendered_figures":900,
    "newly_rendered_figures":[
        "N04","N05"
    ],
    "previously_verified_figures_copied":[
        "N01","N02","N03","N06","N07"
    ],
    "independent_external_acquisitions":1,
    "potsdam_patches":5,
    "perturbations_per_patch":17,
    "source_frozen_records":255,
    "source_posthoc_records":170,
    "maximum_cross_step_difference":(
        maximum_cross_step_difference
    ),
    "registration_estimators_rerun":False,
    "fusion_checkpoints_rerun":False,
    "historical_files_modified":False,
    "figure_removal_decisions_made":False,
    "manuscript_revision_performed":False,
    "source_hashes":{
        "step40_report":sha256(report_paths["step40"]),
        "step41c_report":sha256(report_paths["step41c"]),
        "step42_report":sha256(report_paths["step42"]),
        "step43_report":sha256(report_paths["step43"]),
        "step44c_report":sha256(report_paths["step44c"]),
        "step45_report":sha256(report_paths["step45"]),
        "frozen_results":sha256(FROZEN_CSV),
        "diagnostic_results":sha256(DIAGNOSTIC_CSV),
        "step30_protocol":sha256(PROTOCOL_PATH)
    },
    "limitations":[
        "One independent EnMAP Potsdam acquisition",
        "Five spatially distinct patches from that one scene",
        "Synthetic paired MSI observations",
        "Dependent 17-shift repeated conditions",
        "Post hoc calibration-bias omission diagnostics",
        "No external fusion checkpoint evaluation",
        "Full manuscript must be reviewed before figure removal"
    ],
    "elapsed_seconds":round(time.monotonic()-START,2),
    "output_directory":str(OUT)
}

save_json(
    OUT/"Source_Provenance.json",
    manifest
)

zip_path = OUT.parent / (
    OUT.name + ".zip"
)

if zip_path.exists():
    raise FileExistsError(zip_path)

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6
) as z:

    for path in sorted(OUT.rglob("*")):

        if not path.is_file():
            continue

        z.write(
            path,
            arcname=str(
                Path(OUT.name)
                /path.relative_to(OUT)
            )
        )

# Validate ZIP integrity.
with zipfile.ZipFile(zip_path,"r") as z:

    bad = z.testzip()

    if bad is not None:
        raise RuntimeError(
            "ZIP integrity failure: "+bad
        )

print("\n"+"="*90)
print("STEP 46 — FINAL REPORT")
print("="*90)

print(json.dumps({
    "status":manifest["status"],
    "figures":manifest["figures"],
    "tables":manifest["table_csv_files"],
    "zip_file":str(zip_path),
    "output_folder":str(OUT),
    "elapsed_seconds":manifest["elapsed_seconds"]
},indent=2))

print("\nSTEP 46 COMPLETE")
print("Original Augsburg and Potsdam outputs preserved.")
