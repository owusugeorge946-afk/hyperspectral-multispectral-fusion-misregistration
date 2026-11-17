
# =============================================================================
# STEP 44C — FINAL VERIFIED POTSDAM SPATIAL FIGURES
# =============================================================================
#
# COMPLETE CORRECTED VERSION
#
# Fixes:
#   1. Rasterio band indexes converted to Python list
#   2. Step41C vs Step42 records joined on exact case keys
#   3. CSV numerical differences checked using 1e-9 tolerance
#   4. Spectral band order and image provenance verified
#   5. Patch footprints derived from original GeoTIFF transform
#   6. Common image stretches and error color scales
#   7. Frozen vs post hoc results identified separately
#
# Figures:
#   F01  Actual Potsdam EnMAP scene and five patch footprints
#   F02  Real MSI registration image comparison (two patches)
#   F03  Four-band pixelwise image difference maps
#   F04  Spatial patch-level mean registration EPE maps
#
# Outputs: 900-dpi PNG + vector PDF, CSV audits, captions, JSON report.
#
# Does NOT:
#   - Execute original registration algorithms again
#   - Retrain fusion checkpoints
#   - Refit SRF weights or bias offsets
#   - Use external ground truth to select new methods
#   - Overwrite any historical output
#
# LIMITATION:
#   All five Potsdam patches are from ONE EnMAP acquisition.
#   Bias-omitted registration results remain POST HOC DIAGNOSTICS.
# =============================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import shift as ndi_shift
import rasterio

# =============================================================================
# A — PROJECT PATHS AND OUTPUT PROTECTION
# =============================================================================

START = time.monotonic()

PROJECT = Path(
    "/content/drive/MyDrive/"
    "MDAS_Registration_Aware_Revision_2026"
)

S27 = PROJECT / "Step27_External_EnMAP_Pilot"
S30 = PROJECT / "Step30_Frozen_Potsdam_Benchmark"
S41 = PROJECT / "Step41C_Frozen_Potsdam_Transfer"
S42 = PROJECT / "Step42_External_Calibration_Transfer_Audit"

SOURCE_TIF = S27 / "Imagery" / "enmap_potsdam.tif"
PROTOCOL_PATH = S30 / "Step30_Frozen_Forward_Model.json"
RESPONSE_PATH = S30 / "Step30_Spectral_Response_Matrix.csv"

FROZEN_CSV = S41 / "Step41C_Registration_Results.csv"
FROZEN_REPORT = S41 / "Step41C_Final_Report.json"

DIAGNOSTIC_CSV = S42 / "Step42_Bias_Attribution_Cases.csv"
DIAGNOSTIC_REPORT = S42 / "Step42_Final_Report.json"

OUT = PROJECT / "Step44C_Final_Verified_Spatial_Figures"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_OUTPUTS = [
    "F01_EnMAP_Scene_Patches.png",
    "F01_EnMAP_Scene_Patches.pdf",
    "F02_Registration_Image_Examples.png",
    "F02_Registration_Image_Examples.pdf",
    "F03_Four_Band_Difference_Maps.png",
    "F03_Four_Band_Difference_Maps.pdf",
    "F04_Spatial_Patch_Registration_EPE.png",
    "F04_Spatial_Patch_Registration_EPE.pdf",
    "Step44C_Frozen_Comparison_Audit.csv",
    "Step44C_Patch_Footprints.csv",
    "Step44C_Example_Values.csv",
    "Step44C_Patch_EPE.csv",
    "Step44C_Figure_Captions.txt",
    "Step44C_Final_Report.json"
]

already_exists = [
    f for f in EXPECTED_OUTPUTS if (OUT / f).exists()
]

if already_exists:
    raise FileExistsError(
        "Step44C output files already exist. "
        "Do not overwrite earlier results. Change OUT to "
        "a new directory name before rerunning:\n"
        + "\n".join(already_exists)
    )

print("=" * 90)
print("STEP 44C — FINAL VERIFIED POTSDAM SPATIAL FIGURES")
print("=" * 90)


# =============================================================================
# B — HELPERS
# =============================================================================

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj, indent=2, allow_nan=False, default=str
        ),
        encoding="utf-8"
    )


def require_finite(name, values):
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"Nonfinite values: {name}")


def export_figure(fig, stem):
    png = OUT / (stem + ".png")
    pdf = OUT / (stem + ".pdf")

    fig.savefig(
        png,
        dpi=900,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.10
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.10
    )

    print("Saved:", png.name, "and", pdf.name, flush=True)
    plt.show()
    plt.close(fig)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# =============================================================================
# C — LOAD AND VERIFY EXPERIMENTAL RECORDS
# =============================================================================

paths = {
    "source_tif": SOURCE_TIF,
    "protocol": PROTOCOL_PATH,
    "response": RESPONSE_PATH,
    "frozen_csv": FROZEN_CSV,
    "frozen_report": FROZEN_REPORT,
    "diagnostic_csv": DIAGNOSTIC_CSV,
    "diagnostic_report": DIAGNOSTIC_REPORT
}

for label, path in paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")

protocol = load_json(PROTOCOL_PATH)
r41 = load_json(FROZEN_REPORT)
r42 = load_json(DIAGNOSTIC_REPORT)

frozen = pd.read_csv(FROZEN_CSV)
combined = pd.read_csv(DIAGNOSTIC_CSV)
response_df = pd.read_csv(RESPONSE_PATH)

assert r41["status"] == (
    "EXTERNAL_FROZEN_REGISTRATION_EVALUATION_COMPLETE"
)
assert r42["status"] == (
    "POST_HOC_CALIBRATION_TRANSFER_AUDIT_COMPLETED"
)

if r42["source_step41c_results_sha256"] != sha256(FROZEN_CSV):
    raise RuntimeError(
        "Step42 was derived from a different Step41C CSV."
    )

assert len(frozen) == 255
assert len(combined) == 425
assert protocol["source_scene_count"] == 1
assert protocol["retained_band_count"] == 218

NCC = "frozen_ncc_step13"
PHASE = "frozen_phase_step20_size_adapted"
IDENTITY = "identity"
NCC_DIAG = "ncc_prebias_same_gate_diagnostic"
PHASE_DIAG = "phase_prebias_same_gate_diagnostic"

METHODS = [NCC, PHASE, IDENTITY, NCC_DIAG, PHASE_DIAG]

if set(combined["method"]) != set(METHODS):
    raise RuntimeError(
        "Unexpected methods in diagnostic results."
    )

KEYS = ["patch_id", "condition_index"]

for name, df in [
    ("Step41C", frozen),
    ("Step42", combined)
]:
    if df.duplicated(KEYS + ["method"]).any():
        raise RuntimeError(f"Duplicate records in {name}")

if not (combined.groupby("method").size() == 85).all():
    raise RuntimeError(
        "Expected 85 cases per method."
    )

numeric_fields = [
    "true_dy", "true_dx",
    "estimated_dy", "estimated_dx", "epe"
]

for df in [frozen, combined]:
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors="raise")
    require_finite("registration records", df[numeric_fields].values)

# Independently recalculate the endpoint errors.
recalculated = np.hypot(
    combined["estimated_dy"] - combined["true_dy"],
    combined["estimated_dx"] - combined["true_dx"]
)

if not np.allclose(
    recalculated,
    combined["epe"],
    atol=1e-10,
    rtol=0
):
    raise RuntimeError(
        "Stored Step42 endpoint errors are inconsistent."
    )

print("\nC — RECORDS VERIFIED")
print("Frozen records:", len(frozen))
print("Post hoc diagnostic records:", len(combined) - len(frozen))
print("Independent source acquisitions:", 1)


# =============================================================================
# D — CORRECTED CROSS-STEP FLOATING-POINT AUDIT
# =============================================================================

# This is the correction for the earlier Step44B failure.
# Compare matching experiment cases, not separately sorted arrays.
# Allow only negligible CSV serialization differences.

COMPARISON_ATOL = 1e-9

audit_rows = []

for method in [NCC, PHASE, IDENTITY]:

    a = frozen[
        frozen["method"] == method
    ][KEYS + numeric_fields].copy()

    b = combined[
        combined["method"] == method
    ][KEYS + numeric_fields].copy()

    if len(a) != 85 or len(b) != 85:
        raise RuntimeError(
            f"Incorrect frozen case count: {method}"
        )

    comparison = a.merge(
        b,
        on=KEYS,
        how="outer",
        suffixes=("_41C", "_42"),
        indicator=True,
        validate="one_to_one"
    )

    if len(comparison) != 85:
        raise RuntimeError(f"Case join failed: {method}")

    if not comparison["_merge"].eq("both").all():
        raise RuntimeError(
            f"Step41C and Step42 have different case keys: {method}"
        )

    for field in numeric_fields:

        original = comparison[
            field + "_41C"
        ].to_numpy(dtype=float)

        saved_copy = comparison[
            field + "_42"
        ].to_numpy(dtype=float)

        difference = np.abs(original - saved_copy)

        require_finite(
            f"{method}/{field}",
            difference
        )

        max_difference = float(difference.max())

        audit_rows.append({
            "method": method,
            "field": field,
            "matched_cases": len(comparison),
            "max_absolute_difference": max_difference,
            "mean_absolute_difference": float(difference.mean()),
            "accepted_tolerance": COMPARISON_ATOL,
            "passed": bool(max_difference <= COMPARISON_ATOL)
        })

        print(
            f"{method} | {field} | "
            f"max difference = {max_difference:.3e}"
        )

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    OUT / "Step44C_Frozen_Comparison_Audit.csv",
    index=False
)

if not audit["passed"].all():
    failures = audit[~audit["passed"]]

    print("\nCROSS-STEP INCONSISTENCIES")
    print(failures.to_string(index=False))

    raise RuntimeError(
        "Step41C and Step42 differ beyond the "
        "permitted 1e-9 CSV tolerance. "
        "Review the saved audit instead of ignoring the difference."
    )

print("\nD — FROZEN RECORD CONSISTENCY: PASSED")
print("All 255 original method-case records reconciled.")


# =============================================================================
# E — LOAD THE ORIGINAL GEOGRAPHIC ENMAP RASTER
# =============================================================================

EXPECTED_TIF_SHA = (
    "4ba14fd27ac80c9169f3d733010dbdaa986584b3a062dc497578c3e114733329"
)

if sha256(SOURCE_TIF) != EXPECTED_TIF_SHA:
    raise RuntimeError(
        "EnMAP source TIFF does not match the recorded source hash."
    )

required_response_columns = [
    "original_hsi_band",
    "wavelength_nm",
    "synthetic_blue_weight",
    "synthetic_green_weight",
    "synthetic_red_weight",
    "synthetic_nir_weight"
]

if not set(required_response_columns).issubset(
    response_df.columns
):
    raise RuntimeError(
        "Unexpected Step30 spectral response schema."
    )

retained_ids = np.asarray(
    protocol["retained_band_ids_1_based"],
    dtype=int
)

source_band_ids = response_df[
    "original_hsi_band"
].to_numpy(dtype=int)

if not np.array_equal(retained_ids, source_band_ids):
    raise RuntimeError("Spectral band ordering mismatch.")

# Correct Rasterio indexing: Python list, not NumPy array.
band_indexes = [int(b) for b in retained_ids.tolist()]

with rasterio.open(SOURCE_TIF) as src:

    if (src.count, src.height, src.width) != (224, 200, 330):
        raise RuntimeError(
            "Unexpected original EnMAP image dimensions."
        )

    if src.crs is None or not src.crs.is_projected:
        raise RuntimeError(
            "Expected a projected georeferenced EnMAP raster."
        )

    TRANSFORM = src.transform
    BOUNDS = src.bounds
    CRS = str(src.crs)

    if not (
        np.isclose(TRANSFORM.b, 0)
        and np.isclose(TRANSFORM.d, 0)
        and TRANSFORM.a > 0
        and TRANSFORM.e < 0
    ):
        raise RuntimeError(
            "This plotting routine requires a north-up raster."
        )

    cube = src.read(
        indexes=band_indexes
    ).astype(np.float32)

assert cube.shape == (218, 200, 330)
require_finite("EnMAP spectral cube", cube)

weights = response_df[
    [
        "synthetic_blue_weight",
        "synthetic_green_weight",
        "synthetic_red_weight",
        "synthetic_nir_weight"
    ]
].to_numpy(np.float64).T

if weights.shape != (4, 218):
    raise RuntimeError("Unexpected response matrix dimensions.")

if not np.allclose(
    weights.sum(axis=1),
    1,
    atol=1e-7,
    rtol=0
):
    raise RuntimeError("Frozen spectral weights are not normalized.")

# Original saved response weights; nothing is refitted.
scene_msi = np.einsum(
    "bc,chw->bhw",
    weights,
    cube.astype(np.float64),
    optimize=True
).astype(np.float32)

# Synthetic channel order: blue, green, red, NIR.
# Display RGB order: red, green, blue.
scene_rgb_raw = np.moveaxis(
    scene_msi[[2, 1, 0]],
    0,
    -1
)

def get_limits(rgb):
    return (
        np.percentile(rgb, 2, axis=(0, 1)),
        np.percentile(rgb, 98, axis=(0, 1))
    )

def rgb_stretch(rgb, limits):
    lo, hi = limits
    out = (rgb - lo) / np.maximum(hi - lo, 1e-12)
    return np.clip(out, 0, 1) ** 0.85

scene_display = rgb_stretch(
    scene_rgb_raw,
    get_limits(scene_rgb_raw)
)

XMIN, XMAX = BOUNDS.left / 1000, BOUNDS.right / 1000
YMIN, YMAX = BOUNDS.bottom / 1000, BOUNDS.top / 1000

print("\nE — ACTUAL ENMAP RASTER VERIFIED")
print("Original size:", (224, 200, 330))
print("Retained bands:", cube.shape[0])
print("CRS:", CRS)
print("Display RGB size:", scene_display.shape)


# =============================================================================
# F — BUILD TRUE GEOGRAPHIC PATCH FOOTPRINTS
# =============================================================================

PATCH_SIZE = 96

def parse_patch_id(pid):
    match = re.fullmatch(
        r"potsdam_r(\d+)_c(\d+)",
        str(pid)
    )
    if match is None:
        raise ValueError(f"Invalid patch ID: {pid}")
    return int(match.group(1)), int(match.group(2))

PATCH_IDS = sorted(
    protocol["selected_patch_ids"],
    key=parse_patch_id
)

if len(PATCH_IDS) != 5:
    raise RuntimeError("Expected five Potsdam patches.")

def patch_extent(pid):
    row, col = parse_patch_id(pid)
    x0, y0 = TRANSFORM * (col, row)
    x1, y1 = TRANSFORM * (
        col + PATCH_SIZE,
        row + PATCH_SIZE
    )
    return (
        min(x0, x1) / 1000,
        max(x0, x1) / 1000,
        min(y0, y1) / 1000,
        max(y0, y1) / 1000
    )

footprint_rows = []

for i, pid in enumerate(PATCH_IDS, start=1):

    row, col = parse_patch_id(pid)

    if (
        row < 0 or col < 0
        or row + PATCH_SIZE > 200
        or col + PATCH_SIZE > 330
    ):
        raise RuntimeError(
            f"Patch outside original scene: {pid}"
        )

    x0, x1, y0, y1 = patch_extent(pid)

    footprint_rows.append({
        "label": f"P{i}",
        "patch_id": pid,
        "row_start": row,
        "col_start": col,
        "row_end_exclusive": row + PATCH_SIZE,
        "col_end_exclusive": col + PATCH_SIZE,
        "xmin_km": x0,
        "xmax_km": x1,
        "ymin_km": y0,
        "ymax_km": y1,
        "crs": CRS
    })

for i, pid1 in enumerate(PATCH_IDS):
    r1, c1 = parse_patch_id(pid1)
    for pid2 in PATCH_IDS[i+1:]:
        r2, c2 = parse_patch_id(pid2)
        overlaps = (
            r1 < r2 + PATCH_SIZE
            and r2 < r1 + PATCH_SIZE
            and c1 < c2 + PATCH_SIZE
            and c2 < c1 + PATCH_SIZE
        )
        if overlaps:
            raise RuntimeError(
                f"Overlapping patches: {pid1}, {pid2}"
            )

pd.DataFrame(footprint_rows).to_csv(
    OUT / "Step44C_Patch_Footprints.csv",
    index=False
)

def scene_axes(ax):
    ax.imshow(
        scene_display,
        extent=[XMIN, XMAX, YMIN, YMAX],
        origin="upper",
        interpolation="nearest"
    )
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(top=True, right=True, direction="out")

def draw_patches(ax, patch_values=None, norm=None, cmap=None):
    for i, pid in enumerate(PATCH_IDS, 1):
        x0, x1, y0, y1 = patch_extent(pid)

        face = (
            "none"
            if patch_values is None
            else cmap(norm(patch_values[pid]))
        )
        edge = "#F8EEA1" if patch_values is None else "white"

        ax.add_patch(Rectangle(
            (x0, y0),
            x1-x0,
            y1-y0,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
            alpha=1.0 if patch_values is None else 0.87,
            zorder=4
        ))

        label = f"P{i}"
        if patch_values is not None:
            label += f"\n{patch_values[pid]:.2f}"

        ax.text(
            (x0+x1)/2,
            (y0+y1)/2,
            label,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.87,
                pad=1.5
            ),
            zorder=6
        )

def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.95, 0.96),
        xytext=(0.95, 0.83),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        color="white",
        fontweight="bold",
        fontsize=11,
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=1.8
        ),
        zorder=10
    )

def add_scale_bar(ax, length_km=1.0):
    width = XMAX-XMIN
    height = YMAX-YMIN

    x = XMIN + 0.07*width
    y = YMIN + 0.07*height
    half = length_km/2
    bar_height = 0.012*height

    for i, color in enumerate(["black", "white"]):
        ax.add_patch(Rectangle(
            (x+i*half, y),
            half,
            bar_height,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=8
        ))

    ax.text(
        x, y+bar_height+0.015*height,
        "0",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
    )
    ax.text(
        x+length_km,
        y+bar_height+0.015*height,
        f"{length_km:g} km",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
    )

# =============================================================================
# F01 — ACTUAL ENMAP SCENE + PATCH LOCATIONS
# =============================================================================

print("\nF01 — ENMAP SCENE AND PATCH FOOTPRINTS", flush=True)

fig, ax = plt.subplots(figsize=(10.2, 6.0))
scene_axes(ax)
draw_patches(ax)
add_north_arrow(ax)
add_scale_bar(ax)

ax.text(
    0.02, 0.97,
    "(a)",
    transform=ax.transAxes,
    va="top",
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.85
    )
)

fig.tight_layout()
export_figure(fig, "F01_EnMAP_Scene_Patches")


# =============================================================================
# G — READ SAVED MSI OBSERVATIONS AND REGISTRATION VALUES
# =============================================================================

def load_patch(pid):
    path = S30 / "Paired_Observations" / f"{pid}.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        ref = z["hrmsi"].copy()
        shifted = z["shifted_msi"].copy()
        offsets = z["shift_offsets_dy_dx"].copy()
        ids = z["retained_band_ids"].copy()

    if ref.shape != (4, 96, 96):
        raise RuntimeError(f"Invalid reference shape for {pid}")
    if shifted.shape != (17, 4, 96, 96):
        raise RuntimeError(f"Invalid shifted data for {pid}")
    if offsets.shape != (17, 2):
        raise RuntimeError(f"Invalid shifts for {pid}")

    if not np.array_equal(ids.astype(int), retained_ids):
        raise RuntimeError(f"Band ID mismatch for {pid}")

    require_finite(pid, ref)
    require_finite(pid, shifted)

    aligned = np.flatnonzero(
        np.all(np.isclose(offsets, 0, atol=1e-10), axis=1)
    )
    if len(aligned) != 1:
        raise RuntimeError(f"Invalid aligned condition: {pid}")

    if not np.array_equal(
        shifted[int(aligned[0])],
        ref
    ):
        raise RuntimeError(f"Aligned source mismatch for {pid}")

    return ref, shifted, offsets

def find_condition(offsets, dy, dx):
    found = np.flatnonzero(
        np.all(
            np.isclose(offsets, [dy, dx], atol=1e-8),
            axis=1
        )
    )
    if len(found) != 1:
        raise RuntimeError(f"Shift ({dy},{dx}) not uniquely found")
    return int(found[0])

def saved_estimate(pid, condition, method):
    matches = combined[
        (combined["patch_id"] == pid)
        & (combined["condition_index"] == condition)
        & (combined["method"] == method)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Missing estimate for {pid}, {condition}, {method}"
        )
    row = matches.iloc[0]
    return (
        float(row["estimated_dy"]),
        float(row["estimated_dx"]),
        float(row["epe"])
    )

def resample_for_display(observed, dy, dx):
    # Important: negative displacement corrects the observed shift.
    # This is visualization resampling, NOT new registration inference.
    return ndi_shift(
        observed.astype(np.float32),
        shift=(0, -dy, -dx),
        order=1,
        mode="nearest",
        prefilter=False
    )

def msi_rgb(chw, limits):
    rgb = np.moveaxis(chw[[2, 1, 0]], 0, -1)
    return rgb_stretch(rgb, limits)

def limits_for_msi(chw):
    rgb = np.moveaxis(chw[[2, 1, 0]], 0, -1)
    return get_limits(rgb)

# Fixed example selection: first and last spatial patches.
EXAMPLE_PATCHES = [PATCH_IDS[0], PATCH_IDS[-1]]
EXAMPLE_SHIFT = (1.5, 0.0)

CORRECTION_METHODS = [
    (NCC, "Frozen NCC"),
    (NCC_DIAG, "NCC, bias omitted*"),
    (PHASE, "Frozen phase"),
    (PHASE_DIAG, "Phase, bias omitted*")
]

example_audit = []

# =============================================================================
# F02 — REAL REGISTRATION IMAGE EXAMPLES
# =============================================================================

print("\nF02 — ACTUAL MSI REGISTRATION EXAMPLES", flush=True)

fig, axes = plt.subplots(
    2, 6,
    figsize=(15.3, 5.8),
    constrained_layout=True
)

column_titles = [
    "Aligned reference",
    "Displaced MSI",
    "Frozen NCC",
    "NCC, bias omitted*",
    "Frozen phase",
    "Phase, bias omitted*"
]

for r, pid in enumerate(EXAMPLE_PATCHES):

    reference, shifted, offsets = load_patch(pid)
    idx = find_condition(offsets, *EXAMPLE_SHIFT)
    observed = shifted[idx]
    limits = limits_for_msi(reference)

    images = [reference, observed]
    epe_labels = [None, float(np.hypot(*EXAMPLE_SHIFT))]

    for method, title in CORRECTION_METHODS:
        dy, dx, epe = saved_estimate(pid, idx, method)
        corrected = resample_for_display(observed, dy, dx)

        images.append(corrected)
        epe_labels.append(epe)

        example_audit.append({
            "figure": "F02",
            "patch_id": pid,
            "condition_index": idx,
            "method": method,
            "true_dy": EXAMPLE_SHIFT[0],
            "true_dx": EXAMPLE_SHIFT[1],
            "estimated_dy": dy,
            "estimated_dx": dx,
            "saved_epe": epe
        })

    for c, image in enumerate(images):
        ax = axes[r, c]
        ax.imshow(
            msi_rgb(image, limits),
            interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        if r == 0:
            ax.set_title(column_titles[c], fontsize=8.4, pad=7)

        if epe_labels[c] is not None:
            ax.text(
                0.035, 0.045,
                f"EPE {epe_labels[c]:.2f} px",
                transform=ax.transAxes,
                fontsize=7.4,
                color="white",
                bbox=dict(
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.72,
                    pad=2
                )
            )

    axes[r, 0].set_ylabel(pid, fontsize=8.1)

fig.supxlabel(
    "* Post hoc bias-omission diagnostic; corrections "
    "resampled for visualization only.",
    fontsize=9
)

export_figure(fig, "F02_Registration_Image_Examples")

pd.DataFrame(example_audit).to_csv(
    OUT / "Step44C_Example_Values.csv",
    index=False
)


# =============================================================================
# F03 — ACTUAL FOUR-BAND PIXELWISE DIFFERENCE MAPS
# =============================================================================

print("\nF03 — FOUR-BAND IMAGE DIFFERENCE MAPS", flush=True)

pid = PATCH_IDS[0]
reference, shifted, offsets = load_patch(pid)
idx = find_condition(offsets, *EXAMPLE_SHIFT)
observed = shifted[idx]

image_columns = [("Displaced MSI", observed)]

for method, label in CORRECTION_METHODS:
    dy, dx, epe = saved_estimate(pid, idx, method)
    image_columns.append(
        (label, resample_for_display(observed, dy, dx))
    )

def four_band_difference(ref, predicted):
    return np.mean(
        np.abs(
            ref.astype(np.float64)
            - predicted.astype(np.float64)
        ),
        axis=0
    )

BORDER = 6

difference_maps = [
    four_band_difference(reference, image)[
        BORDER:-BORDER, BORDER:-BORDER
    ]
    for label, image in image_columns
]

for image in difference_maps:
    if image.shape != (84, 84):
        raise RuntimeError("Unexpected difference-map shape")
    require_finite("MSI difference map", image)

max_color_value = float(
    np.percentile(
        np.concatenate([x.ravel() for x in difference_maps]),
        99.5
    )
)
max_color_value = max(max_color_value, 1e-12)

fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 3.6),
    constrained_layout=True
)

for i, (ax, (name, _), data) in enumerate(
    zip(axes, image_columns, difference_maps)
):
    im = ax.imshow(
        data,
        cmap="magma",
        vmin=0,
        vmax=max_color_value,
        interpolation="nearest"
    )
    ax.set_title(
        f"({chr(97+i)}) {name}",
        fontsize=8.2
    )
    ax.set_xticks([])
    ax.set_yticks([])

cb = fig.colorbar(
    im,
    ax=axes,
    location="right",
    shrink=0.80,
    pad=0.012
)
cb.set_label(
    "Mean absolute four-band signal difference",
    fontsize=9
)

export_figure(fig, "F03_Four_Band_Difference_Maps")


# =============================================================================
# F04 — SPATIAL PATCH-LEVEL REGISTRATION ERROR
# =============================================================================

print("\nF04 — SPATIAL PATCH-LEVEL ENDPOINT ERROR", flush=True)

PLOT_METHODS = [
    (NCC, "Frozen NCC"),
    (PHASE, "Frozen phase"),
    (NCC_DIAG, "NCC, bias omitted*"),
    (PHASE_DIAG, "Phase, bias omitted*")
]

patch_epe = (
    combined[
        combined["method"].isin(
            [m for m, label in PLOT_METHODS]
        )
    ]
    .groupby(["method", "patch_id"], as_index=False)
    .agg(
        cases=("epe", "size"),
        mean_epe=("epe", "mean"),
        median_epe=("epe", "median"),
        max_epe=("epe", "max")
    )
)

if not (patch_epe["cases"] == 17).all():
    raise RuntimeError(
        "Each patch-method pair must have 17 conditions."
    )

patch_epe.to_csv(
    OUT / "Step44C_Patch_EPE.csv",
    index=False
)

lookup = {}

for method, title in PLOT_METHODS:
    sub = patch_epe[patch_epe["method"] == method]
    lookup[method] = dict(
        zip(
            sub["patch_id"],
            sub["mean_epe"].astype(float)
        )
    )
    if set(lookup[method]) != set(PATCH_IDS):
        raise RuntimeError(
            f"Missing patch means for {method}"
        )

vmax = max(1.5, float(patch_epe["mean_epe"].max()))
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.get_cmap("YlOrRd")

fig, axes = plt.subplots(
    2, 2,
    figsize=(11.6, 8.7),
    constrained_layout=True
)

for i, (ax, (method, title)) in enumerate(
    zip(axes.flat, PLOT_METHODS)
):
    scene_axes(ax)
    draw_patches(
        ax,
        patch_values=lookup[method],
        norm=norm,
        cmap=cmap
    )
    ax.set_title(
        f"({chr(97+i)}) {title}",
        loc="left",
        fontsize=10.5
    )
    ax.set_xlabel("Easting (km)", fontsize=8)
    ax.set_ylabel("Northing (km)", fontsize=8)
    ax.tick_params(labelsize=7)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cb = fig.colorbar(
    sm,
    ax=axes.ravel().tolist(),
    shrink=0.7,
    pad=0.015
)
cb.set_label(
    "Patch mean endpoint error (pixels)",
    fontsize=9
)

export_figure(fig, "F04_Spatial_Patch_Registration_EPE")


# =============================================================================
# H — CAPTIONS AND FINAL PROVENANCE
# =============================================================================

captions = """
F01. Spatial distribution of five non-overlapping
96 × 96-pixel evaluation patches in the original Potsdam
EnMAP scene. The displayed RGB composite uses the frozen
four-band spectral-response weights from Step 30.
Patch extents were computed from the source GeoTIFF
geotransform and recorded pixel coordinates.

F02. Image-registration examples using actual synthetic
four-channel MSI observations from two Potsdam patches.
For each patch, an imposed 1.5-pixel vertical displacement
is shown alongside image corrections corresponding to
the frozen MDAS NCC and phase estimators and their post hoc
bias-omission diagnostics. Endpoint errors are obtained
from the saved numerical evaluation. Image resampling is
performed for visualization only, not as a new validation
experiment.

F03. Pixelwise mean absolute difference between an aligned
synthetic four-band MSI reference and the shifted or
visually corrected observations for one Potsdam patch.
All panels share the same display color scale and exclude
a six-pixel visualization border. The maps describe
synthetic MSI signal differences, not hyperspectral
fusion reconstruction errors.

F04. Spatial distribution of patch-level mean endpoint
error for frozen original NCC, frozen bounded phase
correlation, and their post hoc bias-omission diagnostics.
Each patch mean aggregates 17 synthetic displacement
conditions. The five non-overlapping patches belong to
one EnMAP acquisition; they do not represent five
independent geographic test scenes. Bias-omission results
are exploratory and were examined after viewing the
frozen external-transfer results.
"""

(OUT / "Step44C_Figure_Captions.txt").write_text(
    captions.strip(),
    encoding="utf-8"
)

report = {
    "experiment": "Verified Potsdam spatial figure generation",
    "status": "SPATIAL_FIGURES_COMPLETE",
    "figures_generated": 4,
    "png_dpi": 900,
    "vector_pdf_exports": 4,
    "source_acquisitions": 1,
    "source_patches": 5,
    "conditions_per_patch": 17,
    "source_crs": CRS,
    "source_image_shape": [224, 200, 330],
    "retained_hsi_bands": 218,
    "synthetic_msi_bands": 4,
    "cross_step_tolerance": COMPARISON_ATOL,
    "cross_step_max_difference": float(
        audit["max_absolute_difference"].max()
    ),
    "historical_registration_algorithms_rerun": False,
    "registration_parameters_tuned": False,
    "fusion_model_inference_performed": False,
    "bias_omission_methods_posthoc": True,
    "pixelwise_maps_type": (
        "four-band synthetic MSI intensity differences"
    ),
    "source_hashes": {
        name: sha256(path)
        for name, path in paths.items()
    },
    "limitations": [
        "Only one independent external EnMAP scene.",
        "Five spatially distinct patches are not independent scenes.",
        "Seventeen shifts per patch are repeated conditions.",
        "The MSI observations and translations are synthetic.",
        "F02/F03 image correction uses visualization resampling.",
        "F03 does not show hyperspectral reconstruction error.",
        "Post hoc bias-omission diagnostics are exploratory.",
        "No external fusion-model evaluation was performed."
    ],
    "elapsed_seconds": round(time.monotonic()-START, 2),
    "output_directory": str(OUT)
}

write_json(
    OUT / "Step44C_Final_Report.json",
    report
)

print("\n" + "=" * 90)
print("STEP 44C — FINAL REPORT")
print("=" * 90)
print(json.dumps(report, indent=2))

print("\nSTEP 44C COMPLETE")
print("Four spatial figures generated from actual Potsdam data.")
print("Original frozen results unchanged.")
print("Output:", OUT)
