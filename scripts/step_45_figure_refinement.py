
# =============================================================================
# STEP 45 — FINAL SPATIAL FIGURE REFINEMENT
# =============================================================================
#
# FIGURE F02:
#   Real Potsdam MSI observations and registration corrections
#   - 2 spatial patches
#   - 6 comparison columns
#   - Enlarged central details
#   - Reference/observation edge overlays
#
# FIGURE F04:
#   Compact 2x2 spatial patch mean-EPE comparison
#   - Actual georeferenced EnMAP background
#   - Five nonoverlapping 96x96 patch footprints
#   - Shared color scale
#   - Frozen and exploratory methods clearly separated
#
# SOURCE DATA:
#   Step27 original EnMAP TIFF
#   Step30 frozen observations / spectral response
#   Step41C frozen registration evaluation
#   Step42 exploratory bias-omission diagnostics
#
# No methods rerun; no estimates or parameters modified.
# Figures are descriptive visualizations of existing experiments.
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
import matplotlib.patheffects as pe

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import sobel
import rasterio

START = time.monotonic()

ROOT = Path("/content/drive/MyDrive")
PROJECT = ROOT / "MDAS_Registration_Aware_Revision_2026"

S27 = PROJECT / "Step27_External_EnMAP_Pilot"
S30 = PROJECT / "Step30_Frozen_Potsdam_Benchmark"
S41 = PROJECT / "Step41C_Frozen_Potsdam_Transfer"
S42 = PROJECT / "Step42_External_Calibration_Transfer_Audit"

TIF = S27 / "Imagery" / "enmap_potsdam.tif"
PROTOCOL_PATH = S30 / "Step30_Frozen_Forward_Model.json"
RESPONSE_PATH = S30 / "Step30_Spectral_Response_Matrix.csv"

FROZEN_CSV = S41 / "Step41C_Registration_Results.csv"
DIAGNOSTIC_CSV = S42 / "Step42_Bias_Attribution_Cases.csv"
STEP42_REPORT = S42 / "Step42_Final_Report.json"

OUT = PROJECT / "Step45_Refined_Spatial_Manuscript_Figures"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT_NAMES = [
    "F02_Refined_Registration_And_Edge_Overlays.png",
    "F02_Refined_Registration_And_Edge_Overlays.pdf",
    "F04_Refined_Compact_Spatial_EPE.png",
    "F04_Refined_Compact_Spatial_EPE.pdf",
    "Step45_Figure_Data_Audit.csv",
    "Step45_Patch_EPE.csv",
    "Step45_Figure_Captions.txt",
    "Step45_Final_Report.json",
]

existing = [x for x in OUTPUT_NAMES if (OUT / x).exists()]

if existing:
    raise FileExistsError(
        "Step45 outputs already exist. Use a new output "
        "directory rather than overwrite previous results:\n"
        + "\n".join(existing)
    )

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str),
        encoding="utf-8"
    )

def save_fig(fig, stem):
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=900, bbox_inches="tight",
        pad_inches=0.08, facecolor="white"
    )
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.08, facecolor="white"
    )
    print("Saved:", stem, "(PNG and PDF)", flush=True)
    plt.show()
    plt.close(fig)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

print("="*88)
print("STEP 45 — REFINED SCIENTIFIC SPATIAL FIGURES")
print("="*88)

# =============================================================================
# A — PROVENANCE AND CASE INTEGRITY
# =============================================================================

for path in [
    TIF, PROTOCOL_PATH, RESPONSE_PATH,
    FROZEN_CSV, DIAGNOSTIC_CSV, STEP42_REPORT
]:
    if not path.is_file():
        raise FileNotFoundError(path)

protocol = load_json(PROTOCOL_PATH)
r42 = load_json(STEP42_REPORT)

frozen = pd.read_csv(FROZEN_CSV)
combined = pd.read_csv(DIAGNOSTIC_CSV)
response_df = pd.read_csv(RESPONSE_PATH)

assert len(frozen) == 255
assert len(combined) == 425
assert protocol["source_scene_count"] == 1

if r42["source_step41c_results_sha256"] != file_sha(FROZEN_CSV):
    raise RuntimeError("Step42 does not match Step41C source CSV")

NCC = "frozen_ncc_step13"
PHASE = "frozen_phase_step20_size_adapted"
IDENTITY = "identity"
NCC_DIAG = "ncc_prebias_same_gate_diagnostic"
PHASE_DIAG = "phase_prebias_same_gate_diagnostic"

METHODS = [NCC, PHASE, IDENTITY, NCC_DIAG, PHASE_DIAG]

assert set(combined["method"]) == set(METHODS)
assert (combined.groupby("method").size() == 85).all()

KEYS = ["patch_id", "condition_index"]

if combined.duplicated(KEYS + ["method"]).any():
    raise RuntimeError("Duplicated registration records")

numeric_cols = [
    "true_dy", "true_dx",
    "estimated_dy", "estimated_dx", "epe"
]

for df in [frozen, combined]:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[numeric_cols].to_numpy(float)).all():
        raise RuntimeError("Nonfinite saved results")

# Match frozen results by case keys, avoiding row-order assumptions.
for method in [NCC, PHASE, IDENTITY]:
    a = frozen[frozen["method"] == method][KEYS + numeric_cols]
    b = combined[combined["method"] == method][KEYS + numeric_cols]

    matched = a.merge(
        b, on=KEYS,
        suffixes=("_frozen", "_diagnostic_copy"),
        validate="one_to_one"
    )

    if len(matched) != 85:
        raise RuntimeError("Frozen case keys do not reconcile")

    for col in numeric_cols:
        difference = np.abs(
            matched[f"{col}_frozen"].to_numpy(float)
            - matched[f"{col}_diagnostic_copy"].to_numpy(float)
        )
        if float(difference.max()) > 1e-9:
            raise RuntimeError(
                f"Cross-step result changed: {method} / {col}"
            )

print("\n45A — Frozen records verified unchanged")

# =============================================================================
# B — DISPLAY FUNCTIONS AND FROZEN OBSERVATIONS
# =============================================================================

def patch_row_col(pid):
    match = re.fullmatch(r"potsdam_r(\d+)_c(\d+)", pid)
    if match is None:
        raise ValueError(f"Bad patch identifier: {pid}")
    return int(match.group(1)), int(match.group(2))

PATCH_IDS = sorted(
    protocol["selected_patch_ids"], key=patch_row_col
)
assert len(PATCH_IDS) == 5

retained_ids = np.asarray(
    protocol["retained_band_ids_1_based"], dtype=int
)

assert retained_ids.shape == (218,)
assert np.array_equal(
    response_df["original_hsi_band"].to_numpy(int),
    retained_ids
)

RESPONSE = response_df[
    [
        "synthetic_blue_weight",
        "synthetic_green_weight",
        "synthetic_red_weight",
        "synthetic_nir_weight"
    ]
].to_numpy(np.float64).T

assert RESPONSE.shape == (4,218)
assert np.allclose(RESPONSE.sum(axis=1), 1, atol=1e-7, rtol=0)

def stretch(rgb, limits):
    lo, hi = limits
    return np.clip(
        (rgb.astype(np.float64)-lo)
        / np.maximum(hi-lo, 1e-12),
        0, 1
    ) ** 0.85

def display_rgb(msi, limits):
    rgb = np.moveaxis(msi[[2,1,0]], 0, -1)
    return stretch(rgb, limits)

def display_limits(msi):
    rgb = np.moveaxis(msi[[2,1,0]], 0, -1)
    return (
        np.percentile(rgb, 2, axis=(0,1)),
        np.percentile(rgb, 98, axis=(0,1))
    )

def load_patch(pid):
    path = S30 / "Paired_Observations" / f"{pid}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        ref = z["hrmsi"].copy()
        shifted = z["shifted_msi"].copy()
        offsets = z["shift_offsets_dy_dx"].copy()
        bandids = z["retained_band_ids"].copy()

    assert ref.shape == (4,96,96)
    assert shifted.shape == (17,4,96,96)
    assert offsets.shape == (17,2)
    assert np.array_equal(bandids.astype(int), retained_ids)

    aligned_indices = np.flatnonzero(
        np.all(np.isclose(offsets, 0, atol=1e-10), axis=1)
    )
    if len(aligned_indices) != 1:
        raise RuntimeError("Aligned observation ambiguous")
    if not np.array_equal(shifted[aligned_indices[0]], ref):
        raise RuntimeError("Aligned MSI does not match reference")
    return ref, shifted, offsets

def get_index(offsets, shift=(1.5, 0.0)):
    indexes = np.flatnonzero(
        np.all(np.isclose(offsets, shift, atol=1e-8), axis=1)
    )
    if len(indexes) != 1:
        raise RuntimeError("Chosen frozen shift not found")
    return int(indexes[0])

def get_estimate(pid, j, method):
    q = combined[
        (combined["patch_id"] == pid)
        & (combined["condition_index"] == j)
        & (combined["method"] == method)
    ]
    if len(q) != 1:
        raise RuntimeError("Missing saved displacement")
    row = q.iloc[0]
    return (
        float(row["estimated_dy"]),
        float(row["estimated_dx"]),
        float(row["epe"])
    )

def warp_for_display(observed, dy, dx):
    # Display only: inverse estimated translation.
    return ndi_shift(
        observed.astype(np.float32),
        shift=(0,-dy,-dx),
        order=1,
        mode="nearest",
        prefilter=False
    )

def gray_from_msi(msi):
    # Normalized visible-band intensity used only for overlays.
    return np.mean(
        msi[[0,1,2]].astype(np.float64),
        axis=0
    )

def edge_strength(msi):
    gray = gray_from_msi(msi)
    gy = sobel(gray, axis=0)
    gx = sobel(gray, axis=1)
    return np.hypot(gy, gx)

def binary_edges(msi, threshold):
    return edge_strength(msi) >= threshold

def overlay_edges(reference, comparison, threshold):
    # Reference edges: magenta; comparison edges: cyan.
    # White pixels indicate overlap of edge responses.
    r = binary_edges(reference, threshold)
    c = binary_edges(comparison, threshold)

    out = np.zeros((*r.shape, 3), dtype=float) + 0.06
    out[r] = (1.0, 0.28, 0.63)
    out[c] = (0.15, 0.85, 0.92)
    out[r & c] = (1.0, 1.0, 1.0)
    return out

# =============================================================================
# C — F02: FULL IMAGE, MAGNIFIED DETAIL, AND EDGE OVERLAY
# =============================================================================

print("\n45B — Building improved F02")

# Select patches by position, not by model performance.
EXAMPLES = [PATCH_IDS[0], PATCH_IDS[-1]]
SHIFT = (1.5, 0.0)

# The same inset region is used for all methods on each patch.
CROP_Y0, CROP_Y1 = 25, 65
CROP_X0, CROP_X1 = 25, 65

DISPLAY_METHODS = [
    (None, "Aligned reference"),
    ("shifted", "Displaced MSI"),
    (NCC, "Frozen NCC"),
    (NCC_DIAG, "NCC, bias omitted*"),
    (PHASE, "Frozen phase"),
    (PHASE_DIAG, "Phase, bias omitted*")
]

audit_rows = []

# Six columns and three rows per example:
# row 1: true-color full patch
# row 2: 40x40 central zoom
# row 3: magnified edge overlay.
# Two separate figures, assembled in one 2-page conceptual layout
# by stacking two per-patch blocks.

fig = plt.figure(figsize=(17, 13.1), constrained_layout=True)
outer = fig.add_gridspec(
    2, 1,
    height_ratios=[1,1],
    hspace=0.10
)

for block_index, pid in enumerate(EXAMPLES):
    reference, shifted, offsets = load_patch(pid)
    j = get_index(offsets, SHIFT)
    observed = shifted[j]
    limits = display_limits(reference)

    reference_edges = edge_strength(reference)
    # Fixed per-patch threshold determined solely from reference.
    EDGE_THRESHOLD = float(np.percentile(
        reference_edges[8:-8,8:-8], 85
    ))

    images = [reference, observed]
    epe_values = [None, float(np.hypot(*SHIFT))]

    for method, label in DISPLAY_METHODS[2:]:
        dy, dx, epe = get_estimate(pid,j,method)
        images.append(warp_for_display(observed,dy,dx))
        epe_values.append(epe)

        audit_rows.append({
            "patch_id": pid,
            "condition_index": j,
            "method": method,
            "estimated_dy": dy,
            "estimated_dx": dx,
            "registration_epe": epe,
            "edge_threshold_reference_85pct": EDGE_THRESHOLD
        })

    block_grid = outer[block_index].subgridspec(
        3, 6,
        height_ratios=[1.0, 0.88, 0.88],
        hspace=0.06, wspace=0.04
    )

    for col, ((method, label), image) in enumerate(
        zip(DISPLAY_METHODS, images)
    ):
        for row in range(3):
            ax = fig.add_subplot(block_grid[row,col])

            if row == 0:
                shown = display_rgb(image, limits)
                ax.imshow(shown, interpolation="nearest")
                ax.add_patch(Rectangle(
                    (CROP_X0-0.5, CROP_Y0-0.5),
                    CROP_X1-CROP_X0,
                    CROP_Y1-CROP_Y0,
                    fill=False,
                    edgecolor="#F6D55C",
                    linewidth=1.2
                ))

            elif row == 1:
                shown = display_rgb(image, limits)
                ax.imshow(
                    shown[CROP_Y0:CROP_Y1,CROP_X0:CROP_X1],
                    interpolation="nearest"
                )

            else:
                # The first column is reference edge map.
                if col == 0:
                    edge = binary_edges(reference, EDGE_THRESHOLD)
                    edge = edge[CROP_Y0:CROP_Y1,CROP_X0:CROP_X1]
                    ax.imshow(edge, cmap="gray", vmin=0, vmax=1)
                else:
                    overlay = overlay_edges(
                        reference, image, EDGE_THRESHOLD
                    )
                    ax.imshow(
                        overlay[CROP_Y0:CROP_Y1,CROP_X0:CROP_X1],
                        interpolation="nearest"
                    )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#777777")

            if row == 0:
                ax.set_title(
                    label, fontsize=8.5,
                    pad=7
                )
                if epe_values[col] is not None:
                    ax.text(
                        0.025, 0.035,
                        f"EPE {epe_values[col]:.2f} px",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=7.5,
                        bbox=dict(
                            facecolor="black",
                            alpha=0.72,
                            edgecolor="none",
                            pad=1.6
                        )
                    )

            if col == 0:
                row_labels = [
                    f"{pid}\nFull patch",
                    "40×40 detail",
                    "Reference edges" if row == 2 else ""
                ]
                ax.set_ylabel(
                    row_labels[row],
                    fontsize=8
                )

fig.text(
    0.5, 0.01,
    "Edge overlay: magenta = reference, cyan = comparison, "
    "white = overlap. * Post hoc diagnostic; warping shown for visualization.",
    ha="center", va="bottom", fontsize=9
)

save_fig(
    fig,
    "F02_Refined_Registration_And_Edge_Overlays"
)

pd.DataFrame(audit_rows).to_csv(
    OUT / "Step45_Figure_Data_Audit.csv",
    index=False
)

# =============================================================================
# D — LOAD ACTUAL GEOGRAPHIC ENMAP IMAGERY
# =============================================================================

print("\n45C — Loading actual EnMAP scene for F04")

EXPECTED_TIF_SHA = (
    "4ba14fd27ac80c9169f3d733010dbdaa986584b3a062dc497578c3e114733329"
)

if file_sha(TIF) != EXPECTED_TIF_SHA:
    raise RuntimeError("Source EnMAP TIFF checksum mismatch")

with rasterio.open(TIF) as src:
    assert (src.count, src.height, src.width) == (224,200,330)
    assert src.crs is not None
    CRS = str(src.crs)
    assert CRS == "EPSG:32633"
    TRANSFORM = src.transform
    BOUNDS = src.bounds

    if not (
        np.isclose(TRANSFORM.b, 0)
        and np.isclose(TRANSFORM.d, 0)
    ):
        raise RuntimeError("Unexpected rotated image grid")

    # Rasterio indexes must be a plain Python list.
    CUBE = src.read(
        indexes=[int(i) for i in retained_ids]
    ).astype(np.float32)

assert CUBE.shape == (218,200,330)

scene_msi = np.einsum(
    "bc,chw->bhw",
    RESPONSE,
    CUBE.astype(np.float64),
    optimize=True
)

scene_rgb = np.moveaxis(
    scene_msi[[2,1,0]], 0, -1
)

scene_limits = (
    np.percentile(scene_rgb,2,axis=(0,1)),
    np.percentile(scene_rgb,98,axis=(0,1))
)

SCENE_DISPLAY = stretch(scene_rgb,scene_limits)

XMIN, XMAX = BOUNDS.left/1000, BOUNDS.right/1000
YMIN, YMAX = BOUNDS.bottom/1000, BOUNDS.top/1000

def patch_box(pid):
    row,col = patch_row_col(pid)
    x0,y0 = TRANSFORM*(col,row)
    x1,y1 = TRANSFORM*(col+96,row+96)
    return (
        min(x0,x1)/1000,
        max(x0,x1)/1000,
        min(y0,y1)/1000,
        max(y0,y1)/1000
    )

# =============================================================================
# E — COMPACT F04 PATCH-LEVEL ERROR
# =============================================================================

plot_methods = [
    (NCC, "Frozen NCC"),
    (PHASE, "Frozen bounded phase"),
    (NCC_DIAG, "NCC, bias omitted*"),
    (PHASE_DIAG, "Phase, bias omitted*")
]

patch_means = (
    combined[
        combined["method"].isin([m for m,_ in plot_methods])
    ]
    .groupby(["method","patch_id"],as_index=False)
    .agg(
        cases=("epe","size"),
        mean_epe=("epe","mean")
    )
)

if not (patch_means["cases"] == 17).all():
    raise RuntimeError("Expected 17 conditions for every patch")

patch_means.to_csv(
    OUT / "Step45_Patch_EPE.csv",
    index=False
)

lookup = {}
for method,label in plot_methods:
    sub = patch_means[patch_means["method"] == method]
    lookup[method] = dict(zip(
        sub["patch_id"],sub["mean_epe"].astype(float)
    ))

cmap = plt.get_cmap("YlOrRd")
norm = Normalize(
    vmin=0,
    vmax=max(1.7, float(patch_means["mean_epe"].max()))
)

fig, axes = plt.subplots(
    2,2,
    figsize=(12.6,8.4),
    gridspec_kw={
        "wspace":0.08,
        "hspace":0.16
    }
)

for k,(ax,(method,label)) in enumerate(
    zip(axes.flat,plot_methods)
):
    ax.imshow(
        SCENE_DISPLAY,
        extent=[XMIN,XMAX,YMIN,YMAX],
        origin="upper",
        interpolation="nearest",
        alpha=0.91
    )

    for i,pid in enumerate(PATCH_IDS,1):
        x0,x1,y0,y1 = patch_box(pid)
        val = lookup[method][pid]

        ax.add_patch(Rectangle(
            (x0,y0),x1-x0,y1-y0,
            facecolor=cmap(norm(val)),
            edgecolor="white",
            linewidth=1.25,
            alpha=0.64,
            zorder=4
        ))
        ax.text(
            (x0+x1)/2,(y0+y1)/2,
            f"P{i}\n{val:.2f}",
            ha="center",va="center",
            fontsize=8.5,fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.88,
                pad=1.3
            ),
            zorder=7
        )

    ax.set_aspect("equal")
    ax.set_xlim(XMIN,XMAX)
    ax.set_ylim(YMIN,YMAX)
    ax.set_title(
        f"({chr(97+k)}) {label}",
        loc="left",
        fontsize=11,
        pad=6
    )

    ax.tick_params(
        direction="out",
        labelsize=7,
        top=True,right=True
    )

    if k in [2,3]:
        ax.set_xlabel("Easting (km)")
    else:
        ax.tick_params(labelbottom=False)

    if k in [0,2]:
        ax.set_ylabel("Northing (km)")
    else:
        ax.tick_params(labelleft=False)

fig.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.08,
    top=0.94,
    hspace=0.15,
    wspace=0.08
)

cax = fig.add_axes([0.90,0.22,0.022,0.55])

sm = ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])

cb = fig.colorbar(sm,cax=cax)
cb.set_label(
    "Patch mean endpoint error (pixels)",
    fontsize=9
)

save_fig(
    fig,
    "F04_Refined_Compact_Spatial_EPE"
)

# =============================================================================
# F — CAPTIONS AND FINAL REPORT
# =============================================================================

captions = """
F02. Registration and edge-overlap examples for two fixed
Potsdam patches under a synthetic 1.5-pixel vertical translation.
For each patch, full RGB composites, 40 × 40-pixel central
details and reference/comparison edge overlays are displayed.
All corrected images are visualized using inverse shifts of
the saved Step41C/Step42 displacement estimates. Edge thresholds
are computed from each aligned reference image and held
fixed across methods. Magenta indicates reference edges, cyan
indicates comparison edges, and white marks overlapping edge
responses. Endpoint errors derive from the saved registration
evaluation rather than newly computed image differences.
The bias-omission variants are post hoc diagnostics.

F04. Compact spatial summary of mean registration endpoint error
across five non-overlapping Potsdam patches. Each patch value
summarizes 17 synthetic displacement conditions from a single
EnMAP acquisition. The upper panels show the frozen original
MDAS NCC and bounded-phase transfer results; the lower panels
show exploratory bias-omission counterfactuals. All four maps
use the same projected coordinates and shared error scale.
The colour overlay summarizes patch-level errors rather than
pixelwise registration fields.
"""

(OUT / "Step45_Figure_Captions.txt").write_text(
    captions.strip(),encoding="utf-8"
)

report = {
    "status":"REFINED_SPATIAL_FIGURES_GENERATED",
    "figure_count":2,
    "png_dpi":900,
    "pdf_vector_exports":2,
    "original_enmap_used":True,
    "source_crs":CRS,
    "source_acquisitions":1,
    "patch_count":5,
    "synthetic_conditions_per_patch":17,
    "registration_estimators_rerun":False,
    "registration_parameters_modified":False,
    "image_warping_for_visualization_only":True,
    "posthoc_diagnostics_explicit":True,
    "source_sha256":{
        "enmap":file_sha(TIF),
        "protocol":file_sha(PROTOCOL_PATH),
        "response":file_sha(RESPONSE_PATH),
        "step41c":file_sha(FROZEN_CSV),
        "step42":file_sha(DIAGNOSTIC_CSV)
    },
    "output_directory":str(OUT),
    "elapsed_seconds":round(time.monotonic()-START,2)
}

save_json(OUT/"Step45_Final_Report.json",report)

print("\n"+"="*88)
print("STEP 45 — FINAL REPORT")
print("="*88)
print(json.dumps(report,indent=2))
print("\nSTEP 45 COMPLETE")
print("Previous figure files and experimental outputs unchanged.")
