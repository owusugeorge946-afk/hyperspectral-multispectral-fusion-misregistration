
# =============================================================================
# STEP 41C — FROZEN MDAS REGISTRATION TRANSFER TO POTSDAM
#             WITH SPECTRAL METADATA RECONCILIATION
#
# Changes from Step41B:
#   - Explicitly measure CSV-versus-NPZ wavelength differences.
#   - Verify the 218 original band IDs in both files.
#   - Permit only small metadata rounding differences (<= 0.1 nm).
#   - Use saved Step30 spectral-response WEIGHTS without modification.
#
# Original registration algorithms/parameters remain frozen.
# No external parameter tuning and no fusion-model inference.
#
# Outputs saved only to a NEW Step41C folder.
# =============================================================================

from google.colab import drive
from pathlib import Path
from scipy.ndimage import gaussian_filter, sobel

import ast
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive")
PROJECT = ROOT / "MDAS_Registration_Aware_Revision_2026"

NOTEBOOK = (
    ROOT / "Colab Notebooks"
    / "Sensitivity of Hyperspectral and Multispectral Image .ipynb"
)

S13 = PROJECT / "Step13_Identity_Preserving_Gate"
S20 = PROJECT / "Step20_Bounded_Phase_Correlation"
S30 = PROJECT / "Step30_Frozen_Potsdam_Benchmark"
S39 = PROJECT / "Step39_Original_Estimator_Numerical_Replay"
S40 = PROJECT / "Step40_Reproduction_Discrepancy_Audit"

OUT = PROJECT / "Step41C_Frozen_Potsdam_Transfer"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT_NAMES = [
    "Step41C_Wavelength_Audit.csv",
    "Step41C_Registration_Results.csv",
    "Step41C_Method_Summary.csv",
    "Step41C_Shift_Summary.csv",
    "Step41C_Patch_Summary.csv",
    "Step41C_Paired_Comparison.csv",
    "Step41C_Provenance.json",
    "Step41C_Final_Report.json",
    "Step41C_Registration_Comparison.png",
    "Step41C_Registration_Comparison.pdf",
]

if any((OUT / name).exists() for name in OUTPUT_NAMES):
    raise FileExistsError(
        "Previous Step41C output exists. "
        "Preserve it and use a new output folder."
    )

START = time.monotonic()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def text_sha(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=str
        ),
        encoding="utf-8"
    )

print("=" * 92, flush=True)
print("STEP 41C — FROZEN ORIGINAL-METHOD POTSDAM TRANSFER", flush=True)
print("=" * 92, flush=True)


# =============================================================================
# A — VERIFY FROZEN INPUTS
# =============================================================================

FILES = {
    "notebook": NOTEBOOK,
    "gate": S13 / "Step13_Frozen_Gate.json",
    "phase": S20 / "Step20_Frozen_Method.json",
    "protocol": S30 / "Step30_Frozen_Forward_Model.json",
    "response": S30 / "Step30_Spectral_Response_Matrix.csv",
    "manifest": S30 / "Step30_Paired_Observation_Manifest.csv",
    "functions": S39 / "Step39_Function_Provenance.csv",
    "reproduction": S40 / "Step40_Final_Report.json",
}

for label, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")

gate_cfg = read_json(FILES["gate"])
phase_cfg = read_json(FILES["phase"])
protocol = read_json(FILES["protocol"])
reproduction = read_json(FILES["reproduction"])

assert reproduction["status"] == "NUMERICAL_REPRODUCTION_CONSISTENT"
assert reproduction["ncc_matches_at_1e_minus_5"] == 680
assert reproduction["phase_matches_at_3e_minus_5"] == 680
assert reproduction["phase_identity_snap_decisions_changed"] == 0

EXPECTED_PROTOCOL_SHA = (
    "dddd0df0d1bd0aeca0976e5bd22821ac4d07c3b1e8ed96c64eb3db4969a94eca"
)

assert file_sha(FILES["protocol"]) == EXPECTED_PROTOCOL_SHA

assert protocol["source_scene_count"] == 1
assert protocol["retained_band_count"] == 218
assert len(protocol["selected_patch_ids"]) == 5

GATE_NCC = float(gate_cfg["threshold_pixels"])
GATE_PHASE = float(phase_cfg["tau"])

RADIUS_NCC = 5
RADIUS_PHASE = int(phase_cfg["radius"])

BIAS_NCC_DY = -1
BIAS_NCC_DX = -1

BIAS_PHASE_DY = float(phase_cfg["bias_dy"])
BIAS_PHASE_DX = float(phase_cfg["bias_dx"])

assert GATE_NCC == 0.225
assert GATE_PHASE == 0.25
assert RADIUS_PHASE == 5

print("\n41C-A — Historical provenance VERIFIED", flush=True)


# =============================================================================
# B — RECOVER EXACT ORIGINAL FUNCTION DEFINITIONS
# =============================================================================

nb = read_json(NOTEBOOK)

prior = pd.read_csv(FILES["functions"])

expected_hashes = {
    (int(r.cell_index), str(r.function_name)): str(r.source_sha256)
    for r in prior.itertuples(index=False)
}

NCC_NAMES = [
    "registration_features",
    "integer_shift_np",
    "ncc",
    "score_surface",
    "quadratic_vertex",
    "estimate_both",
]

PHASE_NAMES = [
    "standardized_gray",
    "feature_map",
    "phase_surface",
    "vertex",
    "select_bounded",
]

def source_cell(index):
    cell = nb["cells"][index]

    if cell.get("cell_type") != "code":
        raise RuntimeError(f"Cell {index} is not code")

    text = cell.get("source", "")

    return (
        "".join(text)
        if isinstance(text, list)
        else str(text)
    )

def original_node(index, function_name):
    source = source_cell(index)
    tree = ast.parse(source)
    lines = source.splitlines()

    matches = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == function_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely recover {function_name} "
            f"from cell {index}"
        )

    node = matches[0]

    if node.decorator_list:
        raise RuntimeError(
            f"Unexpected decorators: {function_name}"
        )

    extracted = "\n".join(
        lines[node.lineno - 1:node.end_lineno]
    )

    actual = text_sha(extracted)
    expected = expected_hashes.get(
        (index, function_name)
    )

    if expected is None or actual != expected:
        raise RuntimeError(
            f"Historical source mismatch: {function_name}"
        )

    return node

def install_functions(nodes, namespace, label):
    module = ast.Module(
        body=nodes,
        type_ignores=[]
    )
    ast.fix_missing_locations(module)

    exec(
        compile(module, label, "exec"),
        namespace
    )

ncc_nodes = [
    original_node(15, name)
    for name in NCC_NAMES
]

phase_nodes = [
    original_node(24, name)
    for name in PHASE_NAMES
]

ncc_env = {
    "np": np,
    "gaussian_filter": gaussian_filter,
    "sobel": sobel,
    "RADIUS": RADIUS_NCC,
    "BIAS_DY": BIAS_NCC_DY,
    "BIAS_DX": BIAS_NCC_DX,
    "CROP_MARGIN": 8,
    "BLUR_SIGMA": 1.2,
    "INTENSITY_WEIGHT": 0.7,
    "GRADIENT_WEIGHT": 0.3,
}

phase_env = {
    "np": np,
    "cv2": cv2,
}

install_functions(
    ncc_nodes, ncc_env, "mdas_original_ncc"
)

install_functions(
    phase_nodes, phase_env, "mdas_original_phase"
)

print(
    "41C-B — 11 original function definitions verified",
    flush=True
)


# =============================================================================
# C — VERIFIED 72->96 PHASE SURFACE ADAPTATION
# =============================================================================

original_phase_surface = phase_env["phase_surface"]

def generalized_phase_surface(reference, observed):

    reference = np.asarray(reference)
    observed = np.asarray(observed)

    if (
        reference.ndim != 2
        or reference.shape != observed.shape
    ):
        raise ValueError(
            "Phase inputs must be matching 2-D arrays"
        )

    h, w = reference.shape

    if min(h, w) < 2 * RADIUS_PHASE + 3:
        raise ValueError(
            "Phase image dimensions are too small"
        )

    window = np.outer(
        np.hanning(h),
        np.hanning(w)
    ).astype(np.float64)

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)

    a = (a - a.mean()) * window
    b = (b - b.mean()) * window

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    cross = fb * np.conj(fa)
    magnitude = np.abs(cross)

    scale = max(
        float(magnitude.max()),
        1e-12
    )

    normalized = cross / np.maximum(
        magnitude,
        scale * 1e-8
    )

    correlation = np.fft.fftshift(
        np.fft.ifft2(normalized).real
    )

    if not np.isfinite(correlation).all():
        raise RuntimeError(
            "Nonfinite phase-correlation surface"
        )

    return correlation

rng = np.random.default_rng(4102)

surface_differences = []

for _ in range(5):
    a = rng.normal(size=(72, 72)).astype(np.float32)
    b = rng.normal(size=(72, 72)).astype(np.float32)

    original = original_phase_surface(a, b)
    adapted = generalized_phase_surface(a, b)

    surface_differences.append(
        float(np.max(np.abs(original - adapted)))
    )

max_surface_difference = max(surface_differences)

if max_surface_difference > 1e-12:
    raise RuntimeError(
        "Phase adaptation failed 72x72 equivalence"
    )

toy = rng.normal(
    size=(96, 96)
).astype(np.float32)

toy_shifted = np.roll(
    toy,
    shift=(2, -3),
    axis=(0, 1)
)

toy_peak = phase_env["select_bounded"](
    generalized_phase_surface(
        toy,
        toy_shifted
    ),
    RADIUS_PHASE
)

assert np.allclose(
    [
        toy_peak["raw_dy"],
        toy_peak["raw_dx"]
    ],
    [2, -3],
    atol=0.1,
    rtol=0
)

phase_env["phase_surface"] = generalized_phase_surface

print("\n41C-C — Spatial adaptation VERIFIED", flush=True)
print(
    "72x72 maximum difference:",
    max_surface_difference
)
print(
    "96x96 known-shift estimate:",
    toy_peak["raw_dy"],
    toy_peak["raw_dx"]
)


# =============================================================================
# D — LOAD FROZEN SPECTRAL RESPONSE WITHOUT ALTERATION
# =============================================================================

response_df = pd.read_csv(
    FILES["response"]
)

weight_columns = [
    "synthetic_blue_weight",
    "synthetic_green_weight",
    "synthetic_red_weight",
    "synthetic_nir_weight",
]

required_columns = [
    "original_hsi_band",
    "wavelength_nm",
] + weight_columns

if not set(required_columns).issubset(
    response_df.columns
):
    raise RuntimeError(
        "Unexpected Step30 response schema:\n"
        + str(response_df.columns.tolist())
    )

if len(response_df) != 218:
    raise RuntimeError(
        "Unexpected response band count"
    )

retained_ids = np.asarray(
    protocol["retained_band_ids_1_based"],
    dtype=int
)

csv_ids = response_df[
    "original_hsi_band"
].to_numpy(dtype=int)

if not np.array_equal(csv_ids, retained_ids):
    raise RuntimeError(
        "Response CSV original band IDs do not "
        "match the frozen protocol"
    )

csv_wavelengths = response_df[
    "wavelength_nm"
].to_numpy(dtype=np.float64)

if not np.isfinite(csv_wavelengths).all():
    raise RuntimeError(
        "Nonfinite CSV wavelengths"
    )

RESPONSE = response_df[
    weight_columns
].to_numpy(dtype=np.float64).T

assert RESPONSE.shape == (4, 218)
assert np.isfinite(RESPONSE).all()

if not np.allclose(
    RESPONSE.sum(axis=1),
    1.0,
    rtol=0,
    atol=1e-7
):
    raise RuntimeError(
        "Response weights are not normalized"
    )

print("\n41C-D — Frozen response loaded", flush=True)
print("Original bands:", len(retained_ids))
print("Response shape:", RESPONSE.shape)
print("Channel sums:", RESPONSE.sum(axis=1))


# =============================================================================
# E — PRE-FLIGHT ALL PATCHES AND AUDIT WAVELENGTH DIFFERENCES
# =============================================================================

WAVELENGTH_MAX_TOLERANCE_NM = 0.1

patch_files = {}
patch_hashes = {}
audit_rows = []

print("\n41C-E — WAVELENGTH PROVENANCE AUDIT", flush=True)

for patch_id in protocol["selected_patch_ids"]:

    path = (
        S30 / "Paired_Observations"
        / f"{patch_id}.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    patch_files[patch_id] = path
    patch_hashes[patch_id] = file_sha(path)

    with np.load(
        path,
        allow_pickle=False
    ) as z:
        ids = z["retained_band_ids"].astype(int)
        wavelengths = z["wavelengths_nm"].astype(np.float64)

        shapes = {
            "lrhsi": z["lrhsi"].shape,
            "hrmsi": z["hrmsi"].shape,
            "shifted_msi": z["shifted_msi"].shape,
            "shift_offsets_dy_dx": z["shift_offsets_dy_dx"].shape,
        }

    assert ids.shape == (218,)
    assert wavelengths.shape == (218,)

    assert shapes["lrhsi"] == (218, 32, 32)
    assert shapes["hrmsi"] == (4, 96, 96)
    assert shapes["shifted_msi"] == (17, 4, 96, 96)
    assert shapes["shift_offsets_dy_dx"] == (17, 2)

    ids_match = np.array_equal(
        ids, retained_ids
    )

    if not np.isfinite(wavelengths).all():
        raise RuntimeError(
            f"Nonfinite wavelengths in {patch_id}"
        )

    differences = np.abs(
        wavelengths - csv_wavelengths
    )

    for i, difference in enumerate(differences):

        audit_rows.append({
            "patch_id": patch_id,
            "band_index_0based": i,
            "original_band_id": int(ids[i]),
            "band_id_matches_protocol": bool(
                ids[i] == retained_ids[i]
            ),
            "npz_wavelength_nm": float(
                wavelengths[i]
            ),
            "csv_wavelength_nm": float(
                csv_wavelengths[i]
            ),
            "absolute_difference_nm": float(
                difference
            )
        })

    max_diff = float(differences.max())
    median_diff = float(np.median(differences))

    print(
        f"  {patch_id} | "
        f"IDs match: {ids_match} | "
        f"max wavelength difference: {max_diff:.8f} nm | "
        f"median: {median_diff:.8f} nm",
        flush=True
    )

wavelength_audit = pd.DataFrame(
    audit_rows
)

wavelength_audit.to_csv(
    OUT / "Step41C_Wavelength_Audit.csv",
    index=False
)

max_wavelength_difference = float(
    wavelength_audit["absolute_difference_nm"].max()
)

all_ids_match = bool(
    wavelength_audit[
        "band_id_matches_protocol"
    ].all()
)

if not all_ids_match:
    raise RuntimeError(
        "Actual original spectral band IDs are inconsistent. "
        "Do not run registration until resolved."
    )

if max_wavelength_difference > WAVELENGTH_MAX_TOLERANCE_NM:

    worst = wavelength_audit.nlargest(
        10,
        "absolute_difference_nm"
    )

    print("\nLARGEST WAVELENGTH DISCREPANCIES")
    print(
        worst.to_string(index=False)
    )

    raise RuntimeError(
        "Wavelength differences exceed 0.1 nm. "
        "This cannot yet be classified as CSV rounding. "
        "Review the saved Step41C_Wavelength_Audit.csv."
    )

print("\nSpectral metadata audit PASSED.")
print(
    "Maximum absolute wavelength difference:",
    max_wavelength_difference,
    "nm"
)

# This audit accepts small metadata differences only.
# The response weights are never recomputed or modified.


# =============================================================================
# F — BUILD EXTERNAL REGISTRATION REFERENCE
# =============================================================================

def registration_reference(lrhsi):

    if lrhsi.shape != (218, 32, 32):
        raise ValueError(
            f"Unexpected LRHSI shape: {lrhsi.shape}"
        )

    lowres_msi = np.einsum(
        "bc,chw->bhw",
        RESPONSE,
        lrhsi.astype(np.float64),
        optimize=True
    ).astype(np.float32)

    x = torch.from_numpy(
        np.ascontiguousarray(
            lowres_msi[None]
        )
    )

    with torch.inference_mode():
        y = F.interpolate(
            x,
            size=(96, 96),
            mode="bicubic",
            align_corners=False
        )

    output = (
        y[0].permute(1, 2, 0)
        .cpu().numpy()
        .astype(np.float32)
    )

    assert output.shape == (96, 96, 4)
    assert np.isfinite(output).all()

    return output

def msi_hwc(msi):

    if msi.shape != (4, 96, 96):
        raise ValueError(
            f"Unexpected MSI shape: {msi.shape}"
        )

    result = np.ascontiguousarray(
        np.moveaxis(msi, 0, -1),
        dtype=np.float32
    )

    assert np.isfinite(result).all()

    return result


# =============================================================================
# G — ORIGINAL REGISTRATION WRAPPERS
# =============================================================================

def original_ncc(target_features, observed):

    out = ncc_env["estimate_both"](
        target_features,
        observed
    )

    dy = float(out["refined_dy"])
    dx = float(out["refined_dx"])

    snapped = bool(
        np.hypot(dy, dx) < GATE_NCC
    )

    return {
        "raw_dy": dy,
        "raw_dx": dx,
        "estimated_dy": 0.0 if snapped else dy,
        "estimated_dx": 0.0 if snapped else dx,
        "snapped": snapped
    }

def original_phase(target_features, observed):

    obs_features = phase_env["feature_map"](
        observed,
        "phase_gradient"
    )

    surface = phase_env["phase_surface"](
        target_features,
        obs_features
    )

    out = phase_env["select_bounded"](
        surface,
        RADIUS_PHASE
    )

    raw_dy = float(out["raw_dy"])
    raw_dx = float(out["raw_dx"])

    corrected_dy = raw_dy - BIAS_PHASE_DY
    corrected_dx = raw_dx - BIAS_PHASE_DX

    snapped = bool(
        np.hypot(
            corrected_dy,
            corrected_dx
        ) < GATE_PHASE
    )

    return {
        "raw_dy": raw_dy,
        "raw_dx": raw_dx,
        "estimated_dy": (
            0.0 if snapped else corrected_dy
        ),
        "estimated_dx": (
            0.0 if snapped else corrected_dx
        ),
        "snapped": snapped
    }

def record_result(
    patch_id,
    condition_index,
    method,
    true_dy,
    true_dx,
    estimate
):

    true_dy = float(true_dy)
    true_dx = float(true_dx)

    dy = float(
        estimate["estimated_dy"]
    )

    dx = float(
        estimate["estimated_dx"]
    )

    epe = float(
        np.hypot(
            dy - true_dy,
            dx - true_dx
        )
    )

    return {
        "patch_id": patch_id,
        "condition_index": int(condition_index),
        "method": method,
        "true_dy": true_dy,
        "true_dx": true_dx,
        "shift_magnitude": float(
            np.hypot(true_dy, true_dx)
        ),
        "raw_dy": float(estimate["raw_dy"]),
        "raw_dx": float(estimate["raw_dx"]),
        "estimated_dy": dy,
        "estimated_dx": dx,
        "snapped": bool(estimate["snapped"]),
        "error_dy": dy - true_dy,
        "error_dx": dx - true_dx,
        "epe": epe,
        "within_025": bool(epe <= 0.25),
        "within_050": bool(epe <= 0.50),
        "aligned_preserved": bool(
            true_dy == 0
            and true_dx == 0
            and dy == 0
            and dx == 0
        )
    }


# =============================================================================
# H — REGISTRATION EVALUATION
# =============================================================================

print("\n41C-F — POTSDAM REGISTRATION EVALUATION", flush=True)

rows = []

for patch_number, patch_id in enumerate(
    protocol["selected_patch_ids"], 1
):

    with np.load(
        patch_files[patch_id],
        allow_pickle=False
    ) as z:

        lrhsi = z["lrhsi"].copy()
        hrmsi = z["hrmsi"].copy()
        shifted = z["shifted_msi"].copy()
        offsets = z["shift_offsets_dy_dx"].copy()

    if not (
        np.isfinite(lrhsi).all()
        and np.isfinite(hrmsi).all()
        and np.isfinite(shifted).all()
    ):
        raise RuntimeError(
            f"Nonfinite input data: {patch_id}"
        )

    aligned_indices = np.flatnonzero(
        np.all(offsets == 0, axis=1)
    )

    if len(aligned_indices) != 1:
        raise RuntimeError(
            f"Expected one aligned condition: {patch_id}"
        )

    if not np.array_equal(
        shifted[int(aligned_indices[0])],
        hrmsi
    ):
        raise RuntimeError(
            f"Aligned MSI is inconsistent: {patch_id}"
        )

    reference = registration_reference(lrhsi)

    ncc_reference_features = (
        ncc_env["registration_features"](
            reference
        )
    )

    phase_reference_features = (
        phase_env["feature_map"](
            reference,
            "phase_gradient"
        )
    )

    for condition_index, (dy, dx) in enumerate(offsets):

        observed = msi_hwc(
            shifted[condition_index]
        )

        ncc_out = original_ncc(
            ncc_reference_features,
            observed
        )

        phase_out = original_phase(
            phase_reference_features,
            observed
        )

        identity_out = {
            "raw_dy": 0.0,
            "raw_dx": 0.0,
            "estimated_dy": 0.0,
            "estimated_dx": 0.0,
            "snapped": False
        }

        for method, estimate in [
            ("frozen_ncc_step13", ncc_out),
            (
                "frozen_phase_step20_size_adapted",
                phase_out
            ),
            ("identity", identity_out)
        ]:

            rows.append(
                record_result(
                    patch_id,
                    condition_index,
                    method,
                    dy,
                    dx,
                    estimate
                )
            )

    print(
        f"  {patch_number}/5 {patch_id}: "
        "17 conditions × 3 methods complete",
        flush=True
    )

results = pd.DataFrame(rows)

assert len(results) == 255
assert results["patch_id"].nunique() == 5
assert (results.groupby("method").size() == 85).all()

assert not results.duplicated(
    ["patch_id", "condition_index", "method"]
).any()

assert np.isfinite(
    results[["estimated_dy", "estimated_dx", "epe"]]
    .to_numpy(float)
).all()

results.to_csv(
    OUT / "Step41C_Registration_Results.csv",
    index=False
)


# =============================================================================
# I — PERFORMANCE SUMMARY
# =============================================================================

summary_rows = []

for method, group in results.groupby("method"):

    aligned = group[
        group["shift_magnitude"] == 0
    ]

    summary_rows.append({
        "method": method,
        "cases": len(group),
        "mean_epe": float(group["epe"].mean()),
        "median_epe": float(group["epe"].median()),
        "max_epe": float(group["epe"].max()),
        "within_025": float(group["within_025"].mean()),
        "within_050": float(group["within_050"].mean()),
        "aligned_preserved": int(
            aligned["aligned_preserved"].sum()
        ),
        "aligned_mean_epe": float(
            aligned["epe"].mean()
        ),
        "mean_error_dy": float(
            group["error_dy"].mean()
        ),
        "mean_error_dx": float(
            group["error_dx"].mean()
        )
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "Step41C_Method_Summary.csv",
    index=False
)

shift_summary = (
    results.groupby(
        ["method", "shift_magnitude"],
        as_index=False
    )
    .agg(
        cases=("epe", "size"),
        mean_epe=("epe", "mean"),
        median_epe=("epe", "median"),
        max_epe=("epe", "max"),
        within_025=("within_025", "mean")
    )
)

shift_summary.to_csv(
    OUT / "Step41C_Shift_Summary.csv",
    index=False
)

patch_summary = (
    results.groupby(
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
    OUT / "Step41C_Patch_Summary.csv",
    index=False
)

ncc = results[
    results["method"] == "frozen_ncc_step13"
][["patch_id", "condition_index", "epe"]].rename(
    columns={"epe": "ncc_epe"}
)

phase = results[
    results["method"] == "frozen_phase_step20_size_adapted"
][["patch_id", "condition_index", "epe"]].rename(
    columns={"epe": "phase_epe"}
)

paired = ncc.merge(
    phase,
    on=["patch_id", "condition_index"],
    validate="one_to_one"
)

paired["phase_advantage_px"] = (
    paired["ncc_epe"] - paired["phase_epe"]
)

paired["better_method"] = np.select(
    [
        paired["ncc_epe"] < paired["phase_epe"],
        paired["phase_epe"] < paired["ncc_epe"]
    ],
    ["ncc", "phase"],
    default="tie"
)

paired.to_csv(
    OUT / "Step41C_Paired_Comparison.csv",
    index=False
)

print("\n" + "=" * 92)
print("41C-G — EXTERNAL REGISTRATION RESULTS")
print("=" * 92)

print(
    summary.round(6).to_string(index=False)
)

print("\nPaired method comparison:")
print(
    paired["better_method"]
    .value_counts()
    .to_string()
)

print(
    "\nMean phase advantage:",
    float(paired["phase_advantage_px"].mean()),
    "pixels"
)


# =============================================================================
# J — PUBLICATION-QUALITY FIGURE
# =============================================================================

styles = {
    "frozen_ncc_step13": (
        "Frozen NCC + gate", "#225F8D", "o"
    ),
    "frozen_phase_step20_size_adapted": (
        "Frozen bounded phase", "#D0782C", "s"
    ),
    "identity": (
        "No correction", "#6B6B6B", "^"
    )
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8.2, 5.1))

for method, (label, color, marker) in styles.items():

    g = shift_summary[
        shift_summary["method"] == method
    ].sort_values("shift_magnitude")

    ax.plot(
        g["shift_magnitude"],
        g["mean_epe"],
        color=color,
        marker=marker,
        linewidth=1.9,
        markersize=6,
        label=label
    )

ax.set_xlabel(
    "Applied displacement magnitude (pixels)"
)

ax.set_ylabel(
    "Mean endpoint error (pixels)"
)

ax.set_xticks(
    sorted(
        shift_summary["shift_magnitude"].unique()
    )
)

ax.grid(
    axis="y",
    alpha=0.2,
    linewidth=0.7
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=9
)

fig.tight_layout()

fig.savefig(
    OUT / "Step41C_Registration_Comparison.png",
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT / "Step41C_Registration_Comparison.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# =============================================================================
# K — PROVENANCE AND SCIENTIFIC STATUS
# =============================================================================

provenance = {
    "experiment": (
        "Frozen MDAS registration on Potsdam "
        "with documented spatial and spectral interfaces"
    ),
    "historical_notebook_sha256": file_sha(NOTEBOOK),
    "historical_source_function_count": 11,
    "historical_ncc_reproduction": "680/680 at 1e-5 pixels",
    "historical_phase_reproduction": "680/680 at 3e-5 pixels",
    "phase_72x72_max_surface_difference": (
        max_surface_difference
    ),
    "phase_96x96_known_translation": {
        "actual": [2.0, -3.0],
        "estimated": [
            float(toy_peak["raw_dy"]),
            float(toy_peak["raw_dx"])
        ]
    },
    "frozen_ncc_settings": {
        "radius": RADIUS_NCC,
        "bias_dy": BIAS_NCC_DY,
        "bias_dx": BIAS_NCC_DX,
        "intensity_weight": 0.7,
        "gradient_weight": 0.3,
        "crop_margin": 8,
        "blur_sigma": 1.2,
        "identity_gate": GATE_NCC
    },
    "frozen_phase_settings": {
        "radius": RADIUS_PHASE,
        "bias_dy": BIAS_PHASE_DY,
        "bias_dx": BIAS_PHASE_DX,
        "identity_gate": GATE_PHASE
    },
    "spectral_interface": (
        "Direct frozen Step30 spectral response weights "
        "from original band IDs to four synthetic MSI channels"
    ),
    "maximum_wavelength_metadata_difference_nm": (
        max_wavelength_difference
    ),
    "accepted_wavelength_metadata_tolerance_nm": (
        WAVELENGTH_MAX_TOLERANCE_NM
    ),
    "original_band_id_order_verified": True,
    "response_weights_modified": False,
    "registration_parameters_modified": False,
    "external_ground_truth_used_for_tuning": False,
    "source_scene_count": 1,
    "source_patch_count": 5,
    "conditions_per_patch": 17,
    "step30_protocol_sha256": file_sha(FILES["protocol"]),
    "step30_response_sha256": file_sha(FILES["response"]),
    "step40_report_sha256": file_sha(FILES["reproduction"]),
    "patch_npz_sha256": patch_hashes,
}

save_json(
    OUT / "Step41C_Provenance.json",
    provenance
)

report = {
    "status": "EXTERNAL_FROZEN_REGISTRATION_EVALUATION_COMPLETE",
    "source_scenes": 1,
    "source_patches": 5,
    "shift_conditions_per_patch": 17,
    "total_records": len(results),
    "method_summary": summary.to_dict("records"),
    "paired_comparison": {
        "ncc_better": int(
            (paired["better_method"] == "ncc").sum()
        ),
        "phase_better": int(
            (paired["better_method"] == "phase").sum()
        ),
        "ties": int(
            (paired["better_method"] == "tie").sum()
        ),
        "mean_phase_advantage_px": float(
            paired["phase_advantage_px"].mean()
        )
    },
    "max_wavelength_metadata_difference_nm": (
        max_wavelength_difference
    ),
    "registration_parameters_retained": True,
    "external_spectral_interface_adapted": True,
    "phase_spatial_dimensions_adapted": True,
    "external_fusion_checkpoint_tested": False,
    "limitations": [
        "Only one independent Potsdam acquisition.",
        "Five patches are not independent source scenes.",
        "MSI and translations are synthetic.",
        "Original MDAS spectral ridge projection cannot "
        "directly accept the 218-band EnMAP input.",
        "The frozen Step30 response defines the external "
        "four-channel sensor interface.",
        "The original MDAS validation-derived biases "
        "are deliberately retained.",
        "The 72x72 phase surface requires a documented "
        "96x96 dimensional generalization.",
        "No external fusion model performance is evaluated."
    ],
    "elapsed_seconds": round(
        time.monotonic() - START, 2
    ),
    "output_directory": str(OUT)
}

save_json(
    OUT / "Step41C_Final_Report.json",
    report
)

print("\n" + "=" * 92)
print("STEP 41C — FINAL REPORT")
print("=" * 92)
print(json.dumps(report, indent=2))

print("\nSTEP 41C COMPLETE")
print("Historical algorithms and all previous outputs preserved.")
