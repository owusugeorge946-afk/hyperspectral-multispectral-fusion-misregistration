# Hyperspectral–Multispectral Fusion Under Misregistration

**Authors (as listed on the supplied manuscript title page):** George Owusu Amoah, Rubayet Bin Mostafiz, Rachel Olawoyin, and Francis Quayson.  
**Correspondence:** George Owusu Amoah (george.amoah003@stu.ucc.edu.gh).  
**Manuscript:** *Hyperspectral–Multispectral Fusion Under Misregistration: Augmentation Sensitivity and Cross-Scene Registration Calibration*.  
**Target journal:** *Remote Sensing of Environment* (submission preparation; not a claim of publication or acceptance).

Affiliations from the manuscript title page are transcribed in [`AUTHORS.md`](AUTHORS.md). The author list, order and individual CRediT contributions require coauthor confirmation before public release.

Research-code archive accompanying the manuscript **“Hyperspectral–Multispectral Fusion Under Misregistration: Augmentation Sensitivity and Cross-Scene Registration Calibration.”**

## Cite this code

This repository contains the research code associated with the manuscript above. Use the repository's GitHub release URL for a software citation. A `CITATION.cff` file supplies the author metadata for GitHub's **Cite this repository** action. No paper DOI, Zenodo DOI or publication year is claimed here; these can be added once assigned. GitHub's default generated citation should not be treated as a verified journal citation.

## Contents

| Path | Description |
| --- | --- |
| `notebooks/01_mdas_historical_analysis.ipynb` | Original multi-step MDAS analysis source, with bulky notebook outputs removed |
| `notebooks/02_mdas_final_validation.ipynb` | Subsequent validation notebook (Steps 40–46), with bulky notebook outputs removed |
| `scripts/step_40_provenance_replay.py` | Provenance and historical estimator source audit |
| `scripts/step_41c_frozen_potsdam_transfer.py` | **Actual Step 41C transfer execution cell**, including bicubic registration-reference construction |
| `scripts/step_42_calibration_diagnostics.py` | Exploratory, post hoc diagnostics |
| `scripts/step_43_validation_report.py` | Follow-up validation and reporting |
| `scripts/step_44c_spatial_figures.py`, `scripts/step_45_figure_refinement.py`, `scripts/step_46_manuscript_figures.py` | Figure generation and refinement cells |
| `tests/test_interpolation_source.py` | Static source check of the registration-reference interpolation |
| `docs/DATA_REQUIREMENTS.md` | Prerequisite files and restrictions |
| `docs/METHODS_VERIFICATION.md` | Verified implementation facts and limitations |

## Important scope distinction

1. **Augsburg fusion-sensitivity experiment:** SCSRNet trained under three shift-augmentation regimes (0%, 20%, 50%) and evaluated on 40 held-out MDAS patches under 0, 1, 2, and 4-pixel cardinal HR-MSI shifts. The manuscript's comparison is based on **one training run per regime**; it is not evidence of across-seed stability.
2. **MDAS historical registration replay:** NCC and phase correlation were reconstructed from archived source and compared against historical 680-case displacement records for each estimator. The agreement tolerances measure numerical reproducibility, not registration accuracy.
3. **Potsdam frozen external pilot:** one EnMAP acquisition, five patches and 17 synthetic translations per patch. Original MDAS registration biases and identity gates were retained. The Potsdam 4-channel MSI is synthetic, and the original 242-band fusion networks were **not** evaluated on Potsdam.
4. **Post hoc bias omission:** exploratory diagnostic performed after inspecting transfer errors; not independently validated.

## What this archive does and does not provide

- Contains selected execution-cell sources exported from two research notebooks. The `.py` files were **not refactored into standalone CLI applications**; several execute top-level code, mount Google Drive, and expect historical paths and files. Read scripts before running.
- Does not include raw MDAS or EnMAP imagery, full upstream SCSRNet source, trained checkpoints, model weights, historical calibration inputs, or large CSV/NPZ outputs.
- The **Augsburg 0%/20%/50% model-training workflow** is retained in the historical notebook rather than supplied as a standalone, tested training script.
- Source validation tests check only selected code contracts; they do not demonstrate full study reproducibility.
- The 85-case external summary and 680-case replay metrics are historical recorded results and were **not recalculated during release preparation**.

## Reproduction guide (Google Colab + Drive)

The code originates from research notebooks and is provided as an **archival research-code release** rather than a turnkey standalone library. Paths and intermediate names are kept unchanged for source traceability.

1. Obtain the MDAS/SCSRNet data and the original project artifacts using the appropriate data holders and licenses. See [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md).
2. Open `notebooks/01_mdas_historical_analysis.ipynb` to inspect the historical analysis steps and `notebooks/02_mdas_final_validation.ipynb` for the final replay, frozen transfer, and visualizations. Running all cells sequentially is **not** recommended: the notebooks contain historical exploratory and corrective steps.
3. Mount Google Drive in Colab and ensure the prerequisite Step 13, Step 20, Step 30, Step 39 and Step 40 files exist under the configured project root.
4. Execute the relevant validated step from the original Colab source after inspecting paths and output safety guards. Step 41C refuses to overwrite recorded outputs. It requires the exact historical source-notebook file and archived calibration assets.
5. Review generated CSV/JSON reports against recorded results and check spectral band ID/wavelength audit. The packaging process has not rerun the numerical analyses.

Install the Python dependencies in an environment compatible with your PyTorch and CUDA installation:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

This `unittest` command verifies **only the stored Step 41C source contract**. Passing it does **not** reproduce model training or the 85-case numerical evaluation.

## Verified interpolation

SCSRNet uses bicubic upsampling for 24→72 spatial resolution. The historical Step 41C transfer code also uses `torch.nn.functional.interpolate(..., size=(96,96), mode="bicubic", align_corners=False)` for the *low-resolution HSI-derived registration reference*. See [`docs/METHODS_VERIFICATION.md`](docs/METHODS_VERIFICATION.md).

## Release and archive status

This repository is an **archival source snapshot**, not an independently reproduced package. Before a public GitHub or Zenodo release, use [`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md). The reproducible checks in `tests/` cover syntax, preserved static code contracts, notebook output stripping, and metadata completeness; they do not rerun the scientific analyses.

No journal publication DOI, GitHub release URL, or Zenodo DOI is asserted. Do not cite a DOI until it exists. A future Zenodo archived release should use the **exact released commit and version** rather than treating the editable default branch as an immutable reference.

## Source attribution and availability

Source-code snapshots in this package were extracted from the two author-supplied notebooks. Third-party data, trained checkpoint files and upstream source repositories are **not** included. The original SCSRNet repository referenced by the historical notebook is [zsspo/SCSRNet](https://github.com/zsspo/SCSRNet). Its upstream code is available under the MIT License; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the original copyright notice and license text. Dataset and model-weight terms must be checked separately.

**Licensing:** Original repository code and documentation are released under the [MIT License](LICENSE), subject to the rights of their respective authors. SCSRNet-derived materials retain the original upstream MIT notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository's MIT license does not cover third-party imagery, model weights, or materials for which the contributors lack relicensing rights. Coauthor approval and any additional source-ownership checks remain necessary before public release.

**Research integrity:** The included schematic figures and code are not independent validation of results. This archive does not claim a successful independent fusion-model transfer to Potsdam. The post hoc bias-omission analysis must remain identified as exploratory.

**Pre-publication check:** Author name, corresponding email and affiliation details were taken from the supplied manuscript title page; they are not independently verified against author ORCID profiles. The permanent repository and Zenodo identifiers must only be added once actually assigned.
