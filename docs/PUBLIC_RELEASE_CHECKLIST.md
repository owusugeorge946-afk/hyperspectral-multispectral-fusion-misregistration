# Public release and Zenodo archiving checklist

This file documents items requiring a decision or evidence from the authors. It is not a declaration that the repository has been cleared for publication.

## Authorship and permissions

- Obtain approval from all four listed coauthors for public repository release, author names and order, affiliations and citation metadata.
- Obtain agreement on CRediT contributions for the journal manuscript separately. Contribution roles are not inferred from names or file ownership.
- Verify which code is original and which derives from upstream SCSRNet or other packages. Check source licenses before redistributing any material.
- A repository-level MIT LICENSE and upstream SCSRNet attribution have been added. Confirm all contributors' rights and coauthor approval before changing visibility; the license does not cover third-party data or materials.
- Do not publish raw MDAS/EnMAP imagery or checkpoints unless their redistribution terms explicitly permit this.

## Reproducibility

- Run `python -m unittest discover -s tests -v` (static tests only).
- Inspect historical absolute Drive paths; they document the original environment and may need local modification for execution.
- Supply the separately held Step13, Step20, Step30, Step39 and Step40 artifacts and data only where sharing is authorized.
- If promising a complete numerical reproduction, rerun the pipeline from a clean environment and save outputs with hashes. This archive has **not** passed that test.

## GitHub and Zenodo

- Review every file in the planned commit; do not add any `.env`, source imagery, weight files or private inputs.
- The private GitHub repository is created and the audited files have been committed. Verify the current `main` branch before any release.
- The proposed first release is `v1.0.0`; use [`RELEASE_NOTES_v1.0.0.md`](RELEASE_NOTES_v1.0.0.md) as the GitHub Release description. Only publish publicly after approvals and third-party checks.
- If Zenodo is enabled for the repository, archive that release and use the **actual** Zenodo DOI returned by Zenodo.
- Add that verified DOI to `CITATION.cff` and the manuscript's Code and Data Availability section only after deposition. Publish any subsequent DOI changes in a new release if necessary.
- Ensure the submitted paper, GitHub citation and Zenodo authorship metadata agree.

## Scientific limitations to preserve

- One trained checkpoint per augmentation regime; no across-seed uncertainty.
- One external Potsdam acquisition containing five windows; no general cross-scene inference.
- Synthetic four-channel MSI, not independently acquired MSI.
- No Potsdam testing of the 242-band fusion checkpoints.
- Bias-omission results are post hoc diagnostics, not validated improvements.
