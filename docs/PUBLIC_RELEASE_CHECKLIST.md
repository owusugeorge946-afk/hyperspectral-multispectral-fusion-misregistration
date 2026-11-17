# Public release and Zenodo archiving checklist

This file documents items requiring a decision or evidence from the authors. It is not a declaration that the repository has been cleared for publication.

## Authorship and permissions

- Obtain approval from all four listed coauthors for public repository release, author names and order, affiliations and citation metadata.
- Obtain agreement on CRediT contributions for the journal manuscript separately. Contribution roles are not inferred from names or file ownership.
- Verify which code is original and which derives from upstream SCSRNet or other packages. Check source licenses before redistributing any material.
- Select a software license **only after** confirming rights and coauthor approval. Until then, the archive does not grant an open-source license.
- Do not publish raw MDAS/EnMAP imagery or checkpoints unless their redistribution terms explicitly permit this.

## Reproducibility

- Run `python -m unittest discover -s tests -v` (static tests only).
- Inspect historical absolute Drive paths; they document the original environment and may need local modification for execution.
- Supply the separately held Step13, Step20, Step30, Step39 and Step40 artifacts and data only where sharing is authorized.
- If promising a complete numerical reproduction, rerun the pipeline from a clean environment and save outputs with hashes. This archive has **not** passed that test.

## GitHub and Zenodo

- Review every file in the planned commit; do not add any `.env`, source imagery, weight files or private inputs.
- Create a GitHub repository and commit the audited files. The code has not been uploaded by this package preparation.
- After an approved public release, tag an immutable version (e.g. `v1.0.0`) and make a GitHub Release.
- If Zenodo is enabled for the repository, archive that release and use the **actual** Zenodo DOI returned by Zenodo.
- Add that verified DOI to `CITATION.cff` and the manuscript's Code and Data Availability section only after deposition. Publish any subsequent DOI changes in a new release if necessary.
- Ensure the submitted paper, GitHub citation and Zenodo authorship metadata agree.

## Scientific limitations to preserve

- One trained checkpoint per augmentation regime; no across-seed uncertainty.
- One external Potsdam acquisition containing five windows; no general cross-scene inference.
- Synthetic four-channel MSI, not independently acquired MSI.
- No Potsdam testing of the 242-band fusion checkpoints.
- Bias-omission results are post hoc diagnostics, not validated improvements.
