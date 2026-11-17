# Release notes — v1.0.0

## Hyperspectral–multispectral fusion under misregistration

Initial versioned archival release of the research code associated with
*Hyperspectral–Multispectral Fusion Under Misregistration: Augmentation
Sensitivity and Cross-Scene Registration Calibration*.

### Included
- Two output-stripped research notebooks for the historical MDAS analysis
  and subsequent registration-replay/Potsdam validation stages
- Python execution-cell scripts, provenance inspection and diagnostics
- Frozen Potsdam registration-transfer code (Step 41C)
- Static source-contract and packaging tests
- Documentation of required input files, source verification and scientific limitations
- Authors and citation metadata, MIT license and third-party attribution

### Scope and reproducibility
- Augsburg SCSRNet fusion evaluation considered 0%, 20% and 50% shift-training
  regimes, with one recorded training run per regime and 40 held-out test patches.
- Historical NCC and bounded phase-correlation agreement was assessed against
  680 archived MDAS cases per estimator; replay agreement is not external
  registration accuracy.
- The Potsdam transfer pilot involved one EnMAP acquisition, five windows
  and 85 synthetic translations per method. Its MSI channels were simulated.
- Post hoc removal of the historical registration bias is an exploratory
  diagnostic, not independent validation.
- No Potsdam fusion-network inference was performed.

The archive does **not** provide the complete upstream SCSRNet implementation,
raw MDAS/EnMAP imagery, trained checkpoints, or all historical intermediate
assets. Scripts are preserved research execution cells, not a tested
end-to-end application. The numerical experiments have not been independently
rerun from this release.

### Licensing and attribution
Original code in this repository is offered under its MIT LICENSE, subject
to the rights of the contributors. Upstream SCSRNet is MIT-licensed; its
original copyright and license are preserved in THIRD_PARTY_NOTICES.md.
Third-party data, dependencies and weights remain subject to their separate
terms.

### Before public release and Zenodo archiving
Obtain coauthor approval, check third-party content in all notebooks, and
confirm redistribution permissions. The repository must be public for a
standard GitHub–Zenodo integration workflow. Do not insert a Zenodo DOI into
the manuscript or citation metadata until Zenodo has actually minted it.

### Reference
- Upstream SCSRNet: https://github.com/zsspo/SCSRNet
- Source implementation publication: https://doi.org/10.1016/j.jag.2026.105547
