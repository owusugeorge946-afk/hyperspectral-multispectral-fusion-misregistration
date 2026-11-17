# Required existing project assets

The original notebook code uses Google Colab with Drive mounted at
`/content/drive`, and a project root under `MyDrive/MDAS_Registration_Aware_Revision_2026`.
These paths are historical execution paths, not portable defaults.

Step 41C requires artifacts produced by earlier steps:

- `Step13_Identity_Preserving_Gate/Step13_Frozen_Gate.json`
- `Step20_Bounded_Phase_Correlation/Step20_Frozen_Method.json`
- `Step30_Frozen_Potsdam_Benchmark/Step30_Frozen_Forward_Model.json`
- `Step30_Frozen_Potsdam_Benchmark/Step30_Spectral_Response_Matrix.csv`
- `Step30_Frozen_Potsdam_Benchmark/Step30_Paired_Observation_Manifest.csv`
- `Step30_Frozen_Potsdam_Benchmark/Paired_Observations/` (NPZ data)
- `Step39_Original_Estimator_Numerical_Replay/Step39_Function_Provenance.csv`
- `Step40_Reproduction_Discrepancy_Audit/Step40_Final_Report.json`
- The historical notebook whose source functions were SHA-256 checked.

The scripts deliberately refuse to overwrite existing Step 41C outputs.
For a fresh reproduction, preserve source data and calibrated settings,
then execute in an isolated Drive project copy or new output folder.

Access and licensing for MDAS, EnMAP and the upstream SCSRNet code
must be checked separately before redistribution. This package includes
only the two notebooks supplied for packaging and extracted execution
cells, not third-party repository source files or image data.
