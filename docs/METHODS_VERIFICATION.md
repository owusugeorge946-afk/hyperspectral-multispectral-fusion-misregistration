# Implementation verification notes

## SCSRNet fusion input preparation
The supplied `SCSRNet.py` source inspected during this research uses
`F.interpolate(lr_hsi, scale_factor=self.factor, mode='bicubic')`.
MDAS inputs are 24 x 24 x 242 LR-HSI and 72 x 72 x 4 HR-MSI; the
network predicts 72 x 72 x 242 HR-HSI. The architecture uses two
small-kernel and two large-kernel residual blocks in its default configuration.

## Step 41C Potsdam registration reference
The source bundled in `scripts/step_41c_frozen_potsdam_transfer.py`
contains a `registration_reference(lrhsi)` function. It uses the frozen
4 x 218 spectral-response matrix to project LR-HSI (218,32,32) into
a four-band 32 x 32 observation and calls:

```python
y = F.interpolate(x, size=(96, 96), mode="bicubic", align_corners=False)
```

The returned reference has shape (96,96,4). This is independently
verifiable by inspecting the submitted execution-cell source.

## Provenance and external scope
The Step 41C report provided by the authors records one Potsdam EnMAP
acquisition, five nonoverlapping patches and 17 synthetic translation
conditions per patch. NCC and bounded phase registration were assessed
on 85 conditions each; the no-correction baseline has another 85.
Mean endpoint errors reported in that execution: NCC 1.429562777 px,
phase 1.414961584 px, no correction 1.793251846 px. The frozen
registration methods preserved none of the five truly aligned patches.

The transferred spectral interface is based on the previously frozen
Potsdam response weights; it is not a fully unchanged Augsburg sensor
observation model. Phase-correlation handling was adapted for 96 x 96
arrays. Original 242-band fusion checkpoints were **not** tested on
Potsdam. The post hoc bias-omission diagnostics are **not independently
validated** and must not be confused with frozen-transfer results.

## Limits of verification
The uploaded execution source and reported results were checked for
consistency; this repository does not itself ship the original TIFF
assets, trained checkpoints, frozen calibration files or processed NPZ
patches. The numerical experiments have **not been rerun** by this
GitHub packaging step.
