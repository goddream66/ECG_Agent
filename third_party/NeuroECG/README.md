# NeuroECG preprocessing source snapshot

Source: https://github.com/goddream66/NeuroECG

Pinned commit: `95409f8df36737014a4be922f6464fe31f975fcb`

Retrieved on: 2026-11-13 (Asia/Shanghai).

Only `process_easy_500hz.py` was downloaded from upstream. It is preserved
byte-for-byte; this README and `source.json` are local provenance notes.
No datasets, pickle caches, pretrained weights, model code or training pipeline
were downloaded. No new dependencies were installed and no preprocessing job
was run. The CardioAgent runtime is unchanged.

## Included processing

- ECG channel selection, retaining ECG1 and ECG2 when both are present.
- Third-order Butterworth 0.5-40 Hz bandpass using `sosfiltfilt`.
- Polyphase resampling to 500 Hz.
- Per-channel median centering, 95th-percentile absolute scaling, clipping to
  [-5, 5], and float32 output shaped [channels, time].
- Per-channel quality masks. In this revision only low standard deviation
  determines good/bad; HR, RR variability and other metrics are diagnostics.
- Default QC windows: 5,000 samples (10 s); stride: 10,000 samples (20 s).
- Metadata, NPY signal output, NPZ masks, and index construction.
- Polarity normalization helpers exist but are disabled in the active pipeline.

Imports: Python standard library, numpy, scipy, wfdb, tqdm. The script does not
import any other NeuroECG module.

## Before connecting to AFDB or the streaming agent

The upstream entry point is specialized for I-CARE: hard-coded Linux input/output
paths, `_ECG.hea` filename matching, numeric patient-directory assumptions, and
I-CARE-specific time metadata. AFDB requires a separate adapter; executing the
unmodified main entry point is not an AFDB preprocessing command.

Filtering and robust normalization currently use the entire recording. This is
an offline implementation, not a causal streaming implementation. Any online
adapter must constrain processing to elapsed retained samples and record the
window used; whole-record normalization must not expose future samples.

Non-finite values are replaced with zero upstream. An adapter should retain an
independent missing-sample mask rather than interpret those values as measured
ECG. Default QC samples 10 s every 20 s, so its masks do not cover every instant.
The low-standard-deviation criterion is not a complete signal-quality validator.

This preprocessing recipe is from NeuroECG, not evidence that the released
ECGFounder 150-class head was validated with this exact recipe on AFDB.
Normalization does not restore AFDB's missing physical-amplitude calibration.
Keep the raw data, original channel names, sampling rate, and annotation times.

## License status

No LICENSE/COPYING file was present in the inspected upstream tree at this
commit. No permissive license is asserted by this project. Retain attribution
and resolve reuse/redistribution terms before publishing the copied source.
