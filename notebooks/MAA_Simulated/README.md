# Multi-Antenna Array (Simulated) KLT Pipeline

This directory contains an end-to-end, **simulated** multi-antenna pipeline: it generates
synthetic multi-antenna baseband data with `setigen`, reads it back, runs the KLT per
frequency channel, and checks the results against the known injected ground truth.

It is the controlled, synthetic counterpart to [`../Meerkat_Real_Data/`](../Meerkat_Real_Data/):
here every signal is planted deliberately, so the pipeline's output (which channels carry a
signal, how many, and where) can be validated against the truth rather than merely
inspected.

---

## What the pipeline simulates

Two kinds of injected signal are modelled, mirroring the physical distinction that matters
for a real array:

* **`coherent`** — a sky signal placed on the background stream. It receives the geometric
  inter-antenna delays and is therefore **coherent across all antennas** — exactly the
  structure the MR-KLT's dominant spatial eigenvector is designed to capture.
* **`per_antenna`** — local interference (RFI), injected on a subset of antennas **without**
  geometric delays. It is spatially incoherent and serves as a realistic confounder.

Each generated scenario is written alongside a **`.meta.json` sidecar** recording the
injected ground truth (channels, signal kinds, drift rates, levels), so the analysis stage
can be scored against it automatically.

---

## Notebook Overview

### `multiantenna_klt_pipeline.ipynb`
The notebook is organised into numbered sections:

1. **Generation** — helper machinery to synthesise multi-antenna GUPPI raw baseband data.
2. **Define scenarios** — declare the signals to inject (kind, channel, drift rate, level,
   and, for `per_antenna`, which antennas).
3. **Load a scenario** — read a generated scenario back together with its `.meta.json`.
4. **KLT analysis** — run the KLT per frequency channel (DC channel excluded), storing the
   full eigenspectrum of every channel, and estimate the per-channel rank with the
   Kritchman-Nadler estimator (`KN_rankEst`).
5. **Waterfall & PSD** — original vs. reconstructed time–frequency views.
6. **Eigenvalue heatmap & rank per channel** — the full eigenspectrum across channels as a
   heatmap, with `K_est` per channel and the injected ground-truth channels marked.
7. **Single-channel eigenspectrum** — a detailed look at the channel with the strongest
   dominant eigenvalue.

---

## Related material

The MR-KLT theory and the RMT / Kritchman-Nadler background are documented in
[`../Kritchman_Nadler/README.md`](../Kritchman_Nadler/README.md); the `setigen` `NPOL`
header patch used when generating GUPPI files is described in the top-level
[`README.md`](../../README.md).
