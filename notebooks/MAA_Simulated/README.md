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

## Folder layout

| File | Role |
|---|---|
| `multiantenna_klt_pipeline.ipynb` | Orchestration only — configure, generate, analyse, plot. |
| `helpers.py` | Scenario **generation and loading** (`setigen` + `katpoint`). |
| `seti_klt.scan_channels` | The **analysis** — per-channel MR-KLT sweep, shared with `../Meerkat_Real_Data/`. |

The split is deliberate. `helpers.py` fabricates *test data*, so it stays local to this
study — exactly as `Kritchman_Nadler/helpers.py` keeps `make_spiked_data` local. The
per-channel MR-KLT sweep, by contrast, is genuine library code used by both the simulated
and the real-data pipeline, so it lives in the `seti_klt` package as
[`multichannel.py`](../../src/seti_klt/multichannel.py). The notebook itself is left thin
and readable.

### Outputs

The notebook follows the project's standard I/O convention (`seti_klt.utils.io`):

* generated scenarios (`.raw` + `.meta.json`) → `outputs/raw_data/`
* figures → `outputs/figures/` (via `save_figure`)
* run log → `outputs/logs/maa_pipeline.log` (via `SimpleLogger`)

Everything under `outputs/` is git-ignored.

---

## Notebook Overview

### `multiantenna_klt_pipeline.ipynb`
The notebook is organised into numbered sections:

1. **Setup & configuration** — logger, output directories, analysis parameters.
   GPU acceleration for `setigen` is opt-in via `ENABLE_GPU` (needs the `[gpu]` extra).
2. **Define & generate scenarios** — declare the signals to inject (kind, channel, drift
   rate, level, and for `per_antenna` which antennas), then call
   `helpers.generate_scenario`.
3. **Load a scenario** — `helpers.load_scenario` returns a `(K, F, N_A)` cube plus its
   ground-truth metadata.
4. **Per-channel MR-KLT sweep** — `scan_channels` runs the MR-KLT, the Kritchman-Nadler
   rank estimate, and the reconstruction for every channel (DC excluded), then the result
   is **scored against the ground truth** (detected / missed / false positives).
5. **Waterfall & PSD** — original vs. reconstructed time–frequency views.
6. **Eigenvalue heatmap & rank per channel** — the full eigenspectrum across channels as a
   heatmap (normalised by each channel's noise floor), with `K_est` per channel and the
   injected ground-truth channels marked.
7. **Single-channel eigenspectrum** — a detailed look at the channel with the strongest
   dominant eigenvalue.

---

## Related material

The MR-KLT theory and the RMT / Kritchman-Nadler background are documented in
[`../Kritchman_Nadler/README.md`](../Kritchman_Nadler/README.md); the `setigen` `NPOL`
header patch used when generating GUPPI files is described in the top-level
[`README.md`](../../README.md).
