# MR-KLT on Real MeerKAT Data

This directory applies the **Multiple-Realisation Karhunen-Loève Transform (MR-KLT)** to
real interferometric data from the **MeerKAT** array, as recorded by the Breakthrough Listen
BLUSE backend. Unlike the single-dish GUPPI studies elsewhere in this repository, MeerKAT
provides many antennas observing simultaneously, which is the *native* setting for the
MR-KLT: the covariance is built across antennas (spatial), and the dominant eigenvector is a
direction of arrival.

> **Status: work in progress.** The notebook is an exploratory pipeline. It requires the
> `seticore` package and MeerKAT `.stamps` data placed under `data/stamps_data/`
> (git-ignored, not shipped with the repository).

---

## Background: from stamps to a covariance matrix

`seticore` reads Breakthrough Listen **`.stamps`** files, each holding one or more *stamps*
— small time–frequency cutouts around a candidate signal. A stamp's complex voltage array
has shape:

```
(T, F, P, A) = (timesteps, frequency channels, polarisations, antennas)
```

To run MR-KLT on a single frequency channel and polarisation, we take the antenna vector at
each timestep:

```python
x = data[:, freq_ch, pol, :]   # shape (T, A) = (K, N_A)
```

which is exactly the `(K, N_A)` matrix expected by `KLT.apply_mrklt`. Here `K = T`
realisations are **physical** (timesteps) and `N_A = A` is the number of antennas, so the
RMT aspect ratio is `γ = N_A / (K − 1)`.

Because MeerKAT stamps are already channelised (each timestep is a coherent average of many
FFTs), the `K` rows are effectively i.i.d. Gaussian, and the RMT / Tracy-Widom machinery
used elsewhere in the package applies unchanged.

---

## Notebook Overview

### `MRKLT_Meerkat.ipynb`
1. **Setup:** logger, output directories, analysis parameters (`POL`, `KN_ALPHA`).
2. **Loading:** reads a `.stamps` file via `seticore.viewer` and reports the per-stamp
   shape and RMT aspect ratio `γ`.
3. **Channel diagnostics:** computes the average power per frequency channel to locate the
   channel carrying the strongest signal.
4. **MR-KLT:** applies `apply_mrklt` on the dominant channel and inspects the leading
   spatial eigenvector `|v₁|` across antennas.
5. **Per-channel sweep:** a single call to `seti_klt.scan_channels` runs MR-KLT plus the
   Kritchman-Nadler rank estimator (`KN_rankEst`) on every channel, and
   `ChannelScanResult.eigenvector_coherence()` measures how strongly each channel's
   dominant eigenvector aligns with the reference channel's — which is how one physical
   emitter spread across adjacent channels reveals itself.
6. **Visualisation:** original vs. reconstructed waterfalls, integrated PSD comparison, and
   the MR-KLT eigenspectrum for the reference channel.

### Outputs

The notebook follows the project's standard I/O convention (`seti_klt.utils.io`): figures
go to `outputs/figures/` via `save_figure`, and the run log to
`outputs/logs/mrklt_meerkat.log` via `SimpleLogger`. Everything under `outputs/` is
git-ignored.

---

## Related material

The per-channel sweep is [`seti_klt.scan_channels`](../../src/seti_klt/multichannel.py),
shared with the simulated multi-antenna pipeline in
[`../MAA_Simulated/`](../MAA_Simulated/) — the same analysis code runs on synthetic data
with known ground truth and on this real data.

The theory behind MR-KLT, the spiked-covariance model, and the Kritchman-Nadler estimator
is documented in [`../Kritchman_Nadler/README.md`](../Kritchman_Nadler/README.md). The
`apply_mrklt` method itself is documented in `src/seti_klt/KLT.py`.
