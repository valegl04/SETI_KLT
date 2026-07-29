# C-KLT Phase-Transition Sweep

A **simulated, single-receiver** study of how the Covariance-KLT (C-KLT) detection pipeline
behaves as a function of the injected signal SNR, and how that empirical behaviour compares to
the **Baik–Ben Arous–Péché (BBP) phase transition** predicted by Random Matrix Theory.

For each run the pipeline generates a GUPPI raw voltage file with `setigen` (one synthetic
antenna), injects a drifting CW tone at a controlled SNR, reconstructs the signal with the
C-KLT, and uses the **Kritchman & Nadler (2009)** rank estimator to decide how many eigenvalues
are signal — which then sets how many eigenvectors are kept for the reconstruction. Results are
appended to a CSV so a long, expensive sweep is crash-safe and resumable.

---

## What it measures

The C-KLT chops a 1-D voltage stream into `K` windows of `W` samples and diagonalises the
`W×W` sample covariance. The RMT roles map onto the MeerKAT beamformed-stamp dimensions used
elsewhere in the project:

| C-KLT | MR-KLT / MeerKAT stamp | role |
|---|---|---|
| `W` (window size) | `N_A` antennas | covariance dimension |
| `K` (# windows) | timesteps | # realisations |
| `γ = W/(K-1)` | `γ = N_A/(K-1)` | aspect ratio |

The defaults `W = 63`, `K = 293` give `γ ≈ 0.216`, matching the real MeerKAT stamps. The
**phase-transition threshold** is `√γ ≈ 0.46` (in units of σ²): a population spike
`ℓ = λ_s/σ²` only escapes the Marchenko–Pastur bulk — and becomes detectable — when `ℓ > √γ`.

Detection and reconstruction are driven **entirely by Kritchman & Nadler** — there is no
Bayesian / Tracy-Widom detector in the loop. `ℓ̂`, `supercritical`, and `√γ` are kept only as
the theoretical phase-transition reference to compare K&N against.

---

## Folder layout

| File | Role |
|---|---|
| `CKLT_PhaseTransition.ipynb` | Configure → single-run demo → sweep → analysis, plus an "injected vs. effective SNR" section. |
| `helpers.py` | Data generation (`setigen`), the C-KLT + K&N analysis, and the crash-safe sweep. |
| `README.md` | This file. |
| `outputs/` | Generated raw files and the results CSV (git-ignored). |

The *analysis* is the real `seti_klt` library (`KLT.apply_cklt`, `KN_rankEst`,
`TracyWidomTW2.invert_bbp_for_l_pop`); `helpers.py` only wraps generation and orchestration.

---

## `helpers.py` at a glance

* `CKLTConfig` — all fixed instrument / processing parameters (sample rate, PFB, window `W`,
  `K`, target channel, GPU toggle, K&N level, …). Reproduces the MeerKAT covariance geometry.
* `generate_cklt_raw` — write one GUPPI raw file with a drifting CW tone injected at a target
  per-sample SNR, frequency offset, and drift rate.
* `analyze_cklt` — run the C-KLT, compute RMT / phase-transition statistics, run K&N to count
  signal eigenvalues, reconstruct with that many eigenvectors, and measure recovery quality.
* `run_sweep` — loop over decreasing SNR × random (frequency, drift), appending each result to
  a CSV immediately (crash-safe).
* `append_result_row`, `load_results`, `RESULT_FIELDS` — the results table I/O and its columns.
* `spectral_snr` — narrowband spectral SNR of a 1-D voltage vector (CuPy-optional FFT).

### Recorded columns (`RESULT_FIELDS`)

Per run the CSV stores the injection (SNR, frequency offset, drift, seed), the geometry
(`W`, `K`, `γ`, `√γ`), the eigen-spectrum / RMT quantities (`sigma2_hat`, `lambda1`,
`lambda_max_MP`, `l_pop_hat`, `snr_effective`, `supercritical`), the Kritchman & Nadler outputs
(`n_signals_KN`, `sigma_hat_KN`, `n_eig_recon`), reconstruction quality (`input_snr_db`,
`recon_snr_db`, `recon_gain_db`, `freq_err_hz`, `freq_recovered`), and the decisions
(`signal_found`, `signal_confirmed`, `kn_agrees_bbp`).

---

## Notebook overview (`CKLT_PhaseTransition.ipynb`)

1. **Configuration** — `CKLTConfig`; CPU by default, GPU (CuPy `setigen` backend) opt-in.
2. **Single-run demonstration** — one strong injection with eigenspectrum + PSD diagnostics.
3. **The sweep** — decreasing SNR × random frequency/drift, results appended to CSV.
4. **Analysis** — empirical detectability vs. the BBP prediction: detection-rate curves and the
   estimated spike `ℓ̂` against `√γ`, plus reconstruction gain and K&N-vs-BBP agreement.
5. **Injected vs. effective SNR** — measures how the PFB (and quantisation) transform the SNR:
   the injected per-sample SNR is *not* the SNR in the recorded channel — the PFB adds a nearly
   constant channelisation gain (~+29 dB for the default geometry) before recording.

---

## Running it

Run the cells top to bottom. The default sweep straddles the calibrated transition
(~-44 to -50 dB per-sample injected SNR). Notes:

* the CSV is appended row-by-row, so an interrupted sweep can be resumed (re-running appends);
* GPU generation works but pays a one-off CuPy kernel-compilation cost on the first call in a
  fresh kernel; the CPU path is a reliable ~3 s/run with no warmup;
* raising `W` (at fixed `K`) increases `γ` and pushes `√γ` up, making detection harder — a
  second axis for the phase-transition study.

---

## What the results show

* A sharp **phase transition** in K&N detection around **-44 to -50 dB** injected per-sample
  SNR; the empirical 50%-detection point tracks the theoretical BBP crossing (`ℓ̂ → √γ`) to
  within ~1 dB.
* The naive "`λ₁ > MP edge`" test **over-calls** at low SNR (noise fluctuations push `λ₁` over
  the asymptotic edge); K&N is the robust decision.
* The **injected SNR is not the data SNR**: the PFB concentrates the tone into one channel while
  spreading the noise, adding a near-constant gain before recording — quantified in §5.

## Related material

* `seti_klt` — `KLT` (C-KLT / RMT statistics), `KN_rankEst`, `TracyWidomTW2`.
* `../Injected_Signal/` — the single-injection C-KLT demonstration this study is built on.
