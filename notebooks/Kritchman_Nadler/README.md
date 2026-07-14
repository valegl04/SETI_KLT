# Counting signals in noise: the Kritchman–Nadler estimator, deconstructed

This folder deconstructs the **signal-counting** step of the
[`SETI_KLT`](https://github.com/LuigiSotgiu/SETI_KLT) pipeline — the step that answers
*"how many distinct signals are hiding in this covariance matrix, and where is the boundary
below which we are simply blind?"*

The focus is the **Kritchman & Nadler (2009)** estimator — the classical
Random-Matrix-Theory (RMT) rank estimator — ported faithfully from the authors' MATLAB code
and taken apart piece by piece on synthetic data with known ground truth. It is presented as
the **classical baseline** against which this project's own detector (the exact Tracy–Widom₂
Bayesian detector implemented in `seti_klt`, see `KLT.py` and `tracy_widom.py`) is meant to
be compared.

> **Contents of this folder:**
> - `README.md` (this file) — shared theory and background.
> - `01_Kritchman_Nadler_Deconstructed.ipynb` — the estimator, deconstructed.
> - `helpers.py` — shared boilerplate (path setup, synthetic-data generator, plotting).
>
> **Reading order:** this `README.md` first, then the notebook. The shared theory below is
> not repeated inside the notebook; it assumes it.

---

## 1. The problem

A radio telescope hands us a stream of complex baseband samples. After the KLT step
(below) we hold a small **sample covariance matrix (SCM)** and its eigenvalues. Pure
thermal noise scatters its energy roughly evenly across all eigenvalue "slots"; a coherent
signal concentrates energy into one dominant eigenvalue — a *spike*. Counting signals is
therefore counting how many eigenvalues have genuinely escaped the noise floor rather than
fluctuated above it by chance.

This is the *spiked covariance model* of Johnstone (2001). Kritchman & Nadler is one
principled way of drawing the line between "spike" and "fluctuation"; the Bayesian detector
in `seti_klt` is another.

---

## 2. The shared front-end: C-KLT and MR-KLT

Both estimators consume **eigenvalues of an SCM**; how that SCM is built is common to both.

### 2.1 Why windowing / multiple realisations

Classical RMT signal counting assumes an array of `N_A` physical sensors, giving an
`N_A × N_A` spatial covariance averaged over time samples. A single-dish or beamformed
stream is one-dimensional, which breaks that construction. Two equivalent fixes restore an
SCM (Trudu et al.; Maccone's KLT-for-SETI programme):

* **C-KLT (Covariance KLT)** — cut the 1-D stream into `K` non-overlapping windows of `W`
  samples, forming a `(K, W)` matrix. The `W × W` covariance captures the *temporal*
  structure of the channel. Realisations `K` are *artificial* (windows).
* **MR-KLT (Multiple-Realisation KLT)** — for multi-antenna data, stack the `N_A`-vector
  measured at each of `K` timesteps into a `(K, N_A)` matrix. The `N_A × N_A` covariance
  captures the *spatial* structure across antennas, and the dominant eigenvector is a
  direction of arrival (beamforming). Realisations `K` are *physical* (timesteps).

The two are the same linear-algebra object with a relabelling:

| | C-KLT | MR-KLT |
|---|---|---|
| input | 1-D stream `(N,)` | matrix `(K, N_A)` |
| realisations `K` | windows, `K = N // W` | timesteps |
| dimension | window size `W` | antennas `N_A` |
| covariance captures | temporal structure | spatial structure |
| dominant eigenvector | dominant spectral mode | direction of arrival |
| aspect ratio `γ` | `W/(K-1)` | `N_A/(K-1)` |

In code (identical downstream for both):

```python
klt = KLT(logger=quiet_logger)
klt.load_signal_from_array(x)      # (K, N_A) for MR-KLT
klt.apply_mrklt(n_eigenvectors=1)  # or apply_cklt(window_size=W) for a 1-D stream
eigenvalues = klt.get_eigenvalues()   # descending
sigma2_hat  = klt.sigma2_hat           # trace(Σ)/W  (see §2.3)
```

### 2.2 The SCM and its eigenvalues

The SCM is `Σ = np.cov(matrix, rowvar=False)` (division by `K−1`), a `W × W` complex
Hermitian matrix, diagonalised with a Hermitian solver. Eigenvalues are returned in
descending order; `λ₁` is the candidate spike.

### 2.3 Noise variance: `σ̂² = trace(Σ)/W`

The `seti_klt` pipeline estimates the noise floor as the **trace of the SCM divided by its
dimension**. It is unbiased for complex Hermitian (GUE-family) covariance and, crucially,
does not require knowing the rank in advance. A rank-`r` signal contaminates only an
`O(r/W)` fraction of the trace, so the estimate stays usable in the presence of spikes —
though *strong* spikes at small `W` still inflate it noticeably, and that is exactly where
Kritchman–Nadler's bias-correcting `KN_noiseEst` has an edge. This trade-off is one of the
hinges of the comparison.

### 2.4 The RMT background every estimator relies on

Set the aspect ratio `γ = W/(K−1)` (require `K ≥ W`, informative regime `γ ≲ 0.25`).

* **Marchenko–Pastur law** — pure-noise eigenvalues fill a bulk with upper edge
  `λ₊ = σ²(1+√γ)²`.
* **Baik–Ben Arous–Péché (BBP) transition** — a population spike of strength `ℓ`
  (in units of `σ²`) produces a detached sample eigenvalue **only if `ℓ > √γ`**. Below
  that threshold the spike stays inside the bulk and *every* method is provably blind.
  This is not a shortcoming of either algorithm; it is an information-theoretic wall.
* **Tracy–Widom₂ (TW₂)** — the largest *noise* eigenvalue, suitably centred and scaled
  (Johnstone's constants), fluctuates according to the order-2 Tracy–Widom law for
  **complex** (I/Q, GUE) data. Both estimators use `β = 2`; where they differ is *how*
  they evaluate the TW₂ tail (asymptotic vs. exact — see §3).

These theorems are used, not re-derived; see the references for proofs.

---

## 3. The estimator, and the baseline it is compared against

**Kritchman & Nadler** (deconstructed in notebook 01) tests eigenvalues one at a time
against a Tracy–Widom threshold, re-estimating the noise variance at every step with a
*self-consistent* iteration that corrects for the energy each detected spike "steals" from
the noise eigenvalues. It returns a single integer `K̂`. Rigorous and, when its iteration
converges, very accurate — but it uses an *asymptotic* approximation to the TW tail (valid
only for `α ≪ 1`, with a hard `α < 1.99%` cliff for `β = 2`) and re-solves a nonlinear
system at each step. It is ported in `seti_klt.kritchman_nadler` as
`KN_rankEst` / `KN_noiseEst` / `KN_mu_sigma` / `KN_s_Wishart`.

**This project's Bayesian detector** (implemented in `seti_klt.KLT` on top of
`seti_klt.tracy_widom`) keeps the same sequential skeleton but (i) fixes a single
`trace(Σ)/W` noise estimate for the whole window, (ii) evaluates the TW₂ null *exactly* by
integrating the Painlevé II ODE, (iii) models the signal hypothesis with the BBP
supercritical Gaussian law, and (iv) returns a **continuous posterior** `P(H1 | λ₁)`, not
just an integer. Its methods (`compute_rmt_statistics`, `compute_tw2_likelihood`,
`compute_h1_likelihood`, `compute_posterior`) are documented in `KLT.py`.

The comparison is meant to be **honest**: each method wins in a different regime. Notebook 01
runs the *actual ported* Kritchman–Nadler code on synthetic data with a *known* number of
signals (via `helpers.make_spiked_data`) — the only way to say objectively which estimate is
"right" — and ends with a candid account of where the estimator is strong and where it is
fragile.

---

## 4. Running the notebook

**Dependencies.** The synthetic content needs only `numpy`, `scipy`, `matplotlib`, and the
`seti_klt` package.

**Making `seti_klt` importable.** The notebook starts by calling
`helpers.setup_seti_klt_path()`, which finds `../src/seti_klt` automatically if this folder
lives inside the repository (e.g. under `SETI_KLT/notebooks/`). Otherwise set the
`SETI_KLT_SRC` environment variable to the directory containing `seti_klt/`, or
`pip install` the package.

**Ground truth.** The notebook uses `helpers.make_spiked_data(...)`, which plants a *known*
number of signals — the only way to say objectively which estimate is "right".

**Outputs.** The notebook follows the project's standard I/O convention
(`seti_klt.utils.io`): the figure goes to `outputs/figures/` via `save_figure`, and the run
log to `outputs/logs/kritchman_nadler.log` via `SimpleLogger`. Everything under `outputs/`
is git-ignored.

---

## 5. References

Foundations (used, not re-proved):

* I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components
  analysis*, Ann. Statist. **29**(2), 2001.
* J. Baik, G. Ben Arous, S. Péché, *Phase transition of the largest eigenvalue for
  non-null complex sample covariance matrices*, Ann. Probab. **33**(5), 2005.
* C. Tracy, H. Widom, *Level-spacing distributions and the Airy kernel*, Comm. Math. Phys.
  **159**, 1994.

The estimators:

* S. Kritchman, B. Nadler, *Non-parametric detection of the number of signals: hypothesis
  testing and random matrix theory*, IEEE Trans. Signal Process. **57**(10), 2009.
  Original MATLAB: `wisdom.weizmann.ac.il/~nadler/Rank_Estimation/`.
* This project: `github.com/LuigiSotgiu/SETI_KLT` — modules `kritchman_nadler.py`,
  `tracy_widom.py`, `KLT.py`.
