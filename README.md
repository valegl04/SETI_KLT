# SETI KLT Project: Signal Denoising with Karhunen-Loève Transform

This repository contains a Python framework developed during an internship to apply various Karhunen-Loève Transform (KLT) methodologies to radio astronomy data. The primary goal of this project is to process and analyze time-voltage `GUPPI RAW` files (commonly used in SETI research) to test the extraction and reconstruction of weak continuous-wave (CW) signals submerged in thermal noise.

## Background: Why KLT?

In traditional radio astronomy, signals are often detected by accumulating power in frequency bins over time (using FFTs). However, if a signal is extremely weak, the variance of the thermal noise can mask it completely.

This project explores the **Karhunen-Loève Transform (KLT)** as an alternative algorithmic denoising step. Instead of looking directly at frequencies, the KLT analyzes the data in the *covariance domain*. By calculating the covariance matrix of the incoming voltage time-series and performing an Eigendecomposition, the KLT separates the data into:
1. **The Signal Subspace:** Captured by the dominant eigenvalues and eigenvectors (correlated signals).
2. **The Noise Subspace:** The remaining minor eigenvalues (uncorrelated thermal noise).

### Implemented KLT Variants
The core `KLT` class exposes two structural approaches:
* **C-KLT (Covariance KLT):** Segments a 1D baseband stream into 2D windows to build the temporal covariance matrix. Applied with `apply_cklt`.
* **MR-KLT (Multiple-Realization KLT):** For multi-antenna or multi-polarization arrays where data is inherently 2D, building an antenna × antenna spatial covariance matrix. Applied with `apply_mrklt`.

A third variant, **T-KLT (Toeplitz KLT)** — building a deterministic Toeplitz matrix from the 1D autocorrelation of the signal — is scaffolded but currently commented out pending validation.

On top of the transforms, the class also implements an RMT-based detection layer (Marchenko–Pastur / BBP thresholds and an exact Tracy–Widom₂ Bayesian posterior) to decide whether the dominant eigenvalue is a genuine signal or a noise fluctuation.

---

### Deep Dive: Covariance KLT (C-KLT) Framework

Following the formulation analyzed in our reference paper (*Trudu et al., 2020*), the C-KLT models a 1D discrete time-series voltage signal $x[n]$ by projecting it onto an adaptive, data-driven orthogonal basis rather than using static trigonometric kernels like the Fourier Transform.

The algorithmic pipeline implemented in this package follows four mathematical steps:

1. **Trajectory Matrix Formulation** The 1D complex baseband voltage vector of total length $N$ is segmented into $K$ blocks of length $W$ (where $W$ represents the `WINDOW_SIZE`). These blocks are arranged as rows to build the trajectory data matrix $X \in \mathbb{C}^{K \times W}$:

$$
X = \begin{bmatrix} 
x[0] & x[1] & \cdots & x[W-1] \\ 
x[W] & x[W+1] & \cdots & x[2W-1] \\ 
\vdots & \vdots & \ddots & \vdots \\ 
x[(K-1)W] & x[(K-1)W+1] & \cdots & x[KW-1] 
\end{bmatrix}
$$

2. **Covariance Matrix Estimation** The empirical spatial covariance matrix $R_{xx} \in \mathbb{C}^{W \times W}$ is estimated from the rows of $X$. In the standard C-KLT implementation, data is centered by subtracting the column-wise mean vector $\mu_X$ to isolate purely stochastic and alternating components:
   $$R_{xx} = \frac{1}{K} (X - \mu_X)^H (X - \mu_X)$$

3. **Eigendecomposition & Subspace Separation** We solve the characteristic equation $R_{xx}V = V\Lambda$, where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_W)$ contains the real-valued eigenvalues sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_W$). 
   * **Signal Subspace:** The first $M$ dominant eigenvalues ($\lambda_1 \dots \lambda_M$) capture the deterministic, highly-correlated energy of the cosmic or artificial carrier wave.
   * **Noise Subspace:** The remaining $W - M$ eigenvalues form a flat plateau representing the uncorrelated variance ($\sigma^2$) of the thermal noise floor.

4. **Low-Rank Truncated Reconstruction** By selecting only the sub-matrix of primary eigenvectors $V_M \in \mathbb{C}^{W \times M}$, the centered data matrix is projected onto the signal subspace and transformed back into the time domain. Re-adding the mean vector yields the denoised time-series matrix $\hat{X}$, which is then flattened back into a clean 1D baseband signal.

*Note: The **T-KLT** (Toeplitz autocorrelation) variant is scaffolded but currently commented out, pending baseline validation. It will be documented here once activated.*

---

## Repository Structure

The project is structured as an installable Python package (`seti_klt`) accompanied by practical test notebooks.

```text
.
├── pyproject.toml              # Package definition and dependencies (managed via uv)
├── setigen_npol_fix.patch      # Patch for the setigen library (NPOL header fix)
│
├── src/seti_klt/               # CORE PACKAGE
│   ├── KLT.py                  # Main KLT class: C-KLT / MR-KLT + RMT/Bayesian detection
│   ├── multichannel.py         # Per-channel MR-KLT sweep over a multi-antenna cube
│   ├── tracy_widom.py          # Exact Tracy–Widom₂ law (Painlevé II) and BBP helpers
│   ├── kritchman_nadler.py     # Faithful port of the Kritchman & Nadler (2009) estimator
│   └── utils/io.py             # I/O helpers, dynamic paths, and a structured logger
│
├── notebooks/                  # TESTS & ANALYSIS (one subfolder per study, each with a README)
│   ├── Injected_Signal/        # Synthetic CW signal tests + the "Synchronous Phase" study
│   ├── Voyager1_Signal/        # Real GBT telemetry extraction (Voyager 1 case study)
│   ├── Kritchman_Nadler/       # Signal-counting: Kritchman–Nadler vs. Bayesian MR-KLT
│   ├── MAA_Simulated/          # Multi-antenna simulated baseband KLT pipeline
│   └── Meerkat_Real_Data/      # MR-KLT on real MeerKAT / Breakthrough Listen stamp data
│
├── data/                       # RAW DATA (git-ignored, local heavy files)
└── outputs/                    # RESULTS (locally generated figures, logs, CSVs)
```

> **Note:** For detailed explanations of specific studies (the "Synchronous Phase" phenomenon, Voyager 1 decoding, or the signal-counting comparison), see the `README.md` inside each `notebooks/` subdirectory.

---

## Installation & Setup

This project uses `uv` (or standard `pip` + `venv`) for dependency management. To ensure paths resolve correctly and local imports work inside the notebooks, the project must be installed in **editable mode**.

**1. Clone the repository:**
```bash
git clone https://github.com/LuigiSotgiu/seti_klt.git
cd seti_klt
```

**2. Apply the Setigen patch:**
*Note: A bug in the `setigen` library handles the `NPOL` header incorrectly when generating GUPPI outputs. Apply our patch to your local setigen folder before generating synthetic files.*
```bash
# Assuming you cloned setigen next to this repo or downloaded it locally
cd path/to/local/setigen
git apply path/to/seti_klt/setigen_npol_fix.patch
```

**3. Install the environment and the local package:**
```bash
# Install the patched setigen locally
uv pip install -e path/to/local/setigen

# Install this package (seti_klt) in editable mode
uv pip install -e .
```

---

## Quick Start Example

Once installed, you can use the package in your scripts or Jupyter Notebooks seamlessly:

```python
from seti_klt.KLT import KLT
from seti_klt.utils.io import get_data_path

# 1. Initialize the KLT pipeline
klt = KLT()

# 2. Load Raw Voltage Data
#    file_stem is the GUPPI stem (e.g. "obs" for obs.0000.raw), used by
#    setigen to recover the band parameters.
data_path = get_data_path() / "blc3_guppi_57386_VOYAGER1_0004.0000.raw"
klt.load_data_from_guppi(
    file_path=str(data_path),
    file_stem="blc3_guppi_57386_VOYAGER1_0004",
    channel=0,
    num_samples=2**20,
    polarization=0,
)

# 3. Apply Covariance KLT
klt.apply_cklt(window_size=1024, n_eigenvectors=3)

# 4. Extract and visualize results
klt.plot_eigenspectrum(n_components=10)
klt.plot_waterfall_comparison(channel=0, fftlength=1024)
```