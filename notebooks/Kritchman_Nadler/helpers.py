"""
helpers.py
==========
Shared utilities for the expository notebook in this folder
(01 Kritchman-Nadler, deconstructed).

It contains only the *boilerplate* that would otherwise be repeated in the
notebook:

* :func:`setup_seti_klt_path` -- locate and import the ``seti_klt`` package
  regardless of where this folder is placed inside the repository;
* :func:`make_spiked_data` -- generate synthetic complex data with a known
  number of planted signals (the "ground truth" the notebooks compare
  against);
* :func:`eig_from_data` -- run the real MR-KLT eigendecomposition on a data
  matrix and return everything the two estimators need;
* a few small plotting helpers used in more than one notebook.

Nothing algorithmic lives here: the estimators are called straight from the
real ``seti_klt`` package, so the notebook demonstrates the *actual* pipeline,
not a re-implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Make the real seti_klt package importable
# ---------------------------------------------------------------------------

def setup_seti_klt_path() -> None:
    """Ensure ``import seti_klt`` works from wherever this folder lives.

    Resolution order:

    1. If ``seti_klt`` already imports (e.g. installed, or already on
       ``PYTHONPATH``), do nothing.
    2. Otherwise walk upward from this file looking for ``src/seti_klt`` (the
       layout of the LuigiSotgiu/SETI_KLT repository) and add ``src`` to
       ``sys.path``.
    3. As a last resort, honour the ``SETI_KLT_SRC`` environment variable.
    """
    try:
        import seti_klt  # noqa: F401
        return
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "src" / "seti_klt"
        if candidate.exists():
            sys.path.insert(0, str(parent / "src"))
            return

    import os
    env = os.environ.get("SETI_KLT_SRC")
    if env and (Path(env) / "seti_klt").exists():
        sys.path.insert(0, env)
        return

    raise ImportError(
        "Could not locate the seti_klt package. Place this folder inside the "
        "SETI_KLT repository (so that ../src/seti_klt exists), pip-install the "
        "package, or set the SETI_KLT_SRC environment variable to the directory "
        "that contains seti_klt/."
    )


# ---------------------------------------------------------------------------
# 2. Synthetic spiked-covariance data with known ground truth
# ---------------------------------------------------------------------------

def make_spiked_data(
    K: int,
    W: int,
    spikes: tuple[float, ...] = (),
    sigma2: float = 1.0,
    seed: int | None = 0,
) -> np.ndarray:
    r"""Generate a complex data matrix with a known number of planted signals.

    The output has shape ``(K, W)`` and follows the *spiked covariance* model
    that underpins both estimators (Johnstone 2001):

        x_k = n_k + sum_j sqrt(l_j * sigma2) * g_{k,j} * u_j

    where ``n_k`` is circularly-symmetric complex Gaussian noise with per-
    component variance ``sigma2``, the ``u_j`` are orthonormal spatial
    directions, and ``g_{k,j}`` are unit-variance complex Gaussian signal
    amplitudes.  The resulting population covariance is

        Sigma = sigma2 * I  +  sigma2 * sum_j l_j * u_j u_j^H,

    i.e. exactly ``len(spikes)`` population eigenvalues sit above the noise
    floor, each at ``sigma2 * (1 + l_j)``.  ``l_j`` is therefore the population
    spike strength in units of ``sigma2`` -- the same ``l_pop`` used by the
    BBP formulae in ``seti_klt.tracy_widom``.  The Baik-Ben Arous-Peche
    threshold is ``l_c = sqrt(gamma)`` with ``gamma = W / (K - 1)``: spikes
    below it are provably invisible.

    Args:
        K:      Number of realisations (rows).  Plays the role of timesteps in
                MR-KLT, or pseudo-realisations in C-KLT.
        W:      Dimension (columns).  Plays the role of the number of antennas
                ``N_A`` in MR-KLT, or the window size in C-KLT.
        spikes: Population strengths ``l_j`` of the planted signals (empty for
                pure noise).  The number of entries is the ground-truth rank.
        sigma2: Noise variance per complex component (default 1.0).
        seed:   RNG seed for reproducibility.

    Returns:
        Complex ``(K, W)`` array, ready for ``KLT.load_signal_from_array`` +
        ``KLT.apply_mrklt`` (or ``apply_cklt`` if you prefer to treat it as a
        1-D stream reshaped to windows).
    """
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((K, W)) + 1j * rng.standard_normal((K, W))) * np.sqrt(sigma2 / 2.0)
    if spikes:
        # Orthonormal spatial directions via QR of a random complex matrix.
        A = rng.standard_normal((W, len(spikes))) + 1j * rng.standard_normal((W, len(spikes)))
        Q, _ = np.linalg.qr(A)
        for j, l in enumerate(spikes):
            g = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2.0)
            X += np.sqrt(l * sigma2) * np.outer(g, Q[:, j].conj())
    return X


# ---------------------------------------------------------------------------
# 3. Run the real MR-KLT eigendecomposition
# ---------------------------------------------------------------------------

def eig_from_data(X: np.ndarray, logger=None) -> dict:
    """Run the real ``KLT.apply_mrklt`` on ``X`` and return its RMT outputs.

    This calls the *actual* seti_klt code path (``np.cov`` -> Hermitian
    eigendecomposition -> ``trace(Sigma)/W`` noise estimate), so the numbers
    the notebooks show are the numbers the pipeline produces.

    Args:
        X:      Complex ``(K, W)`` data matrix.
        logger: Optional SimpleLogger; a silent one is used if ``None``.

    Returns:
        Dict with ``eigenvalues`` (descending), ``sigma2_hat``, ``K``, ``W``,
        ``gamma`` (= W/(K-1)), and the full ``rmt`` statistics dict.
    """
    setup_seti_klt_path()
    from seti_klt import KLT
    from seti_klt.utils.io import SimpleLogger, LogLevel

    if logger is None:
        logger = SimpleLogger(level=LogLevel.ERROR, console_output=False)

    K, W = X.shape
    klt = KLT(logger=logger)
    klt.load_signal_from_array(X)
    klt.apply_mrklt(n_eigenvectors=1)
    rmt = klt.compute_rmt_statistics()
    return {
        "eigenvalues": klt.get_eigenvalues(),
        "sigma2_hat": klt.sigma2_hat,
        "K": K,
        "W": W,
        "gamma": W / (K - 1),
        "rmt": rmt,
        "klt": klt,
    }


# ---------------------------------------------------------------------------
# 4. Small plotting helpers (shared look-and-feel)
# ---------------------------------------------------------------------------

def plot_eigenspectrum(ax, eigenvalues, sigma2_hat, gamma, n_show=20, title=""):
    """Bar plot of the top eigenvalues with the Marchenko-Pastur edge marked."""
    import numpy as np

    ev = np.asarray(eigenvalues)[:n_show]
    mp_edge = sigma2_hat * (1.0 + np.sqrt(gamma)) ** 2
    colors = ["#2a9d8f" if e > mp_edge else "#adb5bd" for e in ev]
    ax.bar(range(1, len(ev) + 1), ev, color=colors, alpha=0.9)
    ax.axhline(mp_edge, color="#e63946", ls="--", lw=1.5,
               label=f"MP edge $\\sigma^2(1+\\sqrt{{\\gamma}})^2$ = {mp_edge:.3f}")
    ax.set_xlabel("eigenvalue index $k$")
    ax.set_ylabel("$\\lambda_k$")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")


def summary_line(tag, K_est, extra=""):
    """Uniform one-line result print used across notebooks."""
    return f"[{tag:>18}]  K_est = {K_est}   {extra}"
