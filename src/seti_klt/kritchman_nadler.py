"""
kritchman_nadler.py
--------------------
Python port of the Kritchman & Nadler (2009) RMT-based estimator for the
number of signals embedded in noise.

This is a faithful translation of the original MATLAB code by Shira
Kritchman and Boaz Nadler (Weizmann Institute of Science, 2008), available
at http://www.wisdom.weizmann.ac.il/~nadler/Rank_Estimation/, accompanying:

    S. Kritchman and B. Nadler, "Non-Parametric Detection of the Number of
    Signals: Hypothesis Testing and Random Matrix Theory," IEEE Trans.
    Signal Process., 57(10):3930-3941, 2009.

Provided here purely as a faithful reference implementation, to allow a
direct, apples-to-apples comparison against the KLT-RMT Bayesian detector
developed in seti_klt (Phases B-E). Variable names and structure intentionally
mirror the original MATLAB source (KN_rankEst.m, KN_noiseEst.m,
KN_mu_sigma.m, KN_s_Wishart.m) rather than seti_klt's own conventions, to
keep the port auditable line-by-line against the original.

Key difference in scaling convention vs. seti_klt.tracy_widom
---------------------------------------------------------------
KN's mu_np/sigma_np (KN_mu_sigma) are the Johnstone centering/scaling
constants for the *unnormalised* Wishart matrix n*S_n (not the sample
covariance matrix S_n = (1/n) X^H X divided by n).  Consequently
KN_rankEst compares `n * ell(kk)` against `sigma_hat * (mu_np + s*sigma_np)`,
not `ell(kk) / sigma_hat` directly.  This is a different normalisation from
seti_klt.tracy_widom.TracyWidomTW2.johnstone_params, which is calibrated for
the SCM as computed by numpy.cov (divided by K-1).  Do not mix the two
conventions when comparing results.

Threshold approximation
------------------------
KN_s_Wishart uses the *asymptotic* (large-x) approximation to the
Tracy-Widom tail, valid only for small alpha (alpha << 1, as stated in the
original paper).  This is a deliberate speed/simplicity choice by the
authors, distinct from seti_klt.tracy_widom.TracyWidomTW2, which integrates
the full Painleve II ODE for an exact (not asymptotic) TW2 LUT.  This
difference is a relevant point of comparison between the two methods, not
a bug to "fix" in this port.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# KN_s_Wishart.m
# ---------------------------------------------------------------------------

def KN_s_Wishart(alpha: float, beta: int) -> float:
    """Approximate inverse of the Tracy-Widom tail (large-x asymptotic).

    Faithful port of KN_s_Wishart.m.  Returns ``s`` such that, for a TW_beta
    random variable X, ``Pr{X > s} ~= alpha/100``.  Valid only for small
    alpha (alpha << 1), per the asymptotics of Baik, Buckingham & DiFranco
    (2008).

    Args:
        alpha: Confidence level in **percentage points** (e.g. 0.1 for a
                0.1% false-alarm rate) -- matches the original MATLAB
                convention (alpha/100 is the probability).
        beta:  1 for real-valued noise (TW1, GOE), 2 for complex-valued
                noise (TW2, GUE).

    Returns:
        Approximate Tracy-Widom threshold s_Wishart. Returns NaN (matching
        the original MATLAB's silent behaviour under a fractional power of
        a negative number) if alpha is too large for the asymptotic
        approximation to be valid for the given beta -- see warning below.

    Raises:
        ValueError: If beta is not 1 or 2.
    """
    # The asymptotic formula requires its log-argument to be <= 1, i.e.
    # alpha <= 100/(16*pi) =~ 1.99% for beta=2, or alpha <= 100/(4*sqrt(pi))
    # =~ 14.1% for beta=1. Outside this range the original MATLAB code
    # silently returns a complex/NaN value (fractional power of a negative
    # base); this port preserves that numerical behaviour exactly (no
    # logic change), but surfaces it as an explicit warning rather than a
    # silent NaN, since this is easy to trip over (e.g. by naively trying
    # alpha=50 to "match" an unrelated 50%-confidence threshold elsewhere).
    alpha_max = 100.0 / (16.0 * np.pi) if beta == 2 else 100.0 / (4.0 * np.sqrt(np.pi))
    if alpha >= alpha_max:
        import warnings
        warnings.warn(
            f"KN_s_Wishart: alpha={alpha} is outside the valid range for this "
            f"asymptotic approximation (beta={beta} requires alpha < {alpha_max:.4f}). "
            "The original paper states this formula holds only for alpha << 1. "
            "The result will be NaN (matching the original MATLAB's behaviour "
            "under a fractional power of a negative number) rather than a "
            "meaningful threshold.",
            stacklevel=2,
        )

    if beta == 2:
        return float((-3.0 / 4.0 * np.log(16.0 * np.pi * alpha / 100.0)) ** (2.0 / 3.0))
    if beta == 1:
        return float((-3.0 / 2.0 * np.log(4.0 * np.sqrt(np.pi) * alpha / 100.0)) ** (2.0 / 3.0))
    raise ValueError(f"beta must be 1 (real) or 2 (complex), got {beta}.")


# ---------------------------------------------------------------------------
# KN_mu_sigma.m
# ---------------------------------------------------------------------------

def KN_mu_sigma(n: int, p: int, beta: int) -> tuple[float, float]:
    """Johnstone centering/scaling constants for the *unnormalised* Wishart matrix.

    Faithful port of KN_mu_sigma.m.  These normalise the largest eigenvalue
    ell_1 of n*S_n (the unnormalised Wishart matrix, NOT divided by n):

        Pr{ell_1 > (mu_np + s * sigma_np)} --> F_beta(s)   as n, p -> infinity

    See I.M. Johnstone (2006) and N. El Karoui (2006) for derivation
    details.

    Args:
        n:    Number of samples.
        p:    Dimension of samples.
        beta: 1 for real-valued, 2 for complex-valued observations.

    Returns:
        ``(mu_np, sigma_np)``.

    Raises:
        ValueError: If beta is not 1 or 2.
    """
    if beta == 1:
        mu_np = (np.sqrt(n - 0.5) + np.sqrt(p - 0.5)) ** 2
        sigma_np = np.sqrt(mu_np) * (1.0 / np.sqrt(n - 0.5) + 1.0 / np.sqrt(p - 0.5)) ** (1.0 / 3.0)
        return float(mu_np), float(sigma_np)

    if beta == 2:
        P = min(n, p)
        N = max(n, p)
        N_plus = N + 0.5
        P_plus = P + 0.5
        Nm1_plus = N - 1 + 0.5
        Pm1_plus = P - 1 + 0.5

        mu_Nm1P_temp = (np.sqrt(Nm1_plus) + np.sqrt(P_plus)) ** 2
        mu_NPm1_temp = (np.sqrt(N_plus) + np.sqrt(Pm1_plus)) ** 2
        sigma_Nm1P_temp = (np.sqrt(Nm1_plus) + np.sqrt(P_plus)) * (
            1.0 / np.sqrt(Nm1_plus) + 1.0 / np.sqrt(P_plus)
        ) ** (1.0 / 3.0)
        sigma_NPm1_temp = (np.sqrt(N_plus) + np.sqrt(Pm1_plus)) * (
            1.0 / np.sqrt(N_plus) + 1.0 / np.sqrt(Pm1_plus)
        ) ** (1.0 / 3.0)

        gamma_NP = (mu_Nm1P_temp * sigma_NPm1_temp ** 0.5) / (
            mu_NPm1_temp * sigma_Nm1P_temp ** 0.5
        )
        sigma_np = (1.0 + gamma_NP) / (1.0 / sigma_Nm1P_temp + gamma_NP / sigma_NPm1_temp)
        mu_np = (1.0 / sigma_Nm1P_temp ** 0.5 + 1.0 / sigma_NPm1_temp ** 0.5) / (
            1.0 / (mu_Nm1P_temp * sigma_Nm1P_temp ** 0.5)
            + 1.0 / (mu_NPm1_temp * sigma_NPm1_temp ** 0.5)
        )
        return float(mu_np), float(sigma_np)

    raise ValueError(f"beta must be 1 (real) or 2 (complex), got {beta}.")


# ---------------------------------------------------------------------------
# KN_noiseEst.m
# ---------------------------------------------------------------------------

def KN_noiseEst(
    ell: np.ndarray,
    n: int,
    kk: int,
    max_iter: int = 30,
    eps_threshold: float = 1e-5,
) -> float:
    """Self-consistent noise-variance estimator, assuming *kk* signals.

    Faithful port of KN_noiseEst.m.  Solves iteratively for sigma^2 under
    the assumption that the first *kk* sample eigenvalues correspond to
    signals and the remainder to pure noise, correcting for the upward
    bias each spike eigenvalue introduces relative to its "true" (bias-free)
    population counterpart.

    At each iteration, for the current sigma^2 guess, a quadratic equation
    is solved (closed form, larger root) for each of the kk spike
    eigenvalues to obtain an implied "rho" — an estimate of the bias-free
    contribution. sigma^2 is then re-estimated as the mean of the
    (rank p-kk) noise-only eigenvalues plus the *removed* spike bias,
    divided by (p-kk). Iterates until convergence or max_iter is reached.

    Args:
        ell:           Eigenvalues of the SCM, sorted in **descending**
                        order, length p.
        n:             Number of samples.
        kk:            Assumed number of signals (rank).
        max_iter:      Maximum number of fixed-point iterations (default 30,
                        matching the original).
        eps_threshold: Relative convergence tolerance (default 1e-5,
                        matching the original).

    Returns:
        Estimate of the (squared) noise variance, sigma_hat^2.
    """
    ell = np.asarray(ell, dtype=float)
    p = len(ell)

    sigma_0 = (1.0 / (p - kk)) * np.sum(ell[kk:]) * 1.0 / (1.0 - kk / n)

    for _ in range(max_iter):
        tmp = ell[:kk] + sigma_0 - (p - kk) / n * sigma_0

        discriminant = tmp ** 2 - 4.0 * ell[:kk] * sigma_0
        if np.min(discriminant) < 0:
            # Complex-valued solution would result; stop and keep current guess.
            break

        Rho_est = (tmp + np.sqrt(discriminant)) / 2.0

        if np.min(ell[:kk] - Rho_est) < -1e-7:
            import warnings
            warnings.warn(
                f"KN_noiseEst: consistency check failed for kk={kk} "
                f"(min diff={np.min(ell[:kk] - Rho_est):.2e}). "
                "This triggers the 'MAJOR ERROR' from the original MATLAB code, "
                "but execution will continue."
            )

        Delta_l_rho = np.maximum(ell[:kk] - Rho_est, 0.0)
        sigma_new = (1.0 / (p - kk)) * (np.sum(ell[kk:]) + np.sum(Delta_l_rho))

        if abs(sigma_new - sigma_0) / sigma_0 < eps_threshold:
            sigma_0 = sigma_new
            break
        sigma_0 = sigma_new

    return float(sigma_0)


# ---------------------------------------------------------------------------
# KN_rankEst.m
# ---------------------------------------------------------------------------

def KN_rankEst(
    ell: np.ndarray,
    n: int,
    beta: int,
    alpha: float = 0.5,
    max_kk: int | None = None,
) -> tuple[int, float]:
    """Estimate the number of signals (pseudo-rank) via sequential hypothesis testing.

    Faithful port of KN_rankEst.m.  For kk = 1, 2, ..., max_kk, tests
    "at most kk-1 signals" vs. "at least kk signals" by checking whether
    n*ell(kk) exceeds a Tracy-Widom-derived threshold (scaled by the
    self-consistent noise estimate KN_noiseEst(ell, n, kk)). Stops at the
    first kk for which the null hypothesis (no further signal) is accepted.

    Args:
        ell:    Eigenvalues of the SCM, sorted in **descending** order.
        n:      Number of samples.
        beta:   1 for real-valued, 2 for complex-valued observations.
        alpha:  Confidence level in **percentage points** (e.g. 0.1 for a
                0.1% false-alarm rate per test). Default 0.5 (matches the
                MATLAB default).
        max_kk: Maximum number of signals to test for. Defaults to
                min(p, n) - 1.

    Returns:
        ``(K, sigma_hat)``: estimated number of signals and the
        corresponding noise-variance estimate.
    """
    ell = np.asarray(ell, dtype=float)
    p = len(ell)

    if max_kk is None:
        max_kk = min(p, n) - 1
    max_kk = min(max_kk, min(p, n) - 1)

    s_Wishart = KN_s_Wishart(alpha, beta)

    sigma_arr = np.zeros(max_kk)
    kk = 0
    for kk in range(1, max_kk + 1):
        mu_np, sigma_np = KN_mu_sigma(n, p - kk, beta)
        sig_hat_kk = KN_noiseEst(ell, n, kk)
        sigma_arr[kk - 1] = sig_hat_kk

        at_least_kk_signals = n * ell[kk - 1] > sig_hat_kk * (mu_np + s_Wishart * sigma_np)
        if not at_least_kk_signals:
            break

    K = kk - 1
    if K > 0:
        sigma_hat = sigma_arr[K - 1]
    else:
        sigma_hat = float(np.sum(ell) / p)

    return int(K), float(sigma_hat)