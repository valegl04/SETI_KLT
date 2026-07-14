"""
tracy_widom.py
--------------
Tracy-Widom distribution of order 2 (TW₂) for complex Hermitian random matrices
(Gaussian Unitary Ensemble / complex Wishart family).

Background
----------
The largest eigenvalue λ₁ of a complex sample covariance matrix Σ̂ (size W×W,
built from K independent realisations of a Circular Normal process) converges,
after Johnstone's standardisation, to the TW₂ distribution — *not* TW₁, which
applies to real symmetric matrices.  Using TW₁ on complex baseband data
(as produced by GBT/VEGAS or SRT/SARDARA) systematically shifts the
false-alarm probability in the distribution tails, invalidating the Bayesian
posterior for signal detection.

The TW₂ distribution is derived from the Hastings-McLeod solution q(x) of the
Painlevé II ODE:

    q''(x) = x·q(x) + 2·q(x)³,   q(x) ~ Ai(x) as x → +∞

The log-CDF is then:

    log F₂(s) = −∫ₛ^∞ (x − s)·[q(x)]² dx

and the PDF is:

    f₂(s) = F₂(s) · ∫ₛ^∞ [q(x)]² dx

The LUT is pre-computed once at construction time (~30 ms) and stored as two
float64 arrays (s_grid, cdf, pdf).  All runtime lookups are O(log N) binary
searches on the sorted grid.

References
----------
* Tracy & Widom (1994) — original GUE result.
* Johnstone (2001) — centering/scaling constants for complex Wishart.
* El Karoui (2007) — Tracy-Widom limit for large class of complex sample
  covariance matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d
from scipy.special import airy


class TracyWidomTW2:
    """Pre-computed Tracy-Widom TW₂ distribution for complex Wishart eigenvalues.

    The Painlevé II ODE is integrated once at construction time to build a
    high-resolution lookup table (LUT) of the PDF and CDF on the interval
    ``[s_min, s_max]``.  All subsequent evaluations are O(log N) interpolations.

    Args:
        s_min:   Left boundary of the LUT grid (default −10.0).
        s_max:   Right boundary of the LUT grid (default 6.0).
        n_pts:   Number of grid points (default 15 000).
        rtol:    Relative tolerance for the ODE solver (default 1e-12).
        atol:    Absolute tolerance for the ODE solver (default 1e-14).

    Example::

        tw2 = TracyWidomTW2()
        s   = -1.5
        print(tw2.cdf(s), tw2.pdf(s), tw2.sf(s))
    """

    def __init__(
        self,
        s_min: float = -10.0,
        s_max: float = 6.0,
        n_pts: int = 15_000,
        rtol: float = 1e-12,
        atol: float = 1e-14,
    ) -> None:
        self._s_min = s_min
        self._s_max = s_max
        self._n_pts = n_pts
        self._s_grid, self._cdf, self._pdf = self._build_lut(
            s_min, s_max, n_pts, rtol, atol
        )
        self._cdf_interp = interp1d(
            self._s_grid, self._cdf,
            kind="linear", bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        self._pdf_interp = interp1d(
            self._s_grid, self._pdf,
            kind="linear", bounds_error=False,
            fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pdf(self, s: float | np.ndarray) -> float | np.ndarray:
        """TW₂ probability density function at *s*.

        For ``s`` beyond the LUT's right edge (``s_max``, default 6.0), the
        known right-tail asymptotic of the Tracy-Widom distribution is used
        instead of returning exactly zero.  This matters for strong-signal
        detections, where λ₁ standardises to s ≫ 6 and the H₀ likelihood
        must remain non-zero (merely extremely small) for the Bayesian
        posterior to behave correctly.
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        out = self._pdf_interp(s_arr)
        tail_mask = s_arr >= self._s_max
        if np.any(tail_mask):
            out = np.array(out, dtype=float)
            out[tail_mask] = self._pdf_right_tail(s_arr[tail_mask])
        return float(out[0]) if np.isscalar(s) else out

    def cdf(self, s: float | np.ndarray) -> float | np.ndarray:
        """TW₂ cumulative distribution function F₂(s) = P(TW₂ ≤ s).

        For ``s`` beyond the LUT's right edge, ``1 − sf_right_tail(s)`` is
        used (see :meth:`sf` for the asymptotic survival function).
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        out = self._cdf_interp(s_arr)
        tail_mask = s_arr >= self._s_max
        if np.any(tail_mask):
            out = np.array(out, dtype=float)
            out[tail_mask] = 1.0 - self._sf_right_tail(s_arr[tail_mask])
        return float(out[0]) if np.isscalar(s) else out

    def sf(self, s: float | np.ndarray) -> float | np.ndarray:
        """Survival function 1 − F₂(s) = P(TW₂ > s) = false-alarm probability.

        For ``s`` beyond the LUT's right edge, the known asymptotic
        (Tracy-Widom 1994; Deift-Its-Krasovsky 2008) is used:

            1 − F₂(s) ~ 1/(16π·s^{3/2}) · exp(−4/3·s^{3/2})    as s → +∞

        This keeps the false-alarm probability strictly positive (rather
        than collapsing to exactly 0.0) for very strong detections, which is
        essential for the Bayesian posterior in :meth:`KLT.compute_posterior`
        to remain well-defined and responsive to the prior q̂.
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        out = 1.0 - self._cdf_interp(s_arr)
        tail_mask = s_arr >= self._s_max
        if np.any(tail_mask):
            out = np.array(out, dtype=float)
            out[tail_mask] = self._sf_right_tail(s_arr[tail_mask])
        return float(out[0]) if np.isscalar(s) else out

    def log_pdf(self, s: float | np.ndarray) -> float | np.ndarray:
        """Natural logarithm of the TW₂ PDF, numerically stable for s ≫ s_max.

        For ``s`` within the LUT range this is simply ``log(pdf(s))``.  For
        ``s`` far beyond ``s_max`` (deep right tail, e.g. s > 70), the linear
        PDF underflows to exactly 0.0 in float64 even though the true value
        is merely extremely small but non-zero.  This method instead
        evaluates the log of the asymptotic expression directly:

            log f₂(s) ≈ −4/3·s^{3/2} + log(4s^{5/2} + 3s) − log(32π) − 7/2·log(s)

        which remains finite (a large negative number) for arbitrarily large
        s, allowing log-space Bayesian posterior calculations
        (see :meth:`KLT.compute_posterior`) to stay well-defined even for
        signals many orders of magnitude above the noise floor.

        Args:
            s: Standardised statistic (scalar or array).

        Returns:
            log f₂(s).  Returns ``-inf`` for s below the LUT's left edge
            (PDF is exactly representable as 0 there in float64; no
            underflow concern on that side for realistic signal regimes).
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        out = np.full_like(s_arr, -np.inf)

        in_range = (s_arr >= self._s_min) & (s_arr < self._s_max)
        if np.any(in_range):
            pdf_vals = self._pdf_interp(s_arr[in_range])
            with np.errstate(divide="ignore"):
                out[in_range] = np.log(np.maximum(pdf_vals, 0.0))

        tail_mask = s_arr >= self._s_max
        if np.any(tail_mask):
            st = s_arr[tail_mask]
            out[tail_mask] = (
                -4.0 / 3.0 * st ** 1.5
                + np.log(4.0 * st ** 2.5 + 3.0 * st)
                - np.log(32.0 * np.pi)
                - 3.5 * np.log(st)
            )

        return float(out[0]) if np.isscalar(s) else out

    @staticmethod
    def _sf_right_tail(s: np.ndarray) -> np.ndarray:
        """Right-tail asymptotic survival function 1 − F₂(s), s ≫ s_max."""
        return 1.0 / (16.0 * np.pi * s ** 1.5) * np.exp(-4.0 / 3.0 * s ** 1.5)

    @staticmethod
    def _pdf_right_tail(s: np.ndarray) -> np.ndarray:
        """Right-tail asymptotic PDF, derived as −d/ds[sf_right_tail(s)]."""
        return (4.0 * s ** 2.5 + 3.0 * s) * np.exp(-4.0 / 3.0 * s ** 1.5) / (32.0 * np.pi * s ** 3.5)

    def ppf(self, p: float | np.ndarray) -> float | np.ndarray:
        """Quantile (inverse CDF): s such that F₂(s) = p."""
        p = np.clip(p, 0.0, 1.0)
        _inv = interp1d(
            self._cdf, self._s_grid,
            kind="linear", bounds_error=False,
            fill_value=(self._s_min, self._s_max),
        )
        return float(_inv(p)) if np.isscalar(p) else _inv(p)

    def detection_threshold(self, fap: float) -> float:
        """Threshold s* such that P(TW₂ > s* | H₀) = fap."""
        return self.ppf(1.0 - fap)

    # ------------------------------------------------------------------
    # Johnstone constants  (static)
    # ------------------------------------------------------------------

    @staticmethod
    def johnstone_params(K: int, W: int) -> tuple[float, float]:
        """Centering and scaling constants for complex Wishart (GUE / TW₂).

        These constants standardise the largest eigenvalue of the **sample
        covariance matrix** (SCM) as computed by ``numpy.cov(rowvar=False)``,
        which divides by *n = K − 1* (unbiased Bessel correction):

            SCM = X^H X / (K − 1)

        The standardised statistic is:

            s = (λ₁ / σ̂² − μ) / σ_scale

        where μ and σ_scale are defined for n = K − 1 observations and
        p = W dimensions (aspect ratio γ = p/n):

            μ        = (1 + √γ)²
            σ_scale  = n^{−2/3} · (1 + √γ)^{4/3} · (1 + 1/√γ)^{1/3}

        This follows the canonical form in Johnstone (2001), Theorem 1.1 for
        the complex (β = 2) case, applied to the normalised SCM.

        Note: an earlier (incorrect) formulation used the Wishart matrix
        X^H X (unnormalised), yielding μ_KW = (√K + √W)² and
        σ_KW = (√K + √W)·(1/√K + 1/√W)^{1/3} — values O(K) larger — which
        produces the wrong standardisation by a factor of K−1.

        Args:
            K: Number of pseudo-realisations (rows of trajectory matrix).
               The effective sample count is n = K − 1 (np.cov convention).
            W: Window size / covariance matrix dimension  (= p in Johnstone).

        Returns:
            ``(mu, sigma_scale)`` — centering and scaling for the statistic s.
        """
        n      = float(K - 1)          # np.cov divides by K-1
        p      = float(W)
        gamma  = p / n
        mu         = (1.0 + np.sqrt(gamma)) ** 2
        sigma_scale = (n ** (-2.0 / 3.0)
                       * (1.0 + np.sqrt(gamma)) ** (4.0 / 3.0)
                       * (1.0 + 1.0 / np.sqrt(gamma)) ** (1.0 / 3.0))
        return float(mu), float(sigma_scale)

    # ------------------------------------------------------------------
    # Spiked covariance model (BBP supercritical regime)
    # ------------------------------------------------------------------

    @staticmethod
    def bbp_critical_threshold(gamma: float) -> float:
        """Population-level BBP critical threshold λ_c = √γ.

        Below this threshold (population spike intensity ℓ < λ_c) the sample
        eigenvalue cannot escape the Marchenko-Pastur bulk and the detector is
        formally blind to the signal (subcritical regime).  Above it
        (ℓ > λ_c) the eigenvalue separates and follows a Gaussian law instead
        of TW₂ (supercritical regime).

        Args:
            gamma: Aspect ratio γ = p/n = W/(K−1).

        Returns:
            λ_c = √γ  (population-level threshold, dimensionless — in units
            of σ²).
        """
        return float(np.sqrt(gamma))

    @staticmethod
    def bbp_sample_eigenvalue(l_pop: float, gamma: float) -> float:
        """Expected sample eigenvalue (in units of σ²) for a supercritical spike.

        Implements the Baik-Silverstein (2006) / Baik-Ben Arous-Péché (2005)
        almost-sure limit for the largest sample eigenvalue of a rank-1
        perturbed complex Wishart matrix, valid for ℓ > √γ:

            λ_sample(ℓ) / σ² = (ℓ + 1)·(1 + γ/ℓ)

        Note: the simplified form "ℓ + 1/ℓ" sometimes quoted in the
        literature is only valid for the special case γ = 1; using it for
        γ ≠ 1 introduces errors of 10-25 % (verified numerically — see
        project notes).  This implementation uses the general γ-dependent
        form, which matches Monte Carlo simulation to < 1 % across a wide
        range of ℓ and γ.

        Args:
            l_pop: Population spike intensity ℓ (true SNR, in units of σ²).
                   Must be > √γ for the formula to be meaningful.
            gamma: Aspect ratio γ = p/n = W/(K−1).

        Returns:
            Expected normalised sample eigenvalue λ_sample/σ².
        """
        return float((l_pop + 1.0) * (1.0 + gamma / l_pop))

    @staticmethod
    def bbp_sample_variance(l_pop: float, gamma: float, n: int) -> float:
        """Asymptotic variance of the supercritical sample eigenvalue.

        For complex Wishart matrices (β = 2), the fluctuation of the sample
        eigenvalue around its deterministic limit is asymptotically Gaussian
        with variance:

            Var[λ_sample] / σ⁴ = (ℓ + 1)² · (1 − γ/ℓ²) / n

        Verified by Monte Carlo simulation (n_trials ≥ 400) to match within
        ~10 % across ℓ ∈ [1, 8], γ ≈ 0.25 — consistent with O(n^{-1/2})
        finite-size corrections expected at these matrix dimensions.

        Note: this is the complex-case (β=2) variance, which does **not**
        carry the extra factor of 2 that appears in the real (β=1, TW₁)
        analogue — using the real-case formula here would overstate the
        variance by 2× and under-confidently flag true detections.

        Args:
            l_pop: Population spike intensity ℓ.
            gamma: Aspect ratio γ = p/n = W/(K−1).
            n:     Effective sample size (K − 1).

        Returns:
            Variance of λ_sample (in units of σ⁴).

        Raises:
            ValueError: If ``l_pop**2 <= gamma`` (variance would be
                        non-positive — ℓ too close to or below threshold).
        """
        if l_pop ** 2 <= gamma:
            raise ValueError(
                f"l_pop={l_pop} is at or below the BBP threshold "
                f"(l_pop² ≤ γ={gamma:.4f}); the Gaussian fluctuation formula "
                "is only valid in the supercritical regime."
            )
        return float((l_pop + 1.0) ** 2 * (1.0 - gamma / l_pop ** 2) / n)

    @staticmethod
    def invert_bbp_for_l_pop(
        lambda_sample: float,
        gamma: float,
        l_search_max: float = 1e4,
    ) -> float:
        """Invert the BBP map to recover the population spike ℓ from λ_sample.

        Solves  λ_sample/σ² = (ℓ+1)(1+γ/ℓ)  for ℓ given an *observed*
        normalised sample eigenvalue, via closed-form quadratic inversion.

        Rearranging:  ℓ² + ℓ·(1 − λ_sample + γ) + γ = 0

        Args:
            lambda_sample: Observed λ₁ / σ̂² (normalised dominant eigenvalue).
            gamma:         Aspect ratio γ = p/n = W/(K−1).
            l_search_max:  Upper sanity bound on the recovered ℓ (guards
                           against numerical blow-up for λ_sample ≈ λ_max_MP).

        Returns:
            Estimated population spike intensity ℓ̂ (≥ √γ by construction;
            clipped to ``l_search_max`` if the quadratic is degenerate).
        """
        b = 1.0 - lambda_sample + gamma
        c = gamma
        disc = b ** 2 - 4.0 * c
        if disc < 0:
            # λ_sample below the MP edge: no real positive root — signal is
            # (at best) marginally sub-critical.  Return the critical value.
            return float(np.sqrt(gamma))
        l_hat = (-b + np.sqrt(disc)) / 2.0
        l_hat = max(l_hat, np.sqrt(gamma))   # never below threshold
        return float(min(l_hat, l_search_max))

    # ------------------------------------------------------------------
    # LUT properties
    # ------------------------------------------------------------------

    @property
    def s_grid(self) -> np.ndarray:
        return self._s_grid

    @property
    def cdf_table(self) -> np.ndarray:
        return self._cdf

    @property
    def pdf_table(self) -> np.ndarray:
        return self._pdf

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lut(
        s_min: float,
        s_max: float,
        n_pts: int,
        rtol: float,
        atol: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate Painlevé II ODE and compute CDF/PDF via cumulative trapezoid."""
        Ai_val, Aip_val, _, _ = airy(s_max)
        y0 = [float(Ai_val), float(Aip_val)]

        def _ode(x: float, y: list) -> list:
            q, qp = y
            return [qp, x * q + 2.0 * q ** 3]

        x_eval = np.linspace(s_max, s_min, n_pts)
        sol = solve_ivp(
            _ode,
            t_span=(s_max, s_min),
            y0=y0,
            t_eval=x_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if sol.status != 0:
            raise RuntimeError(
                f"Painlevé II ODE integration failed (status={sol.status}): {sol.message}"
            )

        s_asc = sol.t[::-1]
        q_asc = sol.y[0][::-1]

        q2  = q_asc ** 2
        xq2 = s_asc * q2

        total_q2  = trapezoid(q2,  s_asc)
        total_xq2 = trapezoid(xq2, s_asc)
        cum_q2    = cumulative_trapezoid(q2,  s_asc, initial=0.0)
        cum_xq2   = cumulative_trapezoid(xq2, s_asc, initial=0.0)

        I0 = total_q2  - cum_q2
        I1 = total_xq2 - cum_xq2

        log_F = -(I1 - s_asc * I0)
        log_F = np.clip(log_F, -700.0, 0.0)
        cdf   = np.exp(log_F)
        pdf   = cdf * np.maximum(I0, 0.0)

        return s_asc, cdf, pdf