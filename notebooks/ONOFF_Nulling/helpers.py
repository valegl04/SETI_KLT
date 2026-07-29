"""
helpers.py
==========
Spatial (covariance-level) interference nulling for the ON/OFF beam pipeline.

Goal
----
Around a **target** pointing we want to detect signals of interest while
rejecting interference that comes from *other* directions. Instead of forming
explicit OFF beams and subtracting them, we do the equivalent operation directly
on the **antenna covariance** (the "covariance nulling" idea):

1. estimate the target's spatial signature ``a_t`` (steering vector);
2. estimate the **interference subspace** ``U_i`` from the dominant covariance
   eigenvectors that are *not* the target (this is the MR-KLT eigenstructure);
3. project the target beamformer orthogonal to that subspace
   ``w = (I - U_i U_iᴴ) a_t`` (or the MVDR weights ``w ∝ R⁻¹ a_t``), forming a
   beam that keeps the target direction but places nulls on the interferers;
4. run the KLT detector on the resulting single beam.

This module holds only the *spatial* primitives. Data fabrication is reused from
``../MAA_Simulated/helpers.py`` (setigen + katpoint), and the KLT analysis is the
real ``seti_klt`` library — nothing about the pipeline is re-implemented here.

Contents
--------
* :func:`maa`               -- import the MAA_Simulated simulation module.
* :func:`antenna_cov`       -- antenna covariance of a ``(T, A)`` slice.
* :func:`eig_desc`          -- Hermitian eigdecomposition, descending.
* :func:`beamform`          -- apply beamformer weights to a ``(T, A)`` slice.
* :func:`beam_response`     -- ``|wᴴ a|²`` gain of weights toward a direction.
* :func:`interference_subspace` -- interference eigenvectors of a covariance.
* :func:`null_weights`      -- project a steering vector off the interf. subspace.
* :func:`mvdr_weights`      -- minimum-variance distortionless-response weights.
* :func:`beam_snr`          -- narrowband spectral SNR of a 1-D beam.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import katpoint

from seti_klt import KLT, KN_rankEst

try:  # optional GPU
    import cupy as cp
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# Reuse the MAA_Simulated data fabrication (setigen + katpoint)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def maa():
    """Import and return the ``MAA_Simulated/helpers.py`` module.

    Loaded under a unique module name (``maa_sim``) so it does not clash with
    this folder's own ``helpers`` module.  Cached: the module body is executed
    once per session, since ``steering`` calls this on every invocation and
    re-executing it (setigen/blimpy imports) would dominate the runtime.

    Returns:
        The imported module, exposing ``build_antennas``, ``compute_delays``,
        ``generate_scenario``, ``load_scenario``, ``signal_channels``.
    """
    path = Path(__file__).resolve().parent.parent / "MAA_Simulated" / "helpers.py"
    spec = importlib.util.spec_from_file_location("maa_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Directional narrowband simulation
# ---------------------------------------------------------------------------
#
# For the ON/OFF nulling study we need sources arriving from *arbitrary,
# controllable* directions (a target plus a nearby interferer), which the
# ``MAA_Simulated`` setigen path does not expose (it only injects toward the
# pointing or as delay-free per-antenna RFI).  We therefore synthesise the
# per-channel antenna cube directly with a narrowband model, reusing the
# *geometry* from ``MAA_Simulated`` (``build_antennas`` / ``compute_delays``) so
# the array layout and delays are identical to the realistic pipeline:
#
#     x[t, a] = Σ_src  s_src[t] · exp(-2πi f_src · τ_a(dir_src))  +  noise
#
# where ``τ_a(dir)`` is the geometric delay of antenna *a* toward ``dir`` and
# ``s_src[t]`` is the source's complex baseband tone.  This isolates the spatial
# physics the beamformer/nuller act on; swap in the setigen path when PFB/quant
# realism matters.


@dataclass
class SimConfig:
    """Array + band parameters for the directional narrowband simulation."""

    num_antennas: int = 32
    array_radius_m: float = 300.0
    sample_rate: float = 3.0e9
    fch1_hz: float = 1.4e9
    num_branches: int = 1024
    num_chans: int = 16
    n_time: int = 4096
    noise_std: float = 1.0
    target_ra_deg: float = 128.836
    target_dec_deg: float = -45.176
    timestamp: float = 1.70e9        # fixed UNIX time → reproducible geometry
    seed: int = 42

    chan_bw: float = field(init=False)

    def __post_init__(self) -> None:
        self.chan_bw = self.sample_rate / self.num_branches

    def channel_freq(self, ch: int, offset_frac: float = 0.0) -> float:
        """Absolute frequency of a point in coarse channel *ch* (offset in [0,1))."""
        return self.fch1_hz + ch * self.chan_bw + offset_frac * self.chan_bw


def build_geometry(cfg: SimConfig):
    """Return ``(ref_ant, antennas)`` for the array (reuses ``MAA`` layout)."""
    return maa().build_antennas(cfg.num_antennas, cfg.array_radius_m, cfg.seed)


def _radec(ra_deg: float, dec_deg: float) -> katpoint.Target:
    return katpoint.construct_radec_target(np.deg2rad(ra_deg), np.deg2rad(dec_deg))


def steering(cfg: SimConfig, ref_ant, antennas, ra_deg: float, dec_deg: float,
            freq_hz: float) -> np.ndarray:
    """Narrowband steering vector ``a_i = exp(-2πi f τ_i)`` toward a direction.

    Args:
        cfg:       Simulation config (for timestamp / sample_rate).
        ref_ant:   Reference antenna.
        antennas:  Antenna list.
        ra_deg,dec_deg: Source direction.
        freq_hz:   Absolute RF frequency the delays are evaluated at.

    Returns:
        Complex steering vector ``(A,)`` (unit-modulus entries).
    """
    _, delays_sec = maa().compute_delays(
        ref_ant, antennas, _radec(ra_deg, dec_deg), cfg.timestamp, cfg.sample_rate
    )
    return np.exp(-2j * np.pi * freq_hz * delays_sec)


def offset_direction(cfg: SimConfig, d_ra_deg: float = 0.0,
                     d_dec_deg: float = 0.0) -> tuple[float, float]:
    """A direction offset from the target pointing by ``(d_ra_deg, d_dec_deg)``.

    The RA offset is scaled by ``1/cos(dec)`` so the offset is a true angular
    separation on the sky.
    """
    dec = cfg.target_dec_deg + d_dec_deg
    ra = cfg.target_ra_deg + d_ra_deg / np.cos(np.deg2rad(cfg.target_dec_deg))
    return ra, dec


def simulate_cube(cfg: SimConfig, sources: list[dict]) -> tuple[np.ndarray, dict]:
    """Synthesise a ``(T, F, A)`` antenna cube from directional sources + noise.

    Each source dict has:
        * ``ra_deg``, ``dec_deg`` -- arrival direction;
        * ``channel``             -- coarse channel it lands in;
        * ``offset_frac``         -- fine position within the channel (∈[0,1)),
                                     setting its FFT bin (= ``offset_frac·T``);
        * ``level``               -- per-antenna tone amplitude;
        * ``drift_bins``          -- optional linear drift, in fine bins over the
                                     whole record (default 0);
        * ``name``                -- label for the ground truth.

    Args:
        cfg:     Simulation config.
        sources: List of source dicts (see above).

    Returns:
        ``(cube, truth)``: complex64 cube ``(n_time, num_chans, num_antennas)``
        and a ground-truth dict holding the array geometry and, per source, its
        steering vector, channel, and fine bin.
    """
    rng = np.random.default_rng(cfg.seed)
    ref_ant, antennas = build_geometry(cfg)
    T, F, A = cfg.n_time, cfg.num_chans, cfg.num_antennas

    # Circularly-symmetric complex noise, variance noise_std² per antenna.
    s = cfg.noise_std / np.sqrt(2.0)
    cube = (rng.normal(0, s, (T, F, A)) + 1j * rng.normal(0, s, (T, F, A))).astype(np.complex64)

    t = np.arange(T)
    truth = {"ref_ant": ref_ant, "antennas": antennas, "sources": []}
    for src in sources:
        ch = src["channel"]
        off = src["offset_frac"]
        f_abs = cfg.channel_freq(ch, off)
        a = steering(cfg, ref_ant, antennas, src["ra_deg"], src["dec_deg"], f_abs)  # (A,)

        drift = src.get("drift_bins", 0.0)
        # Baseband tone: fine bin = offset_frac·T, plus optional linear drift.
        phase = 2 * np.pi * (off * t + 0.5 * drift * t * t / T)
        s_t = src["level"] * np.exp(1j * phase)                                     # (T,)
        cube[:, ch, :] += (s_t[:, None] * a[None, :]).astype(np.complex64)

        truth["sources"].append({
            "name": src.get("name", "src"),
            "ra_deg": src["ra_deg"], "dec_deg": src["dec_deg"],
            "channel": ch, "offset_frac": off, "level": src["level"],
            "fine_bin": int(round(off * T)), "steering": a,
        })
    return cube, truth


# ---------------------------------------------------------------------------
# Covariance / eigenstructure
# ---------------------------------------------------------------------------

def antenna_cov(x: np.ndarray) -> np.ndarray:
    """Antenna covariance ``R = (1/(T-1)) Σ_t x_t x_tᴴ`` of a ``(T, A)`` slice.

    Uses ``np.cov`` conventions (mean-subtracted, normalised by ``T-1``), matching
    what the MR-KLT / :class:`seti_klt.KLT` use.

    Args:
        x: Complex antenna voltages for one channel, shape ``(T, A)``.

    Returns:
        Hermitian covariance matrix ``(A, A)``.
    """
    return np.cov(np.asarray(x), rowvar=False)


def eig_desc(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition sorted by **descending** eigenvalue.

    Args:
        R: Hermitian matrix ``(A, A)``.

    Returns:
        ``(vals, vecs)`` with ``vals`` descending and ``vecs[:, k]`` the
        eigenvector for ``vals[k]``.
    """
    vals, vecs = np.linalg.eigh(np.asarray(R))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


# ---------------------------------------------------------------------------
# Beamforming
# ---------------------------------------------------------------------------

def beamform(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply beamformer weights to a ``(T, A)`` slice → ``(T,)`` beam.

    The beam is ``b_t = wᴴ x_t`` (the standard convention where ``w`` is the
    steering vector of the look direction).

    Args:
        x: Antenna voltages ``(T, A)``.
        w: Complex weights ``(A,)``.

    Returns:
        Complex beam time series ``(T,)``.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    return x @ np.conj(w)


def beam_response(w: np.ndarray, a: np.ndarray) -> float:
    """Power gain ``|wᴴ a|²`` of weights *w* toward a direction with steering *a*.

    A perfect null toward *a* gives 0; a distortionless response toward the look
    direction (``w = a / ‖a‖²`` convention) gives 1.

    Args:
        w: Beamformer weights ``(A,)``.
        a: Steering vector of the probed direction ``(A,)``.

    Returns:
        ``|wᴴ a|²`` as a float.
    """
    return float(np.abs(np.vdot(w, a)) ** 2)


# ---------------------------------------------------------------------------
# Covariance-level nulling
# ---------------------------------------------------------------------------

def interference_subspace(
    R: np.ndarray,
    a_t: np.ndarray,
    *,
    rank: int | None = None,
    energy: float = 0.90,
    target_align: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the interference subspace from a covariance, excluding the target.

    Diagonalises ``R``, drops the eigenvector most aligned with the (known) target
    steering ``a_t``, and returns the remaining dominant eigenvectors as the
    interference subspace ``U_i``. This is the MR-KLT eigenstructure with the
    target mode removed.

    Args:
        R:            Antenna covariance ``(A, A)`` of the target channel.
        a_t:          Target steering vector ``(A,)`` (from a target-only
                      reference or from the array geometry).
        rank:         Fixed number of interference eigenvectors to keep. When
                      ``None``, the rank is chosen automatically as the leading
                      eigenvectors (target excluded) holding *energy* of the
                      non-target spike energy.
        energy:       Fraction of non-target spike energy to capture when
                      *rank* is ``None``.
        target_align: An eigenvector counts as "the target" if its normalised
                      overlap ``|vᴴ a_t| / ‖a_t‖`` exceeds this — used to pick
                      which single eigenvector to exclude.

    Returns:
        ``(U_i, info)``: the ``(A, r)`` interference basis and a small dict with
        the eigenvalues, the excluded target index, and the chosen rank.
    """
    R = np.asarray(R)
    a_t = np.asarray(a_t)
    a_hat = a_t / np.linalg.norm(a_t)
    vals, vecs = eig_desc(R)

    # Identify the eigenvector that is the target (largest overlap with a_t).
    overlaps = np.abs(vecs.conj().T @ a_hat)          # (A,)
    tgt_idx = int(np.argmax(overlaps))

    # Candidate interference eigenvectors: everything except the target mode.
    cand = [k for k in range(vecs.shape[1]) if k != tgt_idx]
    cand_vals = vals[cand]

    if rank is None:
        # Keep leading non-target eigenvectors above the noise floor holding
        # `energy` of the non-target spike energy. Noise floor ≈ median eigval.
        noise = np.median(vals)
        spike = np.clip(cand_vals - noise, 0.0, None)
        if spike.sum() <= 0:
            r = 0
        else:
            csum = np.cumsum(spike) / spike.sum()
            r = int(np.searchsorted(csum, energy) + 1)
            r = min(r, len(cand))
    else:
        r = min(int(rank), len(cand))

    U_i = vecs[:, cand[:r]] if r > 0 else np.zeros((vecs.shape[0], 0), dtype=vecs.dtype)
    info = {
        "eigenvalues": vals,
        "target_index": tgt_idx,
        "target_overlap": float(overlaps[tgt_idx]),
        "rank": r,
    }
    return U_i, info


def null_weights(a_t: np.ndarray, U_i: np.ndarray) -> np.ndarray:
    """Project the target steering off the interference subspace and normalise.

    ``w = (I - U_i U_iᴴ) a_t``, then scaled so ``wᴴ a_t = 1`` (distortionless
    toward the target). The result places nulls along every column of ``U_i``.

    Args:
        a_t: Target steering vector ``(A,)``.
        U_i: Interference basis ``(A, r)`` (orthonormal columns from
             :func:`interference_subspace`).

    Returns:
        Nulled beamformer weights ``(A,)``.
    """
    a_t = np.asarray(a_t)
    if U_i.shape[1] == 0:
        w = a_t.copy()
    else:
        w = a_t - U_i @ (U_i.conj().T @ a_t)          # orthogonal projection
    denom = np.vdot(w, a_t)                            # wᴴ a_t
    if abs(denom) > 0:
        w = w / np.conj(denom)                         # distortionless: wᴴ a_t = 1
    return w


def mvdr_weights(R: np.ndarray, a_t: np.ndarray, diagonal_load: float = 1e-3) -> np.ndarray:
    """Minimum-variance distortionless-response (Capon) beamformer weights.

    ``w = R⁻¹ a_t / (a_tᴴ R⁻¹ a_t)`` with a little diagonal loading for
    stability. MVDR automatically nulls interferers present in ``R`` while
    keeping unit gain toward ``a_t`` — the "optimal" covariance null.

    Args:
        R:             Antenna covariance ``(A, A)``.
        a_t:           Target steering vector ``(A,)``.
        diagonal_load: Loading added as ``diagonal_load · tr(R)/A · I``.

    Returns:
        MVDR weights ``(A,)`` (satisfying ``wᴴ a_t = 1``).
    """
    R = np.asarray(R)
    a_t = np.asarray(a_t)
    A = R.shape[0]
    Rl = R + diagonal_load * (np.trace(R).real / A) * np.eye(A, dtype=R.dtype)
    Rinv_a = np.linalg.solve(Rl, a_t)
    return Rinv_a / np.conj(np.vdot(a_t, Rinv_a))


# ---------------------------------------------------------------------------
# Beam SNR
# ---------------------------------------------------------------------------

def beam_pattern(
    cfg: SimConfig, ref_ant, antennas, w: np.ndarray, freq_hz: float,
    d_dec_deg: np.ndarray, d_ra_deg: float = 0.0,
) -> np.ndarray:
    """Power response ``|wᴴ a(θ)|²`` of weights *w* across a strip of sky.

    Sweeps the declination offset from the target and returns the beam gain,
    so the target main lobe and the interferer null are directly visible.

    Args:
        cfg:        Simulation config.
        ref_ant:    Reference antenna.
        antennas:   Antenna list.
        w:          Beamformer weights ``(A,)``.
        freq_hz:    Frequency to evaluate the steering at.
        d_dec_deg:  Array of declination offsets from the target (degrees).
        d_ra_deg:   Fixed RA offset (degrees).

    Returns:
        Beam power gain at each offset, shape ``d_dec_deg.shape``.
    """
    g = np.empty(len(d_dec_deg), dtype=float)
    for k, dd in enumerate(d_dec_deg):
        ra, dec = offset_direction(cfg, d_ra_deg=d_ra_deg, d_dec_deg=float(dd))
        a = steering(cfg, ref_ant, antennas, ra, dec, freq_hz)
        g[k] = beam_response(w, a)
    return g


def beam_pattern_2d(
    cfg: SimConfig, ref_ant, antennas, w: np.ndarray, freq_hz: float,
    d_ra_deg: np.ndarray, d_dec_deg: np.ndarray,
) -> np.ndarray:
    """2-D beam power response ``|wᴴ a(θ)|²`` over a grid of sky offsets.

    Evaluates the beam gain on a ``(ΔRA × Δdec)`` grid around the target, so the
    full main-lobe shape (e.g. its elongation for an irregular array) is visible,
    not just a 1-D cut.

    Args:
        cfg:        Simulation config.
        ref_ant:    Reference antenna.
        antennas:   Antenna list.
        w:          Beamformer weights ``(A,)``.
        freq_hz:    Frequency to evaluate the steering at.
        d_ra_deg:   1-D array of RA offsets from the target [deg] (columns).
        d_dec_deg:  1-D array of declination offsets [deg] (rows).

    Returns:
        Gain map of shape ``(len(d_dec_deg), len(d_ra_deg))``.
    """
    G = np.empty((len(d_dec_deg), len(d_ra_deg)), dtype=float)
    for i, dd in enumerate(d_dec_deg):
        for j, dr in enumerate(d_ra_deg):
            ra, dec = offset_direction(cfg, d_ra_deg=float(dr), d_dec_deg=float(dd))
            a = steering(cfg, ref_ant, antennas, ra, dec, freq_hz)
            G[i, j] = beam_response(w, a)
    return G


def beam_grid(
    cfg: SimConfig, ref_ant, antennas, freq_hz: float,
    r_off_deg: float, n_off: int,
) -> list[dict]:
    """Build an **ON** beam at the target plus a ring of **OFF** beams around it.

    This is the literal ON/OFF geometry: one beam on the target and ``n_off``
    contiguous beams on a circle of radius ``r_off_deg`` around it. The OFF beams
    act as an interference reference — they see the RFI but (mostly) not the
    target, because the target sits in their sidelobes.

    Args:
        cfg:        Simulation config (supplies the target pointing).
        ref_ant:    Reference antenna.
        antennas:   Antenna list.
        freq_hz:    Frequency the steering vectors are evaluated at.
        r_off_deg:  Radius of the OFF ring, in degrees. Pick ≳1.5 beamwidths so
                    target leakage into the OFF beams stays small.
        n_off:      Number of OFF beams on the ring.

    Returns:
        List of beam dicts (``ON`` first, then ``OFF0..OFFn``), each with
        ``name``, ``d_ra``/``d_dec`` offsets [deg], the ``steer`` vector, and
        distortionless weights ``w`` (``wᴴ a = 1``).
    """
    beams = []
    a_on = steering(cfg, ref_ant, antennas, cfg.target_ra_deg, cfg.target_dec_deg, freq_hz)
    beams.append(dict(name="ON", d_ra=0.0, d_dec=0.0, steer=a_on,
                      w=a_on / np.vdot(a_on, a_on)))
    for k in range(n_off):
        phi = 2 * np.pi * k / n_off
        d_dec, d_ra = r_off_deg * np.cos(phi), r_off_deg * np.sin(phi)
        ra, dec = offset_direction(cfg, d_ra_deg=d_ra, d_dec_deg=d_dec)
        a = steering(cfg, ref_ant, antennas, ra, dec, freq_hz)
        beams.append(dict(name=f"OFF{k}", d_ra=d_ra, d_dec=d_dec, steer=a,
                          w=a / np.vdot(a, a)))
    return beams


def ls_subtract(b_on: np.ndarray, ref: np.ndarray):
    """Least-squares removal of one or several reference waveforms from the ON beam.

    Classic reference-channel canceller. The references (RFI waveforms
    reconstructed from the OFF beams) enter the ON beam with unknown, complex
    amplitude/phase, which are fitted.

    * A single reference ``(T,)`` → scalar ``α = rᴴb / rᴴr``, residual ``b − α r``.
    * Several references ``(T, r)`` → vector ``α`` from the multi-reference least
      squares ``argminₐ ‖b − R α‖²`` (each interferer gets its own coefficient),
      residual ``b − R α``.

    Args:
        b_on: ON-beam time series ``(T,)``.
        ref:  Reference waveform(s), ``(T,)`` or ``(T, r)``.

    Returns:
        ``(b_clean, alpha)`` — ``alpha`` is a scalar for a 1-D reference, else a
        length-``r`` vector.
    """
    R = np.asarray(ref)
    if R.ndim == 1:
        denom = np.vdot(R, R)
        alpha = (np.vdot(R, b_on) / denom) if abs(denom) > 0 else 0.0 + 0.0j
        return b_on - alpha * R, alpha
    if R.shape[1] == 0:                       # no references → nothing to remove
        return np.asarray(b_on).copy(), np.zeros(0, dtype=complex)
    alpha, *_ = np.linalg.lstsq(R, b_on, rcond=None)
    return b_on - R @ alpha, alpha


def beam_snr(beam: np.ndarray, *, use_gpu: bool = False) -> tuple[float, float]:
    """Narrowband spectral SNR of a 1-D complex beam (peak bin / noise floor).

    Args:
        beam:    Complex beam time series ``(T,)``.
        use_gpu: Run the FFT on the GPU via CuPy when available.

    Returns:
        ``(snr_linear, peak_bin_fraction)``.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    b = xp.asarray(beam)
    b = b - b.mean()
    p = xp.abs(xp.fft.fft(b)) ** 2
    p[0] = 0.0
    peak = int(xp.argmax(p))
    floor = float(xp.median(p[1:]))
    snr = float(p[peak]) / floor if floor > 0 else float("inf")
    return snr, peak / p.shape[0]


# ---------------------------------------------------------------------------
# Full analysis pipeline (no data generation) — for benchmarking
# ---------------------------------------------------------------------------

def process_channel(
    x_ch: np.ndarray,
    Wbeams: np.ndarray,
    *,
    window_size: int = 64,
    n_off_modes=1,
    kn_alpha: float = 0.5,
    on_steer: np.ndarray | None = None,
    soi_overlap: float = 0.5,
    logger=None,
    timed: bool = False,
) -> dict:
    """Run the ON/OFF → MR-KLT → subtract → C-KLT pipeline on one channel.

    This is the **analysis-only** pipeline (no signal generation): everything
    that would run per channel on real recorded data. Refactored out of the
    notebook so it can be called and benchmarked cleanly.

    Steps and their leading cost (A antennas, B beams, T samples, W window):
      * beamform  ``B = x Wᴴ``                    -- O(T·A·B)
      * MR-KLT on the OFF beams + reconstruction  -- O(T·B² + B³)
      * least-squares subtraction                 -- O(T)
      * C-KLT on the cleaned ON beam              -- O(T·W + W³)

    Args:
        x_ch:        Antenna voltages for one channel, ``(T, A)``.
        Wbeams:      Beam weights ``(A, 1+n_off)`` (column 0 = ON, rest = OFF).
        window_size: C-KLT window ``W``.
        n_off_modes: How many MR-KLT interference modes to reconstruct and remove.
                     An int fixes the rank; ``"auto"`` lets **Kritchman & Nadler**
                     (:func:`seti_klt.KN_rankEst`) estimate the number of RFI from
                     the OFF-beam eigenvalues. Each detected mode is removed with
                     its own least-squares coefficient (multi-reference canceller).
        kn_alpha:    K&N false-alarm level per test (percent). Only used for
                     ``n_off_modes="auto"``.
        logger:      Quiet logger for the KLT objects.
        timed:       When ``True``, also return a ``times`` dict (seconds per step).

    Returns:
        Dict with ``b_on``, ``b_clean``, ``alpha``, ``klt_on`` (the fitted C-KLT),
        ``off_eigenvalues``, ``n_off_used`` (rank actually removed), and, if
        *timed*, ``times``.
    """
    import time as _time
    t = {}

    # --- beamform (ON + OFF) ---
    s = _time.perf_counter()
    Bmat = x_ch @ np.conj(Wbeams)                 # (T, 1+n_off)
    b_on, B_off = Bmat[:, 0], Bmat[:, 1:]
    t["beamform"] = _time.perf_counter() - s

    # --- MR-KLT on the OFF beams + interference reconstruction ---
    s = _time.perf_counter()
    klt_off = KLT(logger=logger)
    klt_off.load_signal_from_array(np.ascontiguousarray(B_off))
    klt_off.apply_mrklt(n_eigenvectors=1)          # need the full eigenspectrum
    off_vals = np.asarray(klt_off.get_eigenvalues(), dtype=float)
    vecs = klt_off.get_eigenvectors()

    if n_off_modes == "auto":
        # Kritchman & Nadler → number of RFI. K&N assumes *white* noise, but the
        # OFF beams overlap (shared antennas) so their noise is coloured with the
        # known covariance σ²·(Wᴴ_off W_off). We therefore feed K&N the
        # generalised eigenvalues of (R_beam, Wᴴ_off W_off), which whiten that
        # colour: the noise floor becomes flat (σ²) and only true RFI stay as
        # spikes. np.cov normalises by (T-1) ⇒ n=T-1; complex ⇒ β=2.
        import scipy.linalg as _sla
        T_ = B_off.shape[0]
        W_off = Wbeams[:, 1:]
        G = W_off.conj().T @ W_off
        G = G + 1e-6 * (np.trace(G).real / G.shape[0]) * np.eye(G.shape[0])
        gen_vals = _sla.eigh(klt_off.covariance_matrix, G, eigvals_only=True)[::-1]
        n_det, _ = KN_rankEst(np.asarray(gen_vals, dtype=float), n=T_ - 1,
                              beta=2, alpha=kn_alpha)
        r = int(n_det)
    else:
        r = int(n_off_modes)
    r = max(0, min(r, B_off.shape[1]))

    # The SOI leaks weakly into the OFF beams, so it can appear as one of the
    # detected modes. Removing it would self-subtract the signal, so we drop the
    # detected mode whose OFF-beam signature matches the target direction (its
    # signature there is W_offᴴ·a_on, known from the geometry) and keep only the
    # genuine RFI modes as the reference.
    n_detected = r
    soi_excluded = False
    keep = list(range(r))
    if r >= 1 and on_steer is not None:
        W_off = Wbeams[:, 1:]
        c_soi = W_off.conj().T @ np.asarray(on_steer)
        nrm = np.linalg.norm(c_soi)
        if nrm > 0:
            c_soi = c_soi / nrm
            ov = np.abs(vecs[:, :r].conj().T @ c_soi)          # overlap of each mode with SOI
            j = int(np.argmax(ov))
            if ov[j] > soi_overlap:
                keep = [k for k in range(r) if k != j]
                soi_excluded = True

    if keep:
        V = vecs[:, keep]
        coeff = (B_off - B_off.mean(axis=0)) @ np.conj(V)      # (T, len(keep)) one col / RFI
        ref = coeff[:, 0] if len(keep) == 1 else coeff         # scalar- or multi-reference
    else:
        ref = np.zeros((B_off.shape[0], 0), dtype=B_off.dtype)  # no RFI to remove
    r = len(keep)
    t["mrklt+recon"] = _time.perf_counter() - s

    # --- least-squares subtraction (one coefficient per detected RFI) ---
    s = _time.perf_counter()
    b_clean, alpha = ls_subtract(b_on, ref)
    t["ls_subtract"] = _time.perf_counter() - s

    # --- C-KLT on the cleaned ON beam ---
    s = _time.perf_counter()
    klt_on = KLT(logger=logger)
    klt_on.load_signal_from_array(np.ascontiguousarray(b_clean))
    klt_on.apply_cklt(window_size=window_size, n_eigenvectors=1)
    t["cklt"] = _time.perf_counter() - s

    out = {"b_on": b_on, "b_clean": b_clean, "alpha": alpha, "klt_on": klt_on,
           "off_eigenvalues": off_vals, "n_off_used": r,
           "n_off_detected": n_detected, "soi_excluded": soi_excluded}
    if timed:
        t["total"] = sum(t.values())
        out["times"] = t
    return out


# ---------------------------------------------------------------------------
# One-call experiment harness (change parameters -> get diagnostics)
# ---------------------------------------------------------------------------

def _bin_power(beam: np.ndarray, frac: float) -> float:
    """Peak power in a small window around fine-bin fraction *frac* of a beam."""
    T = len(beam)
    P = np.abs(np.fft.fft(beam - beam.mean())) ** 2
    k = int(round(frac * T)) % T
    return max(P[(k + d) % T] for d in range(-2, 3))


def run_experiment(
    *,
    sources: list[dict],
    num_antennas: int = 64,
    array_radius_m: float = 1000.0,
    num_branches: int = 16384,
    num_chans: int = 32,
    n_time: int = 4096,
    noise_std: float = 1.0,
    seed: int = 1,
    channel: int = 8,
    off_radius_deg: float = 0.05,
    n_off: int = 8,
    n_off_modes=1,
    kn_alpha: float = 0.5,
    cklt_window: int = 64,
    logger=None,
) -> dict:
    """Run the whole ON/OFF -> MR-KLT -> subtract -> C-KLT pipeline from parameters.

    A single entry point so different physical situations can be tried by editing
    one parameter block. Sources are given as **offsets from the target pointing**.

    Each entry of *sources* is a dict:
        * ``name``  -- label;
        * ``role``  -- ``"signal"`` (the SOI to recover) or ``"interferer"`` (RFI);
        * ``d_ra``, ``d_dec`` -- direction offset from the target [deg];
        * ``bin``   -- fine-bin fraction in the channel; sets its FFT bin;
        * ``level`` -- per-antenna amplitude;
        * ``drift`` -- optional linear drift, in fine bins over the record.

    Returns:
        A diagnostics dict (pass it to :func:`experiment_report`).
    """
    cfg = SimConfig(num_antennas=num_antennas, array_radius_m=array_radius_m,
                    num_branches=num_branches, num_chans=num_chans, n_time=n_time,
                    noise_std=noise_std, seed=seed)
    ref, ants = build_geometry(cfg)
    fc = cfg.channel_freq(channel, 0.25)

    specs = []
    for src in sources:
        ra, dec = offset_direction(cfg, d_ra_deg=src.get("d_ra", 0.0),
                                   d_dec_deg=src.get("d_dec", 0.0))
        specs.append(dict(name=src["name"], ra_deg=ra, dec_deg=dec, channel=channel,
                          offset_frac=src["bin"], level=src["level"],
                          drift_bins=src.get("drift", 0.0)))
    cube, truth = simulate_cube(cfg, specs)
    x = cube[:, channel, :]

    beams = beam_grid(cfg, ref, ants, fc, r_off_deg=off_radius_deg, n_off=n_off)
    Wbeams = np.stack([b["w"] for b in beams], axis=1)

    out = process_channel(x, Wbeams, window_size=cklt_window,
                          n_off_modes=n_off_modes, kn_alpha=kn_alpha,
                          on_steer=beams[0]["steer"], logger=logger, timed=True)
    b_on, b_clean = out["b_on"], out["b_clean"]

    A_on, A_off = Wbeams[:, 0], Wbeams[:, 1:]
    srcinfo = []
    for src, ts in zip(sources, truth["sources"]):
        a = ts["steering"]; frac = ts["offset_frac"]
        off_gains = [beam_response(A_off[:, k], a) for k in range(A_off.shape[1])]
        d = dict(
            name=src["name"], role=src.get("role", "interferer"),
            d_ra=src.get("d_ra", 0.0), d_dec=src.get("d_dec", 0.0), bin=frac,
            level=src["level"],
            gain_on=beam_response(A_on, a), gain_off_max=max(off_gains),
            P_before=_bin_power(b_on, frac), P_after=_bin_power(b_clean, frac),
        )
        d["suppression_db"] = (10 * np.log10(d["P_before"] / d["P_after"])
                               if d["P_after"] > 0 else np.inf)
        srcinfo.append(d)

    res = dict(cfg=cfg, beams=beams, truth=truth, sources=srcinfo,
               b_on=b_on, b_clean=b_clean, alpha=out["alpha"],
               off_eigenvalues=out["off_eigenvalues"], times=out["times"],
               channel=channel, off_radius_deg=off_radius_deg, n_off=n_off,
               n_off_used=out["n_off_used"], n_off_detected=out["n_off_detected"],
               soi_excluded=out["soi_excluded"],
               n_interferers_true=sum(1 for d in srcinfo if d["role"] == "interferer"))

    sig = [d for d in srcinfo if d["role"] == "signal"]
    itf = [d for d in srcinfo if d["role"] == "interferer"]
    if sig and itf:
        s0 = sig[0]
        strongest = max(itf, key=lambda d: d["P_before"])
        res["si_before"] = 10 * np.log10(s0["P_before"] / strongest["P_before"])
        res["si_after"] = 10 * np.log10(s0["P_after"] / strongest["P_after"])
        res["soi_loss_db"] = 10 * np.log10(s0["P_before"] / s0["P_after"])
        res["rfi_suppression_db"] = strongest["suppression_db"]

    rec = np.asarray(out["klt_on"].get_reconstructed_signal())
    Pr = np.abs(np.fft.fft(rec - rec.mean())) ** 2
    Pr[0] = 0.0
    peak = int(np.argmax(Pr)) / len(Pr)
    nearest = min(srcinfo, key=lambda d: min(abs(peak - d["bin"]), 1 - abs(peak - d["bin"])))
    res["cklt_peak"] = peak
    res["cklt_lock"] = (nearest["name"]
                        if min(abs(peak - nearest["bin"]), 1 - abs(peak - nearest["bin"])) < 0.02
                        else "?")
    return res


def experiment_report(res: dict, plot: bool = True):
    """Pretty-print the diagnostics of :func:`run_experiment` (and optionally plot)."""
    import matplotlib.pyplot as plt

    cfg = res["cfg"]
    ev = res["off_eigenvalues"]
    ratio = ev[0] / ev[1] if len(ev) > 1 and ev[1] > 0 else np.inf
    print(f"array {cfg.num_antennas} ant, {cfg.array_radius_m:.0f} m | "
          f"channel {res['channel']} ({cfg.chan_bw/1e3:.0f} kHz) | "
          f"OFF ring {res['off_radius_deg']} deg x {res['n_off']}")
    print(f"\n{'source':8s}{'role':11s}{'d_ra':>7s}{'d_dec':>7s}{'bin':>6s}"
          f"{'g@ON':>8s}{'g@OFFmax':>9s}{'suppr[dB]':>10s}")
    for d in res["sources"]:
        print(f"{d['name']:8s}{d['role']:11s}{d['d_ra']:+7.3f}{d['d_dec']:+7.3f}"
              f"{d['bin']:6.2f}{d['gain_on']:8.4f}{d['gain_off_max']:9.4f}{d['suppression_db']:10.1f}")
    if "si_before" in res:
        print(f"\nS/I in ON : {res['si_before']:+.1f} dB  ->  {res['si_after']:+.1f} dB   "
              f"(RFI suppressed {res['rfi_suppression_db']:.1f} dB, SOI loss {res['soi_loss_db']:+.2f} dB)")
    a = np.atleast_1d(res["alpha"])
    a_str = ", ".join(f"{v:.3f}" for v in a) if a.size else "-"
    soi_note = " (incl. SOI leakage → excluded)" if res.get("soi_excluded") else ""
    print(f"K&N on OFF : {res['n_off_detected']} spatial modes detected{soi_note} → "
          f"{res['n_off_used']} RFI removed  (true interferers: {res['n_interferers_true']})")
    print(f"MR-KLT OFF : dominant/2nd = {ratio:.0f}x | LS alpha = [{a_str}]")
    print(f"C-KLT on cleaned ON locks onto: {res['cklt_lock']} (peak bin {res['cklt_peak']:.3f})")
    print(f"cost       : {res['times']['total']*1e3:.1f} ms/channel (KLT-object pipeline)")

    if not plot:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for beam, col, lab in [(res["b_on"], "#888888", "ON before"),
                           (res["b_clean"], "#009E73", "ON after")]:
        P = np.abs(np.fft.fft(beam - beam.mean())) ** 2
        axes[0].semilogy(np.fft.fftshift(np.fft.fftfreq(len(beam))),
                         np.fft.fftshift(P) + 1e-9, color=col,
                         lw=1.5 if "after" in lab else 1.0, label=lab)
    for d in res["sources"]:
        c = "#0072B2" if d["role"] == "signal" else "#D55E00"
        axes[0].axvline(d["bin"], color=c, ls=":", lw=1.2)
    axes[0].set_xlim(0, 0.5)
    axes[0].set_xlabel("fine-bin fraction")
    axes[0].set_ylabel("beam power")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("ON beam: before vs after subtraction")
    axes[1].plot(np.arange(len(ev)), ev, "o-", color="#888888")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("eigenvalue index")
    axes[1].set_ylabel("eigenvalue")
    axes[1].set_title("MR-KLT eigenspectrum (OFF beams)")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stress-test: robustness of the auto pipeline across randomised scenarios
# ---------------------------------------------------------------------------

def _random_sources(n_rfi, rfi_level, off_radius_deg, rng, soi_level=1.0):
    """SOI at the target + *n_rfi* interferers at random azimuth on the OFF ring,
    all on distinct fine bins (so each is separable in the beam spectrum)."""
    grid = np.round(np.arange(0.12, 0.45, 0.03), 3)
    bins = rng.choice(grid, size=n_rfi + 1, replace=False)
    src = [dict(name="SOI", role="signal", d_ra=0.0, d_dec=0.0,
                bin=float(bins[0]), level=soi_level)]
    for k in range(n_rfi):
        phi = rng.uniform(0, 2 * np.pi)
        src.append(dict(name=f"RFI{k}", role="interferer",
                        d_ra=off_radius_deg * np.sin(phi),
                        d_dec=off_radius_deg * np.cos(phi),
                        bin=float(bins[k + 1]), level=rfi_level))
    return src


def stress_point(*, n_rfi=1, rfi_level=15.0, off_radius_deg=0.05, kn_alpha=1.0,
                 n_trials=15, seed0=0, logger=None, **base):
    """Run *n_trials* randomised realisations of one scenario and aggregate.

    Each trial re-randomises the noise, the interferer directions (random azimuth
    on the OFF ring) and the fine bins, runs :func:`run_experiment` with automatic
    K&N counting, and scores the outcome against the injected truth.

    Args:
        n_rfi:          True number of interferers.
        rfi_level:      Per-antenna interferer amplitude (SOI is fixed at 1.0).
        off_radius_deg: OFF ring radius.
        kn_alpha:       K&N false-alarm level [%].
        n_trials:       Realisations to average over.
        seed0:          Base RNG seed for reproducibility.
        base:           Fixed array/channel knobs forwarded to
                        :func:`run_experiment` (num_antennas, n_time, …).

    Returns:
        Dict with ``count_acc`` (fraction of trials whose removed-RFI count equals
        *n_rfi*), ``over``/``under`` (mean over-/under-count), ``soi_loss``,
        ``si_gain``, ``rfi_supp`` (means over trials), and ``n_trials``.
    """
    rng = np.random.default_rng(seed0)
    correct, over, under = [], [], []
    soi_loss, si_gain, supp = [], [], []
    for _ in range(n_trials):
        src = _random_sources(n_rfi, rfi_level, off_radius_deg, rng)
        res = run_experiment(sources=src, off_radius_deg=off_radius_deg,
                             n_off_modes="auto", kn_alpha=kn_alpha,
                             seed=int(rng.integers(1_000_000_000)),
                             logger=logger, **base)
        used = res["n_off_used"]
        correct.append(used == n_rfi)
        over.append(max(0, used - n_rfi))
        under.append(max(0, n_rfi - used))
        if "soi_loss_db" in res:
            soi_loss.append(res["soi_loss_db"])
            si_gain.append(res["si_after"] - res["si_before"])
            supp.append(res["rfi_suppression_db"])
    m = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return dict(count_acc=float(np.mean(correct)), over=m(over), under=m(under),
                soi_loss=m(soi_loss), si_gain=m(si_gain), rfi_supp=m(supp),
                n_trials=n_trials)
