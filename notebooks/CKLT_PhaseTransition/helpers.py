"""
helpers.py
==========
Single-receiver **C-KLT phase-transition sweep**.

This module drives an injection–recovery experiment for the Covariance KLT
(C-KLT) pipeline: it generates GUPPI raw voltage files with ``setigen`` (one
synthetic receiver), injects a drifting CW tone at a controlled SNR, then runs
the C-KLT reconstruction and the Kritchman & Nadler (2009) rank estimator to
decide how many signal eigenvalues are present.  The goal is to map, as a
function of the injected SNR, when the signal's dominant eigenvalue crosses the
Baik–Ben Arous–Péché (BBP) phase transition and becomes detectable — and to
compare that empirical behaviour against the theoretical threshold ``√γ``.

Why C-KLT (and not MR-KLT)
--------------------------
The MR-KLT notebooks work on ``(T, F, P, A)`` multi-antenna cubes.  Here we
deliberately simulate a **single receiver**: a 1-D complex baseband voltage
stream.  The C-KLT chops that stream into ``K`` non-overlapping windows of
``W`` samples, builds the ``W×W`` sample covariance, and looks for a spike.  The
RMT roles map cleanly onto the MeerKAT stamp dimensions:

    C-KLT  W (window size)  ↔  MR-KLT  N_A (antennas)   → covariance dimension
    C-KLT  K (# windows)    ↔  MR-KLT  timesteps        → # realisations
    γ = W / (K - 1)         (same aspect ratio in both)

With the defaults below (``W = 63``, ``K = 293``) we get ``γ = 63/292 ≈ 0.216``,
matching the real MeerKAT beamformed stamps used elsewhere in the project.

The "synchronous phase" caveat
------------------------------
The C-KLT subtracts the per-column (per-window-position) mean before forming the
covariance.  If the tone completes an **integer** number of cycles per window,
every window is an identical copy of the signal, the column mean absorbs all of
it, and the centering step erases it.  We therefore (a) inject at a *fractional*
baseband offset and (b) reject offsets whose cycles-per-window land too close to
an integer.  See ``notebooks/Injected_Signal/CKLT_Injected_Signal.ipynb`` for a
longer discussion.

Public API
----------
* :class:`CKLTConfig`       -- all fixed instrument / processing parameters.
* :func:`generate_cklt_raw` -- write one raw file with an injected tone.
* :func:`analyze_cklt`      -- run C-KLT + K&N + phase-transition diagnostics.
* :func:`spectral_snr`      -- narrowband spectral SNR of a 1-D voltage vector.
* :data:`RESULT_FIELDS`     -- ordered column names for the results table.
* :func:`append_result_row` -- append one result dict to a CSV (crash-safe).
* :func:`load_results`      -- read a results CSV back into a list of dicts.
* :func:`run_sweep`         -- convenience loop over (SNR × random freq/drift).
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from astropy import units as u
import setigen as stg

from seti_klt import KLT, KN_rankEst, TracyWidomTW2
from seti_klt.utils.io import SimpleLogger, LogLevel

# CuPy is optional: the sweep runs on CPU if it (or a GPU) is unavailable.
try:  # pragma: no cover - depends on runtime hardware
    import cupy as cp

    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CKLTConfig:
    """Fixed instrument and processing parameters for the C-KLT sweep.

    The defaults reproduce the *dimensionality* of a MeerKAT beamformed stamp
    (γ = W/(K-1) ≈ 0.216) without trying to be physically faithful in every
    detail — only the covariance geometry matters for the RMT statistics.

    Attributes:
        sample_rate:   Baseband (real) sample rate fed to the antenna, in Hz.
        num_taps:      Polyphase filterbank taps.
        num_branches:  PFB branches → coarse channel count / bandwidth.
        fch1_hz:       Centre frequency of coarse channel 0, in Hz.
        num_chans:     Number of coarse channels recorded to the raw file.
        num_pols:      Polarisations synthesised (C-KLT uses ``pol`` only).
        noise_std:     Thermal-noise voltage std per polarisation.
        target_channel: Coarse channel the tone is injected into / analysed.
        pol:           Polarisation index analysed by the C-KLT.
        window_size:   C-KLT window length W (covariance dimension).
        k_windows:     Target number of windows K (⇒ ``num_samples = W*K``).
        block_size:    setigen raw block size in bytes (kept small for speed).
        use_gpu:       Use the CuPy setigen backend when a GPU is present.
        kn_alpha:      Kritchman & Nadler false-alarm level (percent).
        kn_beta:       1 (real) or 2 (complex). Baseband voltages ⇒ 2.
        freq_tol_frac: Fractional-bandwidth tolerance for "frequency recovered".
        out_root:      Root directory for generated raw files and results.
    """

    # --- instrument / PFB ---
    sample_rate: float = 3.0e9
    num_taps: int = 8
    num_branches: int = 1024
    fch1_hz: float = 1.5e9
    num_chans: int = 64
    num_pols: int = 2
    noise_std: float = 1.0

    # --- injection / analysis target ---
    target_channel: int = 30
    pol: int = 0

    # --- C-KLT geometry (MeerKAT-like: W=63, K=293 ⇒ γ≈0.216) ---
    window_size: int = 63
    k_windows: int = 293

    # --- raw generation ---
    block_size: int = 2 ** 23  # 8 MiB → ~32k time samples/chan at these dims
    # GPU (CuPy setigen backend) is opt-in: it works when a GPU is present, but
    # the first generation in a fresh process pays a one-off kernel-compilation
    # cost. CPU generation is a reliable ~3 s/run with no warmup.
    use_gpu: bool = False

    # --- detection parameters ---
    kn_alpha: float = 0.01
    kn_beta: int = 2
    freq_tol_frac: float = 0.03

    # --- paths ---
    out_root: str = "outputs"

    # --- derived (populated in __post_init__) ---
    chan_bw: float = field(init=False)
    num_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.chan_bw = self.sample_rate / self.num_branches
        self.num_samples = self.window_size * self.k_windows

    @property
    def gamma(self) -> float:
        """RMT aspect ratio γ = W / (K - 1)."""
        return self.window_size / (self.k_windows - 1)

    @property
    def bbp_threshold(self) -> float:
        """Population-level BBP phase-transition threshold √γ (in σ² units)."""
        return float(np.sqrt(self.gamma))

    def channel_center_hz(self, channel: int | None = None) -> float:
        """Absolute centre frequency of a coarse channel (ascending band)."""
        ch = self.target_channel if channel is None else channel
        return self.fch1_hz + ch * self.chan_bw


# ---------------------------------------------------------------------------
# Signal-to-noise helper
# ---------------------------------------------------------------------------


def spectral_snr(
    voltage: np.ndarray,
    *,
    use_gpu: bool = False,
) -> tuple[float, float]:
    """Narrowband spectral SNR of a 1-D complex baseband vector.

    Computes the power spectrum via a single FFT, then reports the ratio of the
    strongest (non-DC) bin to the median (noise-floor) bin.  This is a cheap,
    plot-free proxy for how far a tone stands above the noise — used both for
    the noisy input and for the C-KLT reconstruction.

    Args:
        voltage: 1-D complex (or real) voltage samples.
        use_gpu: Run the FFT on the GPU via CuPy when available.

    Returns:
        ``(snr_linear, peak_bin_fraction)`` where ``peak_bin_fraction`` is the
        location of the peak as a fraction of the band (0–1), useful for
        checking that the recovered tone sits at the injected offset.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    v = xp.asarray(voltage)
    v = v - v.mean()
    power = xp.abs(xp.fft.fft(v)) ** 2
    n = power.shape[0]

    power[0] = 0.0  # suppress DC / residual mean
    peak_idx = int(xp.argmax(power))
    peak = float(power[peak_idx])

    # Noise floor: median over all non-DC bins (robust to the single spike).
    floor = float(xp.median(power[1:]))
    snr_lin = peak / floor if floor > 0 else float("inf")
    return snr_lin, peak_idx / n


def _to_db(x: float) -> float:
    """Linear power ratio → dB (with a floor to avoid log(0))."""
    return 10.0 * np.log10(max(x, 1e-30))


# ---------------------------------------------------------------------------
# Raw-file generation
# ---------------------------------------------------------------------------


def generate_cklt_raw(
    snr_db: float,
    freq_offset_frac: float,
    drift_rate_hz_s: float,
    *,
    config: CKLTConfig,
    file_stem: str,
    seed: int | None = None,
) -> dict:
    """Generate one GUPPI raw file with a drifting CW tone injected.

    The tone is a polarised constant-power carrier placed at a *fractional*
    offset inside ``config.target_channel``.  Its amplitude is set so that the
    per-sample (pre-PFB) SNR equals ``snr_db``, following the same convention as
    ``notebooks/Injected_Signal/CKLT_Injected_Signal.ipynb``:

        level = noise_std · √(2 · 10^(snr_db/10))

    Args:
        snr_db:           Injected per-sample SNR in dB (the swept control).
        freq_offset_frac: Baseband offset within the channel, in [0, 1).  The
                          cycles-per-window (= offset · W) must stay away from an
                          integer or the C-KLT centering erases the tone.
        drift_rate_hz_s:  Linear drift rate of the tone, in Hz/s.
        config:           Fixed instrument / processing parameters.
        file_stem:        Output stem; the raw file is ``<file_stem>.0000.raw``.
        seed:             Optional seed for setigen's noise (reproducibility).

    Returns:
        Metadata dict consumed by :func:`analyze_cklt` (file paths, injected
        parameters, absolute signal frequency, tone amplitude, and geometry).
    """
    Path(file_stem).parent.mkdir(parents=True, exist_ok=True)

    if config.use_gpu and _HAS_CUPY:
        stg.voltage.set_backend("cupy")
    else:
        stg.voltage.set_backend("numpy")

    if seed is not None:
        np.random.seed(seed)
        if _HAS_CUPY:
            cp.random.seed(seed)

    snr_lin = 10.0 ** (snr_db / 10.0)
    level = config.noise_std * np.sqrt(2.0 * snr_lin)

    f_signal = config.channel_center_hz() + freq_offset_frac * config.chan_bw

    antenna = stg.voltage.Antenna(
        sample_rate=config.sample_rate,
        fch1=config.fch1_hz * u.Hz,
        ascending=True,
        num_pols=config.num_pols,
    )

    antenna.x.add_noise(v_mean=0, v_std=config.noise_std)
    antenna.x.add_constant_signal(
        f_start=f_signal, drift_rate=drift_rate_hz_s * u.Hz / u.s, level=level, phase=0
    )
    if config.num_pols == 2:
        antenna.y.add_noise(v_mean=0, v_std=config.noise_std)
        antenna.y.add_constant_signal(
            f_start=f_signal,
            drift_rate=drift_rate_hz_s * u.Hz / u.s,
            level=level,
            phase=np.pi / 2,
        )

    digitizer = stg.voltage.RealQuantizer(target_fwhm=32, num_bits=8)
    filterbank = stg.voltage.PolyphaseFilterbank(
        num_taps=config.num_taps, num_branches=config.num_branches
    )
    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32, num_bits=8)

    rvb = stg.voltage.RawVoltageBackend(
        antenna,
        digitizer=digitizer,
        filterbank=filterbank,
        requantizer=requantizer,
        start_chan=0,
        num_chans=config.num_chans,
        block_size=config.block_size,
        blocks_per_file=128,
        num_subblocks=32,
    )

    rvb.record(
        output_file_stem=file_stem,
        num_blocks=1,
        length_mode="num_blocks",
        header_dict={"TELESCOP": "MEERKAT_SIM", "SRC_NAME": "CKLT_PT"},
        verbose=False,
    )

    return {
        "file_path": f"{file_stem}.0000.raw",
        "file_stem": file_stem,
        "snr_db": float(snr_db),
        "snr_lin": float(snr_lin),
        "level": float(level),
        "freq_offset_frac": float(freq_offset_frac),
        "drift_rate_hz_s": float(drift_rate_hz_s),
        "f_signal_hz": float(f_signal),
        "cycles_per_window": float(freq_offset_frac * config.window_size),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


# Ordered columns of the results table.  Kept explicit so the CSV schema is
# stable across runs (crash-safe appends rely on a fixed header).
RESULT_FIELDS: list[str] = [
    # injection
    "snr_db", "snr_lin", "level", "freq_offset_frac", "drift_rate_hz_s",
    "f_signal_hz", "cycles_per_window", "seed",
    # geometry
    "W", "K", "gamma", "bbp_threshold_sqrt_gamma",
    # eigen-spectrum / RMT
    "sigma2_hat", "lambda1", "lambda2", "gap_ratio",
    "lambda_max_MP", "lambda_c", "l_pop_hat", "margin_l_minus_sqrt_gamma",
    "snr_effective",
    # phase transition (theory)
    "supercritical",
    # Kritchman & Nadler
    "n_signals_KN", "sigma_hat_KN", "n_eig_recon",
    # reconstruction quality
    "input_snr_db", "recon_snr_db", "recon_gain_db",
    "freq_err_hz", "freq_recovered",
    # decisions / agreement
    "signal_found", "signal_confirmed", "kn_agrees_bbp",
    # bookkeeping
    "elapsed_s",
]


def analyze_cklt(
    gen_meta: dict,
    *,
    config: CKLTConfig,
    logger: SimpleLogger | None = None,
) -> dict:
    """Run the C-KLT detection pipeline on one generated raw file.

    Loads the target channel, applies the C-KLT, computes RMT / phase-transition
    statistics, and runs the **Kritchman & Nadler** rank estimator.  The K&N
    signal count directly sets how many dominant eigenvectors are kept for the
    C-KLT reconstruction — there is no Bayesian (TW₂) detector in the loop.
    Reconstruction quality (spectral SNR gain and recovered tone frequency) is
    then measured on that K&N-sized reconstruction.

    Args:
        gen_meta: Dict returned by :func:`generate_cklt_raw`.
        config:   Fixed instrument / processing parameters.
        logger:   Optional logger; a quiet WARNING-level one is used if ``None``.

    Returns:
        A flat dict keyed by :data:`RESULT_FIELDS` (ready for
        :func:`append_result_row`).  Booleans are stored as ``0``/``1`` ints so
        the CSV stays purely numeric where possible.
    """
    t0 = time.perf_counter()

    if logger is None:
        logger = SimpleLogger(
            level=LogLevel.WARNING, console_output=False, file_output=False
        )

    klt = KLT(logger=logger)
    klt.load_data_from_guppi(
        gen_meta["file_path"],
        gen_meta["file_stem"],
        channel=config.target_channel,
        num_samples=config.num_samples,
        polarization=config.pol,
    )

    # First pass: eigendecomposition. n_eigenvectors here only affects the
    # provisional reconstruction, not the eigenvalues K&N consumes.
    klt.apply_cklt(window_size=config.window_size, n_eigenvectors=1)

    rmt = klt.compute_rmt_statistics()
    eigvals = np.asarray(klt.get_eigenvalues(), dtype=float)

    W, K, gamma = rmt["W"], rmt["K"], rmt["gamma"]
    lambda1 = rmt["lambda1"]
    lambda2 = float(eigvals[1]) if eigvals.size > 1 else float("nan")
    sigma2_hat = rmt["sigma2_hat"]

    # Estimated population spike ℓ̂ (invert BBP map on the observed λ₁/σ̂²).
    # This is the RMT phase-transition estimate, not a Bayesian statistic.
    l_pop_hat = TracyWidomTW2.invert_bbp_for_l_pop(lambda1 / sigma2_hat, gamma)

    # --- Kritchman & Nadler rank estimate (complex ⇒ β = 2) --------------
    # np.cov normalises by K-1, so the matching sample count for K&N is n=K-1.
    n_signals_kn, sigma_hat_kn = KN_rankEst(
        eigvals, n=K - 1, beta=config.kn_beta, alpha=config.kn_alpha
    )

    # --- reconstruction with the K&N-chosen number of eigenvectors -------
    # Kritchman & Nadler decides how many signal eigenvalues to keep. We rebuild
    # the signal using exactly that many dominant eigenvectors. If K&N finds
    # none, there is no signal subspace to reconstruct and the recovery metrics
    # are undefined (NaN) rather than forced onto the pure-noise floor.
    n_eig_recon = int(n_signals_kn)
    noisy_in = np.asarray(klt._klt_input)
    input_snr_lin, _ = spectral_snr(noisy_in, use_gpu=config.use_gpu)
    input_snr_db = _to_db(input_snr_lin)

    if n_eig_recon >= 1:
        klt.apply_cklt(window_size=config.window_size, n_eigenvectors=n_eig_recon)
        recon = np.asarray(klt.get_reconstructed_signal())
        recon_snr_lin, recon_frac = spectral_snr(recon, use_gpu=config.use_gpu)
        recon_snr_db = _to_db(recon_snr_lin)

        # Recovered tone frequency vs injection (fraction of the channel band).
        injected_frac = gen_meta["freq_offset_frac"] % 1.0
        d = abs(recon_frac - injected_frac)
        freq_err_frac = min(d, 1.0 - d)
        freq_err_hz = freq_err_frac * config.chan_bw
        freq_recovered = bool(freq_err_frac < config.freq_tol_frac)
    else:
        recon_snr_db = float("nan")
        freq_err_hz = float("nan")
        freq_recovered = False

    # --- decisions & cross-checks ----------------------------------------
    supercritical = bool(rmt["supercritical"])
    signal_found = bool(n_signals_kn >= 1)
    signal_confirmed = bool(signal_found and freq_recovered)
    kn_agrees_bbp = bool(signal_found == supercritical)

    result = {
        "snr_db": gen_meta["snr_db"],
        "snr_lin": gen_meta["snr_lin"],
        "level": gen_meta["level"],
        "freq_offset_frac": gen_meta["freq_offset_frac"],
        "drift_rate_hz_s": gen_meta["drift_rate_hz_s"],
        "f_signal_hz": gen_meta["f_signal_hz"],
        "cycles_per_window": gen_meta["cycles_per_window"],
        "seed": gen_meta["seed"],
        "W": W,
        "K": K,
        "gamma": gamma,
        "bbp_threshold_sqrt_gamma": float(np.sqrt(gamma)),
        "sigma2_hat": sigma2_hat,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "gap_ratio": lambda1 / lambda2 if lambda2 > 0 else float("inf"),
        "lambda_max_MP": rmt["lambda_max_MP"],
        "lambda_c": rmt["lambda_c"],
        "l_pop_hat": l_pop_hat,
        "margin_l_minus_sqrt_gamma": l_pop_hat - float(np.sqrt(gamma)),
        "snr_effective": rmt["snr_effective"],
        "supercritical": int(supercritical),
        "n_signals_KN": int(n_signals_kn),
        "sigma_hat_KN": float(sigma_hat_kn),
        "n_eig_recon": int(n_eig_recon),
        "input_snr_db": input_snr_db,
        "recon_snr_db": recon_snr_db,
        "recon_gain_db": recon_snr_db - input_snr_db,
        "freq_err_hz": freq_err_hz,
        "freq_recovered": int(freq_recovered),
        "signal_found": int(signal_found),
        "signal_confirmed": int(signal_confirmed),
        "kn_agrees_bbp": int(kn_agrees_bbp),
        "elapsed_s": time.perf_counter() - t0,
    }
    return result


# ---------------------------------------------------------------------------
# Incremental results I/O
# ---------------------------------------------------------------------------


def append_result_row(
    path: str | Path,
    result: dict,
    *,
    fields: list[str] = RESULT_FIELDS,
) -> None:
    """Append one result dict as a CSV row, writing the header if the file is new.

    The file is opened, written, and closed on every call so partial results
    survive an interrupted sweep (the whole point of persisting incrementally
    given the per-run cost).

    Args:
        path:   Destination CSV file.
        result: Result dict from :func:`analyze_cklt`.
        fields: Column order (defaults to :data:`RESULT_FIELDS`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fields})


def load_results(path: str | Path) -> list[dict]:
    """Read a results CSV back into a list of dicts with numeric coercion.

    Args:
        path: Results CSV written by :func:`append_result_row`.

    Returns:
        List of row dicts; values are converted to ``float``/``int`` where
        possible, left as strings otherwise.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
            for k, v in raw.items():
                if v == "" or v is None:
                    row[k] = None
                    continue
                try:
                    f = float(v)
                    row[k] = int(f) if f.is_integer() else f
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Convenience sweep
# ---------------------------------------------------------------------------


def run_sweep(
    snr_db_values,
    *,
    config: CKLTConfig,
    n_trials: int = 3,
    freq_offset_range: tuple[float, float] = (0.12, 0.42),
    drift_range_hz_s: tuple[float, float] = (-5.0, 5.0),
    results_path: str | Path | None = None,
    keep_raw: bool = False,
    rng: np.random.Generator | None = None,
    logger: SimpleLogger | None = None,
) -> list[dict]:
    """Run the full (decreasing-SNR × random freq/drift) sweep.

    For each SNR value and each of ``n_trials`` random draws, a raw file is
    generated, analysed, and its result row appended to ``results_path``
    immediately.

    Args:
        snr_db_values:     Iterable of injected SNRs in dB (pass them decreasing).
        config:            Fixed instrument / processing parameters.
        n_trials:          Random (freq, drift) draws per SNR value.
        freq_offset_range: Uniform range for the baseband offset fraction.
        drift_range_hz_s:  Uniform range for the drift rate.
        results_path:      CSV to append to (default ``<out_root>/results/cklt_phase_transition.csv``).
        keep_raw:          Keep each raw file (default deletes it after analysis
                           to save disk during a long sweep).
        rng:               NumPy Generator for reproducible draws.
        logger:            Optional logger for the per-run pipeline.

    Returns:
        List of result dicts (also written to ``results_path``).
    """
    rng = rng or np.random.default_rng()

    out_root = Path(config.out_root)
    raw_dir = out_root / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = out_root / "results" / "cklt_phase_transition.csv"

    all_results: list[dict] = []
    run_id = 0
    for snr_db in snr_db_values:
        for _ in range(n_trials):
            offset = _draw_safe_offset(rng, freq_offset_range, config.window_size)
            drift = float(rng.uniform(*drift_range_hz_s))
            seed = int(rng.integers(0, 2 ** 31 - 1))

            stem = str(raw_dir / f"cklt_run{run_id:04d}_snr{snr_db:+06.1f}")
            gen_meta = generate_cklt_raw(
                snr_db, offset, drift, config=config, file_stem=stem, seed=seed
            )
            result = analyze_cklt(gen_meta, config=config, logger=logger)
            append_result_row(results_path, result)
            all_results.append(result)

            if not keep_raw:
                Path(gen_meta["file_path"]).unlink(missing_ok=True)
            run_id += 1

    return all_results


def _draw_safe_offset(
    rng: np.random.Generator,
    offset_range: tuple[float, float],
    window_size: int,
    min_int_dist: float = 0.15,
    max_tries: int = 50,
) -> float:
    """Draw a baseband offset whose cycles-per-window avoids integer values.

    An integer number of cycles per window makes every C-KLT window an identical
    copy of the tone, so the column-mean subtraction erases it (the "synchronous
    phase" failure).  We resample until the fractional part of ``offset·W`` is at
    least ``min_int_dist`` away from 0 and 1.

    Args:
        rng:          NumPy Generator.
        offset_range: Uniform range for the offset fraction.
        window_size:  C-KLT window length W.
        min_int_dist: Minimum distance (in cycles) from an integer.
        max_tries:    Give up and return the last draw after this many attempts.

    Returns:
        A baseband offset fraction in ``offset_range``.
    """
    offset = float(rng.uniform(*offset_range))
    for _ in range(max_tries):
        frac = (offset * window_size) % 1.0
        if min_int_dist < frac < 1.0 - min_int_dist:
            return offset
        offset = float(rng.uniform(*offset_range))
    return offset
