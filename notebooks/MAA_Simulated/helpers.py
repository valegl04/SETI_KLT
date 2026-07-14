"""
helpers.py
==========
Scenario generation and loading for the simulated multi-antenna KLT pipeline.

This module holds the *test-data fabrication* side of the study — building a
katpoint antenna array, computing geometric delays, synthesising multi-antenna
GUPPI raw baseband with ``setigen``, and reading it back into an analysis cube.
It is deliberately kept local to this folder (rather than inside the ``seti_klt``
package) for the same reason ``Kritchman_Nadler/helpers.py`` keeps
``make_spiked_data`` local: ``seti_klt`` is a KLT *analysis* library, and pulling
a ``setigen`` + ``katpoint`` simulation harness into it would mix "manufacture
fake data" with "analyse data" and drag heavy simulation dependencies into the
analysis core.

The *analysis* itself is not re-implemented here: the notebook calls
``seti_klt.scan_channels`` directly, so it exercises the real pipeline.

Contents
--------
* :func:`build_antennas`    -- place ``num_antennas`` katpoint antennas in a disk.
* :func:`compute_delays`    -- geometric inter-antenna delays toward a target.
* :func:`generate_scenario` -- write a multi-antenna GUPPI raw + ``.meta.json``
                               ground-truth sidecar.
* :func:`load_scenario`     -- read a scenario back as a ``(T, F, A)`` cube.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import astropy.constants as const
import katpoint
import numpy as np
import setigen as stg
from astropy import units as u
from blimpy import GuppiRaw

from seti_klt.utils import SimpleLogger


# ---------------------------------------------------------------------------
# Antenna array construction and geometric delays
# ---------------------------------------------------------------------------

def build_antennas(
    num_antennas: int,
    array_radius_m: float = 1000.0,
    seed: int = 42,
) -> tuple[katpoint.Antenna, list[katpoint.Antenna]]:
    """Create *num_antennas* katpoint antennas scattered over a disk.

    Antenna 0 sits at the centre and acts as the phase reference; the others are
    placed at uniformly random positions inside a disk of radius *array_radius_m*.

    Args:
        num_antennas:   Number of antennas to create.
        array_radius_m: Radius of the array footprint, in metres.
        seed:           RNG seed, for a reproducible layout.

    Returns:
        ``(ref_ant, antennas)``: the reference antenna and the full antenna list.
    """
    rng = np.random.default_rng(seed)
    ref_ant = katpoint.Antenna("m000, -30:42:39.8, 21:26:38.0, 1035.0, 13.5")

    antennas = []
    for i in range(num_antennas):
        if i == 0:
            e = n = up = 0.0
        else:
            r = array_radius_m * np.sqrt(rng.random())   # uniform over the disk
            th = 2 * np.pi * rng.random()
            e, n, up = r * np.cos(th), r * np.sin(th), 0.0
        antennas.append(
            katpoint.Antenna(
                f"m{i:03d}, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, {e}, {n}, {up}"
            )
        )
    return ref_ant, antennas


def compute_delays(
    ref_ant: katpoint.Antenna,
    antennas: list[katpoint.Antenna],
    target: katpoint.Target,
    timestamp: float,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric delay of each antenna toward *target*, relative to *ref_ant*.

    Delays are converted to whole raw samples and shifted so that the minimum is
    zero (``setigen`` requires non-negative delays).

    Args:
        ref_ant:     Reference (phase-centre) antenna.
        antennas:    Antennas to compute delays for.
        target:      Sky target the array is pointed at.
        timestamp:   UNIX timestamp of the observation.
        sample_rate: Baseband sample rate, in Hz.

    Returns:
        ``(delays_samples, delays_sec)``: delays in integer samples and in seconds.
    """
    delays_sec = np.array([
        target.uvw(ant, timestamp, ref_ant)[2] / const.c.value
        for ant in antennas
    ])
    delays_samples = np.round(delays_sec * sample_rate).astype(int)
    delays_samples -= delays_samples.min()
    return delays_samples, delays_sec


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenario(
    output_stem: str,
    output_dir: str | Path,
    num_antennas: int = 60,
    signals: list[dict] | None = None,
    target_radec: tuple[str, str] = ("08:35:20.6", "-45:10:35.2"),
    sample_rate: float = 3e9,
    fch1_hz: float = 1.4e9,
    num_branches: int = 1024,
    num_taps: int = 8,
    num_chans: int = 64,
    start_chan: int = 0,
    num_pols: int = 2,
    noise_std: float = 1.0,
    samples_per_block: int = 8192,
    num_blocks: int = 1,
    num_subblocks: int = 32,
    array_radius_m: float = 1000.0,
    seed: int = 42,
    logger: SimpleLogger | None = None,
    setigen_progress: bool = False,
) -> dict:
    """Generate a multi-antenna GUPPI raw file plus a ``.meta.json`` sidecar.

    The sidecar records the injected ground truth (which channels carry which
    kind of signal, on how many antennas), so the analysis stage can be scored
    against it rather than merely inspected.

    Each entry of *signals* is a dict with:

    * ``kind``       -- ``"coherent"`` (sky signal: injected on the background
                        streams, so it receives the geometric delays and is
                        coherent across **all** antennas) or ``"per_antenna"``
                        (RFI / local signal: injected on a subset of antennas
                        **without** delays, hence spatially incoherent).
    * ``channel``    -- recorded coarse channel (0..``num_chans``-1).  Alternatively
                        pass ``f_start_hz`` directly.
    * ``drift_rate`` -- drift in Hz/s (default 0).
    * ``level``      -- signal amplitude.
    * ``n_antennas`` / ``antenna_indices`` -- ``per_antenna`` only: how many
                        antennas (chosen at random) or exactly which ones.

    Args:
        output_stem:       Base name of the scenario (e.g. ``"scen_A"``).
        output_dir:        Directory the ``.raw`` and ``.meta.json`` are written to.
        num_antennas:      Number of antennas in the simulated array.
        signals:           Signals to inject (see above).  Empty means noise only.
        target_radec:      Sky position of the pointing, as ``(ra, dec)`` strings.
        sample_rate:       Baseband sample rate, in Hz.
        fch1_hz:           Frequency of coarse channel 0, in Hz.
        num_branches:      PFB branches (sets the coarse channel bandwidth).
        num_taps:          PFB taps.
        num_chans:         Number of coarse channels actually recorded.
        start_chan:        First recorded coarse channel.
        num_pols:          Number of polarisations.
        noise_std:         Per-antenna thermal noise standard deviation.
        samples_per_block: Output time frames per block.
        num_blocks:        Number of blocks to record.
        num_subblocks:     Memory control — higher means less RAM at a time.
        array_radius_m:    Array footprint radius, in metres.
        seed:              RNG seed (antenna layout and antenna subset choice).
        logger:            Optional logger.  Falls back to a console logger.
        setigen_progress:  Forward ``verbose`` to ``setigen``'s ``record()``.  Left
                           ``False`` because setigen prints one line per antenna *per
                           subblock* (e.g. 60 x 32 = ~1900 lines per scenario), which
                           buries our own log.  Set ``True`` only to debug a stalled
                           write.

    Returns:
        The ground-truth metadata dict that was written to ``<stem>.meta.json``.

    Raises:
        AssertionError: If *samples_per_block* is not divisible by *num_subblocks*.
        ValueError:     If a signal has an unknown ``kind``.
    """
    assert samples_per_block % num_subblocks == 0, \
        "samples_per_block must be divisible by num_subblocks"

    if logger is None:
        logger = SimpleLogger(console_output=True, file_output=False)

    signals = signals or []
    rng = np.random.default_rng(seed)
    chan_bw = sample_rate / num_branches

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_path = out_dir / output_stem

    def freq_for_channel(c: int, offset_frac: float = 0.25) -> float:
        # Centre of coarse channel c, offset within the channel to avoid the fine DC bin.
        return fch1_hz + (start_chan + c) * chan_bw + offset_frac * chan_bw

    # --- STEP 1: antennas + geometric delays ---
    ref_ant, antennas = build_antennas(num_antennas, array_radius_m, seed)
    target = katpoint.Target(
        f"target_source, radec, {target_radec[0]}, {target_radec[1]}"
    )
    delays_samples, delays_sec = compute_delays(
        ref_ant, antennas, target, time.time(), sample_rate
    )
    logger.info(
        f"[{output_stem}] {num_antennas} antennas | delay range: "
        f"{delays_sec.min() * 1e9:.1f} .. {delays_sec.max() * 1e9:.1f} ns "
        f"({delays_samples.max()} samples)."
    )

    # --- STEP 2: setigen multi-antenna array ---
    maa = stg.voltage.MultiAntennaArray(
        num_antennas=num_antennas,
        sample_rate=sample_rate,
        fch1=fch1_hz * u.Hz,
        ascending=True,
        num_pols=num_pols,
        delays=delays_samples,
    )

    # --- STEP 3: thermal noise, independent per antenna ---
    for antenna in maa.antennas:
        for stream in antenna.streams:
            stream.add_noise(v_mean=0, v_std=noise_std)

    # --- STEP 4: signal injection ---
    injected = []
    for k, sig in enumerate(signals):
        f_start = sig.get("f_start_hz") or freq_for_channel(sig["channel"])
        chan = sig.get("channel")
        if chan is not None and not (0 <= chan < num_chans):
            logger.warning(
                f"signal {k}: channel {chan} lies outside the {num_chans} recorded "
                "channels and will be invisible."
            )
        drift = sig.get("drift_rate", 0.0) * u.Hz / u.s
        level = sig["level"]
        kind = sig["kind"]

        if kind == "coherent":
            # Sky signal: on bg_streams, so it receives the delays and stays
            # coherent across ALL antennas.
            for stream in maa.bg_streams:
                stream.add_constant_signal(f_start=f_start, drift_rate=drift, level=level)
            ants = list(range(num_antennas))

        elif kind == "per_antenna":
            # RFI / local signal: on a subset of antennas, WITHOUT sky delays.
            ants = sig.get("antenna_indices")
            if ants is None:
                n = sig.get("n_antennas") or num_antennas
                ants = sorted(
                    rng.choice(num_antennas, size=min(n, num_antennas), replace=False).tolist()
                )
            for i in ants:
                for stream in maa.antennas[i].streams:
                    stream.add_constant_signal(f_start=f_start, drift_rate=drift, level=level)
        else:
            raise ValueError(f"unknown kind: {kind!r} (use 'coherent' or 'per_antenna')")

        injected.append({
            "kind": kind,
            "channel": chan,
            "f_start_hz": float(f_start),
            "drift_rate_hz_s": float(sig.get("drift_rate", 0.0)),
            "level": level,
            "n_antennas": len(ants),
            "antennas": ants,
        })
        tag = "ALL" if len(ants) == num_antennas else f"{len(ants)} ant"
        logger.info(f"  + signal {k}: {kind:11s} ch={chan} level={level} -> {tag}")

    # --- STEP 5: channelisation and writing ---
    bytes_per_sample = num_antennas * num_chans * num_pols * 2   # complex 8-bit
    block_size = bytes_per_sample * samples_per_block

    rvb = stg.voltage.RawVoltageBackend(
        maa,
        digitizer=stg.voltage.RealQuantizer(target_fwhm=32, num_bits=8),
        filterbank=stg.voltage.PolyphaseFilterbank(
            num_taps=num_taps, num_branches=num_branches
        ),
        requantizer=stg.voltage.ComplexQuantizer(target_fwhm=32, num_bits=8),
        start_chan=start_chan,
        num_chans=num_chans,
        block_size=block_size,
        blocks_per_file=128,
        num_subblocks=num_subblocks,
    )
    logger.info(f"  writing ({samples_per_block} samp/block x {num_blocks} blocks)...")
    rvb.record(
        output_file_stem=str(stem_path),
        num_blocks=num_blocks,
        length_mode="num_blocks",
        verbose=setigen_progress,
    )

    # --- Ground-truth sidecar ---
    meta = {
        "output_stem": str(stem_path),
        "num_antennas": num_antennas,
        "num_chans": num_chans,
        "num_pols": num_pols,
        "start_chan": start_chan,
        "sample_rate": sample_rate,
        "fch1_hz": fch1_hz,
        "chan_bw": chan_bw,
        "samples_per_block": samples_per_block,
        "num_blocks": num_blocks,
        "signals": injected,
    }
    meta_path = stem_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  OK wrote {stem_path}.0000.raw + {meta_path.name}")
    return meta


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenario(
    stem: str | Path,
    pol: int = 0,
    n_samples: int = 3072,
    remove_dc: bool = True,
    logger: SimpleLogger | None = None,
) -> tuple[np.ndarray, dict]:
    """Read a generated scenario back into a ``(T, F, A)`` cube plus its metadata.

    The GUPPI channel axis stacks antennas, i.e. row index is
    ``antenna * num_chans + channel``; this function undoes that packing and
    returns a cube in the ``(K, F, N_A)`` layout expected by
    :func:`seti_klt.scan_channels`.

    Args:
        stem:      Scenario stem, without the ``.0000.raw`` / ``.meta.json`` suffix.
        pol:       Polarisation index to extract.
        n_samples: Number of time samples to read.
        remove_dc: When ``True``, subtract the per-(channel, antenna) time mean.
                   This kills the zero-frequency bin and the common-mode quantiser
                   DC offset without touching narrowband tones.
        logger:    Optional logger.  Falls back to a console logger.

    Returns:
        ``(data, meta)``: complex cube of shape ``(n_samples, num_chans,
        num_antennas)`` and the ground-truth metadata dict.

    Raises:
        AssertionError: If the raw file's channel count does not match the
                        ``num_antennas * num_chans`` recorded in the metadata.
    """
    if logger is None:
        logger = SimpleLogger(console_output=True, file_output=False)

    stem = str(stem)
    meta = json.load(open(f"{stem}.meta.json"))
    N_A, N_CHANNELS = meta["num_antennas"], meta["num_chans"]

    raw = GuppiRaw(f"{stem}.0000.raw")
    blocks, total = [], 0
    while total < n_samples:
        try:
            _, d = raw.read_next_data_block()
        except StopIteration:
            break
        if d is None:
            break
        blocks.append(d)
        total += d.shape[1]

    raw_data = np.concatenate(blocks, axis=1)[:, :n_samples, :]      # (A*F, T, npol)
    assert raw_data.shape[0] == N_A * N_CHANNELS, (
        f"expected {N_A * N_CHANNELS} channels, found {raw_data.shape[0]}"
    )

    # Channel axis stacks antennas: index = antenna * N_CHANNELS + channel.
    raw_data = raw_data.reshape(N_A, N_CHANNELS, n_samples, raw_data.shape[2])
    data = np.transpose(raw_data[:, :, :, pol], (2, 1, 0))           # (T, F, A)
    data = np.ascontiguousarray(data, dtype=np.complex64)

    if remove_dc:
        data = data - data.mean(axis=0, keepdims=True)

    truth = signal_channels(meta)
    logger.info(
        f"[{Path(stem).name}] data {data.shape} | "
        f"signal channels (ground truth): {truth}"
    )
    return data, meta


def signal_channels(meta: dict) -> list[int]:
    """Return the sorted ground-truth channels that carry an injected signal.

    Args:
        meta: Metadata dict from :func:`generate_scenario` / :func:`load_scenario`.

    Returns:
        Sorted list of channel indices.
    """
    return sorted(
        {s["channel"] for s in meta["signals"] if s["channel"] is not None}
    )
