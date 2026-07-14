"""
multichannel.py
---------------
Per-channel MR-KLT sweep over a channelised multi-antenna data cube.

Both the simulated multi-antenna pipeline (``notebooks/MAA_Simulated``) and the
real MeerKAT stamp pipeline (``notebooks/Meerkat_Real_Data``) need the same
operation: for every frequency channel, build the antenna covariance matrix,
run the MR-KLT, ask the Kritchman-Nadler estimator how many signals are present,
and reconstruct the signal subspace.  This module provides that sweep once.

Typical use::

    from seti_klt import scan_channels

    # data: (K, F, N_A) — realisations x channels x antennas
    scan = scan_channels(data, exclude_channels={0}, alpha=0.01)

    scan.k_est            # (F,) signals detected per channel
    scan.detected_channels()
    scan.power_recon      # (K, F) reconstructed power, ready to plot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .KLT import KLT
from .kritchman_nadler import KN_rankEst
from .utils.io import LogLevel, SimpleLogger


@dataclass
class ChannelScanResult:
    """Outcome of a :func:`scan_channels` sweep.

    Channels excluded from the sweep are represented by ``NaN`` in the float
    arrays and ``0`` in :attr:`k_est`, so every array can be indexed directly by
    absolute channel number.

    Attributes:
        channels:     Indices of the channels that were actually analysed.
        k_est:        ``(F,)`` int — signals detected per channel (Kritchman-Nadler).
        lambda1:      ``(F,)`` — dominant eigenvalue per channel.
        eigenvalues:  ``(F, N_A)`` — full eigenspectrum per channel, descending.
        eigenvector1: ``(F, N_A)`` complex — dominant spatial eigenvector per channel.
        power_orig:   ``(K, F)`` — mean power over antennas, input data.
        power_recon:  ``(K, F)`` — mean power over antennas, MR-KLT reconstruction.
        K:            Number of realisations (rows of the input).
        N_A:          Number of antennas.
        n_channels:   Total number of channels ``F`` in the cube.
    """

    channels: list[int]
    k_est: np.ndarray
    lambda1: np.ndarray
    eigenvalues: np.ndarray
    eigenvector1: np.ndarray
    power_orig: np.ndarray
    power_recon: np.ndarray
    K: int
    N_A: int
    n_channels: int = field(default=0)

    def detected_channels(self, min_signals: int = 1) -> list[int]:
        """Return the channels where at least *min_signals* signals were found.

        Args:
            min_signals: Minimum ``k_est`` for a channel to be reported.

        Returns:
            Sorted list of channel indices.
        """
        return [c for c in self.channels if self.k_est[c] >= min_signals]

    def noise_normalised_eigenvalues(self) -> np.ndarray:
        """Return the eigenspectrum divided by each channel's own noise floor.

        The noise floor is the median eigenvalue of the channel, so a value of
        ``~1`` means "noise" and ``>> 1`` means a coherent component.  This puts
        channels with different absolute scales on a comparable footing, which is
        what makes the per-channel eigenspectrum heatmap readable.

        Returns:
            ``(F, N_A)`` array of eigenvalues in units of the per-channel floor.
            Excluded channels stay ``NaN``.
        """
        # Take the median only over analysed channels: an excluded channel is an
        # all-NaN row, and nanmedian would warn on it and return NaN anyway.
        floor = np.full((self.n_channels, 1), np.nan)
        if self.channels:
            floor[self.channels, 0] = np.nanmedian(
                self.eigenvalues[self.channels], axis=1
            )
        return self.eigenvalues / floor

    def eigenvector_coherence(self, ref_channel: int | None = None) -> np.ndarray:
        """Spatial coherence of each channel's dominant eigenvector with a reference.

        Computes ``|<v1_c, v1_ref>| / (||v1_c|| ||v1_ref||)`` per channel.  A value
        near 1 means the channel sees the *same* spatial mode (i.e. the same
        physical source / direction of arrival) as the reference channel — which is
        how a single narrowband emitter spread over adjacent channels reveals itself.

        Args:
            ref_channel: Reference channel.  Defaults to the channel with the
                         largest dominant eigenvalue.

        Returns:
            ``(F,)`` array of coherences in [0, 1]; ``NaN`` for excluded channels.
        """
        if ref_channel is None:
            ref_channel = int(np.nanargmax(self.lambda1))

        v_ref = self.eigenvector1[ref_channel]
        out = np.full(self.n_channels, np.nan)
        for c in self.channels:
            v = self.eigenvector1[c]
            denom = np.linalg.norm(v) * np.linalg.norm(v_ref)
            if denom > 0:
                out[c] = float(np.abs(np.vdot(v, v_ref)) / denom)
        return out


def scan_channels(
    data: np.ndarray,
    exclude_channels: set[int] | tuple[int, ...] = (),
    alpha: float = 0.01,
    beta: int = 2,
    reconstruct: bool = True,
    logger: SimpleLogger | None = None,
) -> ChannelScanResult:
    """Run the MR-KLT plus Kritchman-Nadler rank estimation on every channel.

    For each frequency channel ``c`` the antenna slice ``data[:, c, :]`` of shape
    ``(K, N_A)`` is fed to :meth:`KLT.apply_mrklt`.  The eigenspectrum is passed to
    :func:`~seti_klt.kritchman_nadler.KN_rankEst` to estimate how many signals the
    channel carries, and (optionally) the signal subspace is reconstructed using
    exactly that many eigenvectors.

    Args:
        data:             Complex cube of shape ``(K, F, N_A)`` — realisations
                          (timesteps) x frequency channels x antennas.  If your
                          data still carries a polarisation axis, select it first
                          (e.g. ``data[:, :, pol, :]``).
        exclude_channels: Channels to skip entirely (typically the DC channel
                          ``{0}``, and possibly the band edges).  Excluded channels
                          appear as ``NaN`` / ``0`` in the result.
        alpha:            Kritchman-Nadler confidence level, in **percentage
                          points** (0.01 = a 0.01% false-alarm rate per test).
        beta:             1 for real, 2 for complex (I/Q) data.  Baseband voltages
                          are complex, so the default is 2.
        reconstruct:      When ``True``, also reconstruct the signal subspace of
                          each channel and fill :attr:`ChannelScanResult.power_recon`.
                          Set to ``False`` to skip the second MR-KLT pass if you only
                          need the eigenvalues and the rank.
        logger:           Optional logger.  A silent one is used when ``None``,
                          because a per-channel sweep would otherwise emit one
                          INFO block per channel.

    Returns:
        :class:`ChannelScanResult` with the per-channel eigenvalues, dominant
        eigenvector, estimated rank, and original/reconstructed power.

    Raises:
        ValueError: If *data* is not a 3-D ``(K, F, N_A)`` array.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3-D (K, F, N_A) cube, got a {data.ndim}-D array with "
            f"shape {data.shape}. If the data still has a polarisation axis "
            "(K, F, P, N_A), select one polarisation first, e.g. data[:, :, pol, :]."
        )

    K, F, N_A = data.shape
    exclude = set(exclude_channels)
    channels = [c for c in range(F) if c not in exclude]

    # A sweep runs one KLT per channel; an INFO-level logger would print a full
    # block for every one of them, so stay silent unless explicitly told otherwise.
    if logger is None:
        logger = SimpleLogger(level=LogLevel.ERROR, console_output=False, file_output=False)

    k_est        = np.zeros(F, dtype=int)
    lambda1      = np.full(F, np.nan)
    eigenvalues  = np.full((F, N_A), np.nan)
    eigenvector1 = np.full((F, N_A), np.nan, dtype=complex)
    power_orig   = np.mean(np.abs(data) ** 2, axis=2)     # (K, F)
    power_recon  = np.full((K, F), np.nan)                # NaN = not analysed

    for c in channels:
        klt = KLT(logger=logger)
        klt.load_signal_from_array(data[:, c, :])
        klt.apply_mrklt(n_eigenvectors=1)

        evals = klt.get_eigenvalues()
        eigenvalues[c, : len(evals)] = evals
        lambda1[c] = evals[0]
        eigenvector1[c] = klt.get_eigenvectors()[:, 0]

        k, _ = KN_rankEst(ell=evals, n=K, beta=beta, alpha=alpha)
        k_est[c] = k

        if not reconstruct:
            continue

        if k > 0:
            # Second pass keeping exactly the rank the estimator found.
            klt.apply_mrklt(n_eigenvectors=k)
            # reconstructed_signal is (K, N_A); average power over the antenna axis.
            power_recon[:, c] = np.mean(np.abs(klt.reconstructed_signal) ** 2, axis=1)
        else:
            # No coherent component detected -> null reconstruction.
            power_recon[:, c] = 0.0

    return ChannelScanResult(
        channels=channels,
        k_est=k_est,
        lambda1=lambda1,
        eigenvalues=eigenvalues,
        eigenvector1=eigenvector1,
        power_orig=power_orig,
        power_recon=power_recon,
        K=K,
        N_A=N_A,
        n_channels=F,
    )
