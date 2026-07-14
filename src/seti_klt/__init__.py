"""
seti_klt
--------
KLT-based denoising and RMT signal-detection toolkit for radio-astronomy
baseband data.

Public API
----------
    from seti_klt import KLT, TracyWidomTW2
    from seti_klt import KN_rankEst, KN_noiseEst, KN_mu_sigma, KN_s_Wishart

* :class:`KLT`            — main load → apply → inspect pipeline (C-KLT / MR-KLT),
                            RMT statistics, and the Tracy-Widom₂ Bayesian detector.
* :class:`TracyWidomTW2`  — exact Tracy-Widom₂ law (Painlevé II) and BBP helpers.
* ``KN_*``                — faithful port of the Kritchman & Nadler (2009) RMT
                            rank estimator, provided for comparison.
"""

from .KLT import KLT
from .tracy_widom import TracyWidomTW2
from .kritchman_nadler import KN_rankEst, KN_noiseEst, KN_mu_sigma, KN_s_Wishart

__all__ = [
    "KLT",
    "TracyWidomTW2",
    "KN_rankEst",
    "KN_noiseEst",
    "KN_mu_sigma",
    "KN_s_Wishart",
]
