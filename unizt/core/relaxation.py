"""Electronic relaxation time models: Bardeen-Shockley Deformation Potential Theory (DPT),
constant relaxation time approximation (CRTA), and flexible power-law sensitivity modes.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import constants as const

from unizt.core.validator import ValidationError

ArrayOrFloat = Union[float, np.ndarray]


def _format_return(val: np.ndarray, is_scalar: bool) -> ArrayOrFloat:
    return float(val.item()) if is_scalar else val


def tau_dpt(
    T: Union[float, np.ndarray],
    m_eff: float,
    E1: float,
    C_3d: float,
) -> ArrayOrFloat:
    """Compute acoustic-phonon relaxation time tau_DPT(T) (s) via 3D Bardeen-Shockley
    Deformation Potential Theory:

        tau_DPT(T) = 2 * sqrt(2*pi) * hbar^4 * C_3d / (3 * (m*)^(3/2) * (k_B * T)^(3/2) * E1^2)

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in K (strictly > 0).
    m_eff : float
        Carrier effective mass in units of free electron mass ``m_0``.
    E1 : float
        Acoustic deformation potential in eV.
    C_3d : float
        3D longitudinal elastic modulus ``C_3d`` in GPa.

    Returns
    -------
    float or np.ndarray
        Relaxation time in seconds, scaling naturally as ``T^(-3/2)``.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0) or not np.all(np.isfinite(T_arr)):
        raise ValidationError("Temperature T must be finite and strictly positive (K).")
    if m_eff <= 0.0:
        raise ValidationError(f"Effective mass m_eff must be positive (in m_0), got {m_eff}.")
    if E1 == 0.0 or not np.isfinite(E1):
        raise ValidationError(f"Deformation potential E1 must be finite and nonzero (eV), got {E1}.")
    if C_3d <= 0.0:
        raise ValidationError(f"3D elastic modulus C_3d must be positive (GPa), got {C_3d}.")

    m_star = float(m_eff) * const.electron_mass
    E1_joule = abs(float(E1)) * const.eV
    C_3d_pa = float(C_3d) * 1e9

    numerator = 2.0 * np.sqrt(2.0 * const.pi) * (const.hbar ** 4) * C_3d_pa
    denominator = (
        3.0
        * (m_star ** 1.5)
        * ((const.k * T_arr) ** 1.5)
        * (E1_joule ** 2)
    )
    tau = numerator / denominator
    return _format_return(tau, is_scalar)


def tau_constant(
    T: Union[float, np.ndarray],
    tau_0: float = 1e-14,
) -> ArrayOrFloat:
    """Return constant relaxation time approximation ``tau(T) = tau_0`` (s)."""
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0):
        raise ValidationError("Temperature T must be strictly positive (K).")
    if tau_0 <= 0.0 or not np.isfinite(tau_0):
        raise ValidationError(f"Constant relaxation time tau_0 must be positive (s), got {tau_0}.")

    tau = np.full_like(T_arr, float(tau_0), dtype=np.float64)
    return _format_return(tau, is_scalar)


def tau_power_law(
    T: Union[float, np.ndarray],
    tau_0: float = 1e-14,
    T_0: float = 300.0,
    exponent: float = -1.5,
) -> ArrayOrFloat:
    """Compute phenomenological power-law relaxation time ``tau_0 * (T / T_0)^p`` (s)."""
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0):
        raise ValidationError("Temperature T must be strictly positive (K).")
    if tau_0 <= 0.0 or T_0 <= 0.0:
        raise ValidationError("tau_0 and T_0 must be strictly positive.")

    tau = float(tau_0) * ((T_arr / float(T_0)) ** float(exponent))
    return _format_return(tau, is_scalar)


def compute_relaxation_time(
    T: Union[float, np.ndarray],
    mode: str = "dpt",
    *,
    m_eff: Optional[float] = None,
    E1: Optional[float] = None,
    C_3d: Optional[float] = None,
    tau_const: float = 1e-14,
    tau_0: float = 1e-14,
    T_0: float = 300.0,
    exponent: float = -1.5,
) -> ArrayOrFloat:
    """Unified relaxation time dispatcher supporting ``'dpt'``, ``'constant'``, and ``'power_law'``."""
    m = mode.strip().lower()
    if m == "dpt":
        missing = [
            name
            for name, val in [("m_eff", m_eff), ("E1", E1), ("C_3d", C_3d)]
            if val is None
        ]
        if missing:
            raise ValidationError(f"DPT relaxation mode requires: {', '.join(missing)}.")
        return tau_dpt(T, m_eff=float(m_eff), E1=float(E1), C_3d=float(C_3d))
    elif m in {"constant", "crta", "const"}:
        return tau_constant(T, tau_0=float(tau_const))
    elif m in {"power_law", "power-law", "power"}:
        return tau_power_law(T, tau_0=float(tau_0), T_0=float(T_0), exponent=float(exponent))
    else:
        raise ValidationError(
            f"Unknown relaxation mode '{mode}'; expected 'dpt', 'constant', or 'power_law'."
        )
