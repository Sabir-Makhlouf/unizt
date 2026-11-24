"""Lattice thermal conductivity models: Slack's anharmonic Umklapp equation,
Cahill-Pohl minimum thermal conductivity, and custom/experimental CSV override.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from scipy import constants as const
from scipy.integrate import quad

from unizt.core.elastic import ElasticProperties
from unizt.core.validator import ValidationError

ArrayOrFloat = Union[float, np.ndarray]


def _format_return(val: np.ndarray, is_scalar: bool) -> ArrayOrFloat:
    return float(val.item()) if is_scalar else val


def slack_kappa_l(
    T: Union[float, np.ndarray],
    M_avg: float,
    vol_atom: float,
    theta_D: float,
    gamma: float,
    n_atoms: int,
    scale_theta_by_n: bool = True,
) -> ArrayOrFloat:
    """Compute lattice thermal conductivity kappa_L(T) (W/m/K) via Slack's equation.

    Uses Julian's Grüneisen-dependent prefactor (Slack 1979; Morelli & Slack 2006):

        kappa_L(T) = A(gamma) * M_avg * theta_a^3 * delta / (gamma^2 * n_atoms^(2/3) * T)

    where:
      - ``A(gamma) = 2.43e-6 / (1 - 0.514/gamma + 0.228/gamma^2)`` (for ``M_avg`` in
        amu, ``delta`` in Angstrom, ``theta_a`` and ``T`` in K, giving W/m/K)
      - ``theta_a = theta_D * n_atoms^(-1/3)`` when ``scale_theta_by_n=True``
        (acoustic-branch cutoff convention)
      - ``delta = (vol_atom)^(1/3) / 1e-10`` (cubic root of atomic volume in Angstrom)

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in K (must be > 0).
    M_avg : float
        Average atomic mass in amu (g/mol).
    vol_atom : float
        Average volume per atom (``delta^3``) in m^3 (SI).
    theta_D : float
        Debye temperature in K.
    gamma : float
        Acoustic Grüneisen parameter (dimensionless, nonzero).
    n_atoms : int
        Number of atoms per primitive unit cell (>= 1).
    scale_theta_by_n : bool, optional
        If True (default, Morelli & Slack convention), scale Anderson's Debye
        temperature ``theta_D`` to the acoustic-branch cutoff ``theta_a = theta_D * n^(-1/3)``.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0) or not np.all(np.isfinite(T_arr)):
        raise ValidationError("Temperature T must be finite and strictly positive (K).")
    if M_avg <= 0.0 or vol_atom <= 0.0 or theta_D <= 0.0 or n_atoms <= 0:
        raise ValidationError(
            "M_avg, vol_atom, theta_D, and n_atoms must all be strictly positive."
        )
    if gamma == 0.0:
        raise ValidationError("Grüneisen parameter gamma cannot be zero.")

    g = float(gamma)
    denom_a = 1.0 - 0.514 / g + 0.228 / (g ** 2)
    if denom_a <= 0.0:
        raise ValidationError(f"Invalid Grüneisen factor denominator for gamma={g}.")

    A_gamma = 2.43e-6 / denom_a
    delta_ang = (float(vol_atom) ** (1.0 / 3.0)) / const.angstrom
    theta_a = float(theta_D) * (float(n_atoms) ** (-1.0 / 3.0)) if scale_theta_by_n else float(theta_D)

    kappa = (
        A_gamma
        * float(M_avg)
        * (theta_a ** 3)
        * delta_ang
        / ((g ** 2) * (float(n_atoms) ** (2.0 / 3.0)) * T_arr)
    )
    return _format_return(kappa, is_scalar)


def _debye_heat_capacity_integral(upper_limit: float) -> float:
    """Evaluate int_0^u x^3 * exp(x) / (exp(x) - 1)^2 dx safely without overflow."""
    if upper_limit <= 0.0:
        return 0.0

    def integrand(x: float) -> float:
        if x < 1e-7:
            return x
        if x > 100.0:
            return 0.0
        ex = np.exp(-x)
        denom = np.expm1(-x)
        return float((x ** 3) * ex / (denom * denom))

    u = min(float(upper_limit), 100.0)
    val, _ = quad(integrand, 0.0, u, epsabs=1e-12, epsrel=1e-10, limit=100)
    return float(val)


def cahill_pohl_kappa_min(
    T: Union[float, np.ndarray],
    v_l: float,
    v_t: float,
    vol_atom: float,
) -> ArrayOrFloat:
    """Compute Cahill-Pohl minimum (amorphous-limit) thermal conductivity kappa_min(T) (W/m/K).

    Follows Cahill, Watson & Pohl, Phys. Rev. B 46, 6131 (1992):

        kappa_min(T) = (pi / 6)^(1/3) * k_B * n^(2/3)
                       * sum_{i=l,t,t} v_i * (T / Theta_i)^2 * int_0^{Theta_i/T} x^3 e^x / (e^x - 1)^2 dx

    where ``n = 1 / vol_atom`` (m^-3) and ``Theta_i = v_i * (hbar / k_B) * (6*pi^2*n)^(1/3)``.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0) or not np.all(np.isfinite(T_arr)):
        raise ValidationError("Temperature T must be finite and strictly positive (K).")
    if v_l <= 0.0 or v_t <= 0.0 or vol_atom <= 0.0:
        raise ValidationError("v_l, v_t, and vol_atom must be strictly positive.")

    n_density = 1.0 / float(vol_atom)
    prefactor = ((const.pi / 6.0) ** (1.0 / 3.0)) * const.k * (n_density ** (2.0 / 3.0))
    q_d = (6.0 * (const.pi ** 2) * n_density) ** (1.0 / 3.0)

    theta_l = float(v_l) * (const.hbar / const.k) * q_d
    theta_t = float(v_t) * (const.hbar / const.k) * q_d
    modes = [(float(v_l), theta_l, 1.0), (float(v_t), theta_t, 2.0)]

    T_flat = np.atleast_1d(T_arr).astype(np.float64)
    T_unique, inv_idx = np.unique(T_flat, return_inverse=True)
    out_unique = np.zeros_like(T_unique, dtype=np.float64)

    for idx, temp in enumerate(T_unique):
        total = 0.0
        for vel, theta_i, weight in modes:
            integral = _debye_heat_capacity_integral(theta_i / temp)
            total += weight * vel * ((temp / theta_i) ** 2) * integral
        out_unique[idx] = prefactor * total

    out = out_unique[inv_idx]
    return float(out.item()) if is_scalar else out.reshape(T_arr.shape)


def load_custom_kappa_csv(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load user-supplied ``temperature_K,kappa_L_W_m_K`` CSV file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Custom kappa_L CSV file not found: {p}")

    temps: list[float] = []
    kappas: list[float] = []
    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            try:
                t_val = float(first)
                k_val = float(row[1].strip())
            except (ValueError, IndexError):
                # Skip header row (e.g. temperature_K,kappa_L_W_m_K)
                if not temps:
                    continue
                raise ValidationError(f"Malformed numeric row in {p}: {row}")
            if t_val <= 0.0 or k_val <= 0.0:
                raise ValidationError(
                    f"Custom kappa_L CSV values must be positive; got T={t_val}, kappa_L={k_val}."
                )
            temps.append(t_val)
            kappas.append(k_val)

    if len(temps) < 2:
        raise ValidationError(
            f"Custom kappa_L CSV '{p}' must contain at least 2 valid data rows."
        )

    order = np.argsort(temps)
    return np.asarray(temps, dtype=np.float64)[order], np.asarray(kappas, dtype=np.float64)[order]


def evaluate_custom_kappa(
    T: Union[float, np.ndarray],
    T_ref: np.ndarray,
    kappa_ref: np.ndarray,
    interpolation: str = "loglog",
    extrapolate: bool = True,
) -> ArrayOrFloat:
    """Interpolate user-provided ``kappa_L(T)`` via ``linear`` or ``loglog`` scaling."""
    T_arr = np.asarray(T, dtype=np.float64)
    is_scalar = T_arr.ndim == 0
    if np.any(T_arr <= 0.0):
        raise ValidationError("Temperature T must be strictly positive (K).")

    order = np.argsort(T_ref)
    t_r = np.asarray(T_ref, dtype=np.float64)[order]
    k_r = np.asarray(kappa_ref, dtype=np.float64)[order]

    interp_mode = interpolation.strip().lower()
    if interp_mode == "linear":
        if extrapolate and len(t_r) >= 2:
            val = np.interp(T_arr, t_r, k_r)
            low_mask = T_arr < t_r[0]
            high_mask = T_arr > t_r[-1]
            if np.any(low_mask):
                slope_lo = (k_r[1] - k_r[0]) / (t_r[1] - t_r[0])
                val = np.where(low_mask, np.maximum(1e-6, k_r[0] + slope_lo * (T_arr - t_r[0])), val)
            if np.any(high_mask):
                slope_hi = (k_r[-1] - k_r[-2]) / (t_r[-1] - t_r[-2])
                val = np.where(high_mask, np.maximum(1e-6, k_r[-1] + slope_hi * (T_arr - t_r[-1])), val)
        else:
            val = np.interp(T_arr, t_r, k_r)
    elif interp_mode in {"loglog", "log-log", "log"}:
        lt_r = np.log(t_r)
        lk_r = np.log(k_r)
        lt = np.log(T_arr)
        lk = np.interp(lt, lt_r, lk_r)
        if extrapolate and len(t_r) >= 2:
            low_mask = lt < lt_r[0]
            high_mask = lt > lt_r[-1]
            if np.any(low_mask):
                slope_lo = (lk_r[1] - lk_r[0]) / (lt_r[1] - lt_r[0])
                lk = np.where(low_mask, lk_r[0] + slope_lo * (lt - lt_r[0]), lk)
            if np.any(high_mask):
                slope_hi = (lk_r[-1] - lk_r[-2]) / (lt_r[-1] - lt_r[-2])
                lk = np.where(high_mask, lk_r[-1] + slope_hi * (lt - lt_r[-1]), lk)
        val = np.exp(lk)
    else:
        raise ValidationError(
            f"Unsupported interpolation mode '{interpolation}'; expected 'linear' or 'loglog'."
        )

    return _format_return(np.asarray(val, dtype=np.float64), is_scalar)


def compute_kappa_l(
    T: Union[float, np.ndarray],
    mode: str = "slack",
    *,
    elastic_props: Optional[ElasticProperties] = None,
    M_avg: Optional[float] = None,
    vol_atom: Optional[float] = None,
    theta_D: Optional[float] = None,
    gamma: Optional[float] = None,
    n_atoms: Optional[int] = None,
    v_l: Optional[float] = None,
    v_t: Optional[float] = None,
    custom_csv: Optional[Union[str, Path]] = None,
    interpolation: str = "loglog",
    scale_theta_by_n: bool = True,
    floor_cahill: bool = False,
) -> ArrayOrFloat:
    """High-level dispatcher for lattice thermal conductivity ``kappa_L(T)``.

    Supported modes: ``"slack"``, ``"cahill"``, and ``"custom"``.
    """
    m = mode.strip().lower()
    if elastic_props is not None:
        if vol_atom is None:
            vol_atom = elastic_props.vol_atom
        if theta_D is None:
            theta_D = elastic_props.theta_D
        if gamma is None:
            gamma = elastic_props.gamma
        if v_l is None:
            v_l = elastic_props.v_l
        if v_t is None:
            v_t = elastic_props.v_t

    if m == "slack":
        missing = [
            name
            for name, val in [
                ("M_avg", M_avg),
                ("vol_atom", vol_atom),
                ("theta_D", theta_D),
                ("gamma", gamma),
                ("n_atoms", n_atoms),
            ]
            if val is None
        ]
        if missing:
            raise ValidationError(f"Slack kappa_L mode requires: {', '.join(missing)}.")
        k_slack = slack_kappa_l(
            T,
            M_avg=float(M_avg),
            vol_atom=float(vol_atom),
            theta_D=float(theta_D),
            gamma=float(gamma),
            n_atoms=int(n_atoms),
            scale_theta_by_n=scale_theta_by_n,
        )
        if floor_cahill and v_l is not None and v_t is not None:
            k_min = cahill_pohl_kappa_min(T, v_l=float(v_l), v_t=float(v_t), vol_atom=float(vol_atom))
            if np.ndim(T) == 0:
                return float(max(k_slack, k_min))
            return np.maximum(k_slack, k_min)
        return k_slack
    elif m in {"cahill", "cahill-pohl", "kappa_min"}:
        missing = [
            name
            for name, val in [("v_l", v_l), ("v_t", v_t), ("vol_atom", vol_atom)]
            if val is None
        ]
        if missing:
            raise ValidationError(f"Cahill-Pohl kappa_min mode requires: {', '.join(missing)}.")
        return cahill_pohl_kappa_min(T, v_l=float(v_l), v_t=float(v_t), vol_atom=float(vol_atom))
    elif m == "custom":
        if custom_csv is None:
            raise ValidationError("Custom kappa_L mode requires 'custom_csv' path.")
        t_ref, k_ref = load_custom_kappa_csv(custom_csv)
        return evaluate_custom_kappa(T, t_ref, k_ref, interpolation=interpolation)
    else:
        raise ValidationError(
            f"Unknown kappa_L mode '{mode}'; choose from 'slack', 'cahill', or 'custom'."
        )
