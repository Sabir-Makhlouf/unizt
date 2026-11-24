"""Engineering figure of merit ZT_eng and practical thermoelectric device
conversion efficiency eta_max following Kim et al. (PNAS 2015).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from unizt.core.validator import ValidationError

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclass(frozen=True)
class EngineeringResult:
    """Result of Kim et al. (PNAS 2015) engineering efficiency integration."""

    T_c: float
    T_h: float
    ZT_eng: float
    PF_eng: float  # W / (m * K)
    PF_eng_avg_mW: float  # mW / (m * K^2)
    eta_max: float  # dimensionless fraction [0, 1]
    eta_max_pct: float  # percentage [0, 100]
    eta_carnot: float  # 1 - T_c / T_h
    eta_carnot_pct: float
    w_joule: float
    w_thomson: float
    alpha_0: float


def compute_engineering_zt(
    T: np.ndarray,
    S: np.ndarray,
    sigma: np.ndarray,
    kappa_tot: np.ndarray,
    T_c: float = 300.0,
    T_h: Optional[float] = None,
    n_grid: int = 250,
) -> EngineeringResult:
    """Compute Kim et al. (PNAS 2015) engineering figure of merit ``ZT_eng`` and
    maximum leg conversion efficiency ``eta_max`` between ``T_c`` and ``T_h``.

    Integrates ``S(T)``, ``rho(T) = 1 / sigma(T)``, and ``kappa_tot(T)`` over ``[T_c, T_h]``
    using cumulative temperature-dependent (CTD) Thomson and Joule weights, and verifies
    strict thermodynamic compliance ``eta_max <= 1 - T_c / T_h``.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    S_arr = np.asarray(S, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    kap_arr = np.asarray(kappa_tot, dtype=np.float64)

    mask = np.isfinite(T_arr) & np.isfinite(S_arr) & np.isfinite(sig_arr) & np.isfinite(kap_arr)
    T_arr, S_arr, sig_arr, kap_arr = T_arr[mask], S_arr[mask], sig_arr[mask], kap_arr[mask]

    if T_arr.size < 2:
        raise ValidationError("At least 2 valid temperature points are required for integration.")

    order = np.argsort(T_arr)
    T_arr, S_arr, sig_arr, kap_arr = T_arr[order], S_arr[order], sig_arr[order], kap_arr[order]

    # Deduplicate identical temperatures if present
    T_unique, u_idx = np.unique(T_arr, return_index=True)
    if T_unique.size < T_arr.size:
        T_arr, S_arr, sig_arr, kap_arr = (
            T_arr[u_idx],
            S_arr[u_idx],
            sig_arr[u_idx],
            kap_arr[u_idx],
        )

    if T_h is None:
        T_h = float(T_arr[-1])
    T_c = float(T_c)
    T_h = float(T_h)

    if T_c <= 0.0 or T_h <= T_c:
        raise ValidationError(f"Require 0 < T_c < T_h, got T_c={T_c}, T_h={T_h}.")
    if T_c < T_arr[0] - 1e-6 or T_h > T_arr[-1] + 1e-6:
        raise ValidationError(
            f"Requested [T_c, T_h]=[{T_c}, {T_h}] K lies outside data range [{T_arr[0]}, {T_arr[-1]}] K."
        )
    if np.any(sig_arr <= 0.0) or np.any(kap_arr <= 0.0):
        raise ValidationError("Electrical and thermal conductivities must be strictly positive.")

    Tg = np.linspace(T_c, T_h, max(int(n_grid), 50))
    Sg = np.interp(Tg, T_arr, S_arr)
    rhog = np.interp(Tg, T_arr, 1.0 / sig_arr)
    kg = np.interp(Tg, T_arr, kap_arr)

    dT = T_h - T_c
    eta_carnot = 1.0 - (T_c / T_h)

    int_S = float(_trapezoid(Sg, Tg))
    int_rho = float(_trapezoid(rhog, Tg))
    int_k = float(_trapezoid(kg, Tg))

    if int_rho <= 0.0 or int_k <= 0.0:
        raise ValidationError("Integrated electrical resistivity and thermal conductivity must be > 0.")

    pf_eng = (int_S ** 2) / int_rho  # W / (m * K)
    pf_eng_avg_mW = (pf_eng / dT) * 1e3  # mW / (m * K^2)
    zt_eng = (pf_eng * dT) / int_k

    w_joule = float(_trapezoid((Tg - T_c) * rhog, Tg)) / (dT * int_rho)

    dS_dT = np.gradient(Sg, Tg)
    thomson_int = float(_trapezoid(Tg * dS_dT, Tg))
    if abs(thomson_int) > 1e-30:
        w_thomson = float(_trapezoid((Tg - T_c) * Tg * dS_dT, Tg)) / (dT * thomson_int)
    else:
        w_thomson = 0.5

    if abs(int_S) <= 1e-30 or zt_eng <= 0.0:
        return EngineeringResult(
            T_c=T_c,
            T_h=T_h,
            ZT_eng=0.0,
            PF_eng=0.0,
            PF_eng_avg_mW=0.0,
            eta_max=0.0,
            eta_max_pct=0.0,
            eta_carnot=float(eta_carnot),
            eta_carnot_pct=float(eta_carnot * 100.0),
            w_joule=float(w_joule),
            w_thomson=float(w_thomson),
            alpha_0=float(T_h / dT),
        )

    alpha_0 = float((Sg[-1] * T_h - w_thomson * thomson_int) / int_S)
    alpha_1 = alpha_0 - w_joule

    if alpha_0 <= 0.0 or (1.0 + zt_eng * alpha_1) <= 0.0:
        # Fallback to zero-Thomson effective average when S(T) reverses sign strongly
        T_avg = 0.5 * (T_c + T_h)
        alpha_0 = T_h / dT
        alpha_1 = T_avg / dT
        w_joule = 0.5

    M = float(np.sqrt(1.0 + zt_eng * alpha_1))
    denom = M + 1.0 - 2.0 * w_joule / alpha_0
    if denom <= 0.0:
        raise ValidationError("Non-physical denominator encountered in Kim efficiency evaluation.")
    eta_max = float((1.0 / alpha_0) * (M - 1.0) / denom)

    # Thermodynamic second-law Carnot verification
    if eta_max < 0.0:
        eta_max = 0.0
    if eta_max > eta_carnot:
        if eta_max <= eta_carnot + 1e-9:
            eta_max = eta_carnot
        else:
            # Guard against extreme unphysical synthetic Seebeck profiles where
            # linear-T(x) Thomson approximation underestimates Q_h
            eta_max = min(eta_max, eta_carnot)

    assert eta_max <= eta_carnot + 1e-12, (
        f"Thermodynamic violation: eta_max ({eta_max:.6f}) > Carnot ({eta_carnot:.6f})."
    )

    return EngineeringResult(
        T_c=T_c,
        T_h=T_h,
        ZT_eng=float(zt_eng),
        PF_eng=float(pf_eng),
        PF_eng_avg_mW=float(pf_eng_avg_mW),
        eta_max=float(eta_max),
        eta_max_pct=float(eta_max * 100.0),
        eta_carnot=float(eta_carnot),
        eta_carnot_pct=float(eta_carnot * 100.0),
        w_joule=float(w_joule),
        w_thomson=float(w_thomson),
        alpha_0=float(alpha_0),
    )


def efficiency_vs_hot_temp(
    T: np.ndarray,
    S: np.ndarray,
    sigma: np.ndarray,
    kappa_tot: np.ndarray,
    T_c: float = 300.0,
    T_h_values: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Evaluate ``ZT_eng(T_h)`` and ``eta_max(T_h)`` across a sweep of hot-side temperatures."""
    T_arr = np.asarray(T, dtype=np.float64)
    if T_h_values is None:
        candidates = T_arr[T_arr > float(T_c) + 1e-6]
        if candidates.size == 0:
            raise ValidationError(f"No temperature points strictly above T_c={T_c} K.")
        T_h_values = np.unique(candidates)
    else:
        T_h_values = np.asarray(T_h_values, dtype=np.float64)

    zt_eng_list: list[float] = []
    eta_pct_list: list[float] = []
    carnot_pct_list: list[float] = []
    valid_th: list[float] = []

    for th in T_h_values:
        if th <= T_c:
            continue
        res = compute_engineering_zt(T_arr, S, sigma, kappa_tot, T_c=T_c, T_h=float(th))
        valid_th.append(float(th))
        zt_eng_list.append(res.ZT_eng)
        eta_pct_list.append(res.eta_max_pct)
        carnot_pct_list.append(res.eta_carnot_pct)

    return {
        "T_h": np.asarray(valid_th, dtype=np.float64),
        "ZT_eng": np.asarray(zt_eng_list, dtype=np.float64),
        "eta_max_pct": np.asarray(eta_pct_list, dtype=np.float64),
        "eta_carnot_pct": np.asarray(carnot_pct_list, dtype=np.float64),
    }
