"""True thermoelectric figure of merit ZT(T, n), power factor PF, and
n-type / p-type optimal carrier concentration extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from unizt.core.validator import ValidationError, validate_transport_data

ScalarOrCallableOrArray = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


def _resolve_temp_dependent(
    value: ScalarOrCallableOrArray, T: np.ndarray, name: str
) -> np.ndarray:
    if callable(value):
        arr = np.asarray(value(T), dtype=np.float64)
    elif np.ndim(value) == 0:
        arr = np.full_like(T, float(value), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != T.shape:
            raise ValidationError(
                f"'{name}' shape {arr.shape} does not match temperature array shape {T.shape}."
            )
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValidationError(f"'{name}' must be finite and strictly positive.")
    return arr


@dataclass
class ZTResult:
    """Complete thermoelectric post-processing output across (T, mu/n) grids."""

    T: np.ndarray
    mu: np.ndarray
    n: np.ndarray
    S: np.ndarray
    sigma_tau: np.ndarray
    kappa_e_tau: np.ndarray
    tau: np.ndarray
    sigma: np.ndarray
    pf: np.ndarray  # mW / (m * K^2)
    kappa_e: np.ndarray  # W / (m * K)
    kappa_l: np.ndarray  # W / (m * K)
    kappa_tot: np.ndarray  # W / (m * K)
    zt: np.ndarray
    T_grid: np.ndarray
    tau_grid: np.ndarray
    kappa_l_grid: np.ndarray
    n_type: dict[str, np.ndarray]
    p_type: dict[str, np.ndarray]
    overall_peak_zt: float
    overall_peak_type: str
    overall_peak_T: float
    overall_peak_n: float

    def summary_table(self) -> list[dict[str, float]]:
        """Return per-temperature summary rows for CSV / LaTeX / CLI reporting."""
        rows: list[dict[str, float]] = []
        for idx, temp in enumerate(self.T_grid):
            rows.append(
                {
                    "T_K": float(temp),
                    "tau_s": float(self.tau_grid[idx]),
                    "kappa_L_W_m_K": float(self.kappa_l_grid[idx]),
                    "ZT_max_n": float(self.n_type["zt_max"][idx]),
                    "n_opt_n": float(self.n_type["n_opt"][idx]),
                    "PF_n_mW_m_K2": float(self.n_type["pf_opt"][idx]),
                    "S_n_uV_K": float(self.n_type["S_opt"][idx] * 1e6),
                    "ZT_max_p": float(self.p_type["zt_max"][idx]),
                    "n_opt_p": float(self.p_type["n_opt"][idx]),
                    "PF_p_mW_m_K2": float(self.p_type["pf_opt"][idx]),
                    "S_p_uV_K": float(self.p_type["S_opt"][idx] * 1e6),
                }
            )
        return rows


def _extract_regime(
    T_grid: np.ndarray,
    T_rounded: np.ndarray,
    mask_regime: np.ndarray,
    mu: np.ndarray,
    n: np.ndarray,
    S: np.ndarray,
    sigma: np.ndarray,
    pf: np.ndarray,
    kappa_e: np.ndarray,
    kappa_l: np.ndarray,
    kappa_tot: np.ndarray,
    zt: np.ndarray,
) -> dict[str, np.ndarray]:
    n_t = len(T_grid)
    out = {
        "T": T_grid.copy(),
        "zt_max": np.full(n_t, np.nan, dtype=np.float64),
        "n_opt": np.full(n_t, np.nan, dtype=np.float64),
        "mu_opt": np.full(n_t, np.nan, dtype=np.float64),
        "S_opt": np.full(n_t, np.nan, dtype=np.float64),
        "sigma_opt": np.full(n_t, np.nan, dtype=np.float64),
        "pf_opt": np.full(n_t, np.nan, dtype=np.float64),
        "kappa_e_opt": np.full(n_t, np.nan, dtype=np.float64),
        "kappa_l_opt": np.full(n_t, np.nan, dtype=np.float64),
        "kappa_tot_opt": np.full(n_t, np.nan, dtype=np.float64),
    }
    for i, temp in enumerate(T_grid):
        idx_candidates = np.where(mask_regime & (T_rounded == np.round(temp, 6)))[0]
        if idx_candidates.size == 0:
            continue
        best_local = idx_candidates[int(np.argmax(zt[idx_candidates]))]
        out["zt_max"][i] = zt[best_local]
        out["n_opt"][i] = n[best_local]
        out["mu_opt"][i] = mu[best_local]
        out["S_opt"][i] = S[best_local]
        out["sigma_opt"][i] = sigma[best_local]
        out["pf_opt"][i] = pf[best_local]
        out["kappa_e_opt"][i] = kappa_e[best_local]
        out["kappa_l_opt"][i] = kappa_l[best_local]
        out["kappa_tot_opt"][i] = kappa_tot[best_local]
    return out


def compute_zt(
    transport_data: Mapping[str, Any],
    tau: ScalarOrCallableOrArray,
    kappa_l: ScalarOrCallableOrArray,
    *,
    vol_uc_m3: Optional[float] = None,
    n_unit: Optional[str] = None,
) -> ZTResult:
    """Combine BoltzTraP2/generic transport coefficients with ``tau(T)`` and ``kappa_L(T)``
    to compute Power Factor (mW/m/K^2), ``kappa_tot``, and true ``ZT(T, n)``.

    Carrier regimes follow BoltzTraP convention:
      - n-type: ``n < 0``
      - p-type: ``n > 0``
    """
    data = validate_transport_data(
        transport_data,
        require_positive_T=True,
        n_unit=n_unit,
        vol_uc_m3=vol_uc_m3,
    )

    required = ("T", "n", "S", "sigma_tau", "kappa_e_tau")
    for field in required:
        if field not in data:
            raise ValidationError(f"Transport data missing required field '{field}'.")

    T = data["T"]
    n = data["n"]
    S = data["S"]
    sigma_tau = data["sigma_tau"]
    kappa_e_tau = data["kappa_e_tau"]
    mu = data["mu"] if "mu" in data else np.zeros_like(T)

    tau_arr = _resolve_temp_dependent(tau, T, "tau")
    kappa_l_arr = _resolve_temp_dependent(kappa_l, T, "kappa_l")

    sigma = sigma_tau * tau_arr
    pf = (S ** 2) * sigma * 1e3  # mW / (m * K^2)
    kappa_e = kappa_e_tau * tau_arr
    kappa_tot = kappa_e + kappa_l_arr

    if np.any(kappa_tot <= 0.0):
        raise ValidationError("Total thermal conductivity (kappa_e + kappa_L) must be positive.")

    zt = (S ** 2) * sigma * T / kappa_tot

    T_rounded = np.round(T, 6)
    T_grid, first_idx = np.unique(T_rounded, return_index=True)
    tau_grid = tau_arr[first_idx]
    kappa_l_grid = kappa_l_arr[first_idx]

    n_type_res = _extract_regime(
        T_grid, T_rounded, n < 0.0, mu, n, S, sigma, pf, kappa_e, kappa_l_arr, kappa_tot, zt
    )
    p_type_res = _extract_regime(
        T_grid, T_rounded, n > 0.0, mu, n, S, sigma, pf, kappa_e, kappa_l_arr, kappa_tot, zt
    )

    best_n_val = np.nanmax(n_type_res["zt_max"]) if np.any(np.isfinite(n_type_res["zt_max"])) else -np.inf
    best_p_val = np.nanmax(p_type_res["zt_max"]) if np.any(np.isfinite(p_type_res["zt_max"])) else -np.inf

    if best_n_val >= best_p_val and np.isfinite(best_n_val):
        idx = int(np.nanargmax(n_type_res["zt_max"]))
        peak_zt = float(n_type_res["zt_max"][idx])
        peak_type = "n-type"
        peak_T = float(T_grid[idx])
        peak_n = float(n_type_res["n_opt"][idx])
    elif np.isfinite(best_p_val):
        idx = int(np.nanargmax(p_type_res["zt_max"]))
        peak_zt = float(p_type_res["zt_max"][idx])
        peak_type = "p-type"
        peak_T = float(T_grid[idx])
        peak_n = float(p_type_res["n_opt"][idx])
    else:
        idx = int(np.argmax(zt))
        peak_zt = float(zt[idx])
        peak_type = "intrinsic"
        peak_T = float(T[idx])
        peak_n = float(n[idx])

    return ZTResult(
        T=T,
        mu=mu,
        n=n,
        S=S,
        sigma_tau=sigma_tau,
        kappa_e_tau=kappa_e_tau,
        tau=tau_arr,
        sigma=sigma,
        pf=pf,
        kappa_e=kappa_e,
        kappa_l=kappa_l_arr,
        kappa_tot=kappa_tot,
        zt=zt,
        T_grid=T_grid,
        tau_grid=tau_grid,
        kappa_l_grid=kappa_l_grid,
        n_type=n_type_res,
        p_type=p_type_res,
        overall_peak_zt=peak_zt,
        overall_peak_type=peak_type,
        overall_peak_T=peak_T,
        overall_peak_n=peak_n,
    )
