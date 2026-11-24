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
        T_unique, inv_idx = np.unique(T, return_inverse=True)
        arr_u = np.asarray(value(T_unique), dtype=np.float64)
        if arr_u.ndim == 0:
            arr = np.full_like(T, float(arr_u), dtype=np.float64)
        elif arr_u.shape == T_unique.shape:
            arr = arr_u[inv_idx]
        else:
            arr = np.asarray(value(T), dtype=np.float64)
            if arr.shape != T.shape:
                raise ValidationError(
                    f"'{name}' shape {arr.shape} does not match temperature array shape {T.shape}."
                )
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


def _align_spin_channel_grid(
    up_data: Mapping[str, Any],
    dn_data: Mapping[str, Any],
    sigma_key: str,
    kappa_key: str,
) -> tuple[np.ndarray, ...]:
    """Align spin-up and spin-down channels onto a common (T, mu) grid.

    Uses exact point-to-point values when (T, mu) grids match; otherwise intersects
    temperatures and linearly interpolates spin-down quantities onto the overlapping
    spin-up chemical potential grid at each temperature.
    """
    T_up = np.asarray(up_data["T"], dtype=np.float64)
    T_dn = np.asarray(dn_data["T"], dtype=np.float64)
    mu_up = np.asarray(up_data["mu"], dtype=np.float64) if "mu" in up_data else np.zeros_like(T_up)
    mu_dn = np.asarray(dn_data["mu"], dtype=np.float64) if "mu" in dn_data else np.zeros_like(T_dn)

    S_up = np.asarray(up_data["S"], dtype=np.float64)
    S_dn = np.asarray(dn_data["S"], dtype=np.float64)
    sig_up = np.asarray(up_data[sigma_key], dtype=np.float64)
    sig_dn = np.asarray(dn_data[sigma_key], dtype=np.float64)
    kap_up = np.asarray(up_data[kappa_key], dtype=np.float64)
    kap_dn = np.asarray(dn_data[kappa_key], dtype=np.float64)
    n_up = np.asarray(up_data["n"], dtype=np.float64) if "n" in up_data else np.zeros_like(T_up)
    n_dn = np.asarray(dn_data["n"], dtype=np.float64) if "n" in dn_data else np.zeros_like(T_dn)
    dos_up = np.asarray(up_data["DOS"], dtype=np.float64) if "DOS" in up_data else np.zeros_like(T_up)
    dos_dn = np.asarray(dn_data["DOS"], dtype=np.float64) if "DOS" in dn_data else np.zeros_like(T_dn)

    if (
        T_up.shape == T_dn.shape
        and np.allclose(T_up, T_dn, rtol=1e-7, atol=1e-6)
        and np.allclose(mu_up, mu_dn, rtol=1e-7, atol=1e-8)
    ):
        return (
            T_up,
            mu_up,
            n_up,
            n_dn,
            dos_up,
            dos_dn,
            S_up,
            S_dn,
            sig_up,
            sig_dn,
            kap_up,
            kap_dn,
        )

    T_up_r = np.round(T_up, 6)
    T_dn_r = np.round(T_dn, 6)
    common_temps = np.intersect1d(T_up_r, T_dn_r)
    if common_temps.size == 0:
        raise ValidationError(
            "Spin-up and spin-down transport data share no common temperatures."
        )

    chunks: list[tuple[np.ndarray, ...]] = []
    for t_val in common_temps:
        u_mask = T_up_r == t_val
        d_mask = T_dn_r == t_val

        u_mu = mu_up[u_mask]
        d_mu = mu_dn[d_mask]
        u_order = np.argsort(u_mu)
        d_order = np.argsort(d_mu)

        u_mu = u_mu[u_order]
        d_mu = d_mu[d_order]

        if u_mu.shape == d_mu.shape and np.allclose(u_mu, d_mu, rtol=1e-7, atol=1e-8):
            chunks.append(
                (
                    T_up[u_mask][u_order],
                    u_mu,
                    n_up[u_mask][u_order],
                    n_dn[d_mask][d_order],
                    dos_up[u_mask][u_order],
                    dos_dn[d_mask][d_order],
                    S_up[u_mask][u_order],
                    S_dn[d_mask][d_order],
                    sig_up[u_mask][u_order],
                    sig_dn[d_mask][d_order],
                    kap_up[u_mask][u_order],
                    kap_dn[d_mask][d_order],
                )
            )
            continue

        mu_lo = max(float(u_mu[0]), float(d_mu[0]))
        mu_hi = min(float(u_mu[-1]), float(d_mu[-1]))
        if mu_hi <= mu_lo:
            raise ValidationError(
                f"No overlapping chemical potential Ef range at T={t_val} K."
            )
        ov = (u_mu >= mu_lo - 1e-12) & (u_mu <= mu_hi + 1e-12)
        mu_c = u_mu[ov]
        if mu_c.size == 0:
            continue

        chunks.append(
            (
                np.full(mu_c.shape, float(T_up[u_mask][0]), dtype=np.float64),
                mu_c,
                n_up[u_mask][u_order][ov],
                np.interp(mu_c, d_mu, n_dn[d_mask][d_order]),
                dos_up[u_mask][u_order][ov],
                np.interp(mu_c, d_mu, dos_dn[d_mask][d_order]),
                S_up[u_mask][u_order][ov],
                np.interp(mu_c, d_mu, S_dn[d_mask][d_order]),
                sig_up[u_mask][u_order][ov],
                np.maximum(0.0, np.interp(mu_c, d_mu, sig_dn[d_mask][d_order])),
                kap_up[u_mask][u_order][ov],
                np.maximum(0.0, np.interp(mu_c, d_mu, kap_dn[d_mask][d_order])),
            )
        )

    if not chunks:
        raise ValidationError("Failed to align spin-up and spin-down grids.")

    return tuple(np.concatenate([c[i] for c in chunks]) for i in range(12))  # type: ignore[return-value]


def combine_spin_channels(
    up_data: Mapping[str, Any],
    dn_data: Mapping[str, Any],
    *,
    tau: Optional[ScalarOrCallableOrArray] = None,
    kappa_l: Optional[ScalarOrCallableOrArray] = None,
    vol_uc_m3: Optional[float] = None,
    n_unit: Optional[str] = None,
) -> dict[str, Any]:
    """Combine spin-up and spin-down transport channels using the exact two-current model.

    Aligns matching ``(T, Ef)`` grids (interpolating spin-down onto overlapping
    spin-up chemical potentials if endpoints differ slightly) and computes:

      1. ``sigma_tot = sigma_up + sigma_dn``
      2. ``S_tot = (sigma_up * S_up + sigma_dn * S_dn) / (sigma_up + sigma_dn)``
         (safely returns ``0.0`` where ``sigma_tot == 0``)
      3. ``kappa_e_tot = kappa_e_up + kappa_e_dn + (sigma_up * sigma_dn / (sigma_up + sigma_dn)) * (S_up - S_dn)^2 * T``
      4. ``PF_tot = S_tot^2 * sigma_tot``
      5. ``ZT_tot = (S_tot^2 * sigma_tot * T) / (kappa_e_tot + kappa_L(T))`` (when ``kappa_l`` is given)

    Supports both BoltzTraP2 ``(sigma_tau, kappa_e_tau)`` inputs and direct
    ``(sigma, kappa_e)`` inputs.
    """
    has_tau_norm = ("sigma_tau" in up_data) and ("sigma_tau" in dn_data)
    has_direct = ("sigma" in up_data) and ("sigma" in dn_data)
    if not has_tau_norm and not has_direct:
        raise ValidationError(
            "Both spin channels must provide either ('sigma_tau', 'kappa_e_tau') or ('sigma', 'kappa_e')."
        )

    sigma_key = "sigma_tau" if has_tau_norm else "sigma"
    kappa_key = "kappa_e_tau" if has_tau_norm else "kappa_e"
    if kappa_key not in up_data or kappa_key not in dn_data:
        raise ValidationError(f"Both spin channels must provide '{kappa_key}'.")

    up_clean = validate_transport_data(
        up_data, require_positive_T=True, n_unit=n_unit, vol_uc_m3=vol_uc_m3
    )
    dn_clean = validate_transport_data(
        dn_data, require_positive_T=True, n_unit=n_unit, vol_uc_m3=vol_uc_m3
    )

    (
        T,
        mu,
        n_up,
        n_dn,
        dos_up,
        dos_dn,
        S_up,
        S_dn,
        sig_up_raw,
        sig_dn_raw,
        kap_up_raw,
        kap_dn_raw,
    ) = _align_spin_channel_grid(up_clean, dn_clean, sigma_key, kappa_key)

    sig_up_raw = np.maximum(0.0, sig_up_raw)
    sig_dn_raw = np.maximum(0.0, sig_dn_raw)
    kap_up_raw = np.maximum(0.0, kap_up_raw)
    kap_dn_raw = np.maximum(0.0, kap_dn_raw)

    sig_tot_raw = sig_up_raw + sig_dn_raw
    with np.errstate(divide="ignore", invalid="ignore"):
        S_tot = np.where(
            sig_tot_raw > 0.0,
            (sig_up_raw * S_up + sig_dn_raw * S_dn) / sig_tot_raw,
            0.0,
        )
        mix_raw = np.where(
            sig_tot_raw > 0.0,
            (sig_up_raw * sig_dn_raw / sig_tot_raw) * ((S_up - S_dn) ** 2) * T,
            0.0,
        )

    kap_tot_raw = kap_up_raw + kap_dn_raw + mix_raw
    pf_raw = (S_tot ** 2) * sig_tot_raw

    combined: dict[str, Any] = {
        "T": T,
        "mu": mu,
        "n": n_up + n_dn,
        "n_up": n_up,
        "n_dn": n_dn,
        "DOS": dos_up + dos_dn,
        "S": S_tot,
        "S_tot": S_tot,
        "S_up": S_up,
        "S_dn": S_dn,
        "n_unit": up_clean.get("n_unit", "m^-3"),
        "spin_polarized": True,
    }

    if has_tau_norm:
        combined["sigma_tau"] = sig_tot_raw
        combined["sigma_tau_up"] = sig_up_raw
        combined["sigma_tau_dn"] = sig_dn_raw
        combined["kappa_e_tau"] = kap_tot_raw
        combined["kappa_e_tau_up"] = kap_up_raw
        combined["kappa_e_tau_dn"] = kap_dn_raw
        combined["kappa_e_mix_tau"] = mix_raw
        combined["pf_tau"] = pf_raw
    else:
        combined["sigma"] = sig_tot_raw
        combined["sigma_tot"] = sig_tot_raw
        combined["sigma_up"] = sig_up_raw
        combined["sigma_dn"] = sig_dn_raw
        combined["kappa_e"] = kap_tot_raw
        combined["kappa_e_tot"] = kap_tot_raw
        combined["kappa_e_up"] = kap_up_raw
        combined["kappa_e_dn"] = kap_dn_raw
        combined["kappa_e_mix"] = mix_raw
        combined["pf"] = pf_raw
        combined["pf_tot"] = pf_raw
        combined["pf_mW"] = pf_raw * 1e3
        combined["sigma_tau"] = sig_tot_raw
        combined["kappa_e_tau"] = kap_tot_raw

    if tau is not None:
        tau_arr = _resolve_temp_dependent(tau, T, "tau")
        sigma_abs = sig_tot_raw * tau_arr
        kappa_e_abs = kap_tot_raw * tau_arr
        pf_abs = (S_tot ** 2) * sigma_abs
        combined["tau"] = tau_arr
        combined["sigma"] = sigma_abs
        combined["sigma_tot"] = sigma_abs
        combined["sigma_up"] = sig_up_raw * tau_arr
        combined["sigma_dn"] = sig_dn_raw * tau_arr
        combined["kappa_e"] = kappa_e_abs
        combined["kappa_e_tot"] = kappa_e_abs
        combined["kappa_e_up"] = kap_up_raw * tau_arr
        combined["kappa_e_dn"] = kap_dn_raw * tau_arr
        combined["kappa_e_mix"] = mix_raw * tau_arr
        combined["pf"] = pf_abs
        combined["pf_tot"] = pf_abs
        combined["pf_mW"] = pf_abs * 1e3

    if kappa_l is not None:
        kappa_l_arr = _resolve_temp_dependent(kappa_l, T, "kappa_l")
        sigma_for_zt = combined.get("sigma", sig_tot_raw)
        kappa_e_for_zt = combined.get("kappa_e", kap_tot_raw)
        kappa_tot = kappa_e_for_zt + kappa_l_arr
        zt_tot = (S_tot ** 2) * sigma_for_zt * T / kappa_tot
        combined["kappa_l"] = kappa_l_arr
        combined["kappa_tot"] = kappa_tot
        combined["zt"] = zt_tot
        combined["zt_tot"] = zt_tot

    return combined


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
