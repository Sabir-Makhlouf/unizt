"""True thermoelectric figure of merit ZT(T, n), physical RBA windowing,
spin-polarized two-channel combination, and n-type / p-type optimal extraction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
from scipy import constants as const

from unizt.core.validator import ValidationError, validate_transport_data

ScalarOrCallableOrArray = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]

RY_TO_EV = float(const.physical_constants["Rydberg constant times hc in eV"][0])


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


def _infer_fermi_energy(
    T: np.ndarray,
    mu: np.ndarray,
    n: np.ndarray,
    e_fermi: Optional[float] = None,
    mu_unit: str = "eV",
) -> tuple[float, float, np.ndarray]:
    """Resolve neutral Fermi level ``E_F`` and compute ``mu - E_F`` in eV."""
    unit_low = mu_unit.strip().lower()
    scale_to_ev = RY_TO_EV if unit_low == "ry" else 1.0

    if e_fermi is not None:
        ef_native = float(e_fermi)
    elif mu.size == 0 or np.allclose(mu, mu[0]):
        ef_native = float(mu[0]) if mu.size else 0.0
    else:
        T_unique = np.unique(np.round(T, 6))
        # Use T closest to 300 K as reference for neutrality crossing
        T_ref = float(T_unique[int(np.argmin(np.abs(T_unique - 300.0)))])
        ref_mask = np.round(T, 6) == T_ref
        mu_ref = mu[ref_mask]
        n_ref = n[ref_mask]
        order = np.argsort(mu_ref)
        mu_ref = mu_ref[order]
        n_ref = n_ref[order]

        max_n = float(np.max(np.abs(n_ref))) if n_ref.size else 0.0
        zero_tol = max(1e-12 * max_n, 1e-30)
        zero_idx = np.where(np.abs(n_ref) <= zero_tol)[0]

        if zero_idx.size >= 2 and np.any(np.diff(zero_idx) == 1):
            # Insulating gap plateau: choose midpoint of zero-carrier plateau
            ef_native = float(0.5 * (mu_ref[zero_idx[0]] + mu_ref[zero_idx[-1]]))
        else:
            cross_idx = np.where(
                ((n_ref[:-1] >= 0.0) & (n_ref[1:] <= 0.0))
                | ((n_ref[:-1] <= 0.0) & (n_ref[1:] >= 0.0))
            )[0]
            if cross_idx.size > 0:
                idx = int(cross_idx[0])
                dn_val = float(n_ref[idx + 1] - n_ref[idx])
                if abs(dn_val) > 0.0:
                    frac = -float(n_ref[idx]) / dn_val
                    ef_native = float(mu_ref[idx] + frac * (mu_ref[idx + 1] - mu_ref[idx]))
                else:
                    ef_native = float(0.5 * (mu_ref[idx] + mu_ref[idx + 1]))
            else:
                ef_native = float(mu_ref[int(np.argmin(np.abs(n_ref)))])

    ef_ev = ef_native * scale_to_ev
    dmu_ev = (mu - ef_native) * scale_to_ev
    return ef_native, ef_ev, dmu_ev


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
    # Physical windowing, Fermi reference, and intrinsic curve metadata
    e_fermi: float = 0.0
    e_fermi_ev: float = 0.0
    mu_unit: str = "eV"
    n_unit: str = "m^-3"
    dmu_ev: Optional[np.ndarray] = None
    ef_window_ev: Optional[float] = None
    max_carrier_density_cm3: Optional[float] = None
    physical_mask: Optional[np.ndarray] = None
    at_ef: Optional[dict[str, np.ndarray]] = None
    unconstrained_peak_zt: Optional[float] = None
    unconstrained_peak_n: Optional[float] = None
    spin_polarized: bool = False
    S_up: Optional[np.ndarray] = None
    S_dn: Optional[np.ndarray] = None
    sigma_up: Optional[np.ndarray] = None
    sigma_dn: Optional[np.ndarray] = None
    kappa_e_up: Optional[np.ndarray] = None
    kappa_e_dn: Optional[np.ndarray] = None
    kappa_e_mix: Optional[np.ndarray] = None

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
                    "dmu_n_eV": float(self.n_type["dmu_opt_eV"][idx]),
                    "PF_n_mW_m_K2": float(self.n_type["pf_opt"][idx]),
                    "S_n_uV_K": float(self.n_type["S_opt"][idx] * 1e6),
                    "ZT_max_p": float(self.p_type["zt_max"][idx]),
                    "n_opt_p": float(self.p_type["n_opt"][idx]),
                    "dmu_p_eV": float(self.p_type["dmu_opt_eV"][idx]),
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
    dmu_ev: np.ndarray,
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
        "dmu_opt_eV": np.full(n_t, np.nan, dtype=np.float64),
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
        out["dmu_opt_eV"][i] = dmu_ev[best_local]
        out["S_opt"][i] = S[best_local]
        out["sigma_opt"][i] = sigma[best_local]
        out["pf_opt"][i] = pf[best_local]
        out["kappa_e_opt"][i] = kappa_e[best_local]
        out["kappa_l_opt"][i] = kappa_l[best_local]
        out["kappa_tot_opt"][i] = kappa_tot[best_local]
    return out


def _extract_at_fermi(
    T_grid: np.ndarray,
    T_rounded: np.ndarray,
    mu: np.ndarray,
    e_fermi: float,
    n: np.ndarray,
    S: np.ndarray,
    sigma: np.ndarray,
    pf: np.ndarray,
    kappa_e: np.ndarray,
    kappa_l: np.ndarray,
    kappa_tot: np.ndarray,
    zt: np.ndarray,
    *,
    S_up: Optional[np.ndarray] = None,
    S_dn: Optional[np.ndarray] = None,
    kappa_e_mix: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Evaluate exact intrinsic/undoped transport curve at ``mu = E_F`` vs ``T``."""
    n_t = len(T_grid)
    out: dict[str, np.ndarray] = {
        "T": T_grid.copy(),
        "n": np.zeros(n_t, dtype=np.float64),
        "S": np.zeros(n_t, dtype=np.float64),
        "S_up": np.full(n_t, np.nan, dtype=np.float64),
        "S_dn": np.full(n_t, np.nan, dtype=np.float64),
        "sigma": np.zeros(n_t, dtype=np.float64),
        "pf": np.zeros(n_t, dtype=np.float64),
        "kappa_e": np.zeros(n_t, dtype=np.float64),
        "kappa_e_mix": np.zeros(n_t, dtype=np.float64),
        "kappa_l": np.zeros(n_t, dtype=np.float64),
        "kappa_tot": np.zeros(n_t, dtype=np.float64),
        "zt": np.zeros(n_t, dtype=np.float64),
    }

    for i, temp in enumerate(T_grid):
        m = T_rounded == np.round(temp, 6)
        mu_k = mu[m]
        order = np.argsort(mu_k)
        mu_s = mu_k[order]

        if mu_s.size == 1 or e_fermi <= mu_s[0] or e_fermi >= mu_s[-1]:
            idx_nearest = int(np.argmin(np.abs(mu_s - e_fermi)))
            interp = lambda arr: float(arr[m][order][idx_nearest])
        else:
            interp = lambda arr: float(np.interp(e_fermi, mu_s, arr[m][order]))

        n_val = interp(n)
        s_val = interp(S)
        sig_val = interp(sigma)
        ke_val = interp(kappa_e)
        kl_val = interp(kappa_l)
        ktot_val = ke_val + kl_val
        pf_val = (s_val ** 2) * sig_val * 1e3
        zt_val = ((s_val ** 2) * sig_val * temp / ktot_val) if ktot_val > 0.0 else 0.0

        out["n"][i] = n_val
        out["S"][i] = s_val
        out["sigma"][i] = sig_val
        out["pf"][i] = pf_val
        out["kappa_e"][i] = ke_val
        out["kappa_l"][i] = kl_val
        out["kappa_tot"][i] = ktot_val
        out["zt"][i] = zt_val

        if S_up is not None and S_dn is not None:
            out["S_up"][i] = interp(S_up)
            out["S_dn"][i] = interp(S_dn)
        if kappa_e_mix is not None:
            out["kappa_e_mix"][i] = interp(kappa_e_mix)

    return out


def _align_spin_channel_grid(
    up_data: Mapping[str, Any],
    dn_data: Mapping[str, Any],
    sigma_key: str,
    kappa_key: str,
) -> tuple[np.ndarray, ...]:
    """Align spin-up and spin-down channels onto a common (T, mu) grid."""
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

    Aligns matching ``(T, Ef)`` grids and computes:

      1. ``sigma_tot = sigma_up + sigma_dn``
      2. ``S_tot = (sigma_up * S_up + sigma_dn * S_dn) / (sigma_up + sigma_dn)``
      3. ``kappa_e_mix = (sigma_up * sigma_dn / (sigma_up + sigma_dn)) * (S_up - S_dn)^2 * T``
      4. ``kappa_e_tot = kappa_e_up + kappa_e_dn + kappa_e_mix``
      5. ``PF_tot = S_tot^2 * sigma_tot``
      6. ``ZT_tot = (S_tot^2 * sigma_tot * T) / (kappa_e_tot + kappa_L(T))`` (when ``kappa_l`` is given)
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
        "mu_unit": up_clean.get("mu_unit", dn_clean.get("mu_unit", "eV")),
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
    ef_window: Optional[float] = None,
    max_carrier_density: Optional[float] = None,
    e_fermi: Optional[float] = None,
    mu_unit: str = "auto",
    extract_at_ef: bool = True,
    warn_unconstrained: bool = True,
) -> ZTResult:
    """Combine BoltzTraP2/generic transport coefficients with ``tau(T)`` and ``kappa_L(T)``
    to compute Power Factor (mW/m/K^2), ``kappa_tot``, and true ``ZT(T, n)``.

    Optional physical Rigid Band Approximation (RBA) windowing:
      - ``ef_window`` (eV): restricts peak optimization to ``|mu - E_F| <= ef_window``.
      - ``max_carrier_density`` (cm^-3): restricts optimization to ``|n| <= max_carrier_density``.
      - ``extract_at_ef``: extracts intrinsic transport curve at ``mu = E_F``.
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

    resolved_n_unit = str(data.get("n_unit", "m^-3"))
    resolved_mu_unit = (
        str(data.get("mu_unit", "eV"))
        if mu_unit.strip().lower() == "auto"
        else mu_unit.strip()
    )

    tau_arr = _resolve_temp_dependent(tau, T, "tau")
    kappa_l_arr = _resolve_temp_dependent(kappa_l, T, "kappa_l")

    sigma = sigma_tau * tau_arr
    pf = (S ** 2) * sigma * 1e3  # mW / (m * K^2)
    kappa_e = kappa_e_tau * tau_arr
    kappa_tot = kappa_e + kappa_l_arr

    if np.any(kappa_tot <= 0.0):
        raise ValidationError("Total thermal conductivity (kappa_e + kappa_L) must be positive.")

    zt = (S ** 2) * sigma * T / kappa_tot

    ef_native, ef_ev, dmu_ev = _infer_fermi_energy(
        T, mu, n, e_fermi=e_fermi, mu_unit=resolved_mu_unit
    )

    phys_mask = np.ones(T.shape, dtype=bool)
    if ef_window is not None:
        if float(ef_window) <= 0.0:
            raise ValidationError("ef_window must be strictly positive (eV).")
        phys_mask &= np.abs(dmu_ev) <= float(ef_window)
    if max_carrier_density is not None:
        if float(max_carrier_density) <= 0.0:
            raise ValidationError("max_carrier_density must be strictly positive (cm^-3).")
        if resolved_n_unit != "m^-3":
            raise ValidationError(
                "max_carrier_density filtering requires carrier density in SI m^-3 "
                "(supply vol_uc_m3 or n_unit='m^-3'/'cm^-3')."
            )
        max_n_m3 = float(max_carrier_density) * 1e6
        phys_mask &= np.abs(n) <= max_n_m3

    if not np.any(phys_mask):
        raise ValidationError(
            "No transport grid points satisfy the physical windowing constraints "
            f"(ef_window={ef_window} eV, max_carrier_density={max_carrier_density} cm^-3)."
        )

    T_rounded = np.round(T, 6)
    T_grid, first_idx = np.unique(T_rounded, return_index=True)
    tau_grid = tau_arr[first_idx]
    kappa_l_grid = kappa_l_arr[first_idx]

    n_type_res = _extract_regime(
        T_grid,
        T_rounded,
        phys_mask & (n < 0.0),
        mu,
        dmu_ev,
        n,
        S,
        sigma,
        pf,
        kappa_e,
        kappa_l_arr,
        kappa_tot,
        zt,
    )
    p_type_res = _extract_regime(
        T_grid,
        T_rounded,
        phys_mask & (n > 0.0),
        mu,
        dmu_ev,
        n,
        S,
        sigma,
        pf,
        kappa_e,
        kappa_l_arr,
        kappa_tot,
        zt,
    )

    best_n_val = (
        np.nanmax(n_type_res["zt_max"]) if np.any(np.isfinite(n_type_res["zt_max"])) else -np.inf
    )
    best_p_val = (
        np.nanmax(p_type_res["zt_max"]) if np.any(np.isfinite(p_type_res["zt_max"])) else -np.inf
    )

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
        cand = np.where(phys_mask)[0]
        idx = int(cand[int(np.argmax(zt[cand]))])
        peak_zt = float(zt[idx])
        peak_type = "intrinsic"
        peak_T = float(T[idx])
        peak_n = float(n[idx])

    raw_best_idx = int(np.argmax(zt))
    raw_peak_zt = float(zt[raw_best_idx])
    raw_peak_n = float(n[raw_best_idx])

    if (
        warn_unconstrained
        and max_carrier_density is None
        and resolved_n_unit == "m^-3"
        and abs(peak_n) > 1e27  # 1e21 cm^-3
    ):
        warnings.warn(
            f"Unconstrained peak ZT ({peak_zt:.3f}) occurs at |n|={abs(peak_n)*1e-6:.2e} cm^-3 "
            "(> 1e21 cm^-3), where the Rigid Band Approximation may break down. "
            "Consider specifying ef_window and max_carrier_density.",
            UserWarning,
            stacklevel=2,
        )

    spin_pol = bool(data.get("spin_polarized", False))
    S_up = data.get("S_up")
    S_dn = data.get("S_dn")
    sig_up = data["sigma_tau_up"] * tau_arr if "sigma_tau_up" in data else data.get("sigma_up")
    sig_dn = data["sigma_tau_dn"] * tau_arr if "sigma_tau_dn" in data else data.get("sigma_dn")
    ke_up = data["kappa_e_tau_up"] * tau_arr if "kappa_e_tau_up" in data else data.get("kappa_e_up")
    ke_dn = data["kappa_e_tau_dn"] * tau_arr if "kappa_e_tau_dn" in data else data.get("kappa_e_dn")
    ke_mix = data["kappa_e_mix_tau"] * tau_arr if "kappa_e_mix_tau" in data else data.get("kappa_e_mix")

    at_ef_res = None
    if extract_at_ef:
        at_ef_res = _extract_at_fermi(
            T_grid,
            T_rounded,
            mu,
            ef_native,
            n,
            S,
            sigma,
            pf,
            kappa_e,
            kappa_l_arr,
            kappa_tot,
            zt,
            S_up=S_up,
            S_dn=S_dn,
            kappa_e_mix=ke_mix,
        )

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
        e_fermi=ef_native,
        e_fermi_ev=ef_ev,
        mu_unit=resolved_mu_unit,
        n_unit=resolved_n_unit,
        dmu_ev=dmu_ev,
        ef_window_ev=float(ef_window) if ef_window is not None else None,
        max_carrier_density_cm3=(
            float(max_carrier_density) if max_carrier_density is not None else None
        ),
        physical_mask=phys_mask,
        at_ef=at_ef_res,
        unconstrained_peak_zt=raw_peak_zt,
        unconstrained_peak_n=raw_peak_n,
        spin_polarized=spin_pol,
        S_up=S_up,
        S_dn=S_dn,
        sigma_up=sig_up,
        sigma_dn=sig_dn,
        kappa_e_up=ke_up,
        kappa_e_dn=ke_dn,
        kappa_e_mix=ke_mix,
    )


def optimize_zt(
    transport_data: Mapping[str, Any],
    tau: ScalarOrCallableOrArray,
    kappa_l: ScalarOrCallableOrArray,
    *,
    vol_uc_m3: Optional[float] = None,
    n_unit: Optional[str] = None,
    ef_window: Optional[float] = 0.5,
    max_carrier_density: Optional[float] = 1e21,
    e_fermi: Optional[float] = None,
    mu_unit: str = "auto",
    extract_at_ef: bool = True,
    warn_unconstrained: bool = True,
) -> ZTResult:
    """Physically constrained optimal ZT extractor (defaults: ``|mu - E_F| <= 0.5 eV``
    and ``|n| <= 1e21 cm^-3``).
    """
    return compute_zt(
        transport_data,
        tau=tau,
        kappa_l=kappa_l,
        vol_uc_m3=vol_uc_m3,
        n_unit=n_unit,
        ef_window=ef_window,
        max_carrier_density=max_carrier_density,
        e_fermi=e_fermi,
        mu_unit=mu_unit,
        extract_at_ef=extract_at_ef,
        warn_unconstrained=warn_unconstrained,
    )
