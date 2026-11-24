"""Researcher-grade numerical output bundle exporter for UniZT.

Generates in the target output directory:
  1) ``report_summary.txt``    — Human-readable report covering elastic, thermal,
                                 intrinsic, constrained optimal, and device metrics.
  2) ``summary_metrics.json``  — Machine-readable JSON of all scalar metrics & provenance.
  3) ``transport_at_ef.csv``   — Temperature-dependent intrinsic curve at ``mu = E_F``.
  4) ``optimal_doping_vs_T.csv`` — Optimal n-type & p-type curves inside physical window.
  5) ``table_summary.tex``     — Publication-ready ``booktabs`` LaTeX summary table.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from unizt import __author__, __version__
from unizt.core.elastic import ElasticProperties
from unizt.core.engineering import EngineeringResult
from unizt.core.kappa_l import cahill_pohl_kappa_min
from unizt.core.zt import ZTResult


def _fmt(val: Optional[float], spec: str = ".3f") -> str:
    if val is None or not np.isfinite(val):
        return "--"
    return format(float(val), spec)


def _fmt_sci(val: Optional[float]) -> str:
    if val is None or not np.isfinite(val):
        return "--"
    return f"{float(val):.2e}"


def _sample_at_temp(T_grid: np.ndarray, values: np.ndarray, target_T: float) -> float:
    if T_grid.size == 0 or values.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(T_grid - target_T)))
    return float(values[idx])


def _best_regime_point(regime: Mapping[str, np.ndarray]) -> dict[str, Optional[float]]:
    zt_arr = regime["zt_max"]
    valid = np.isfinite(zt_arr)
    if not np.any(valid):
        return {
            "ZT_max": None,
            "T_opt_K": None,
            "n_opt_cm3": None,
            "dmu_eV": None,
            "S_uV_K": None,
            "PF_mW_mK2": None,
            "sigma_S_m": None,
            "kappa_tot_W_mK": None,
        }
    idx = int(np.nanargmax(zt_arr))
    return {
        "ZT_max": float(zt_arr[idx]),
        "T_opt_K": float(regime["T"][idx]),
        "n_opt_cm3": float(regime["n_opt"][idx] * 1e-6),
        "dmu_eV": float(regime["dmu_opt_eV"][idx]),
        "S_uV_K": float(regime["S_opt"][idx] * 1e6),
        "PF_mW_mK2": float(regime["pf_opt"][idx]),
        "sigma_S_m": float(regime["sigma_opt"][idx]),
        "kappa_tot_W_mK": float(regime["kappa_tot_opt"][idx]),
    }


def export_research_bundle(
    zt_result: ZTResult,
    outdir: Union[str, Path],
    *,
    elastic_props: Optional[ElasticProperties] = None,
    eng_result: Optional[EngineeringResult] = None,
    eng_result_n: Optional[EngineeringResult] = None,
    eng_result_p: Optional[EngineeringResult] = None,
    kappa_min_grid: Optional[np.ndarray] = None,
    material_name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    tex_filename: str = "table_summary.tex",
    tex_temperatures: Optional[Sequence[float]] = None,
    tau_mode: Optional[str] = None,
    kappa_mode: Optional[str] = None,
) -> dict[str, Path]:
    """Export all 5 researcher-grade numerical tables, JSON metrics, and summary report."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    T_grid = zt_result.T_grid
    if kappa_min_grid is None and elastic_props is not None:
        if elastic_props.v_l and elastic_props.v_t and elastic_props.vol_atom:
            kappa_min_grid = np.asarray(
                cahill_pohl_kappa_min(
                    T_grid,
                    v_l=elastic_props.v_l,
                    v_t=elastic_props.v_t,
                    vol_atom=elastic_props.vol_atom,
                ),
                dtype=np.float64,
            )

    T_300 = float(T_grid[int(np.argmin(np.abs(T_grid - 300.0)))])
    T_max = float(np.max(T_grid))
    k_l_300 = _sample_at_temp(T_grid, zt_result.kappa_l_grid, 300.0)
    k_l_tmax = _sample_at_temp(T_grid, zt_result.kappa_l_grid, T_max)
    tau_300 = _sample_at_temp(T_grid, zt_result.tau_grid, 300.0)
    tau_tmax = _sample_at_temp(T_grid, zt_result.tau_grid, T_max)
    k_min_300 = (
        _sample_at_temp(T_grid, kappa_min_grid, 300.0)
        if kappa_min_grid is not None
        else None
    )
    k_min_tmax = (
        _sample_at_temp(T_grid, kappa_min_grid, T_max)
        if kappa_min_grid is not None
        else None
    )

    best_n = _best_regime_point(zt_result.n_type)
    best_p = _best_regime_point(zt_result.p_type)

    intrinsic_summary: dict[str, Optional[float]] = {
        "ZT_peak": None,
        "T_peak_K": None,
        "S_at_peak_uV_K": None,
        "S_at_300K_uV_K": None,
        "ZT_at_300K": None,
    }
    if zt_result.at_ef is not None and len(zt_result.at_ef["T"]) > 0:
        ef_d = zt_result.at_ef
        i_pk = int(np.argmax(ef_d["zt"]))
        i_300 = int(np.argmin(np.abs(ef_d["T"] - 300.0)))
        intrinsic_summary = {
            "ZT_peak": float(ef_d["zt"][i_pk]),
            "T_peak_K": float(ef_d["T"][i_pk]),
            "S_at_peak_uV_K": float(ef_d["S"][i_pk] * 1e6),
            "S_at_300K_uV_K": float(ef_d["S"][i_300] * 1e6),
            "ZT_at_300K": float(ef_d["zt"][i_300]),
        }

    # 1. transport_at_ef.csv
    at_ef_path = out_path / "transport_at_ef.csv"
    ef_cols = [
        "T_K",
        "S_up_uV_K",
        "S_dn_uV_K",
        "S_tot_uV_K",
        "sigma_tot_S_m",
        "PF_mW_mK2",
        "kappa_e_tot_W_mK",
        "kappa_e_mix_W_mK",
        "kappa_L_W_mK",
        "ZT_tot",
    ]
    with at_ef_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ef_cols)
        writer.writeheader()
        if zt_result.at_ef is not None:
            ef_d = zt_result.at_ef
            for i in range(len(ef_d["T"])):
                writer.writerow(
                    {
                        "T_K": float(ef_d["T"][i]),
                        "S_up_uV_K": (
                            float(ef_d["S_up"][i] * 1e6)
                            if np.isfinite(ef_d["S_up"][i])
                            else float("nan")
                        ),
                        "S_dn_uV_K": (
                            float(ef_d["S_dn"][i] * 1e6)
                            if np.isfinite(ef_d["S_dn"][i])
                            else float("nan")
                        ),
                        "S_tot_uV_K": float(ef_d["S"][i] * 1e6),
                        "sigma_tot_S_m": float(ef_d["sigma"][i]),
                        "PF_mW_mK2": float(ef_d["pf"][i]),
                        "kappa_e_tot_W_mK": float(ef_d["kappa_e"][i]),
                        "kappa_e_mix_W_mK": float(ef_d["kappa_e_mix"][i]),
                        "kappa_L_W_mK": float(ef_d["kappa_l"][i]),
                        "ZT_tot": float(ef_d["zt"][i]),
                    }
                )

    # 2. optimal_doping_vs_T.csv
    opt_path = out_path / "optimal_doping_vs_T.csv"
    opt_cols = [
        "T_K",
        "tau_fs",
        "kappa_L_W_mK",
        "ZT_n",
        "n_opt_n_cm3",
        "dmu_n_eV",
        "S_n_uV_K",
        "PF_n_mW_mK2",
        "ZT_p",
        "n_opt_p_cm3",
        "dmu_p_eV",
        "S_p_uV_K",
        "PF_p_mW_mK2",
    ]
    opt_rows: list[dict[str, float]] = []
    for i, temp in enumerate(T_grid):
        n_scale = 1e-6 if zt_result.n_unit == "m^-3" else 1.0
        opt_rows.append(
            {
                "T_K": float(temp),
                "tau_fs": float(zt_result.tau_grid[i] * 1e15),
                "kappa_L_W_mK": float(zt_result.kappa_l_grid[i]),
                "ZT_n": float(zt_result.n_type["zt_max"][i]),
                "n_opt_n_cm3": float(zt_result.n_type["n_opt"][i] * n_scale),
                "dmu_n_eV": float(zt_result.n_type["dmu_opt_eV"][i]),
                "S_n_uV_K": float(zt_result.n_type["S_opt"][i] * 1e6),
                "PF_n_mW_mK2": float(zt_result.n_type["pf_opt"][i]),
                "ZT_p": float(zt_result.p_type["zt_max"][i]),
                "n_opt_p_cm3": float(zt_result.p_type["n_opt"][i] * n_scale),
                "dmu_p_eV": float(zt_result.p_type["dmu_opt_eV"][i]),
                "S_p_uV_K": float(zt_result.p_type["S_opt"][i] * 1e6),
                "PF_p_mW_mK2": float(zt_result.p_type["pf_opt"][i]),
            }
        )
    with opt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=opt_cols)
        writer.writeheader()
        writer.writerows(opt_rows)

    # 3. table_summary.tex
    tex_path = out_path / tex_filename
    if tex_temperatures is not None:
        t_targets = {round(float(x), 2) for x in tex_temperatures}
        tex_rows = [r for r in opt_rows if round(r["T_K"], 2) in t_targets]
        if not tex_rows:
            tex_rows = opt_rows
    elif len(opt_rows) > 15:
        step = max(1, len(opt_rows) // 10)
        tex_rows = opt_rows[::step]
        if opt_rows[-1] not in tex_rows:
            tex_rows.append(opt_rows[-1])
    else:
        tex_rows = opt_rows

    mat_title = material_name or "Thermoelectric Material"
    cap_constraints = []
    if zt_result.ef_window_ev is not None:
        cap_constraints.append(rf"$|\mu - E_F| \leq {zt_result.ef_window_ev:.2f}\;\mathrm{{eV}}$")
    if zt_result.max_carrier_density_cm3 is not None:
        cap_constraints.append(rf"$|n| \leq {zt_result.max_carrier_density_cm3:.0e}\;\mathrm{{cm}}^{{-3}}$")
    constraint_note = (
        " under RBA constraints (" + ", ".join(cap_constraints) + ")"
        if cap_constraints
        else ""
    )

    tex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{Thermoelectric transport properties of {mat_title} computed with UniZT v{__version__}{constraint_note}.}}",
        r"\label{tab:unizt_summary}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$T$ (K) & $\tau$ (fs) & $\kappa_L$ (W\,m$^{-1}$\,K$^{-1}$) & "
        r"$ZT_{\max}^{(n)}$ & $n_{\mathrm{opt}}^{(n)}$ ($\mathrm{cm}^{-3}$) & $\mathrm{PF}^{(n)}$ & "
        r"$ZT_{\max}^{(p)}$ & $n_{\mathrm{opt}}^{(p)}$ ($\mathrm{cm}^{-3}$) & $\mathrm{PF}^{(p)}$ \\",
        r"\midrule",
    ]
    for r in tex_rows:
        tex_lines.append(
            f"{r['T_K']:.0f} & {r['tau_fs']:.2f} & {r['kappa_L_W_mK']:.3f} & "
            f"{_fmt(r['ZT_n'], '.3f')} & {_fmt_sci(r['n_opt_n_cm3'])} & {_fmt(r['PF_n_mW_mK2'], '.2f')} & "
            f"{_fmt(r['ZT_p'], '.3f')} & {_fmt_sci(r['n_opt_p_cm3'])} & {_fmt(r['PF_p_mW_mK2'], '.2f')} \\\\"
        )

    footer_notes: list[str] = []
    if eng_result_n is not None and eng_result_p is not None:
        footer_notes.append(
            rf"Device ($T_c={eng_result_n.T_c:.0f}\;\mathrm{{K}},\,T_h={eng_result_n.T_h:.0f}\;\mathrm{{K}}$): "
            rf"$n$-type $ZT_{{\mathrm{{eng}}}}={eng_result_n.ZT_eng:.2f}$ ($\eta_{{\max}}={eng_result_n.eta_max_pct:.2f}\%$); "
            rf"$p$-type $ZT_{{\mathrm{{eng}}}}={eng_result_p.ZT_eng:.2f}$ ($\eta_{{\max}}={eng_result_p.eta_max_pct:.2f}\%$)."
        )
    elif eng_result is not None:
        footer_notes.append(
            rf"Device ($T_c={eng_result.T_c:.0f}\;\mathrm{{K}},\,T_h={eng_result.T_h:.0f}\;\mathrm{{K}}$): "
            rf"$ZT_{{\mathrm{{eng}}}}={eng_result.ZT_eng:.2f}$, $\eta_{{\max}}={eng_result.eta_max_pct:.2f}\%$ "
            rf"(Carnot {eng_result.eta_carnot_pct:.2f}\%)."
        )
    if footer_notes:
        tex_lines.append(r"\midrule")
        tex_lines.append(r"\multicolumn{9}{l}{" + " ".join(footer_notes) + r"} \\")

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    # 4. summary_metrics.json
    def _eng_dict(er: Optional[EngineeringResult]) -> Optional[dict[str, float]]:
        if er is None:
            return None
        return {
            "T_c_K": er.T_c,
            "T_h_K": er.T_h,
            "ZT_eng": er.ZT_eng,
            "PF_eng_W_mK": er.PF_eng,
            "eta_max_pct": er.eta_max_pct,
            "eta_carnot_pct": er.eta_carnot_pct,
        }

    elastic_dict: Optional[dict[str, Any]] = None
    if elastic_props is not None:
        elastic_dict = {
            "born_stable": bool(np.all(elastic_props.eigenvalues > 0.0)),
            "min_eigenvalue_GPa": float(np.min(elastic_props.eigenvalues)),
            "B_VRH_GPa": elastic_props.B_H,
            "G_VRH_GPa": elastic_props.G_H,
            "E_GPa": elastic_props.E,
            "poisson_ratio": elastic_props.nu,
            "pugh_ratio_B_G": elastic_props.pugh_ratio,
            "anisotropy_index": elastic_props.anisotropy_index,
            "longitudinal_modulus_C3d_GPa": elastic_props.longitudinal_modulus,
            "density_kg_m3": elastic_props.density,
            "v_l_m_s": elastic_props.v_l,
            "v_t_m_s": elastic_props.v_t,
            "v_m_m_s": elastic_props.v_m,
            "theta_D_K": elastic_props.theta_D,
            "gruneisen_gamma": elastic_props.gamma,
            "vol_atom_m3": elastic_props.vol_atom,
        }

    metrics_payload: dict[str, Any] = {
        "tool": f"UniZT v{__version__}",
        "author": __author__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "material": material_name,
        "spin_polarized": zt_result.spin_polarized,
        "models": {
            "tau_mode": tau_mode,
            "kappa_mode": kappa_mode,
        },
        "constraints": {
            "e_fermi_native": zt_result.e_fermi,
            "e_fermi_eV": zt_result.e_fermi_ev,
            "mu_unit": zt_result.mu_unit,
            "ef_window_eV": zt_result.ef_window_ev,
            "max_carrier_density_cm3": zt_result.max_carrier_density_cm3,
            "valid_grid_points": int(np.sum(zt_result.physical_mask)) if zt_result.physical_mask is not None else int(len(zt_result.T)),
            "total_grid_points": int(len(zt_result.T)),
        },
        "elastic_and_acoustic": elastic_dict,
        "thermal_and_scattering": {
            "T_ref_K": T_300,
            "T_max_K": T_max,
            "kappa_L_300K_W_mK": k_l_300,
            "kappa_L_Tmax_W_mK": k_l_tmax,
            "kappa_min_300K_W_mK": k_min_300,
            "kappa_min_Tmax_W_mK": k_min_tmax,
            "tau_300K_s": tau_300,
            "tau_Tmax_s": tau_tmax,
        },
        "intrinsic_at_ef": intrinsic_summary,
        "constrained_optimal": {
            "overall_peak_ZT": zt_result.overall_peak_zt,
            "overall_peak_type": zt_result.overall_peak_type,
            "overall_peak_T_K": zt_result.overall_peak_T,
            "overall_peak_n_cm3": float(zt_result.overall_peak_n * 1e-6) if zt_result.n_unit == "m^-3" else zt_result.overall_peak_n,
            "n_type": best_n,
            "p_type": best_p,
        },
        "unconstrained_reference": {
            "peak_ZT": zt_result.unconstrained_peak_zt,
            "peak_n_cm3": (
                float(zt_result.unconstrained_peak_n * 1e-6)
                if zt_result.unconstrained_peak_n is not None and zt_result.n_unit == "m^-3"
                else zt_result.unconstrained_peak_n
            ),
        },
        "device_efficiency": {
            "dominant_or_single": _eng_dict(eng_result),
            "n_type": _eng_dict(eng_result_n),
            "p_type": _eng_dict(eng_result_p),
        },
        "provenance": dict(provenance) if provenance is not None else None,
    }
    json_path = out_path / "summary_metrics.json"
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # 5. report_summary.txt
    lines = [
        "=" * 76,
        f"UniZT v{__version__} | Researcher Summary Report"
        + (f" — {material_name}" if material_name else ""),
        "=" * 76,
        f"Spin Mode           : {'Two-channel spin-polarized (up + dn)' if zt_result.spin_polarized else 'Single-channel'}",
        f"Models              : tau={tau_mode or 'custom'} | kappa_L={kappa_mode or 'custom'}",
        f"Neutral Fermi E_F   : {zt_result.e_fermi:.6f} {zt_result.mu_unit} ({zt_result.e_fermi_ev:.4f} eV)",
        (
            f"Physical Window     : |mu - E_F| <= {zt_result.ef_window_ev} eV | "
            f"|n| <= {zt_result.max_carrier_density_cm3:.2e} cm^-3 "
            if zt_result.ef_window_ev is not None and zt_result.max_carrier_density_cm3 is not None
            else f"Physical Window     : ef_window={zt_result.ef_window_ev} eV | max_|n|={zt_result.max_carrier_density_cm3} cm^-3"
        ),
        f"Grid Points Kept    : {metrics_payload['constraints']['valid_grid_points']} / {metrics_payload['constraints']['total_grid_points']}",
        "",
    ]

    if elastic_props is not None:
        lines.extend(
            [
                "[1] Elastic & Acoustic Summary",
                f"  Born Stability    : PASS (min eig = {float(np.min(elastic_props.eigenvalues)):.2f} GPa > 0)",
                f"  VRH Moduli        : B = {elastic_props.B_H:.2f} GPa | G = {elastic_props.G_H:.2f} GPa | E = {elastic_props.E:.2f} GPa",
                f"  Poisson / Pugh    : nu = {elastic_props.nu:.4f} | B/G = {elastic_props.pugh_ratio:.2f}",
                (
                    f"  Sound Velocities  : v_l = {elastic_props.v_l:.1f} m/s | v_t = {elastic_props.v_t:.1f} m/s | v_m = {elastic_props.v_m:.1f} m/s"
                    if elastic_props.v_m
                    else "  Sound Velocities  : N/A (density not supplied)"
                ),
                (
                    f"  Debye & Gruneisen : theta_D = {elastic_props.theta_D:.2f} K | gamma = {elastic_props.gamma:.3f}"
                    if elastic_props.theta_D
                    else f"  Gruneisen Param   : gamma = {elastic_props.gamma:.3f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "[2] Thermal Conductivity & Carrier Scattering",
            f"  kappa_L(T={T_300:.0f} K) : {k_l_300:.4f} W/m/K"
            + (f" | kappa_min = {k_min_300:.4f} W/m/K" if k_min_300 is not None else ""),
            f"  kappa_L(T={T_max:.0f} K) : {k_l_tmax:.4f} W/m/K"
            + (f" | kappa_min = {k_min_tmax:.4f} W/m/K" if k_min_tmax is not None else ""),
            f"  tau(T={T_300:.0f} K)     : {tau_300:.3e} s ({tau_300*1e15:.2f} fs)",
            f"  tau(T={T_max:.0f} K)     : {tau_tmax:.3e} s ({tau_tmax*1e15:.2f} fs)",
            "",
            "[3] Intrinsic Transport at Neutral Fermi Level (mu = E_F)",
            f"  At 300 K          : ZT = {_fmt(intrinsic_summary['ZT_at_300K'], '.4f')} | S = {_fmt(intrinsic_summary['S_at_300K_uV_K'], '.2f')} uV/K",
            f"  Intrinsic Peak ZT : {_fmt(intrinsic_summary['ZT_peak'], '.4f')} at T = {_fmt(intrinsic_summary['T_peak_K'], '.0f')} K (S = {_fmt(intrinsic_summary['S_at_peak_uV_K'], '.2f')} uV/K)",
            "",
            "[4] Physically Constrained Optimal Doping Summary",
            (
                f"  n-type (n < 0)    : ZT_max = {_fmt(best_n['ZT_max'], '.4f')} at T = {_fmt(best_n['T_opt_K'], '.0f')} K | "
                f"n_opt = {_fmt_sci(best_n['n_opt_cm3'])} cm^-3 | dmu = {_fmt(best_n['dmu_eV'], '+.3f')} eV | "
                f"S = {_fmt(best_n['S_uV_K'], '.1f')} uV/K | PF = {_fmt(best_n['PF_mW_mK2'], '.3f')} mW/m/K^2"
            ),
            (
                f"  p-type (n > 0)    : ZT_max = {_fmt(best_p['ZT_max'], '.4f')} at T = {_fmt(best_p['T_opt_K'], '.0f')} K | "
                f"n_opt = {_fmt_sci(best_p['n_opt_cm3'])} cm^-3 | dmu = {_fmt(best_p['dmu_eV'], '+.3f')} eV | "
                f"S = {_fmt(best_p['S_uV_K'], '.1f')} uV/K | PF = {_fmt(best_p['PF_mW_mK2'], '.3f')} mW/m/K^2"
            ),
            (
                f"  Unconstrained Ref : ZT_raw = {_fmt(zt_result.unconstrained_peak_zt, '.3f')} at |n| = "
                f"{_fmt_sci(abs(zt_result.unconstrained_peak_n) * 1e-6 if zt_result.unconstrained_peak_n else None)} cm^-3"
            ),
            "",
            "[5] Kim et al. Device Engineering Efficiency",
        ]
    )

    if eng_result_n is not None or eng_result_p is not None:
        if eng_result_n is not None:
            lines.append(
                f"  n-type ({eng_result_n.T_c:.0f}->{eng_result_n.T_h:.0f} K) : ZT_eng = {eng_result_n.ZT_eng:.3f} | "
                f"eta_max = {eng_result_n.eta_max_pct:.2f}% (Carnot = {eng_result_n.eta_carnot_pct:.2f}%)"
            )
        if eng_result_p is not None:
            lines.append(
                f"  p-type ({eng_result_p.T_c:.0f}->{eng_result_p.T_h:.0f} K) : ZT_eng = {eng_result_p.ZT_eng:.3f} | "
                f"eta_max = {eng_result_p.eta_max_pct:.2f}% (Carnot = {eng_result_p.eta_carnot_pct:.2f}%)"
            )
    elif eng_result is not None:
        lines.append(
            f"  Dominant Leg ({eng_result.T_c:.0f}->{eng_result.T_h:.0f} K): ZT_eng = {eng_result.ZT_eng:.3f} | "
            f"eta_max = {eng_result.eta_max_pct:.2f}% (Carnot = {eng_result.eta_carnot_pct:.2f}%)"
        )
    else:
        lines.append("  Device Efficiency : Insufficient temperature points above T_c.")

    lines.append("=" * 76)
    report_path = out_path / "report_summary.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "report_summary": report_path,
        "summary_metrics": json_path,
        "transport_at_ef": at_ef_path,
        "optimal_doping_vs_T": opt_path,
        "table_summary": tex_path,
    }
