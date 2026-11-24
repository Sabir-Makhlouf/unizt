"""Real-world spin-polarized two-channel thermoelectric case study for cubic Cs2VBr6.

Uses FP-LAPW / BoltzTraP spin-up (``examples/cs2vbr6/up.trace``) and spin-down
(``examples/cs2vbr6/dn.trace``) transport calculations for the vacancy-ordered
double perovskite Cs2VBr6 (Fm-3m, No. 225).

Reference:
  S. Makhlouf, S. E. Amara, A. Amara, M. Cakmak, C. Tayran, H. Kheribot,
  and A. Y. Behlali, "Comprehensive Investigation of Physical Properties of
  Cs2VX6 (X = Cl, I, or Br) Vacancy-Ordered Double Perovskites",
  J. Am. Ceram. Soc. 109(5), e70769 (2026), DOI: 10.1111/jace.70769.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unizt import __author__, __version__
from unizt.core.elastic import compute_elastic_properties
from unizt.core.engineering import compute_engineering_zt
from unizt.core.kappa_l import cahill_pohl_kappa_min, slack_kappa_l
from unizt.core.relaxation import tau_dpt
from unizt.core.zt import combine_spin_channels, compute_zt
from unizt.parsers.boltztrap2 import parse_spin_polarized_traces


RY_TO_EV = 13.605693122994


def main() -> int:
    up_path = ROOT_DIR / "examples" / "cs2vbr6" / "up.trace"
    dn_path = ROOT_DIR / "examples" / "cs2vbr6" / "dn.trace"
    out_dir = ROOT_DIR / "paper_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Structural, elastic, Slack, and DPT parameters for cubic Cs2VBr6 (Fm-3m)
    a_lat_ang = 10.60
    n_atoms_prim = 9  # 2 Cs + 1 V + 6 Br per primitive cell
    vol_prim_m3 = ((a_lat_ang * const.angstrom) ** 3) / 4.0
    vol_atom_m3 = vol_prim_m3 / n_atoms_prim
    m_formula_amu = 2.0 * 132.90545 + 50.9415 + 6.0 * 79.904
    m_avg_amu = m_formula_amu / n_atoms_prim
    density_kg_m3 = (m_formula_amu * 1e-3) / (const.Avogadro * vol_prim_m3)

    # Representative born-stable cubic halide double-perovskite stiffness tensor (GPa)
    C_ij = [
        [39.5, 17.5, 17.5, 0.0, 0.0, 0.0],
        [17.5, 39.5, 17.5, 0.0, 0.0, 0.0],
        [17.5, 17.5, 39.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 11.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 11.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 11.2],
    ]
    elastic = compute_elastic_properties(C_ij, density=density_kg_m3, m_avg=m_avg_amu)
    assert elastic.vol_atom is not None and elastic.theta_D is not None

    # Deformation Potential Theory parameters (effective band mass & deformation potential)
    m_eff_dpt = 0.85
    E1_dpt_eV = 11.0
    C_3d_gpa = elastic.longitudinal_modulus

    def tau_fn(T_arr: np.ndarray) -> np.ndarray:
        return np.asarray(
            tau_dpt(T_arr, m_eff=m_eff_dpt, E1=E1_dpt_eV, C_3d=C_3d_gpa),
            dtype=np.float64,
        )

    def kappa_l_fn(T_arr: np.ndarray) -> np.ndarray:
        k_sl = slack_kappa_l(
            T_arr,
            M_avg=m_avg_amu,
            vol_atom=elastic.vol_atom,
            theta_D=elastic.theta_D,
            gamma=elastic.gamma,
            n_atoms=n_atoms_prim,
            scale_theta_by_n=True,
        )
        k_min = cahill_pohl_kappa_min(
            T_arr,
            v_l=elastic.v_l,
            v_t=elastic.v_t,
            vol_atom=elastic.vol_atom,
        )
        return np.maximum(k_sl, k_min)

    # 2. Load spin-up and spin-down BoltzTraP traces and run two-channel engine
    up_data, dn_data = parse_spin_polarized_traces(up_path, dn_path, vol_uc_m3=vol_prim_m3)
    combined = combine_spin_channels(
        up_data,
        dn_data,
        tau=tau_fn,
        kappa_l=kappa_l_fn,
        vol_uc_m3=vol_prim_m3,
    )

    zt_tot_res = compute_zt(combined, tau=tau_fn, kappa_l=kappa_l_fn, vol_uc_m3=vol_prim_m3)

    # Also compute individual spin-channel ZT on overlapping grid for comparison
    up_aligned = {
        "T": combined["T"],
        "mu": combined["mu"],
        "n": combined["n_up"],
        "S": combined["S_up"],
        "sigma_tau": combined["sigma_tau_up"],
        "kappa_e_tau": combined["kappa_e_tau_up"],
        "n_unit": "m^-3",
    }
    dn_aligned = {
        "T": combined["T"],
        "mu": combined["mu"],
        "n": combined["n_dn"],
        "S": combined["S_dn"],
        "sigma_tau": combined["sigma_tau_dn"],
        "kappa_e_tau": combined["kappa_e_tau_dn"],
        "n_unit": "m^-3",
    }
    zt_up_res = compute_zt(up_aligned, tau=tau_fn, kappa_l=kappa_l_fn, vol_uc_m3=vol_prim_m3)
    zt_dn_res = compute_zt(dn_aligned, tau=tau_fn, kappa_l=kappa_l_fn, vol_uc_m3=vol_prim_m3)

    # Evaluate physical near-Fermi window (|mu - Ef0| <= 0.50 eV, half-metallic t2g manifold)
    ef0_ry = 0.211075
    dmu_ev = (combined["mu"] - ef0_ry) * RY_TO_EV
    hm_mask = np.abs(dmu_ev) <= 0.50
    hm_subset = {
        k: (v[hm_mask] if isinstance(v, np.ndarray) and v.shape == combined["T"].shape else v)
        for k, v in combined.items()
    }
    zt_hm_res = compute_zt(hm_subset, tau=tau_fn, kappa_l=kappa_l_fn, vol_uc_m3=vol_prim_m3)

    # Engineering efficiency between T_c = 300 K and T_h = 1000 K along dominant optimal leg
    def _eval_eng(regime: dict[str, np.ndarray], tc: float = 300.0, th: float = 1000.0):
        valid = np.isfinite(regime["zt_max"]) & (regime["T"] >= tc) & (regime["T"] <= th)
        if np.sum(valid) < 2:
            return None
        return compute_engineering_zt(
            regime["T"][valid],
            regime["S_opt"][valid],
            regime["sigma_opt"][valid],
            regime["kappa_tot_opt"][valid],
            T_c=tc,
            T_h=th,
        )

    eng_n = _eval_eng(zt_tot_res.n_type)
    eng_p = _eval_eng(zt_tot_res.p_type)

    # 3. Export summary CSV & booktabs LaTeX table at selected temperatures
    target_temps = {300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0}
    summary_rows: list[dict[str, Any]] = []
    for row in zt_tot_res.summary_table():
        if round(row["T_K"], 1) in target_temps:
            summary_rows.append(
                {
                    "T_K": int(round(row["T_K"])),
                    "tau_fs": round(row["tau_s"] * 1e15, 2),
                    "kappa_L_W_m_K": round(row["kappa_L_W_m_K"], 3),
                    "ZT_max_n": round(row["ZT_max_n"], 3),
                    "n_opt_n_cm3": float(row["n_opt_n"] * 1e-6),
                    "PF_n_mW_m_K2": round(row["PF_n_mW_m_K2"], 3),
                    "ZT_max_p": round(row["ZT_max_p"], 3),
                    "n_opt_p_cm3": float(row["n_opt_p"] * 1e-6),
                    "PF_p_mW_m_K2": round(row["PF_p_mW_m_K2"], 3),
                }
            )

    csv_path = out_dir / "table_cs2vbr6_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    tex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Spin-polarized two-channel thermoelectric properties of cubic $\mathrm{Cs_2VBr_6}$ ($Fm\bar{3}m$) computed with UniZT v"
        + __version__
        + r" using DPT relaxation time and Cahill--Slack lattice thermal conductivity.}",
        r"\label{tab:cs2vbr6_summary}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$T$ (K) & $\tau_{\mathrm{DPT}}$ (fs) & $\kappa_L$ (W\,m$^{-1}$\,K$^{-1}$) & "
        r"$ZT_{\max}^{(n)}$ & $n_{\mathrm{opt}}^{(n)}$ ($\mathrm{cm}^{-3}$) & $\mathrm{PF}^{(n)}$ & "
        r"$ZT_{\max}^{(p)}$ & $n_{\mathrm{opt}}^{(p)}$ ($\mathrm{cm}^{-3}$) & $\mathrm{PF}^{(p)}$ \\",
        r"\midrule",
    ]
    for r in summary_rows:
        tex_lines.append(
            f"{r['T_K']} & {r['tau_fs']:.2f} & {r['kappa_L_W_m_K']:.3f} & "
            f"{r['ZT_max_n']:.3f} & {r['n_opt_n_cm3']:.2e} & {r['PF_n_mW_m_K2']:.2f} & "
            f"{r['ZT_max_p']:.3f} & {r['n_opt_p_cm3']:.2e} & {r['PF_p_mW_m_K2']:.2f} \\\\"
        )
    tex_lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{9}}{{l}}{{Device metrics ($T_c=300\;\mathrm{{K}},\,T_h=1000\;\mathrm{{K}}$): "
            rf"$n$-type $ZT_{{\mathrm{{eng}}}}={eng_n.ZT_eng:.2f}$ ($\eta_{{\max}}={eng_n.eta_max_pct:.2f}\%$); "
            rf"$p$-type $ZT_{{\mathrm{{eng}}}}={eng_p.ZT_eng:.2f}$ ($\eta_{{\max}}={eng_p.eta_max_pct:.2f}\%$).}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (out_dir / "table_cs2vbr6_summary.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # 4. Write complete provenance JSON
    provenance = {
        "tool": f"UniZT v{__version__}",
        "author": __author__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "material": {
            "formula": "Cs2VBr6",
            "structure": "Vacancy-ordered double perovskite, cubic Fm-3m (No. 225)",
            "reference": "S. Makhlouf et al., J. Am. Ceram. Soc. 109(5), e70769 (2026)",
            "doi": "10.1111/jace.70769",
            "trace_up": str(up_path.relative_to(ROOT_DIR)),
            "trace_dn": str(dn_path.relative_to(ROOT_DIR)),
            "n_rows_up": int(len(up_data["T"])),
            "n_rows_dn": int(len(dn_data["T"])),
            "n_rows_combined": int(len(combined["T"])),
            "neutral_fermi_energy_Ry": ef0_ry,
        },
        "assumed_and_derived_parameters": {
            "lattice_parameter_angstrom": a_lat_ang,
            "n_atoms_primitive": n_atoms_prim,
            "primitive_cell_volume_m3": vol_prim_m3,
            "atomic_volume_m3": elastic.vol_atom,
            "average_atomic_mass_amu": m_avg_amu,
            "density_kg_m3": density_kg_m3,
            "elastic_tensor_GPa": C_ij,
            "B_VRH_GPa": elastic.B_H,
            "G_VRH_GPa": elastic.G_H,
            "E_GPa": elastic.E,
            "poisson_ratio": elastic.nu,
            "pugh_ratio": elastic.pugh_ratio,
            "debye_temperature_K": elastic.theta_D,
            "gruneisen_gamma": elastic.gamma,
            "dpt_m_eff_m0": m_eff_dpt,
            "dpt_E1_eV": E1_dpt_eV,
            "dpt_C3d_GPa": C_3d_gpa,
            "tau_300K_s": float(tau_fn(np.array([300.0]))[0]),
            "tau_900K_s": float(tau_fn(np.array([900.0]))[0]),
            "kappa_L_model": "Slack (n-scaled acoustic cutoff) floored by Cahill-Pohl kappa_min",
            "kappa_L_300K_W_m_K": float(kappa_l_fn(np.array([300.0]))[0]),
        },
        "results_summary": {
            "full_grid_peak_ZT": zt_tot_res.overall_peak_zt,
            "full_grid_peak_type": zt_tot_res.overall_peak_type,
            "full_grid_peak_T_K": zt_tot_res.overall_peak_T,
            "n_type_peak_ZT": float(np.nanmax(zt_tot_res.n_type["zt_max"])),
            "n_type_peak_T_K": float(zt_tot_res.n_type["T"][np.nanargmax(zt_tot_res.n_type["zt_max"])]),
            "p_type_peak_ZT": float(np.nanmax(zt_tot_res.p_type["zt_max"])),
            "p_type_peak_T_K": float(zt_tot_res.p_type["T"][np.nanargmax(zt_tot_res.p_type["zt_max"])]),
            "near_EF_half_metallic_peak_ZT_n": float(np.nanmax(zt_hm_res.n_type["zt_max"])),
            "near_EF_half_metallic_peak_ZT_p": float(np.nanmax(zt_hm_res.p_type["zt_max"])),
            "engineering_300_1000K_n": {
                "ZT_eng": eng_n.ZT_eng if eng_n else None,
                "eta_max_pct": eng_n.eta_max_pct if eng_n else None,
            },
            "engineering_300_1000K_p": {
                "ZT_eng": eng_p.ZT_eng if eng_p else None,
                "eta_max_pct": eng_p.eta_max_pct if eng_p else None,
            },
        },
    }
    (out_dir / "cs2vbr6_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    # 5. Publication 300 DPI Figures
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    # Figure 1: 2x2 Spin-Channel Comparison at T = 600 K vs (mu - E_F) in eV
    T_plot = 600.0
    m600 = (np.round(combined["T"], 6) == T_plot) & (dmu_ev >= -1.1) & (dmu_ev <= 2.3)
    x_ev = dmu_ev[m600]
    ord_x = np.argsort(x_ev)
    x_ev = x_ev[ord_x]

    tau_600 = combined["tau"][m600][ord_x]
    S_up_uV = combined["S_up"][m600][ord_x] * 1e6
    S_dn_uV = combined["S_dn"][m600][ord_x] * 1e6
    S_tot_uV = combined["S_tot"][m600][ord_x] * 1e6

    sig_up_600 = combined["sigma_up"][m600][ord_x]
    sig_dn_600 = combined["sigma_dn"][m600][ord_x]
    sig_tot_600 = combined["sigma_tot"][m600][ord_x]

    pf_up_mW = (combined["S_up"][m600][ord_x] ** 2) * sig_up_600 * 1e3
    pf_dn_mW = (combined["S_dn"][m600][ord_x] ** 2) * sig_dn_600 * 1e3
    pf_tot_mW = combined["pf_mW"][m600][ord_x]

    ke_up_600 = combined["kappa_e_up"][m600][ord_x]
    ke_dn_600 = combined["kappa_e_dn"][m600][ord_x]
    ke_tot_600 = combined["kappa_e_tot"][m600][ord_x]
    ke_mix_600 = combined["kappa_e_mix"][m600][ord_x]

    kl_600 = combined["kappa_l"][m600][ord_x]
    zt_up_600 = (combined["S_up"][m600][ord_x] ** 2) * sig_up_600 * T_plot / (ke_up_600 + kl_600)
    zt_dn_600 = (combined["S_dn"][m600][ord_x] ** 2) * sig_dn_600 * T_plot / (ke_dn_600 + kl_600)
    zt_tot_600 = combined["zt_tot"][m600][ord_x]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
    c_up, c_dn, c_tot = "#0072B2", "#D55E00", "#111827"

    # (a) Seebeck
    ax = axes[0, 0]
    ax.plot(x_ev, S_up_uV, color=c_up, lw=1.6, ls="--", label=r"Spin-up ($\uparrow$)")
    ax.plot(x_ev, S_dn_uV, color=c_dn, lw=1.6, ls=":", label=r"Spin-down ($\downarrow$)")
    ax.plot(x_ev, S_tot_uV, color=c_tot, lw=2.0, label="Two-channel total")
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.axhline(0.0, color="#9CA3AF", lw=0.7)
    ax.set_ylabel(r"Seebeck Coefficient $S$ ($\mu$V K$^{-1}$)")
    ax.set_title(r"(a) Seebeck Coefficient ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(-650, 850)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    # (b) Power Factor
    ax = axes[0, 1]
    ax.plot(x_ev, pf_up_mW, color=c_up, lw=1.6, ls="--", label=r"Spin-up ($\uparrow$)")
    ax.plot(x_ev, pf_dn_mW, color=c_dn, lw=1.6, ls=":", label=r"Spin-down ($\downarrow$)")
    ax.plot(x_ev, pf_tot_mW, color=c_tot, lw=2.0, label="Two-channel total")
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_ylabel(r"Power Factor $\mathrm{PF}$ (mW m$^{-1}$ K$^{-2}$)")
    ax.set_title(r"(b) Power Factor ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)

    # (c) Electronic thermal conductivity + spin-mixing
    ax = axes[1, 0]
    ax.plot(x_ev, ke_up_600, color=c_up, lw=1.6, ls="--", label=r"$\kappa_{e,\uparrow}$")
    ax.plot(x_ev, ke_dn_600, color=c_dn, lw=1.6, ls=":", label=r"$\kappa_{e,\downarrow}$")
    ax.plot(x_ev, ke_tot_600, color=c_tot, lw=2.0, label=r"$\kappa_{e,\mathrm{tot}}$ (incl. mixing)")
    ax.plot(x_ev, ke_mix_600, color="#009E73", lw=1.5, ls="-.", label=r"$\kappa_{e,\mathrm{mix}}$")
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_xlabel(r"Chemical Potential $\mu - E_F$ (eV)")
    ax.set_ylabel(r"Electronic Thermal Cond. $\kappa_e$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(r"(c) Electronic Thermal Conductivity ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.3)

    # (d) True ZT
    ax = axes[1, 1]
    ax.plot(x_ev, zt_up_600, color=c_up, lw=1.6, ls="--", label=r"$ZT_{\uparrow}$")
    ax.plot(x_ev, zt_dn_600, color=c_dn, lw=1.6, ls=":", label=r"$ZT_{\downarrow}$")
    ax.plot(x_ev, zt_tot_600, color=c_tot, lw=2.1, label=r"$ZT_{\mathrm{tot}}$ (two-channel)")
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_xlabel(r"Chemical Potential $\mu - E_F$ (eV)")
    ax.set_ylabel(r"True Figure of Merit $ZT$")
    ax.set_title(r"(d) Two-Channel Figure of Merit ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    fig.suptitle(
        r"Spin-Polarized Two-Channel Transport in Cubic $\mathrm{Cs_2VBr_6}$ ($Fm\bar{3}m$)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_dir / "fig_cs2vbr6_spin_channels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Peak ZT vs Temperature comparing n-type and p-type
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    T_g = zt_tot_res.T_grid
    ax.plot(
        T_g,
        zt_tot_res.p_type["zt_max"],
        color="#D55E00",
        lw=2.2,
        marker="o",
        ms=4.2,
        markevery=3,
        label=r"p-type ($n > 0$, two-channel)",
    )
    ax.plot(
        T_g,
        zt_tot_res.n_type["zt_max"],
        color="#0072B2",
        lw=2.2,
        marker="s",
        ms=4.2,
        markevery=3,
        label=r"n-type ($n < 0$, two-channel)",
    )
    ax.plot(
        zt_hm_res.T_grid,
        zt_hm_res.p_type["zt_max"],
        color="#D55E00",
        lw=1.4,
        ls="--",
        alpha=0.75,
        label=r"p-type near-$E_F$ ($|\mu - E_F| \leq 0.5\;\mathrm{eV}$)",
    )
    ax.plot(
        zt_hm_res.T_grid,
        zt_hm_res.n_type["zt_max"],
        color="#0072B2",
        lw=1.4,
        ls="--",
        alpha=0.75,
        label=r"n-type near-$E_F$ ($|\mu - E_F| \leq 0.5\;\mathrm{eV}$)",
    )
    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel(r"Optimal Figure of Merit $ZT_{\max}(T)$")
    ax.set_title(r"$\mathrm{Cs_2VBr_6}$ Two-Channel Peak $ZT(T)$: n-Type vs. p-Type")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.8)
    fig.savefig(out_dir / "fig_cs2vbr6_zt_vs_t.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Cs2VBr6 two-channel case study completed successfully.")
    print(
        f"Full-grid peak ZT: {zt_tot_res.overall_peak_zt:.3f} ({zt_tot_res.overall_peak_type} "
        f"at T={zt_tot_res.overall_peak_T:.0f} K)"
    )
    print(
        f"  p-type peak ZT = {np.nanmax(zt_tot_res.p_type['zt_max']):.3f} at "
        f"{zt_tot_res.p_type['T'][np.nanargmax(zt_tot_res.p_type['zt_max'])]:.0f} K"
    )
    print(
        f"  n-type peak ZT = {np.nanmax(zt_tot_res.n_type['zt_max']):.3f} at "
        f"{zt_tot_res.n_type['T'][np.nanargmax(zt_tot_res.n_type['zt_max'])]:.0f} K"
    )
    print(
        f"Near-EF (|dmu|<=0.5 eV) peak ZT: n={np.nanmax(zt_hm_res.n_type['zt_max']):.3f}, "
        f"p={np.nanmax(zt_hm_res.p_type['zt_max']):.3f}"
    )
    if eng_n and eng_p:
        print(
            f"Device eta_max (300->1000 K): n-type={eng_n.eta_max_pct:.2f}% (ZT_eng={eng_n.ZT_eng:.2f}), "
            f"p-type={eng_p.eta_max_pct:.2f}% (ZT_eng={eng_p.ZT_eng:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
