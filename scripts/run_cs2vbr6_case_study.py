"""Real-world spin-polarized two-channel thermoelectric case study for cubic Cs2VBr6.

Uses FP-LAPW / BoltzTraP spin-up (``examples/cs2vbr6/up.trace``) and spin-down
(``examples/cs2vbr6/dn.trace``) transport calculations and the exact published
DFT structural and elastic parameters for cubic Cs2VBr6 (Fm-3m, No. 225) from:

  S. Makhlouf, S. E. Amara, A. Amara, M. Cakmak, C. Tayran, H. Kheribot,
  and A. Y. Behlali, "Comprehensive Investigation of Physical Properties of
  Cs2VX6 (X = Cl, I, or Br) Vacancy-Ordered Double Perovskites",
  J. Am. Ceram. Soc. 109(5), e70769 (2026), DOI: 10.1111/jace.70769.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unizt import __version__
from unizt.core.elastic import compute_elastic_properties
from unizt.core.engineering import compute_engineering_zt
from unizt.core.kappa_l import cahill_pohl_kappa_min, slack_kappa_l
from unizt.core.relaxation import tau_constant, tau_dpt
from unizt.core.zt import combine_spin_channels, optimize_zt
from unizt.export import export_research_bundle
from unizt.parsers.boltztrap2 import parse_spin_polarized_traces


def main() -> int:
    up_path = ROOT_DIR / "examples" / "cs2vbr6" / "up.trace"
    dn_path = ROOT_DIR / "examples" / "cs2vbr6" / "dn.trace"
    out_dir = ROOT_DIR / "paper_outputs" / "cs2vbr6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exact published DFT parameters (Makhlouf et al., J. Am. Ceram. Soc. 109, e70769, 2026)
    a0_ang = 11.02
    n_atoms_prim = 9  # 2 Cs + 1 V + 6 Br per primitive cell
    vol_prim_m3 = ((a0_ang * const.angstrom) ** 3) / 4.0
    m_formula_amu = 2.0 * 132.90545 + 50.9415 + 6.0 * 79.904
    m_avg_amu = m_formula_amu / n_atoms_prim
    density_kg_m3 = (m_formula_amu * 1e-3) / (const.Avogadro * vol_prim_m3)

    # Exact cubic elastic stiffness tensor (GPa)
    C_ij = [
        [40.27, 24.67, 24.67, 0.0, 0.0, 0.0],
        [24.67, 40.27, 24.67, 0.0, 0.0, 0.0],
        [24.67, 24.67, 40.27, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 12.06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 12.06, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 12.06],
    ]
    elastic = compute_elastic_properties(C_ij, density=density_kg_m3, m_avg=m_avg_amu)
    assert elastic.vol_atom is not None and elastic.theta_D is not None

    # Deformation Potential Theory parameters
    m_eff_dpt = 0.50
    E1_dpt_eV = 9.0
    C_3d_gpa = elastic.longitudinal_modulus
    published_ref_zt = 0.95

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

    # 2. Load spin-up and spin-down traces and combine via exact two-current engine
    up_data, dn_data = parse_spin_polarized_traces(up_path, dn_path, vol_uc_m3=vol_prim_m3)
    combined = combine_spin_channels(
        up_data,
        dn_data,
        tau=tau_fn,
        kappa_l=kappa_l_fn,
        vol_uc_m3=vol_prim_m3,
    )

    # Primary RBA-constrained optimization: |mu - E_F| <= 0.5 eV and |n| <= 1e21 cm^-3
    ef_window_ev = 0.5
    max_n_cm3 = 1e21
    zt_res = optimize_zt(
        combined,
        tau=tau_fn,
        kappa_l=kappa_l_fn,
        vol_uc_m3=vol_prim_m3,
        ef_window=ef_window_ev,
        max_carrier_density=max_n_cm3,
        extract_at_ef=True,
        warn_unconstrained=False,
    )

    # Also evaluate CRTA (tau = 1e-14 s) under the same RBA constraints for provenance comparison
    zt_crta = optimize_zt(
        combined,
        tau=lambda T_arr: tau_constant(T_arr, tau_0=1e-14),
        kappa_l=kappa_l_fn,
        vol_uc_m3=vol_prim_m3,
        ef_window=ef_window_ev,
        max_carrier_density=max_n_cm3,
        extract_at_ef=True,
        warn_unconstrained=False,
    )

    # Kim et al. device engineering efficiency (300 K -> 1000 K)
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

    eng_n = _eval_eng(zt_res.n_type)
    eng_p = _eval_eng(zt_res.p_type)
    eng_dom = eng_n if zt_res.overall_peak_type == "n-type" else eng_p

    provenance = {
        "reference": "S. Makhlouf et al., J. Am. Ceram. Soc. 109(5), e70769 (2026)",
        "doi": "10.1111/jace.70769",
        "space_group": "Fm-3m (No. 225)",
        "a0_angstrom": a0_ang,
        "V_prim_m3": vol_prim_m3,
        "n_atoms_prim": n_atoms_prim,
        "C11_GPa": 40.27,
        "C12_GPa": 24.67,
        "C44_GPa": 12.06,
        "published_reference_peak_ZT": published_ref_zt,
        "dpt_parameters": {
            "m_eff_m0": m_eff_dpt,
            "E1_eV": E1_dpt_eV,
            "C_3d_GPa": C_3d_gpa,
        },
        "crta_comparison_1e_14s": {
            "constrained_n_peak_ZT": float(np.nanmax(zt_crta.n_type["zt_max"])),
            "constrained_p_peak_ZT": float(np.nanmax(zt_crta.p_type["zt_max"])),
            "intrinsic_ef_peak_ZT": float(np.max(zt_crta.at_ef["zt"])),
        },
    }

    # 3. Export complete researcher-grade bundle inside paper_outputs/cs2vbr6/
    kappa_min_grid = np.asarray(
        cahill_pohl_kappa_min(
            zt_res.T_grid,
            v_l=elastic.v_l,
            v_t=elastic.v_t,
            vol_atom=elastic.vol_atom,
        ),
        dtype=np.float64,
    )
    bundle_paths = export_research_bundle(
        zt_res,
        out_dir,
        elastic_props=elastic,
        eng_result=eng_dom,
        eng_result_n=eng_n,
        eng_result_p=eng_p,
        kappa_min_grid=kappa_min_grid,
        material_name=r"cubic $\mathrm{Cs_2VBr_6}$ ($Fm\bar{3}m$)",
        provenance=provenance,
        tex_filename="table_cs2vbr6_summary.tex",
        tex_temperatures=[300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        tau_mode=f"dpt (m_eff={m_eff_dpt}, E1={E1_dpt_eV} eV)",
        kappa_mode="slack + cahill_floor",
    )

    # 4. Publication 300 DPI Figures
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

    # Figure 1: 2x2 panel focused on physical window -0.6 <= mu - E_F <= +0.6 eV at T = 600 K
    assert zt_res.dmu_ev is not None
    T_plot = 600.0
    m600 = (
        (np.round(zt_res.T, 6) == T_plot)
        & (zt_res.dmu_ev >= -0.6)
        & (zt_res.dmu_ev <= 0.6)
    )
    x_ev = zt_res.dmu_ev[m600]
    ord_x = np.argsort(x_ev)
    x_ev = x_ev[ord_x]

    S_up_uV = combined["S_up"][m600][ord_x] * 1e6
    S_dn_uV = combined["S_dn"][m600][ord_x] * 1e6
    S_tot_uV = zt_res.S[m600][ord_x] * 1e6

    sig_up = combined["sigma_up"][m600][ord_x]
    sig_dn = combined["sigma_dn"][m600][ord_x]
    pf_up_mW = (combined["S_up"][m600][ord_x] ** 2) * sig_up * 1e3
    pf_dn_mW = (combined["S_dn"][m600][ord_x] ** 2) * sig_dn * 1e3
    pf_tot_mW = zt_res.pf[m600][ord_x]

    ke_up = combined["kappa_e_up"][m600][ord_x]
    ke_dn = combined["kappa_e_dn"][m600][ord_x]
    ke_tot = zt_res.kappa_e[m600][ord_x]
    ke_mix = combined["kappa_e_mix"][m600][ord_x]

    kl_600 = zt_res.kappa_l[m600][ord_x]
    zt_up = (combined["S_up"][m600][ord_x] ** 2) * sig_up * T_plot / (ke_up + kl_600)
    zt_dn = (combined["S_dn"][m600][ord_x] ** 2) * sig_dn * T_plot / (ke_dn + kl_600)
    zt_tot = zt_res.zt[m600][ord_x]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
    c_up, c_dn, c_tot = "#0072B2", "#D55E00", "#111827"

    # (a) Seebeck
    ax = axes[0, 0]
    ax.plot(x_ev, S_up_uV, color=c_up, lw=1.7, ls="--", label=r"Spin-up ($\uparrow$)")
    ax.plot(x_ev, S_dn_uV, color=c_dn, lw=1.7, ls=":", label=r"Spin-down ($\downarrow$)")
    ax.plot(x_ev, S_tot_uV, color=c_tot, lw=2.0, label="Two-channel total")
    ax.axvspan(-0.5, 0.5, color="#10B981", alpha=0.06, label=r"RBA window ($\pm 0.5$ eV)")
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.axhline(0.0, color="#9CA3AF", lw=0.7)
    ax.set_ylabel(r"Seebeck Coefficient $S$ ($\mu$V K$^{-1}$)")
    ax.set_title(r"(a) Seebeck Coefficient ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(-520, 750)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)

    # (b) Power Factor
    ax = axes[0, 1]
    ax.plot(x_ev, pf_up_mW, color=c_up, lw=1.7, ls="--", label=r"Spin-up ($\uparrow$)")
    ax.plot(x_ev, pf_dn_mW, color=c_dn, lw=1.7, ls=":", label=r"Spin-down ($\downarrow$)")
    ax.plot(x_ev, pf_tot_mW, color=c_tot, lw=2.0, label="Two-channel total")
    ax.axvspan(-0.5, 0.5, color="#10B981", alpha=0.06)
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_ylabel(r"Power Factor $\mathrm{PF}$ (mW m$^{-1}$ K$^{-2}$)")
    ax.set_title(r"(b) Power Factor ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)

    # (c) Electronic thermal conductivity + spin mixing
    ax = axes[1, 0]
    ax.plot(x_ev, ke_up, color=c_up, lw=1.7, ls="--", label=r"$\kappa_{e,\uparrow}$")
    ax.plot(x_ev, ke_dn, color=c_dn, lw=1.7, ls=":", label=r"$\kappa_{e,\downarrow}$")
    ax.plot(x_ev, ke_tot, color=c_tot, lw=2.0, label=r"$\kappa_{e,\mathrm{tot}}$")
    ax.plot(x_ev, ke_mix, color="#009E73", lw=1.6, ls="-.", label=r"$\kappa_{e,\mathrm{mix}}$")
    ax.axvspan(-0.5, 0.5, color="#10B981", alpha=0.06)
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_xlabel(r"Chemical Potential $\mu - E_F$ (eV)")
    ax.set_ylabel(r"Electronic Thermal Cond. $\kappa_e$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(r"(c) Electronic Thermal Conductivity ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    # (d) True ZT
    ax = axes[1, 1]
    ax.plot(x_ev, zt_up, color=c_up, lw=1.7, ls="--", label=r"$ZT_{\uparrow}$")
    ax.plot(x_ev, zt_dn, color=c_dn, lw=1.7, ls=":", label=r"$ZT_{\downarrow}$")
    ax.plot(x_ev, zt_tot, color=c_tot, lw=2.1, label=r"$ZT_{\mathrm{tot}}$ (two-channel)")
    ax.axvspan(-0.5, 0.5, color="#10B981", alpha=0.06)
    ax.axvline(0.0, color="#6B7280", lw=0.9, ls="-.")
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel(r"Chemical Potential $\mu - E_F$ (eV)")
    ax.set_ylabel(r"True Figure of Merit $ZT$")
    ax.set_title(r"(d) Two-Channel Figure of Merit ($T = 600\;\mathrm{K}$)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    fig.suptitle(
        rf"Spin-Polarized Two-Channel Transport in $\mathrm{{Cs_2VBr_6}}$ ($a_0 = {a0_ang:.2f}\;\mathrm{{\AA}}$, UniZT v{__version__})",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_dir / "fig_cs2vbr6_spin_channels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Physically constrained n-type, p-type, and intrinsic mu = E_F ZT(T) curves
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    T_g = zt_res.T_grid
    ax.plot(
        T_g,
        zt_res.n_type["zt_max"],
        color="#0072B2",
        lw=2.2,
        marker="s",
        ms=4.5,
        markevery=3,
        label=r"n-type RBA ($|\mu-E_F|\leq 0.5\;\mathrm{eV},\;|n|\leq 10^{21}\;\mathrm{cm}^{-3}$)",
    )
    ax.plot(
        T_g,
        zt_res.p_type["zt_max"],
        color="#D55E00",
        lw=2.2,
        marker="o",
        ms=4.5,
        markevery=3,
        label=r"p-type RBA ($|\mu-E_F|\leq 0.5\;\mathrm{eV},\;|n|\leq 10^{21}\;\mathrm{cm}^{-3}$)",
    )
    if zt_res.at_ef is not None:
        ax.plot(
            zt_res.at_ef["T"],
            zt_res.at_ef["zt"],
            color="#009E73",
            lw=2.0,
            ls="-.",
            label=r"Intrinsic / Undoped ($\mu = E_F$)",
        )
    ax.axhline(
        published_ref_zt,
        color="#374151",
        lw=1.5,
        ls="--",
        label=rf"Published Ref. Peak $ZT = {published_ref_zt:.2f}$ (Makhlouf et al. 2026)",
    )
    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel(r"True Figure of Merit $ZT$")
    ax.set_title(r"$\mathrm{Cs_2VBr_6}$ Constrained vs. Intrinsic Two-Channel $ZT(T)$")
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.4)
    fig.savefig(out_dir / "fig_cs2vbr6_zt_vs_t.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(bundle_paths["report_summary"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
