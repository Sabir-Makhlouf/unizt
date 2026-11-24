"""Command-line interface for UniZT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy import constants as const

from unizt import (
    BIBTEX_CITATION,
    CITATION_BANNER,
    __version__,
)
from unizt.core.elastic import compute_elastic_properties
from unizt.core.engineering import compute_engineering_zt, efficiency_vs_hot_temp
from unizt.core.kappa_l import (
    cahill_pohl_kappa_min,
    compute_kappa_l,
    load_custom_kappa_csv,
)
from unizt.core.relaxation import compute_relaxation_time
from unizt.core.zt import ZTResult, combine_spin_channels, compute_zt
from unizt.export import export_research_bundle
from unizt.parsers.elastic_parser import parse_elastic_tensor
from unizt.parsers.registry import get_parser, list_parsers
from unizt.plotting import (
    plot_efficiency_vs_th,
    plot_kappa_l_vs_t,
    plot_zt_heatmap,
    plot_zt_vs_t,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unizt",
        description=(
            f"UniZT v{__version__}: Unified Post-Processing Tool for True Thermoelectric "
            "Figure of Merit ZT, Lattice Thermal Conductivity, and Device Efficiency."
        ),
    )

    # Utility & configuration
    parser.add_argument(
        "--cite",
        action="store_true",
        help="Print academic attribution banner and BibTeX citation, then exit.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON parameter file (overridden by explicit CLI flags).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory for output CSV, LaTeX table, and 300 DPI plots (default: '.').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=None,
        help="Generate publication-ready 300 DPI PNG figures.",
    )

    # Transport & elastic inputs
    parser.add_argument("--trace", type=str, default=None, help="Path to transport trace/table file.")
    parser.add_argument(
        "--trace-up",
        type=str,
        default=None,
        help="Path to spin-up BoltzTraP2/1 trace file (must pair with --trace-dn).",
    )
    parser.add_argument(
        "--trace-dn",
        type=str,
        default=None,
        help="Path to spin-down BoltzTraP2/1 trace file (must pair with --trace-up).",
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help=f"Transport code parser name (available: {', '.join(list_parsers())}; default: boltztrap2).",
    )
    parser.add_argument(
        "--cols",
        type=str,
        default=None,
        help="Column mapping for generic parser (e.g. 'T,n,S,sigma_tau,kappa_e_tau' or 'T=0,n=1,...').",
    )
    parser.add_argument(
        "--n-unit",
        type=str,
        default=None,
        help="Carrier density unit override ('m^-3', 'cm^-3', or 'e/uc').",
    )
    parser.add_argument(
        "--vol-uc",
        type=float,
        default=None,
        help="Primitive unit-cell volume (in m^3, or Angstrom^3 when --vol-unit ang3).",
    )
    parser.add_argument("--elastic", type=str, default=None, help="Path to 6x6 elastic tensor file.")
    parser.add_argument(
        "--elastic-format",
        type=str,
        choices=["auto", "json", "vasp", "wien2k", "qe"],
        default=None,
        help="Elastic tensor format (default: auto).",
    )

    # Crystal & lattice thermal conductivity physics
    parser.add_argument("--density", type=float, default=None, help="Mass density in kg/m^3.")
    parser.add_argument("--m-avg", type=float, default=None, help="Average atomic mass in amu (g/mol).")
    parser.add_argument("--n-atoms", type=int, default=None, help="Atoms per primitive unit cell.")
    parser.add_argument(
        "--vol-atom",
        type=float,
        default=None,
        help="Volume per atom (in m^3, or Angstrom^3 when --vol-unit ang3).",
    )
    parser.add_argument(
        "--vol-unit",
        type=str,
        choices=["m3", "ang3"],
        default=None,
        help="Unit for --vol-atom and --vol-uc (default: m3).",
    )
    parser.add_argument(
        "--kappa-mode",
        type=str,
        choices=["slack", "cahill", "custom"],
        default=None,
        help="Lattice thermal conductivity model (default: slack).",
    )
    parser.add_argument(
        "--kappa-csv",
        type=str,
        default=None,
        help="CSV file (temperature_K,kappa_L_W_m_K) for --kappa-mode custom.",
    )
    parser.add_argument(
        "--kappa-interp",
        type=str,
        choices=["loglog", "linear"],
        default=None,
        help="Interpolation mode for custom kappa_L CSV (default: loglog).",
    )

    # Relaxation time physics
    parser.add_argument(
        "--tau-mode",
        type=str,
        choices=["dpt", "constant", "power_law"],
        default=None,
        help="Electronic relaxation time mode (default: dpt).",
    )
    parser.add_argument("--m-eff", type=float, default=None, help="Effective mass in m_0 (DPT).")
    parser.add_argument("--e1", type=float, default=None, help="Deformation potential in eV (DPT).")
    parser.add_argument("--c3d", type=float, default=None, help="3D elastic modulus C_3d in GPa (DPT).")
    parser.add_argument(
        "--tau-const",
        type=float,
        default=None,
        help="Constant relaxation time in s (default: 1e-14).",
    )
    parser.add_argument("--tau-0", type=float, default=None, help="Reference tau_0 in s (power_law).")
    parser.add_argument("--t0", type=float, default=None, help="Reference T_0 in K (power_law).")
    parser.add_argument("--tau-exp", type=float, default=None, help="Power-law exponent p (default: -1.5).")

    # Device engineering efficiency
    parser.add_argument("--tc", type=float, default=None, help="Cold-side temperature T_c in K (default: 300).")
    parser.add_argument("--th", type=float, default=None, help="Hot-side temperature T_h in K (default: max T).")

    # Physical RBA windowing & Fermi reference
    parser.add_argument(
        "--ef-window",
        type=float,
        default=None,
        help="Restrict peak search to |mu - E_F| <= ef_window in eV (default: 0.5).",
    )
    parser.add_argument(
        "--max-carrier-density",
        type=float,
        default=None,
        help="Restrict peak search to |n| <= max_carrier_density in cm^-3 (default: 1e21).",
    )
    parser.add_argument(
        "--no-physical-limits",
        action="store_true",
        default=None,
        help="Disable physical RBA constraints (--ef-window and --max-carrier-density).",
    )
    parser.add_argument(
        "--at-ef",
        action="store_true",
        default=None,
        help="Display intrinsic/undoped transport curve evaluated at mu = E_F.",
    )
    parser.add_argument(
        "--ef",
        type=float,
        default=None,
        help="Explicit neutral Fermi level E_F in mu units (inferred automatically if omitted).",
    )
    parser.add_argument(
        "--mu-unit",
        type=str,
        choices=["auto", "ry", "ev"],
        default=None,
        help="Chemical potential unit in transport input (default: auto).",
    )

    return parser


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "outdir": ".",
        "plot": False,
        "code": "boltztrap2",
        "elastic_format": "auto",
        "vol_unit": "m3",
        "kappa_mode": "slack",
        "kappa_interp": "loglog",
        "tau_mode": "dpt",
        "tau_const": 1e-14,
        "tau_0": 1e-14,
        "t0": 300.0,
        "tau_exp": -1.5,
        "tc": 300.0,
        "ef_window": 0.5,
        "max_carrier_density": 1e21,
        "no_physical_limits": False,
        "at_ef": False,
        "mu_unit": "auto",
    }
    cfg: dict[str, Any] = dict(defaults)

    if args.config is not None:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config JSON file not found: {cfg_path}")
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a JSON object.")
        for k, v in loaded.items():
            cfg[k.replace("-", "_")] = v

    for key, val in vars(args).items():
        if key in {"cite", "config"}:
            continue
        if val is not None and val is not False:
            cfg[key] = val
        elif key in {"plot", "no_physical_limits", "at_ef"} and getattr(args, key):
            cfg[key] = True

    if cfg.get("no_physical_limits"):
        cfg["ef_window"] = None
        cfg["max_carrier_density"] = None

    return cfg


def _write_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_latex_table(rows: list[dict[str, float]], out_path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Post-processed thermoelectric transport and peak $ZT$ summary computed with UniZT.}",
        r"\label{tab:unizt_results}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$T$ (K) & $\tau$ (fs) & $\kappa_L$ (W\,m$^{-1}$\,K$^{-1}$) & "
        r"$ZT_{\max}^{(n)}$ & $n_{\mathrm{opt}}^{(n)}$ & $\mathrm{PF}^{(n)}$ & "
        r"$ZT_{\max}^{(p)}$ & $n_{\mathrm{opt}}^{(p)}$ & $\mathrm{PF}^{(p)}$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['T_K']:.0f} & {r['tau_s'] * 1e15:.2f} & {r['kappa_L_W_m_K']:.2f} & "
            f"{_fmt(r['ZT_max_n'], '.3f')} & {_fmt_sci(r['n_opt_n'])} & {_fmt(r['PF_n_mW_m_K2'], '.2f')} & "
            f"{_fmt(r['ZT_max_p'], '.3f')} & {_fmt_sci(r['n_opt_p'])} & {_fmt(r['PF_p_mW_m_K2'], '.2f')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(val: float, spec: str) -> str:
    return "--" if not np.isfinite(val) else format(val, spec)


def _fmt_sci(val: float) -> str:
    if not np.isfinite(val):
        return "--"
    return f"{val:.2e}"


def _print_summary(
    cfg: dict[str, Any],
    elastic_props: Any,
    zt_res: ZTResult,
    eng_res: Optional[Any],
) -> None:
    print("=" * 72)
    print(f"UniZT v{__version__} Post-Processing Summary")
    print("=" * 72)
    if elastic_props is not None:
        print(
            f"Elastic Moduli (VRH): B = {elastic_props.B_H:.2f} GPa | "
            f"G = {elastic_props.G_H:.2f} GPa | E = {elastic_props.E:.2f} GPa | "
            f"nu = {elastic_props.nu:.3f}"
        )
        print(
            f"Acoustic Parameters : gamma = {elastic_props.gamma:.3f}"
            + (f" | theta_D = {elastic_props.theta_D:.1f} K" if elastic_props.theta_D else "")
            + (f" | v_m = {elastic_props.v_m:.1f} m/s" if elastic_props.v_m else "")
        )
    print(
        f"Models              : tau_mode = {cfg['tau_mode']} | kappa_mode = {cfg['kappa_mode']}"
        + (" | two-channel spin-polarized" if cfg.get("trace_up") else "")
    )
    if zt_res.ef_window_ev is not None or zt_res.max_carrier_density_cm3 is not None:
        print(
            f"RBA Constraints     : |mu - E_F| <= {zt_res.ef_window_ev} eV | "
            f"|n| <= {zt_res.max_carrier_density_cm3:.2e} cm^-3"
            if zt_res.max_carrier_density_cm3 is not None
            else f"RBA Constraints     : |mu - E_F| <= {zt_res.ef_window_ev} eV"
        )
    print(
        f"Overall Peak ZT     : {zt_res.overall_peak_zt:.3f} ({zt_res.overall_peak_type}) "
        f"at T = {zt_res.overall_peak_T:.0f} K (n_opt = {zt_res.overall_peak_n:.3e})"
    )
    if eng_res is not None:
        print(
            f"Device Efficiency   : ZT_eng = {eng_res.ZT_eng:.3f} | "
            f"eta_max = {eng_res.eta_max_pct:.2f}% "
            f"(Carnot = {eng_res.eta_carnot_pct:.2f}%) over "
            f"[{eng_res.T_c:.0f}, {eng_res.T_h:.0f}] K"
        )
    print("-" * 72)
    print(f"{'T(K)':>6} {'tau(fs)':>9} {'kappa_L':>9} {'ZT_n':>8} {'ZT_p':>8} {'PF_n':>8} {'PF_p':>8}")
    for row in zt_res.summary_table():
        print(
            f"{row['T_K']:6.0f} {row['tau_s']*1e15:9.2f} {row['kappa_L_W_m_K']:9.2f} "
            f"{_fmt(row['ZT_max_n'], '8.3f')} {_fmt(row['ZT_max_p'], '8.3f')} "
            f"{_fmt(row['PF_n_mW_m_K2'], '8.2f')} {_fmt(row['PF_p_mW_m_K2'], '8.2f')}"
        )
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cite:
        print(CITATION_BANNER)
        print()
        print(BIBTEX_CITATION)
        return 0

    cfg = _merge_config(args)
    has_single = bool(cfg.get("trace"))
    has_up = bool(cfg.get("trace_up"))
    has_dn = bool(cfg.get("trace_dn"))
    if has_up != has_dn:
        parser.error("Both --trace-up and --trace-dn must be provided together.")
    if has_single and has_up:
        parser.error("Specify either --trace or the pair (--trace-up, --trace-dn), not both.")
    if not has_single and not has_up:
        parser.error("--trace (or --trace-up and --trace-dn) is required unless --cite is passed.")

    vol_scale = (const.angstrom ** 3) if cfg["vol_unit"] == "ang3" else 1.0
    vol_atom = float(cfg["vol_atom"]) * vol_scale if cfg.get("vol_atom") is not None else None
    vol_uc = float(cfg["vol_uc"]) * vol_scale if cfg.get("vol_uc") is not None else None

    elastic_props = None
    if cfg.get("elastic"):
        C_ij = parse_elastic_tensor(cfg["elastic"], fmt=cfg["elastic_format"])
        elastic_props = compute_elastic_properties(
            C_ij,
            density=cfg.get("density"),
            m_avg=cfg.get("m_avg"),
            vol_atom=vol_atom,
        )
        if vol_atom is None:
            vol_atom = elastic_props.vol_atom

    if vol_atom is None and cfg.get("density") and cfg.get("m_avg"):
        vol_atom = float((float(cfg["m_avg"]) * 1e-3) / (const.Avogadro * float(cfg["density"])))

    if vol_uc is None and vol_atom is not None and cfg.get("n_atoms"):
        vol_uc = float(vol_atom) * int(cfg["n_atoms"])

    # Parse transport data through central registry
    parse_fn = get_parser(cfg["code"])
    parse_kwargs: dict[str, Any] = {}
    if cfg["code"].strip().lower() == "generic":
        if cfg.get("cols") is not None:
            parse_kwargs["cols"] = cfg["cols"]
        if cfg.get("n_unit") is not None:
            parse_kwargs["n_unit"] = cfg["n_unit"]
        if vol_uc is not None:
            parse_kwargs["vol_uc_m3"] = vol_uc
    elif cfg["code"].strip().lower() in {"boltztrap2", "boltztrap"}:
        if vol_uc is not None:
            parse_kwargs["vol_uc_m3"] = vol_uc
        if cfg.get("n_unit") is not None:
            parse_kwargs["n_unit"] = cfg["n_unit"]

    if has_up:
        up_data = parse_fn(cfg["trace_up"], **parse_kwargs)
        dn_data = parse_fn(cfg["trace_dn"], **parse_kwargs)
        transport_data = combine_spin_channels(up_data, dn_data, vol_uc_m3=vol_uc)
    else:
        transport_data = parse_fn(cfg["trace"], **parse_kwargs)

    c3d = cfg.get("c3d")
    if c3d is None and elastic_props is not None:
        c3d = elastic_props.longitudinal_modulus

    def tau_func(T_arr: np.ndarray) -> np.ndarray:
        return np.asarray(
            compute_relaxation_time(
                T_arr,
                mode=cfg["tau_mode"],
                m_eff=cfg.get("m_eff"),
                E1=cfg.get("e1"),
                C_3d=c3d,
                tau_const=float(cfg["tau_const"]),
                tau_0=float(cfg["tau_0"]),
                T_0=float(cfg["t0"]),
                exponent=float(cfg["tau_exp"]),
            ),
            dtype=np.float64,
        )

    def kappa_func(T_arr: np.ndarray) -> np.ndarray:
        return np.asarray(
            compute_kappa_l(
                T_arr,
                mode=cfg["kappa_mode"],
                elastic_props=elastic_props,
                M_avg=cfg.get("m_avg"),
                vol_atom=vol_atom,
                n_atoms=cfg.get("n_atoms"),
                custom_csv=cfg.get("kappa_csv"),
                interpolation=cfg["kappa_interp"],
            ),
            dtype=np.float64,
        )

    zt_res = compute_zt(
        transport_data,
        tau=tau_func,
        kappa_l=kappa_func,
        vol_uc_m3=vol_uc,
        ef_window=cfg.get("ef_window"),
        max_carrier_density=cfg.get("max_carrier_density"),
        e_fermi=cfg.get("ef"),
        mu_unit=cfg.get("mu_unit", "auto"),
        extract_at_ef=True,
        warn_unconstrained=True,
    )

    # Select dominant carrier polarity and individual legs for engineering integration
    eng_res = None
    eng_n = None
    eng_p = None
    eff_sweep = None
    if zt_res.T_grid.size >= 2:
        tc_val = float(cfg["tc"])
        for reg_key, reg_dict in [("n", zt_res.n_type), ("p", zt_res.p_type)]:
            v_reg = np.isfinite(reg_dict["zt_max"])
            if np.sum(v_reg) >= 2:
                T_r = reg_dict["T"][v_reg]
                th_v = float(cfg["th"]) if cfg.get("th") is not None else float(np.max(T_r))
                if tc_val >= float(np.min(T_r)) - 1e-6 and th_v > tc_val:
                    er = compute_engineering_zt(
                        T_r,
                        reg_dict["S_opt"][v_reg],
                        reg_dict["sigma_opt"][v_reg],
                        reg_dict["kappa_tot_opt"][v_reg],
                        T_c=tc_val,
                        T_h=th_v,
                    )
                    if reg_key == "n":
                        eng_n = er
                    else:
                        eng_p = er

        regime = zt_res.n_type if zt_res.overall_peak_type == "n-type" else zt_res.p_type
        valid_reg = np.isfinite(regime["zt_max"])
        if np.sum(valid_reg) >= 2:
            T_reg = regime["T"][valid_reg]
            S_reg = regime["S_opt"][valid_reg]
            sig_reg = regime["sigma_opt"][valid_reg]
            kap_reg = regime["kappa_tot_opt"][valid_reg]
            th_val = float(cfg["th"]) if cfg.get("th") is not None else float(np.max(T_reg))
            if tc_val >= float(np.min(T_reg)) - 1e-6 and th_val > tc_val:
                eng_res = eng_n if zt_res.overall_peak_type == "n-type" else eng_p
                eff_sweep = efficiency_vs_hot_temp(
                    T_reg, S_reg, sig_reg, kap_reg, T_c=tc_val
                )

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = zt_res.summary_table()
    _write_csv(rows, outdir / "zt_results.csv")
    _write_latex_table(rows, outdir / "zt_results.tex")
    export_research_bundle(
        zt_res,
        outdir,
        elastic_props=elastic_props,
        eng_result=eng_res,
        eng_result_n=eng_n,
        eng_result_p=eng_p,
        tau_mode=cfg["tau_mode"],
        kappa_mode=cfg["kappa_mode"],
    )
    _print_summary(cfg, elastic_props, zt_res, eng_res)
    if cfg.get("at_ef") and zt_res.at_ef is not None:
        ef_d = zt_res.at_ef
        print("\nIntrinsic Transport at mu = E_F (selected temperatures):")
        print(f"{'T(K)':>6} {'S_tot(uV/K)':>13} {'sigma(S/m)':>13} {'PF(mW/mK2)':>12} {'kappa_tot':>11} {'ZT(E_F)':>9}")
        step = max(1, len(ef_d["T"]) // 10)
        for idx in range(0, len(ef_d["T"]), step):
            print(
                f"{ef_d['T'][idx]:6.0f} {ef_d['S'][idx]*1e6:13.2f} {ef_d['sigma'][idx]:13.3e} "
                f"{ef_d['pf'][idx]:12.4f} {ef_d['kappa_tot'][idx]:11.4f} {ef_d['zt'][idx]:9.4f}"
            )

    if cfg.get("plot"):
        kappa_cahill = None
        if elastic_props is not None and elastic_props.v_l and elastic_props.v_t and vol_atom:
            kappa_cahill = cahill_pohl_kappa_min(
                zt_res.T_grid,
                v_l=elastic_props.v_l,
                v_t=elastic_props.v_t,
                vol_atom=vol_atom,
            )
        custom_pts = (
            load_custom_kappa_csv(cfg["kappa_csv"])
            if cfg["kappa_mode"] == "custom" and cfg.get("kappa_csv")
            else None
        )
        plot_kappa_l_vs_t(
            zt_res.T_grid,
            zt_res.kappa_l_grid,
            kappa_cahill=kappa_cahill,
            custom_points=custom_pts,
            output_path=outdir / "kappa_l_vs_t.png",
        )

        zt_const = None
        if cfg["tau_mode"] != "constant":
            zt_const = compute_zt(
                transport_data,
                tau=float(cfg["tau_const"]),
                kappa_l=kappa_func,
                vol_uc_m3=vol_uc,
            )
        plot_zt_vs_t(zt_res, zt_const_result=zt_const, output_path=outdir / "zt_vs_t.png")
        plot_zt_heatmap(zt_res, output_path=outdir / "zt_heatmap.png")
        if eff_sweep is not None and len(eff_sweep["T_h"]) >= 2:
            plot_efficiency_vs_th(
                eff_sweep,
                output_path=outdir / "efficiency_vs_th.png",
                T_c=float(cfg["tc"]),
            )

    print()
    print(CITATION_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
