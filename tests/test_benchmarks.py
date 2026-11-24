"""Comprehensive unit and integration benchmark suite for UniZT."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unizt import __author__, __email__, __version__
from unizt.core.elastic import (
    compute_elastic_properties,
    isotropic_tensor_from_moduli,
)
from unizt.core.engineering import compute_engineering_zt
from unizt.core.kappa_l import (
    cahill_pohl_kappa_min,
    compute_kappa_l,
    slack_kappa_l,
)
from unizt.core.relaxation import (
    compute_relaxation_time,
    tau_constant,
    tau_dpt,
    tau_power_law,
)
from unizt.core.validator import (
    BornStabilityError,
    ValidationError,
    validate_elastic_tensor,
    validate_transport_data,
)
from unizt.core.zt import compute_zt
from unizt.parsers.boltztrap2 import parse_boltztrap2_trace
from unizt.parsers.elastic_parser import parse_elastic_tensor
from unizt.parsers.generic import parse_generic_transport
from unizt.parsers.registry import PARSER_REGISTRY, get_parser, register_parser


def test_isotropic_vrh_moduli_and_acoustic_properties():
    """Exact isotropic tensor must satisfy B_V=B_R=B_H, G_V=G_R=G_H, exact E & nu."""
    B_true = 100.0
    G_true = 45.0
    rho = 5000.0
    C_iso = isotropic_tensor_from_moduli(B_true, G_true)

    props = compute_elastic_properties(C_iso, density=rho, m_avg=50.0)
    E_true = (9.0 * B_true * G_true) / (3.0 * B_true + G_true)
    nu_true = (3.0 * B_true - 2.0 * G_true) / (2.0 * (3.0 * B_true + G_true))
    gamma_true = 1.5 * (1.0 + nu_true) / (2.0 - 3.0 * nu_true)

    assert props.B_V == pytest.approx(B_true, rel=1e-10)
    assert props.B_R == pytest.approx(B_true, rel=1e-10)
    assert props.B_H == pytest.approx(B_true, rel=1e-10)
    assert props.G_V == pytest.approx(G_true, rel=1e-10)
    assert props.G_R == pytest.approx(G_true, rel=1e-10)
    assert props.G_H == pytest.approx(G_true, rel=1e-10)
    assert props.E == pytest.approx(E_true, rel=1e-10)
    assert props.nu == pytest.approx(nu_true, rel=1e-10)
    assert props.gamma == pytest.approx(gamma_true, rel=1e-10)
    assert props.anisotropy_index == pytest.approx(0.0, abs=1e-10)
    assert props.v_l > props.v_m > props.v_t > 0.0
    assert props.theta_D > 0.0


def test_born_stability_and_tensor_validation():
    """Stable tensors pass; unstable, asymmetric, or malformed tensors fail."""
    C_stable = isotropic_tensor_from_moduli(80.0, 35.0)
    validated = validate_elastic_tensor(C_stable)
    assert validated.shape == (6, 6)

    # Unstable cubic crystal: C11 < C12 -> negative shear eigenvalue (C11 - C12)
    C_unstable = np.array(
        [
            [50.0, 70.0, 70.0, 0.0, 0.0, 0.0],
            [70.0, 50.0, 70.0, 0.0, 0.0, 0.0],
            [70.0, 70.0, 50.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 30.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 30.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
        ]
    )
    with pytest.raises(BornStabilityError):
        validate_elastic_tensor(C_unstable)

    # Asymmetric tensor
    C_asym = C_stable.copy()
    C_asym[0, 1] += 10.0
    with pytest.raises(ValidationError):
        validate_elastic_tensor(C_asym)

    # Wrong shape
    with pytest.raises(ValidationError):
        validate_elastic_tensor(np.eye(3))


def test_validate_transport_data():
    """Validator filters NaN/Inf rows, rejects T <= 0, and converts cm^-3 -> m^-3."""
    raw = {
        "T": [300.0, 400.0, 500.0],
        "n": [-1e19, np.nan, 2e19],
        "S": [-200e-6, -210e-6, np.inf],
        "sigma_tau": [1e19, 1e19, 1e19],
        "kappa_e_tau": [2.0, 2.0, 2.0],
        "n_unit": "cm^-3",
    }
    with pytest.warns(UserWarning, match="Removed 2"):
        cleaned = validate_transport_data(raw)
    assert len(cleaned["T"]) == 1
    assert cleaned["n"][0] == pytest.approx(-1e25)
    assert cleaned["n_unit"] == "m^-3"

    with pytest.raises(ValidationError):
        validate_transport_data(
            {
                "T": [-50.0, 300.0],
                "n": [-1e25, 1e25],
                "S": [-2e-4, 2e-4],
                "sigma_tau": [1e19, 1e19],
                "kappa_e_tau": [2.0, 2.0],
            }
        )


def test_slack_scaling_cahill_positivity_and_custom_csv(tmp_path: Path):
    """Slack kappa_L obeys 1/T scaling, Cahill kappa_min > 0, custom CSV works."""
    T = np.array([300.0, 600.0, 900.0])
    k_slack = slack_kappa_l(
        T,
        M_avg=90.0,
        vol_atom=2.0e-29,
        theta_D=350.0,
        gamma=1.5,
        n_atoms=3,
    )
    assert k_slack[0] > 0.0
    assert k_slack[1] == pytest.approx(0.5 * k_slack[0], rel=1e-12)
    assert k_slack[2] == pytest.approx((1.0 / 3.0) * k_slack[0], rel=1e-12)

    k_min = cahill_pohl_kappa_min(T, v_l=5000.0, v_t=3000.0, vol_atom=2.0e-29)
    assert np.all(k_min > 0.0)
    assert k_min[2] >= k_min[0]

    csv_file = tmp_path / "kappa_exp.csv"
    csv_file.write_text(
        "temperature_K,kappa_L_W_m_K\n300,4.0\n600,2.0\n",
        encoding="utf-8",
    )
    k_custom = compute_kappa_l(450.0, mode="custom", custom_csv=csv_file, interpolation="linear")
    assert k_custom == pytest.approx(3.0, rel=1e-12)
    k_custom_log = compute_kappa_l(
        np.sqrt(300.0 * 600.0), mode="custom", custom_csv=csv_file, interpolation="loglog"
    )
    assert k_custom_log == pytest.approx(np.sqrt(8.0), rel=1e-10)


def test_dpt_relaxation_scaling_and_modes():
    """Bardeen-Shockley tau_dpt(T) follows exact T^(-3/2) scaling."""
    T = np.array([300.0, 1200.0])
    taus = tau_dpt(T, m_eff=0.35, E1=12.0, C_3d=150.0)
    assert taus[0] > 0.0
    assert taus[1] == pytest.approx(taus[0] * (4.0 ** -1.5), rel=1e-12)

    assert tau_constant(300.0, tau_0=2e-14) == pytest.approx(2e-14)
    assert tau_power_law(600.0, tau_0=1e-14, T_0=300.0, exponent=-1.0) == pytest.approx(0.5e-14)
    assert compute_relaxation_time(300.0, mode="constant", tau_const=3e-14) == pytest.approx(3e-14)


def test_zt_identity_and_regime_extraction():
    """ZT matches S^2 * sigma * T / (kappa_e + kappa_L) and extracts n/p peaks."""
    T = np.array([300.0, 300.0, 300.0, 300.0, 600.0, 600.0, 600.0, 600.0])
    n = np.array([-2e25, -1e25, 1e25, 3e25, -2e25, -1e25, 1e25, 3e25])
    S = np.array([-220e-6, -150e-6, 180e-6, 240e-6, -260e-6, -190e-6, 210e-6, 270e-6])
    sigma_tau = np.array([1.2e19, 0.8e19, 1.0e19, 1.5e19, 1.0e19, 0.7e19, 0.9e19, 1.3e19])
    kappa_e_tau = np.array([0.8e14, 0.5e14, 0.6e14, 1.0e14, 1.4e14, 0.9e14, 1.1e14, 1.6e14])

    tau_val = 1.5e-14
    kappa_l_val = 2.0

    res = compute_zt(
        {"T": T, "n": n, "S": S, "sigma_tau": sigma_tau, "kappa_e_tau": kappa_e_tau},
        tau=tau_val,
        kappa_l=kappa_l_val,
    )

    sigma_exp = sigma_tau * tau_val
    ke_exp = kappa_e_tau * tau_val
    zt_exp = (S ** 2) * sigma_exp * T / (ke_exp + kappa_l_val)
    pf_exp = (S ** 2) * sigma_exp * 1e3

    np.testing.assert_allclose(res.zt, zt_exp, rtol=1e-12)
    np.testing.assert_allclose(res.pf, pf_exp, rtol=1e-12)
    assert res.n_type["n_opt"][0] < 0.0
    assert res.p_type["n_opt"][0] > 0.0
    assert res.n_type["zt_max"][1] == pytest.approx(np.max(zt_exp[4:6]))
    assert res.p_type["zt_max"][1] == pytest.approx(np.max(zt_exp[6:8]))


def test_engineering_zt_and_carnot_bound():
    """Kim et al. ZT_eng & eta_max match Ioffe limit for constant properties and obey Carnot."""
    T = np.linspace(300.0, 900.0, 61)
    S = np.full_like(T, 210e-6)
    sigma = np.full_like(T, 1.1e5)
    kappa = np.full_like(T, 1.8)

    eng = compute_engineering_zt(T, S, sigma, kappa, T_c=300.0, T_h=900.0)
    Z_const = (210e-6 ** 2) * 1.1e5 / 1.8
    T_avg = 0.5 * (300.0 + 900.0)
    eta_carnot = 1.0 - 300.0 / 900.0
    eta_ioffe = eta_carnot * (np.sqrt(1.0 + Z_const * T_avg) - 1.0) / (
        np.sqrt(1.0 + Z_const * T_avg) + 300.0 / 900.0
    )

    assert eng.ZT_eng == pytest.approx(Z_const * 600.0, rel=1e-6)
    assert eng.eta_max == pytest.approx(eta_ioffe, rel=1e-6)
    assert eng.eta_max < eng.eta_carnot


def test_parsers_and_registry(tmp_path: Path):
    """Test boltztrap2, generic, elastic parsers (JSON/VASP/WIEN2k/QE), and PARSER_REGISTRY."""
    bt_file = tmp_path / "case.trace"
    bt_file.write_text(
        "# Ef[Ry] T[K] N DOS(Ef) S[V/K] sigma/tau0[1/(Ohm*m*s)] RH[m^3/C] kappa0[W/(m*K*s)]\n"
        "0.350 300.0 -0.05 1.2 -2.0e-4 1.0e19 -1.0e-8 1.5e14\n"
        "0.320 300.0  0.05 1.1  1.9e-4 0.9e19  1.0e-8 1.3e14\n",
        encoding="utf-8",
    )
    bt_data = parse_boltztrap2_trace(bt_file, vol_uc_m3=5.0e-29)
    assert bt_data["T"][0] == pytest.approx(300.0)
    assert bt_data["n"][0] == pytest.approx(-0.05 / 5.0e-29)
    assert bt_data["S"][1] == pytest.approx(1.9e-4)

    gen_file = tmp_path / "transport.csv"
    gen_file.write_text(
        "300.0,-1.5e25,-2.1e-4,1.1e19,1.2e14\n"
        "600.0,1.2e25,2.3e-4,0.8e19,1.8e14\n",
        encoding="utf-8",
    )
    gen_data = parse_generic_transport(gen_file, cols="T,n,S,sigma_tau,kappa_e_tau")
    assert gen_data["sigma_tau"][0] == pytest.approx(1.1e19)
    assert gen_data["kappa_e_tau"][1] == pytest.approx(1.8e14)

    # Elastic JSON & VASP OUTCAR
    C_iso = isotropic_tensor_from_moduli(90.0, 40.0)
    json_file = tmp_path / "elastic.json"
    json_file.write_text(json.dumps({"C_ij": C_iso.tolist()}), encoding="utf-8")
    np.testing.assert_allclose(parse_elastic_tensor(json_file, fmt="json"), C_iso)

    vasp_file = tmp_path / "OUTCAR"
    c11, c12, c44 = (C_iso[0, 0] * 10.0, C_iso[0, 1] * 10.0, C_iso[3, 3] * 10.0)
    vasp_file.write_text(
        " TOTAL ELASTIC MODULI (kBar)\n"
        " Direction    XX          YY          ZZ          XY          YZ          ZX\n"
        " --------------------------------------------------------------------------------\n"
        f" XX {c11:10.4f} {c12:10.4f} {c12:10.4f} 0.0 0.0 0.0\n"
        f" YY {c12:10.4f} {c11:10.4f} {c12:10.4f} 0.0 0.0 0.0\n"
        f" ZZ {c12:10.4f} {c12:10.4f} {c11:10.4f} 0.0 0.0 0.0\n"
        f" XY 0.0 0.0 0.0 {c44:10.4f} 0.0 0.0\n"
        f" YZ 0.0 0.0 0.0 0.0 {c44:10.4f} 0.0\n"
        f" ZX 0.0 0.0 0.0 0.0 0.0 {c44:10.4f}\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(parse_elastic_tensor(vasp_file, fmt="vasp"), C_iso, atol=1e-4)

    # WIEN2k elast/IRelast format
    wien_file = tmp_path / "elast.output"
    wien_file.write_text(
        f"C11 = {C_iso[0, 0]:.4f} GPa\n"
        f"C12 = {C_iso[0, 1]:.4f} GPa\n"
        f"C44 = {C_iso[3, 3]:.4f} GPa\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(parse_elastic_tensor(wien_file, fmt="wien2k"), C_iso, atol=1e-4)

    # Quantum ESPRESSO thermo_pw format (kbar)
    qe_file = tmp_path / "thermo_pw.out"
    qe_rows = "\n".join("  ".join(f"{val * 10.0:10.4f}" for val in row) for row in C_iso)
    qe_file.write_text(
        "Elastic constants (kbar)\n" + qe_rows + "\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(parse_elastic_tensor(qe_file, fmt="qe"), C_iso, atol=1e-4)

    # Registry checks
    assert "boltztrap2" in PARSER_REGISTRY
    assert "generic" in PARSER_REGISTRY
    register_parser("custom_dummy", lambda p: {"T": np.array([300.0])})
    assert callable(get_parser("custom_dummy"))
    PARSER_REGISTRY.pop("custom_dummy", None)


def test_cli_cite_and_end_to_end(tmp_path: Path):
    """Verify CLI --cite banner/BibTeX and full pipeline execution with --config & --plot."""
    proc_cite = subprocess.run(
        [sys.executable, "-m", "unizt.cli", "--cite"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"UniZT v{__version__}" in proc_cite.stdout
    assert __author__ in proc_cite.stdout
    assert __email__ in proc_cite.stdout
    assert "@software{makhlouf2026unizt" in proc_cite.stdout

    # End-to-end CLI run with --config and --plot
    trace_file = tmp_path / "sample.trace"
    rows = ["# Ef T N DOS S sigma/tau RH kappa0"]
    for T in (300.0, 450.0, 600.0):
        for n_val, s_val in [(-0.15, -2.1e-4), (-0.05, -2.5e-4), (0.06, 2.3e-4), (0.16, 1.9e-4)]:
            rows.append(f"0.3 {T} {n_val} 1.0 {s_val} 1.2e19 0.0 1.4e14")
    trace_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    C_iso = isotropic_tensor_from_moduli(80.0, 40.0)
    elastic_file = tmp_path / "elastic.json"
    elastic_file.write_text(json.dumps({"C_ij": C_iso.tolist()}), encoding="utf-8")

    outdir = tmp_path / "out"
    cfg_file = tmp_path / "params.json"
    cfg_file.write_text(
        json.dumps(
            {
                "trace": str(trace_file),
                "elastic": str(elastic_file),
                "density": 6000.0,
                "m_avg": 80.0,
                "n_atoms": 3,
                "m_eff": 0.4,
                "e1": 12.0,
                "tau_mode": "dpt",
                "kappa_mode": "slack",
                "tc": 300.0,
                "outdir": str(outdir),
                "plot": True,
            }
        ),
        encoding="utf-8",
    )

    proc_run = subprocess.run(
        [sys.executable, "-m", "unizt.cli", "--config", str(cfg_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Overall Peak ZT" in proc_run.stdout
    assert __author__ in proc_run.stdout
    assert (outdir / "zt_results.csv").is_file()
    assert (outdir / "zt_results.tex").is_file()
    assert (outdir / "kappa_l_vs_t.png").is_file()
    assert (outdir / "zt_vs_t.png").is_file()
    assert (outdir / "zt_heatmap.png").is_file()
    assert (outdir / "efficiency_vs_th.png").is_file()
