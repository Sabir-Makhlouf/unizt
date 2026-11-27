# UniZT

[![PyPI version](https://img.shields.io/pypi/v/unizt.svg)](https://pypi.org/project/unizt/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://pypi.org/project/unizt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Sabir-Makhlouf/unizt/blob/main/LICENSE)
[![CI](https://github.com/Sabir-Makhlouf/unizt/actions/workflows/ci.yml/badge.svg)](https://github.com/Sabir-Makhlouf/unizt/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24513282.svg)](https://doi.org/10.5281/zenodo.24513282)

**UniZT (`v0.2.0`)** is a lightweight, code-agnostic, open-source Python package for unified first-principles thermoelectric post-processing. It bridges density functional theory (DFT) elastic tensors and semiclassical Boltzmann transport traces (`BoltzTraP2`/`1`) into realistic material and device-level thermoelectric evaluation—covering mechanical stability, acoustic lattice thermal conductivity, temperature-dependent carrier scattering, exact spin-polarized two-channel transport, physical rigid band approximation (RBA) bounds, and maximum thermoelectric generator leg efficiency.

---

## Key Features

- **Code-Agnostic Parsers & Registry:** Parses `BoltzTraP2`/`1` single and spin-polarized (`up.trace`, `dn.trace`) transport files alongside $6 \times 6$ Voigt stiffness tensors ($C_{ij}$) from **WIEN2k**, **VASP** (`OUTCAR`), **Quantum ESPRESSO** (`thermo_pw`), **JSON**, or custom column-mapped tables via an extensible `PARSER_REGISTRY`.
- **Mechanical Stability & Acoustics:** Validates tensor symmetry and Born mechanical stability ($\lambda_k > 0$ for all six eigenvalues), and computes Voigt–Reuss–Hill (VRH) elastic moduli ($B, G, E, \nu$), acoustic sound velocities ($v_l, v_t, v_m$), acoustic Debye temperature ($\theta_D$), and Grüneisen parameter ($\gamma$).
- **Lattice Thermal Conductivity ($\kappa_L$):** Evaluates $\kappa_L(T)$ via Slack's equation and the Cahill–Pohl minimum (amorphous-limit) thermal conductivity $\kappa_{\min}(T)$, or smoothly interpolates external/anharmonic `Phono3py`/experimental curves (`--kappa-mode custom --kappa-csv`).
- **Physical Carrier Relaxation Time ($\tau$):** Moves beyond arbitrary constant relaxation times using 3D Bardeen–Shockley Deformation Potential Theory (DPT, $\tau \propto T^{-3/2}$), alongside `--tau-mode power_law` and `--tau-mode constant` options for sensitivity benchmarking.
- **Exact Spin-Polarized Two-Channel Model:** Combines majority ($\uparrow$) and minority ($\downarrow$) spin channels on aligned $(T, \mu)$ grids using conductivity-weighted Seebeck coupling and explicitly includes the spin-mixing electronic thermal conductivity term:
  $$\kappa_{e,\mathrm{mix}} = \frac{\sigma_\uparrow \sigma_\downarrow}{\sigma_\uparrow + \sigma_\downarrow} (S_\uparrow - S_\downarrow)^2 T$$
- **Physical RBA Windowing & Intrinsic Extraction:** Enforces configurable physical RBA limits ($|\mu - E_F| \le 0.5\ \mathrm{eV}$ and $|n| \le 10^{21}\ \mathrm{cm^{-3}}$ by default) to prevent artificial $ZT$ inflation at unphysical doping levels, while extracting intrinsic undoped transport directly at $\mu = E_F$ (`--at-ef`).
- **Device-Level Conversion Efficiency:** Integrates temperature-dependent transport across $\Delta T = T_h - T_c$ using the Kim et al. engineering figure of merit ($ZT_{\mathrm{eng}}$) and maximum leg conversion efficiency ($\eta_{\max}$), strictly verified against the Carnot limit ($\eta_C = 1 - T_c / T_h$).
- **Researcher-Ready Export Bundle:** Automatically writes human-readable reports, machine-readable JSON metrics, CSV datasets, publication-ready `booktabs` LaTeX tables, and 300 DPI Matplotlib figures.

---

## Installation

Install the latest stable release directly from PyPI:

```bash
pip install unizt
```

Or install from source (including development and test dependencies):

```bash
git clone https://github.com/Sabir-Makhlouf/unizt.git
cd unizt
pip install -e ".[dev]"
```

**Requirements:** Python $\ge 3.9$, `numpy`, `scipy`, `matplotlib`.

---

## CLI Quickstart

### 1. Print Academic Citation

```bash
unizt --cite
```

### 2. Run Immediately on the Bundled Spin-Polarized Example ($\mathrm{Cs_2VBr_6}$)

Run a complete, reproducible spin-polarized two-channel analysis using the bundled WIEN2k/BoltzTraP2 files in `examples/cs2vbr6/`:

```bash
unizt \
  --trace-up examples/cs2vbr6/up.trace \
  --trace-dn examples/cs2vbr6/dn.trace \
  --elastic examples/cs2vbr6/Cij.json \
  --elastic-format json \
  --density 3951.6 \
  --m-avg 88.46 \
  --n-atoms 9 \
  --vol-atom 37.17 \
  --tau-mode dpt \
  --m-eff 0.50 \
  --e1 9.0 \
  --kappa-mode slack \
  --ef-window 0.5 \
  --max-carrier-density 1e21 \
  --at-ef \
  --tc 300 \
  --th 1000 \
  --outdir results_cs2vbr6 \
  --plot
```

### 3. Template for Non-Magnetic VASP + BoltzTraP2 Post-Processing

```bash
unizt \
  --trace case.trace \
  --code boltztrap2 \
  --elastic OUTCAR \
  --elastic-format vasp \
  --density 7699 \
  --m-avg 89.54 \
  --n-atoms 3 \
  --tau-mode dpt \
  --m-eff 0.45 \
  --e1 14.0 \
  --kappa-mode slack \
  --tc 300 \
  --outdir results \
  --plot
```

> **Windows PowerShell Tip:** Replace line-continuation backslashes (`\`) with backticks (`` ` ``), pass the command on a single line, or supply a JSON configuration file via `unizt --config config.json`.

---

## Input Units Reference

All internal calculations in UniZT are executed strictly in SI units using CODATA constants from `scipy.constants`. Please supply CLI inputs in the units below:

| CLI Flag | Physical Quantity | Required Unit |
| :--- | :--- | :--- |
| `--density` | Crystal mass density ($\rho$) | $\mathrm{kg\,m^{-3}}$ |
| `--m-avg` | Average atomic mass ($\bar{M}$) | $\mathrm{amu}$ ($\mathrm{g\,mol^{-1}}$) |
| `--n-atoms` | Number of atoms in the primitive cell ($n$) | dimensionless integer |
| `--vol-atom` | Average volume per atom ($\delta^3$) | $\mathrm{\AA^3}$ |
| `--m-eff` | Carrier effective mass ($m^*$) | $m_0$ (free-electron mass) |
| `--e1` | Band-edge deformation potential ($E_1$) | $\mathrm{eV}$ |
| `--ef-window` | RBA chemical-potential window ($|\mu - E_F|$) | $\mathrm{eV}$ |
| `--max-carrier-density` | RBA carrier-density cap ($|n|$) | $\mathrm{cm^{-3}}$ |
| `--tc`, `--th` | Cold-side and hot-side temperatures ($T_c, T_h$) | $\mathrm{K}$ |

---

## Output Bundle

Each UniZT run writes a structured researcher bundle to `--outdir`:

| Output File | Description |
| :--- | :--- |
| `report_summary.txt` | Human-readable summary of Born stability, VRH moduli, acoustics, $\kappa_L(T)$, $\tau(T)$, intrinsic ($\mu = E_F$) transport, RBA-constrained optimal n-type/p-type peaks, and $\eta_{\max}$. |
| `summary_metrics.json` | Machine-readable JSON containing scalar performance metrics and provenance. |
| `transport_at_ef.csv` | Temperature-dependent intrinsic transport at $\mu = E_F$ ($S_\uparrow, S_\downarrow, S_{\mathrm{tot}}, \sigma_{\mathrm{tot}}, \mathrm{PF}, \kappa_{e,\mathrm{tot}}, \kappa_{e,\mathrm{mix}}, \kappa_L, ZT_{\mathrm{tot}}$). |
| `optimal_doping_vs_T.csv` | Optimal n-type and p-type carrier concentrations, chemical-potential shifts, Seebeck coefficients, power factors, and $ZT_{\max}(T)$ within the RBA window. |
| `table_summary.tex` | Publication-ready `booktabs` LaTeX summary table. |
| `*.png` / `*.jpg` | Automated 300 DPI publication-grade figures (when `--plot` is enabled). |

---

## Validation & Benchmarks

### Multi-Class Benchmark (5 Crystal Systems)

Executed via `scripts/run_paper_benchmark.py` across five canonical thermoelectric crystal classes ($\mathrm{Bi_2Te_3}$, $\mathrm{PbTe}$, $\mathrm{ZrNiSn}$, $\mathrm{Mg_2Si}$, and $\mathrm{SnSe}$):

- **$\kappa_L(300\ \mathrm{K})$ Parity:** $R^2 = 0.797$, $\mathrm{MAE} = 1.54\ \mathrm{W\,m^{-1}\,K^{-1}}$.
- **Peak $ZT$ Parity:** $R^2 = 0.958$, $\mathrm{MAE} = 0.11$.

### Spin-Polarized Case Study: Half-Metallic $\mathrm{Cs_2VBr_6}$

Evaluated on real WIEN2k/BoltzTraP2 spin-up and spin-down traces (`examples/cs2vbr6/`):

- Reproduces published VRH elastic moduli and acoustic sound velocities ($B = 29.87\ \mathrm{GPa}$, $G = 10.13\ \mathrm{GPa}$, $\theta_D = 160.51\ \mathrm{K}$, $\kappa_L(300\ \mathrm{K}) = 0.317\ \mathrm{W\,m^{-1}\,K^{-1}}$).
- Demonstrates why RBA windowing is essential: constraining the search to $|\mu - E_F| \le 0.5\ \mathrm{eV}$ and $|n| \le 10^{21}\ \mathrm{cm^{-3}}$ prevents artificial $ZT$ inflation (from unconstrained $ZT_{\mathrm{raw}} = 7.72$ at $|n| = 1.21 \times 10^{22}\ \mathrm{cm^{-3}}$) down to physically realizable peaks of $ZT_{\max} = 0.386$ (n-type, $\eta_{\max} = 5.57\%$ for $T_c = 300\ \mathrm{K}, T_h = 1000\ \mathrm{K}$) and $ZT_{\max} = 0.175$ (p-type, $\eta_{\max} = 2.42\%$).

---

## Testing & Repository Structure

Run the full automated verification suite (`11` unit and integration tests covering VRH moduli, Born stability, Slack/Cahill–Pohl $\kappa_L$, DPT scaling, Carnot bounds, spin-mixing invariants, RBA filtering, researcher exports, and CLI execution):

```bash
pytest -v
```

```text
unizt/
├── unizt/
│   ├── core/           # validator, elastic, kappa_l, relaxation, zt, engineering
│   ├── parsers/        # boltztrap2, elastic_parser, generic, registry
│   ├── export.py       # report, JSON, CSV, booktabs LaTeX exporter
│   ├── plotting.py     # 300 DPI plotting engine
│   └── cli.py          # unified command-line interface
├── examples/cs2vbr6/   # real spin-polarized WIEN2k/BoltzTraP2 traces & stiffness tensor
├── scripts/            # reproducible multi-class benchmark scripts
├── paper_outputs/      # archived benchmark tables, figures, and provenance
├── tests/              # automated pytest verification suite
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

---

## Citation

If you use **UniZT** in your research, please cite the software archive (or run `unizt --cite`):

```bibtex
@software{makhlouf_2026_unizt,
  author       = {Makhlouf, Sabir},
  title        = {{UniZT: A Unified Open-Source Python Package for True Thermoelectric Figure of Merit, Spin-Polarized Transport, Lattice Thermal Conductivity, and Device Efficiency from First-Principles Calculations}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.2.0},
  doi          = {10.5281/zenodo.24513282},
  url          = {https://github.com/Sabir-Makhlouf/unizt}
}
```

---

## License & Contact

Released under the **MIT License** — see [LICENSE](LICENSE) for details.  
Developed and maintained by **Dr. Sabir Makhlouf** (`sabir.makhlouf@gmail.com`).
