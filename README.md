# UniZT

**Unified Post-Processing Tool for True Thermoelectric Figure of Merit $ZT$, Lattice Thermal Conductivity, and Device Efficiency from First-Principles DFT & BoltzTraP2 Calculations.**

Developed by **Dr. Sabir Makhlouf** (`sabir.makhlouf@gmail.com`)

---

## Features

- **Elastic & Acoustic Engine (`unizt.core.elastic`, `unizt.core.validator`)**
  - Born mechanical stability verification via stiffness tensor positive-definiteness.
  - Voigt–Reuss–Hill (VRH) bulk ($B$), shear ($G$), Young's ($E$) moduli, and Poisson's ratio ($\nu$).
  - Longitudinal ($v_l$), transverse ($v_t$), and mean ($v_m$) sound velocities, acoustic Debye temperature ($\theta_D$), and Belomestnykh–Leineweber acoustic Grüneisen parameter ($\gamma$).
- **Lattice Thermal Conductivity (`unizt.core.kappa_l`)**
  - Slack's anharmonic Umklapp equation for $\kappa_L(T)$.
  - Cahill–Pohl amorphous-limit minimum thermal conductivity $\kappa_{\min}(T)$.
  - Custom/experimental $\kappa_L(T)$ override from CSV (`temperature_K,kappa_L_W_m_K`) with linear or log–log interpolation.
- **Carrier Relaxation Time (`unizt.core.relaxation`)**
  - Bardeen–Shockley 3D Deformation Potential Theory ($\tau_{\mathrm{DPT}}(T) \propto T^{-3/2}$).
  - Constant ($\tau_0$) and flexible power-law ($\tau_0 (T / T_0)^p$) sensitivity modes.
- **True $ZT(T, n)$ & Device Conversion Efficiency (`unizt.core.zt`, `unizt.core.engineering`)**
  - Combines BoltzTraP2 transport coefficients with $\tau(T)$ and $\kappa_L(T)$ to compute Power Factor $\mathrm{PF} = S^2 \sigma$, $\kappa_{\mathrm{tot}}$, and $ZT(T, n)$.
  - **Spin-Polarized Two-Channel Transport (`combine_spin_channels`)**: Exact two-current synthesis of spin-up and spin-down channels including conductivity-weighted Seebeck $S_{\mathrm{tot}}$ and the spin-mixing / bipolar thermal conductivity term $\frac{\sigma_\uparrow \sigma_\downarrow}{\sigma_\uparrow + \sigma_\downarrow}(S_\uparrow - S_\downarrow)^2 T$.
  - Separates $n$-type ($n < 0$) and $p$-type ($n > 0$) regimes and extracts optimal doping $n_{\mathrm{opt}}(T)$ and peak $ZT_{\max}(T)$.
  - Kim et al. (PNAS 2015) engineering figure of merit $ZT_{\mathrm{eng}}$ and leg conversion efficiency $\eta_{\max}$ with automatic Carnot bound verification ($\eta_{\max} \le 1 - T_c / T_h$).
- **Lightweight Parsers & Extensible Registry (`unizt.parsers`)**
  - Pure stdlib + NumPy parsers for BoltzTraP2/1 `*.trace`, arbitrary whitespace/CSV files (`--cols`), and 6×6 elastic tensors from VASP `OUTCAR`, WIEN2k `elast`/`IRelast`, Quantum ESPRESSO `thermo_pw`, and JSON.
- **Publication Outputs (`unizt.plotting`, `unizt.cli`)**
  - 300 DPI journal-ready PNG figures, CSV summaries, and `booktabs` LaTeX tables (`zt_results.tex`).

---

## Installation

```bash
pip install -e ".[dev]"
```

Or standard installation:

```bash
pip install .
```

---

## CLI Quickstart

```bash
# Print academic attribution and BibTeX citation
unizt --cite

# Full post-processing with BoltzTraP2 trace, VASP elastic tensor, DPT + Slack, and 300 DPI plots
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

# Spin-polarized two-channel post-processing (e.g. magnetic / half-metallic Cs2VBr6)
unizt \
  --trace-up examples/cs2vbr6/up.trace \
  --trace-dn examples/cs2vbr6/dn.trace \
  --density 4440.1 \
  --m-avg 88.464 \
  --n-atoms 9 \
  --tau-mode dpt \
  --m-eff 0.85 \
  --e1 11.0 \
  --c3d 39.67 \
  --kappa-mode cahill \
  --outdir results_cs2vbr6 \
  --plot

# Using a generic CSV file with explicit column mapping and JSON configuration
unizt \
  --trace transport.csv \
  --code generic \
  --cols "T,n,S,sigma_tau,kappa_e_tau" \
  --config params.json \
  --plot
```

---

## Benchmarks & Real-World Case Study

Run the automated zero-recalculation benchmark across $\mathrm{Bi_2Te_3}$, $\mathrm{PbTe}$, $\mathrm{ZrNiSn}$, $\mathrm{Mg_2Si}$, and $\mathrm{SnSe}$:

```bash
python scripts/run_paper_benchmark.py
```

Run the real-world spin-polarized two-channel case study on the vacancy-ordered double perovskite $\mathrm{Cs_2VBr_6}$ ($Fm\bar{3}m$, using `examples/cs2vbr6/up.trace` & `dn.trace`):

```bash
python scripts/run_cs2vbr6_case_study.py
```

Outputs are generated in `paper_outputs/` (including `cs2vbr6_provenance.json`, `table_cs2vbr6_summary.csv`, `table_cs2vbr6_summary.tex`, `fig_cs2vbr6_spin_channels.png`, and `fig_cs2vbr6_zt_vs_t.png`).

---

## License

MIT License — Copyright (c) 2026 Dr. Sabir Makhlouf (`sabir.makhlouf@gmail.com`). See [LICENSE](LICENSE).

## Citation

If you use UniZT in your research, please cite:

```bibtex
@software{makhlouf2026unizt,
  author  = {Makhlouf, Sabir},
  title   = {{UniZT}: A Unified Post-Processing Tool for True Thermoelectric
             Figure of Merit and Device Efficiency from First-Principles Calculations},
  year    = {2026},
  version = {0.2.0},
  license = {MIT},
  email   = {sabir.makhlouf@gmail.com}
}
```
