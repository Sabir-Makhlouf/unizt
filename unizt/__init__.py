"""UniZT: Unified Post-Processing Tool for True Thermoelectric Figure of Merit ZT,
Lattice Thermal Conductivity, and Device Efficiency from First-Principles DFT & BoltzTraP2 Calculations.
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Dr. Sabir Makhlouf"
__email__ = "sabir.makhlouf@gmail.com"
__license__ = "MIT"

CITATION_BANNER = (
    f"UniZT v{__version__} | Developed by {__author__} ({__email__})\n"
    "If you use UniZT in published research, please cite:\n"
    'S. Makhlouf, "UniZT: A Unified Post-Processing Tool for True Thermoelectric\n'
    'Figure of Merit and Device Efficiency from First-Principles Calculations" (2026).'
)

BIBTEX_CITATION = (
    "@software{makhlouf2026unizt,\n"
    "  author  = {Makhlouf, Sabir},\n"
    "  title   = {{UniZT}: A Unified Post-Processing Tool for True Thermoelectric\n"
    "             Figure of Merit and Device Efficiency from First-Principles Calculations},\n"
    "  year    = {2026},\n"
    f"  version = {{{__version__}}},\n"
    "  license = {MIT},\n"
    f"  email   = {{{__email__}}}\n"
    "}"
)

from unizt.core import (
    ElasticProperties,
    EngineeringResult,
    ZTResult,
    cahill_pohl_kappa_min,
    combine_spin_channels,
    compute_elastic_properties,
    compute_engineering_zt,
    compute_kappa_l,
    compute_relaxation_time,
    compute_zt,
    efficiency_vs_hot_temp,
    optimize_zt,
    slack_kappa_l,
    tau_constant,
    tau_dpt,
    tau_power_law,
    validate_elastic_tensor,
    validate_transport_data,
)
from unizt.export import export_research_bundle
from unizt.parsers import (
    PARSER_REGISTRY,
    get_parser,
    list_parsers,
    parse_boltztrap2_trace,
    parse_elastic_tensor,
    parse_generic_transport,
    parse_spin_polarized_traces,
    register_parser,
)

__all__ = [
    "BIBTEX_CITATION",
    "CITATION_BANNER",
    "ElasticProperties",
    "EngineeringResult",
    "PARSER_REGISTRY",
    "ZTResult",
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "cahill_pohl_kappa_min",
    "combine_spin_channels",
    "compute_elastic_properties",
    "compute_engineering_zt",
    "compute_kappa_l",
    "compute_relaxation_time",
    "compute_zt",
    "efficiency_vs_hot_temp",
    "export_research_bundle",
    "get_parser",
    "list_parsers",
    "optimize_zt",
    "parse_boltztrap2_trace",
    "parse_elastic_tensor",
    "parse_generic_transport",
    "parse_spin_polarized_traces",
    "register_parser",
    "slack_kappa_l",
    "tau_constant",
    "tau_dpt",
    "tau_power_law",
    "validate_elastic_tensor",
    "validate_transport_data",
]
