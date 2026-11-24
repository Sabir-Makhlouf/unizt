"""Core physics engine for UniZT."""

from unizt.core.elastic import (
    ElasticProperties,
    compute_elastic_properties,
    debye_temperature,
    gruneisen_from_poisson,
    isotropic_tensor_from_moduli,
    sound_velocities,
)
from unizt.core.engineering import (
    EngineeringResult,
    compute_engineering_zt,
    efficiency_vs_hot_temp,
)
from unizt.core.kappa_l import (
    cahill_pohl_kappa_min,
    compute_kappa_l,
    evaluate_custom_kappa,
    load_custom_kappa_csv,
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
from unizt.core.zt import ZTResult, combine_spin_channels, compute_zt

__all__ = [
    "BornStabilityError",
    "ElasticProperties",
    "EngineeringResult",
    "ValidationError",
    "ZTResult",
    "cahill_pohl_kappa_min",
    "combine_spin_channels",
    "compute_elastic_properties",
    "compute_engineering_zt",
    "compute_kappa_l",
    "compute_relaxation_time",
    "compute_zt",
    "debye_temperature",
    "efficiency_vs_hot_temp",
    "evaluate_custom_kappa",
    "gruneisen_from_poisson",
    "isotropic_tensor_from_moduli",
    "load_custom_kappa_csv",
    "slack_kappa_l",
    "sound_velocities",
    "tau_constant",
    "tau_dpt",
    "tau_power_law",
    "validate_elastic_tensor",
    "validate_transport_data",
]
