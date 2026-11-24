"""Elastic stiffness tensor analysis, Voigt-Reuss-Hill moduli, sound velocities,
Debye temperature, and acoustic Grüneisen parameter calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import constants as const

from unizt.core.validator import ValidationError, validate_elastic_tensor


@dataclass(frozen=True)
class ElasticProperties:
    """Container for crystal elastic, acoustic, and anharmonic properties.

    All moduli are in GPa, density in kg/m^3, velocities in m/s,
    Debye temperature in K, and atomic volume in m^3.
    """

    C_ij: np.ndarray
    S_ij: np.ndarray
    B_V: float
    B_R: float
    B_H: float
    G_V: float
    G_R: float
    G_H: float
    E: float
    nu: float
    pugh_ratio: float
    anisotropy_index: float
    longitudinal_modulus: float
    gamma: float
    eigenvalues: np.ndarray
    density: Optional[float] = None
    v_l: Optional[float] = None
    v_t: Optional[float] = None
    v_m: Optional[float] = None
    theta_D: Optional[float] = None
    vol_atom: Optional[float] = None

    @property
    def B(self) -> float:
        """Hill bulk modulus (GPa)."""
        return self.B_H

    @property
    def G(self) -> float:
        """Hill shear modulus (GPa)."""
        return self.G_H

    def acoustic_debye_temperature(self, n_atoms: int) -> Optional[float]:
        """Return Slack's acoustic-branch cutoff Debye temperature ``theta_a = theta_D * n^(-1/3)``."""
        if self.theta_D is None:
            return None
        if n_atoms <= 0:
            raise ValidationError("n_atoms must be a positive integer.")
        return float(self.theta_D * (float(n_atoms) ** (-1.0 / 3.0)))


def gruneisen_from_poisson(nu: float) -> float:
    """Compute acoustic Grüneisen parameter from Poisson's ratio via the
    Belomestnykh / Leineweber relation: ``gamma = 3/2 * (1 + nu) / (2 - 3*nu)``.
    """
    denom = 2.0 - 3.0 * float(nu)
    if denom <= 0.0:
        raise ValidationError(
            f"Poisson ratio nu={nu:.4f} yields non-positive denominator (2 - 3*nu <= 0)."
        )
    return float(1.5 * (1.0 + float(nu)) / denom)


def sound_velocities(
    B_gpa: float, G_gpa: float, density: float
) -> tuple[float, float, float]:
    """Compute longitudinal (v_l), transverse (v_t), and mean (v_m) sound velocities (m/s).

    Parameters
    ----------
    B_gpa : float
        Bulk modulus in GPa.
    G_gpa : float
        Shear modulus in GPa.
    density : float
        Mass density in kg/m^3.
    """
    if density <= 0.0:
        raise ValidationError(f"Density must be positive (kg/m^3), got {density}.")
    if B_gpa <= 0.0 or G_gpa <= 0.0:
        raise ValidationError(
            f"Bulk and shear moduli must be positive, got B={B_gpa}, G={G_gpa} GPa."
        )
    B_pa = float(B_gpa) * 1e9
    G_pa = float(G_gpa) * 1e9
    rho = float(density)

    v_l = float(np.sqrt((B_pa + (4.0 / 3.0) * G_pa) / rho))
    v_t = float(np.sqrt(G_pa / rho))
    v_m = float(((1.0 / 3.0) * (2.0 / (v_t ** 3) + 1.0 / (v_l ** 3))) ** (-1.0 / 3.0))
    return v_l, v_t, v_m


def debye_temperature(
    v_m: float,
    density: Optional[float] = None,
    m_avg: Optional[float] = None,
    vol_atom: Optional[float] = None,
) -> float:
    """Compute acoustic Debye temperature theta_D (K) using Anderson's formulation.

    Requires either ``vol_atom`` (in m^3) or both ``density`` (kg/m^3) and
    ``m_avg`` (average atomic mass in amu / g/mol).
    """
    if v_m <= 0.0:
        raise ValidationError(f"Mean sound velocity must be positive, got {v_m}.")

    if vol_atom is not None:
        if vol_atom <= 0.0:
            raise ValidationError(f"vol_atom must be positive (m^3), got {vol_atom}.")
        n_density = 1.0 / float(vol_atom)
    elif density is not None and m_avg is not None:
        if density <= 0.0 or m_avg <= 0.0:
            raise ValidationError(
                f"density and m_avg must be positive, got density={density}, m_avg={m_avg}."
            )
        m_avg_kg_per_mol = float(m_avg) * 1e-3
        n_density = (float(density) * const.Avogadro) / m_avg_kg_per_mol
    else:
        raise ValidationError(
            "Either vol_atom (m^3) or both density (kg/m^3) and m_avg (amu) must be provided."
        )

    theta_D = (const.h / const.k) * ((3.0 * n_density) / (4.0 * const.pi)) ** (1.0 / 3.0) * float(v_m)
    return float(theta_D)


def compute_elastic_properties(
    C_ij: Any,
    density: Optional[float] = None,
    m_avg: Optional[float] = None,
    vol_atom: Optional[float] = None,
    sym_tol: float = 1e-3,
    eig_tol: float = 1e-6,
) -> ElasticProperties:
    """Compute Voigt-Reuss-Hill elastic moduli and acoustic properties from a 6x6 stiffness tensor.

    Parameters
    ----------
    C_ij : array-like
        6x6 elastic stiffness tensor in GPa (Voigt notation).
    density : float, optional
        Mass density in kg/m^3. Needed for sound velocities ``v_l, v_t, v_m``.
    m_avg : float, optional
        Average atomic mass in amu (g/mol). Used with ``density`` to infer ``vol_atom``
        and ``theta_D`` when ``vol_atom`` is not provided.
    vol_atom : float, optional
        Volume per atom in m^3 (SI). Used to compute ``theta_D``.
    sym_tol : float, optional
        Symmetry tolerance in GPa passed to ``validate_elastic_tensor``.
    eig_tol : float, optional
        Minimum eigenvalue threshold in GPa for Born mechanical stability.
    """
    C = validate_elastic_tensor(C_ij, sym_tol=sym_tol, eig_tol=eig_tol)
    eigvals = np.linalg.eigvalsh(C)
    S = np.linalg.inv(C)

    C_diag_sum = float(C[0, 0] + C[1, 1] + C[2, 2])
    C_off_sum = float(C[0, 1] + C[0, 2] + C[1, 2])
    C_shear_sum = float(C[3, 3] + C[4, 4] + C[5, 5])

    B_V = (C_diag_sum + 2.0 * C_off_sum) / 9.0
    G_V = (C_diag_sum - C_off_sum + 3.0 * C_shear_sum) / 15.0

    S_diag_sum = float(S[0, 0] + S[1, 1] + S[2, 2])
    S_off_sum = float(S[0, 1] + S[0, 2] + S[1, 2])
    S_shear_sum = float(S[3, 3] + S[4, 4] + S[5, 5])

    B_R = 1.0 / (S_diag_sum + 2.0 * S_off_sum)
    G_R = 15.0 / (4.0 * S_diag_sum - 4.0 * S_off_sum + 3.0 * S_shear_sum)

    B_H = 0.5 * (B_V + B_R)
    G_H = 0.5 * (G_V + G_R)

    E = (9.0 * B_H * G_H) / (3.0 * B_H + G_H)
    nu = (3.0 * B_H - 2.0 * G_H) / (2.0 * (3.0 * B_H + G_H))
    pugh_ratio = B_H / G_H
    anisotropy_index = 5.0 * (G_V / G_R) + (B_V / B_R) - 6.0
    longitudinal_modulus = B_H + (4.0 / 3.0) * G_H
    gamma = gruneisen_from_poisson(nu)

    resolved_vol_atom = vol_atom
    if resolved_vol_atom is None and density is not None and m_avg is not None:
        if density > 0.0 and m_avg > 0.0:
            resolved_vol_atom = float((float(m_avg) * 1e-3) / (const.Avogadro * float(density)))

    v_l = v_t = v_m = theta_D = None
    if density is not None:
        v_l, v_t, v_m = sound_velocities(B_H, G_H, density)
        if resolved_vol_atom is not None:
            theta_D = debye_temperature(v_m, vol_atom=resolved_vol_atom)

    return ElasticProperties(
        C_ij=C,
        S_ij=S,
        B_V=float(B_V),
        B_R=float(B_R),
        B_H=float(B_H),
        G_V=float(G_V),
        G_R=float(G_R),
        G_H=float(G_H),
        E=float(E),
        nu=float(nu),
        pugh_ratio=float(pugh_ratio),
        anisotropy_index=float(anisotropy_index),
        longitudinal_modulus=float(longitudinal_modulus),
        gamma=float(gamma),
        eigenvalues=eigvals,
        density=float(density) if density is not None else None,
        v_l=v_l,
        v_t=v_t,
        v_m=v_m,
        theta_D=theta_D,
        vol_atom=float(resolved_vol_atom) if resolved_vol_atom is not None else None,
    )


def isotropic_tensor_from_moduli(B_gpa: float, G_gpa: float) -> np.ndarray:
    """Construct an exact isotropic 6x6 Voigt stiffness matrix (GPa) from B and G."""
    if B_gpa <= 0.0 or G_gpa <= 0.0:
        raise ValidationError("B_gpa and G_gpa must be positive.")
    lam = float(B_gpa) - (2.0 / 3.0) * float(G_gpa)
    mu = float(G_gpa)
    c11 = lam + 2.0 * mu
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], c11)
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C
