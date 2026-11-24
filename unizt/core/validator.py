"""Validation utilities for elastic stiffness tensors and electronic transport data."""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

import numpy as np


class ValidationError(ValueError):
    """Base validation error for UniZT inputs."""


class BornStabilityError(ValidationError):
    """Raised when an elastic stiffness tensor violates Born mechanical stability."""


def validate_elastic_tensor(
    C_ij: Any,
    sym_tol: float = 1e-3,
    sym_rtol: float = 1e-4,
    eig_tol: float = 1e-6,
) -> np.ndarray:
    """Validate a 6x6 Voigt elastic stiffness tensor (in GPa).

    Checks:
      1. Shape is strictly (6, 6) and all entries are finite.
      2. Matrix is symmetric within tolerance ``atol=sym_tol, rtol=sym_rtol``.
      3. Born mechanical stability (Mouhat & Coudert, PRB 2014): all eigenvalues
         of the symmetrized 6x6 stiffness matrix are strictly positive
         (``eigval > eig_tol``).

    Parameters
    ----------
    C_ij : array-like
        6x6 stiffness matrix in GPa.
    sym_tol : float, optional
        Absolute symmetry tolerance in GPa (default 1e-3).
    sym_rtol : float, optional
        Relative symmetry tolerance (default 1e-4).
    eig_tol : float, optional
        Minimum strictly positive eigenvalue threshold in GPa (default 1e-6).

    Returns
    -------
    np.ndarray
        Validated, symmetrized (6, 6) float64 stiffness matrix.
    """
    C = np.asarray(C_ij, dtype=np.float64)
    if C.ndim != 2 or C.shape != (6, 6):
        raise ValidationError(
            f"Elastic tensor must have shape (6, 6), got {C.shape}."
        )
    if not np.all(np.isfinite(C)):
        raise ValidationError("Elastic tensor contains NaN or Inf values.")

    if not np.allclose(C, C.T, rtol=sym_rtol, atol=sym_tol):
        max_asym = float(np.max(np.abs(C - C.T)))
        raise ValidationError(
            f"Elastic tensor is not symmetric (max |C - C^T| = {max_asym:.4e} GPa)."
        )

    C_sym = 0.5 * (C + C.T)
    eigvals = np.linalg.eigvalsh(C_sym)
    min_eig = float(np.min(eigvals))
    if min_eig <= eig_tol:
        raise BornStabilityError(
            f"Elastic tensor fails Born mechanical stability: "
            f"minimum eigenvalue = {min_eig:.4e} GPa <= {eig_tol:.1e} GPa."
        )
    return C_sym


def validate_transport_data(
    data: Mapping[str, Any],
    require_positive_T: bool = True,
    n_unit: Optional[str] = None,
    vol_uc_m3: Optional[float] = None,
    strict_units: bool = False,
) -> dict[str, Any]:
    """Validate and clean electronic transport data arrays.

    Actions performed:
      - Removes rows containing NaN or Inf across all array columns (emitting a warning).
      - Ensures strictly positive temperatures ``T > 0`` K.
      - Validates or converts carrier concentration ``n`` units (supports SI
        ``m^-3``, ``cm^-3``, and ``e/uc`` when ``vol_uc_m3`` is supplied).

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary of transport arrays (e.g. ``T``, ``mu``, ``n``, ``S``,
        ``sigma_tau``, ``kappa_e_tau``).
    require_positive_T : bool, optional
        If True (default), raise ValidationError on ``T <= 0``. If False, filter out
        non-positive temperatures with a warning.
    n_unit : str, optional
        Explicit carrier concentration unit override (``"m^-3"``, ``"cm^-3"``,
        or ``"e/uc"``). Falls back to ``data.get("n_unit", "m^-3")``.
    vol_uc_m3 : float, optional
        Unit-cell volume in m^3 required to convert ``"e/uc"`` to ``"m^-3"``.
    strict_units : bool, optional
        If True, raise ValidationError instead of warning when carrier
        concentration magnitudes appear inconsistent with ``m^-3``.

    Returns
    -------
    dict[str, Any]
        Cleaned copy of transport dictionary with ``n`` in SI (``m^-3``).
    """
    if not isinstance(data, Mapping):
        raise ValidationError("Transport data must be a mapping/dictionary.")

    cleaned: dict[str, Any] = {}
    array_keys: list[str] = []
    lengths: set[int] = set()

    for key, val in data.items():
        if isinstance(val, (np.ndarray, list, tuple)):
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim != 1:
                raise ValidationError(
                    f"Transport field '{key}' must be 1D, got shape {arr.shape}."
                )
            array_keys.append(key)
            lengths.add(len(arr))
            cleaned[key] = arr
        else:
            cleaned[key] = val

    if not array_keys:
        raise ValidationError("Transport data contains no array columns.")
    if len(lengths) != 1:
        raise ValidationError(
            f"All transport array columns must have equal length; got lengths {sorted(lengths)}."
        )

    n_rows = next(iter(lengths))
    if n_rows == 0:
        raise ValidationError("Transport data contains zero rows.")

    if "T" not in cleaned:
        raise ValidationError("Transport data must contain temperature column 'T'.")

    finite_mask = np.ones(n_rows, dtype=bool)
    for key in array_keys:
        finite_mask &= np.isfinite(cleaned[key])

    n_nonfinite = int(n_rows - np.sum(finite_mask))
    if n_nonfinite > 0:
        warnings.warn(
            f"Removed {n_nonfinite} transport row(s) containing NaN or Inf values.",
            UserWarning,
            stacklevel=2,
        )
        for key in array_keys:
            cleaned[key] = cleaned[key][finite_mask]

    if len(cleaned["T"]) == 0:
        raise ValidationError("No valid finite transport rows remain after filtering.")

    T_arr = cleaned["T"]
    pos_mask = T_arr > 0.0
    if not np.all(pos_mask):
        bad_vals = T_arr[~pos_mask]
        if require_positive_T:
            raise ValidationError(
                f"Transport temperatures must be strictly positive (K); "
                f"found non-positive values (min = {np.min(bad_vals):.4g} K)."
            )
        warnings.warn(
            f"Removed {int(np.sum(~pos_mask))} row(s) with non-positive temperature.",
            UserWarning,
            stacklevel=2,
        )
        for key in array_keys:
            cleaned[key] = cleaned[key][pos_mask]
        if len(cleaned["T"]) == 0:
            raise ValidationError("No valid positive-temperature rows remain.")

    unit = (n_unit or cleaned.get("n_unit") or "m^-3").strip().lower()
    if "n" in cleaned:
        n_arr = cleaned["n"].copy()
        if unit in {"cm^-3", "cm-3", "1/cm3"}:
            n_arr = n_arr * 1e6
            cleaned["n_unit"] = "m^-3"
        elif unit in {"e/uc", "e_per_uc", "uc^-1"}:
            if vol_uc_m3 is None or vol_uc_m3 <= 0.0:
                raise ValidationError(
                    "vol_uc_m3 (> 0) must be provided to convert carrier concentration from 'e/uc' to 'm^-3'."
                )
            n_arr = n_arr / float(vol_uc_m3)
            cleaned["n_unit"] = "m^-3"
        elif unit in {"m^-3", "m-3", "1/m3"}:
            cleaned["n_unit"] = "m^-3"
        else:
            raise ValidationError(f"Unsupported carrier concentration unit: '{unit}'.")

        nonzero = np.abs(n_arr[n_arr != 0.0])
        if nonzero.size > 0:
            max_n = float(np.max(nonzero))
            if max_n < 1e6:
                msg = (
                    f"Carrier concentration max |n|={max_n:.3e} looks like 'e/uc' "
                    "rather than SI m^-3. Provide vol_uc_m3 and n_unit='e/uc' if needed."
                )
                if strict_units:
                    raise ValidationError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
            elif 1e12 <= max_n <= 1e22:
                msg = (
                    f"Carrier concentration max |n|={max_n:.3e} falls in typical cm^-3 "
                    "range (1e12-1e22) rather than SI m^-3 (1e18-1e28). "
                    "Specify n_unit='cm^-3' if values are per cm^3."
                )
                if strict_units:
                    raise ValidationError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)

        cleaned["n"] = n_arr

    return cleaned
