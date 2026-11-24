"""Semantic parser for BoltzTraP2 and BoltzTraP1 ``*.trace`` transport files
(stdlib + NumPy only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


_DEFAULT_BT_COLUMNS = {
    "mu": 0,
    "T": 1,
    "n": 2,
    "DOS": 3,
    "S": 4,
    "sigma_tau": 5,
    "RH": 6,
    "kappa_e_tau": 7,
}


def _identify_header_columns(header_line: str) -> tuple[dict[str, int], Optional[str]]:
    """Map semantic BoltzTraP2/BoltzTraP1 header tokens to column indices."""
    clean = header_line.lstrip("#").strip()
    if not clean:
        return {}, None

    # Split on whitespace while keeping tokens like 'DOS(Ef)' or 'sigma/tau0[1/(Ohm*m*s)]'
    tokens = clean.split()
    mapping: dict[str, int] = {}
    detected_n_unit: Optional[str] = None

    for idx, raw_tok in enumerate(tokens):
        tok = raw_tok.lower()
        base = re.sub(r"[\[\(].*$", "", tok)

        if base in {"ef", "mu", "e_f", "fermi"} or tok.startswith("ef[") or tok.startswith("mu["):
            mapping.setdefault("mu", idx)
        elif base in {"t", "temp", "temperature"} or tok.startswith("t["):
            mapping.setdefault("T", idx)
        elif base == "n" or tok.startswith("n[") or base in {"carrier", "nelec"}:
            mapping.setdefault("n", idx)
            if "cm^-3" in tok or "cm-3" in tok:
                detected_n_unit = "cm^-3"
            elif "m^-3" in tok or "m-3" in tok:
                detected_n_unit = "m^-3"
            elif "e/uc" in tok or "uc" in tok:
                detected_n_unit = "e/uc"
        elif base.startswith("dos"):
            mapping.setdefault("DOS", idx)
        elif base in {"s", "seebeck"} or tok.startswith("s["):
            mapping.setdefault("S", idx)
        elif "sigma" in tok or base in {"s/tau", "sigma_tau", "sigma/tau", "sigma/tau0"}:
            mapping.setdefault("sigma_tau", idx)
        elif base in {"rh", "r_h"} or tok.startswith("rh["):
            mapping.setdefault("RH", idx)
        elif "kappa" in tok:
            mapping.setdefault("kappa_e_tau", idx)

    return mapping, detected_n_unit


def parse_boltztrap2_trace(
    path: Union[str, Path],
    vol_uc_m3: Optional[float] = None,
    n_unit: Optional[str] = None,
) -> dict[str, Any]:
    """Parse a BoltzTraP2 or BoltzTraP1 ``*.trace`` file using stdlib + NumPy only.

    Extracts ``mu`` (Ef), ``T``, ``n`` (carrier density), ``DOS``, ``S``,
    ``sigma_tau`` (sigma/tau), and ``kappa_e_tau`` (kappa_e/tau), plus optional ``RH``.

    Parameters
    ----------
    path : str or Path
        Path to ``*.trace`` file.
    vol_uc_m3 : float, optional
        Unit-cell volume in m^3. When carrier density in the file is in ``e/uc``,
        supplying ``vol_uc_m3`` converts ``n`` directly to SI ``m^-3``.
    n_unit : str, optional
        Explicit override for the file's carrier concentration unit
        (``"e/uc"``, ``"cm^-3"``, or ``"m^-3"``).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"BoltzTraP2 trace file not found: {file_path}")

    header_map: dict[str, int] = {}
    header_n_unit: Optional[str] = None
    rows: list[list[float]] = []

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate_map, candidate_unit = _identify_header_columns(stripped)
                if len(candidate_map) >= 4:
                    header_map = candidate_map
                    if candidate_unit is not None:
                        header_n_unit = candidate_unit
                continue

            parts = stripped.split()
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                if not rows and not header_map:
                    candidate_map, candidate_unit = _identify_header_columns(stripped)
                    if len(candidate_map) >= 4:
                        header_map = candidate_map
                        if candidate_unit is not None:
                            header_n_unit = candidate_unit
                continue
            rows.append(nums)

    if not rows:
        raise ValueError(f"No numeric transport records found in trace file: {file_path}")

    n_cols = min(len(r) for r in rows)
    matrix = np.asarray([r[:n_cols] for r in rows], dtype=np.float64)

    if not header_map:
        if n_cols >= 8:
            col_map = dict(_DEFAULT_BT_COLUMNS)
        elif n_cols == 7:
            # Variant without Hall coefficient RH
            col_map = {
                "mu": 0,
                "T": 1,
                "n": 2,
                "DOS": 3,
                "S": 4,
                "sigma_tau": 5,
                "kappa_e_tau": 6,
            }
        elif n_cols == 6:
            col_map = {
                "mu": 0,
                "T": 1,
                "n": 2,
                "S": 3,
                "sigma_tau": 4,
                "kappa_e_tau": 5,
            }
        else:
            raise ValueError(
                f"Cannot infer BoltzTraP2 trace column layout with {n_cols} columns and no header."
            )
    else:
        col_map = header_map

    required = ("T", "n", "S", "sigma_tau", "kappa_e_tau")
    missing = [k for k in required if k not in col_map or col_map[k] >= n_cols]
    if missing:
        raise ValueError(
            f"Trace file '{file_path}' missing required semantic columns: {', '.join(missing)}."
        )

    result: dict[str, Any] = {}
    for key, idx in col_map.items():
        if idx < n_cols:
            result[key] = matrix[:, idx].copy()

    if "mu" not in result:
        result["mu"] = np.zeros(matrix.shape[0], dtype=np.float64)
    if "DOS" not in result:
        result["DOS"] = np.zeros(matrix.shape[0], dtype=np.float64)

    resolved_unit = n_unit or header_n_unit
    if resolved_unit is None:
        max_n = float(np.max(np.abs(result["n"]))) if result["n"].size else 0.0
        if max_n < 1e4:
            resolved_unit = "e/uc"
        elif 1e12 <= max_n <= 1e22:
            resolved_unit = "cm^-3"
        else:
            resolved_unit = "m^-3"

    if resolved_unit == "e/uc" and vol_uc_m3 is not None and vol_uc_m3 > 0.0:
        result["n"] = result["n"] / float(vol_uc_m3)
        result["n_unit"] = "m^-3"
    elif resolved_unit == "cm^-3":
        result["n"] = result["n"] * 1e6
        result["n_unit"] = "m^-3"
    else:
        result["n_unit"] = resolved_unit

    return result


def parse_spin_polarized_traces(
    up_path: Union[str, Path],
    dn_path: Union[str, Path],
    vol_uc_m3: Optional[float] = None,
    n_unit: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse spin-up and spin-down BoltzTraP2/1 ``*.trace`` files."""
    up_data = parse_boltztrap2_trace(up_path, vol_uc_m3=vol_uc_m3, n_unit=n_unit)
    dn_data = parse_boltztrap2_trace(dn_path, vol_uc_m3=vol_uc_m3, n_unit=n_unit)
    return up_data, dn_data

