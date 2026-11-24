"""Generic delimiter/CSV transport file parser with explicit ``--cols`` mapping
(stdlib + NumPy only).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np


_CANONICAL_ALIASES = {
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "temperature_k": "T",
    "mu": "mu",
    "ef": "mu",
    "e_f": "mu",
    "fermi": "mu",
    "n": "n",
    "carrier": "n",
    "carrier_density": "n",
    "dos": "DOS",
    "s": "S",
    "seebeck": "S",
    "sigma_tau": "sigma_tau",
    "sigma/tau": "sigma_tau",
    "sigma_over_tau": "sigma_tau",
    "sigma0": "sigma_tau",
    "sigma": "sigma_tau",
    "kappa_e_tau": "kappa_e_tau",
    "kappa_e/tau": "kappa_e_tau",
    "kappa/tau": "kappa_e_tau",
    "kappae_tau": "kappa_e_tau",
    "kappa0": "kappa_e_tau",
    "kappa_e": "kappa_e_tau",
    "rh": "RH",
}


def _canonicalize(name: str) -> str:
    clean = name.strip()
    return _CANONICAL_ALIASES.get(clean.lower(), clean)


def _parse_cols_spec(
    cols: Optional[Union[str, Mapping[str, int], list[str], tuple[str, ...]]]
) -> tuple[Optional[dict[str, int]], Optional[list[str]]]:
    """Parse ``cols`` specification into either a name->index dict or ordered column name list."""
    if cols is None:
        return None, None
    if isinstance(cols, Mapping):
        return {_canonicalize(k): int(v) for k, v in cols.items()}, None
    if isinstance(cols, (list, tuple)):
        return None, [_canonicalize(str(c)) for c in cols]

    raw = str(cols).strip()
    if not raw:
        return None, None
    if raw.startswith("{"):
        parsed = json.loads(raw)
        return {_canonicalize(k): int(v) for k, v in parsed.items()}, None

    items = [item.strip() for item in raw.split(",") if item.strip()]
    if any(("=" in it or ":" in it) for it in items):
        mapping: dict[str, int] = {}
        for it in items:
            sep = "=" if "=" in it else ":"
            k, v = it.split(sep, 1)
            mapping[_canonicalize(k)] = int(v.strip())
        return mapping, None

    return None, [_canonicalize(it) for it in items]


def parse_generic_transport(
    path: Union[str, Path],
    cols: Optional[Union[str, Mapping[str, int], list[str], tuple[str, ...]]] = None,
    delimiter: Optional[str] = None,
    n_unit: str = "m^-3",
    vol_uc_m3: Optional[float] = None,
) -> dict[str, Any]:
    """Parse an arbitrary whitespace-separated or CSV transport file.

    Supports flexible ``cols`` mapping via:
      - Ordered comma-separated names, e.g. ``"T,n,S,sigma_tau,kappa_e_tau"``
      - Explicit index mapping, e.g. ``"T=0,n=1,S=2,sigma_tau=3,kappa_e_tau=4"``
        or ``"T:0,S:2,..."``
      - Automatic header detection when ``cols=None``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Generic transport file not found: {file_path}")

    raw_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if delimiter is None:
        sample = "\n".join(raw_lines[:10])
        delimiter = "," if "," in sample else None

    header_tokens: Optional[list[str]] = None
    rows: list[list[float]] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            toks = next(csv.reader([text], delimiter=delimiter)) if delimiter else text.split()
            if toks and any(not _is_float(t) for t in toks):
                header_tokens = [_canonicalize(t) for t in toks]
            continue

        toks = next(csv.reader([stripped], delimiter=delimiter)) if delimiter else stripped.split()
        toks = [t.strip() for t in toks if t.strip() != ""]
        if not toks:
            continue
        if all(_is_float(t) for t in toks):
            rows.append([float(t) for t in toks])
        elif not rows and header_tokens is None:
            header_tokens = [_canonicalize(t) for t in toks]

    if not rows:
        raise ValueError(f"No numeric transport data found in '{file_path}'.")

    n_cols = min(len(r) for r in rows)
    matrix = np.asarray([r[:n_cols] for r in rows], dtype=np.float64)

    explicit_map, ordered_names = _parse_cols_spec(cols)
    if explicit_map is not None:
        col_map = explicit_map
    elif ordered_names is not None:
        col_map = {name: idx for idx, name in enumerate(ordered_names)}
    elif header_tokens is not None:
        col_map = {name: idx for idx, name in enumerate(header_tokens)}
    else:
        raise ValueError(
            "Generic parser requires '--cols' mapping or a descriptive header line."
        )

    for required in ("T", "n", "S", "sigma_tau", "kappa_e_tau"):
        if required not in col_map:
            raise ValueError(
                f"Generic transport mapping missing required column '{required}'. "
                f"Mapped columns: {list(col_map.keys())}"
            )
        if col_map[required] < 0 or col_map[required] >= n_cols:
            raise ValueError(
                f"Column index {col_map[required]} for '{required}' out of range (0..{n_cols - 1})."
            )

    result: dict[str, Any] = {}
    for name, idx in col_map.items():
        if 0 <= idx < n_cols:
            result[name] = matrix[:, idx].copy()

    if "mu" not in result:
        result["mu"] = np.zeros(matrix.shape[0], dtype=np.float64)
    if "DOS" not in result:
        result["DOS"] = np.zeros(matrix.shape[0], dtype=np.float64)

    unit = n_unit.strip().lower()
    if unit == "cm^-3":
        result["n"] = result["n"] * 1e6
        result["n_unit"] = "m^-3"
    elif unit in {"e/uc", "e_per_uc"} and vol_uc_m3 is not None and vol_uc_m3 > 0.0:
        result["n"] = result["n"] / float(vol_uc_m3)
        result["n_unit"] = "m^-3"
    else:
        result["n_unit"] = n_unit

    return result


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
