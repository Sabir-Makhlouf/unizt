"""Parsers for 6x6 elastic stiffness tensors C_ij (GPa) from VASP OUTCAR,
WIEN2k elast/IRelast, Quantum ESPRESSO thermo_pw, and JSON files (stdlib + NumPy only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from unizt.core.validator import validate_elastic_tensor


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")


def _extract_floats(line: str) -> list[float]:
    return [float(m.replace("D", "e").replace("d", "e")) for m in _FLOAT_RE.findall(line)]


def _parse_json_elastic(text: str) -> np.ndarray:
    payload = json.loads(text)
    scale = 1.0
    if isinstance(payload, dict):
        unit = str(payload.get("unit", payload.get("units", "gpa"))).strip().lower()
        if "kbar" in unit:
            scale = 0.1
        elif unit in {"pa"}:
            scale = 1e-9
        elif "mbar" in unit:
            scale = 100.0

        raw = None
        for key in ("C_ij", "c_ij", "elastic_tensor", "stiffness_tensor", "tensor", "C"):
            if key in payload:
                raw = payload[key]
                break
        if raw is None:
            raise ValueError(
                "JSON elastic file must contain 'C_ij', 'elastic_tensor', 'stiffness_tensor', 'tensor', or 'C'."
            )
    else:
        raw = payload

    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 36 and arr.shape != (6, 6):
        arr = arr.reshape((6, 6))
    return arr * scale


def _parse_vasp_outcar(text: str) -> np.ndarray:
    """Parse VASP OUTCAR ``TOTAL ELASTIC MODULI (kBar)`` (reordering XY,YZ,ZX -> Voigt YZ,ZX,XY)."""
    lines = text.splitlines()
    target_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if "TOTAL ELASTIC MODULI" in line.upper():
            target_idx = idx
    if target_idx is None:
        raise ValueError("Could not find 'TOTAL ELASTIC MODULI' block in VASP OUTCAR.")

    vasp_rows: list[list[float]] = []
    for line in lines[target_idx + 1 : target_idx + 20]:
        s = line.strip()
        if not s or s.startswith("---") or s.lower().startswith("direction"):
            continue
        if re.match(r"^(XX|YY|ZZ|XY|YZ|ZX)\b", s, flags=re.IGNORECASE):
            vals = _extract_floats(s)
            if len(vals) >= 6:
                vasp_rows.append(vals[-6:])
        if len(vasp_rows) == 6:
            break

    if len(vasp_rows) != 6:
        raise ValueError("Incomplete 6x6 elastic moduli matrix in VASP OUTCAR.")

    # VASP order: (XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5) -> Standard Voigt: (XX, YY, ZZ, YZ, ZX, XY)
    vasp_mat = np.asarray(vasp_rows, dtype=np.float64) * 0.1  # kBar -> GPa
    perm = [0, 1, 2, 4, 5, 3]
    voigt = vasp_mat[np.ix_(perm, perm)]
    return voigt


def _parse_wien2k_elastic(text: str) -> np.ndarray:
    """Parse WIEN2k elast / IRelast outputs (either key=val constants or 6x6 matrix in GPa)."""
    # First check for named constants like C11 = ..., c12: ...
    kv_matches = re.findall(
        r"\b[cC]_?([1-6]{2})\s*[:=]\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)",
        text,
    )
    if kv_matches:
        c = {k: float(v.replace("D", "e").replace("d", "e")) for k, v in kv_matches}
        scale = 0.1 if ("kbar" in text.lower() and "gpa" not in text.lower()) else 1.0
        for k in c:
            c[k] *= scale

        C = np.zeros((6, 6), dtype=np.float64)
        c11 = c.get("11")
        if c11 is None:
            raise ValueError("WIEN2k elastic text missing C11.")
        c12 = c.get("12", 0.0)
        c44 = c.get("44", 0.0)

        # Infer symmetry defaults when only independent constants are reported
        c22 = c.get("22", c11)
        c33 = c.get("33", c11)
        c13 = c.get("13", c12)
        c23 = c.get("23", c13)
        c55 = c.get("55", c44)
        if "66" in c:
            c66 = c["66"]
        elif ("33" in c or "13" in c) and "22" not in c:
            # Hexagonal / trigonal symmetry relation
            c66 = 0.5 * (c11 - c12)
        else:
            c66 = c44

        C[0, 0], C[1, 1], C[2, 2] = c11, c22, c33
        C[0, 1] = C[1, 0] = c12
        C[0, 2] = C[2, 0] = c13
        C[1, 2] = C[2, 1] = c23
        C[3, 3], C[4, 4], C[5, 5] = c44, c55, c66

        # Fill any additional monoclinic/triclinic off-diagonal components if present
        for ij, val in c.items():
            i, j = int(ij[0]) - 1, int(ij[1]) - 1
            if i != j and (i >= 3 or j >= 3):
                C[i, j] = C[j, i] = val
        return C

    # Fallback: search for 6 consecutive rows with >= 6 floats
    rows: list[list[float]] = []
    for line in text.splitlines():
        vals = _extract_floats(line)
        if len(vals) == 6:
            rows.append(vals)
            if len(rows) == 6:
                scale = 0.1 if ("kbar" in text.lower() and "gpa" not in text.lower()) else 1.0
                return np.asarray(rows, dtype=np.float64) * scale
        else:
            rows.clear()

    raise ValueError("Could not parse elastic tensor from WIEN2k elast/IRelast file.")


def _parse_qe_thermo_pw(text: str) -> np.ndarray:
    """Parse Quantum ESPRESSO ``thermo_pw`` elastic constants output."""
    lines = text.splitlines()
    start_idx = 0
    unit_scale = 1.0
    found_header = False

    for idx, line in enumerate(lines):
        low = line.lower()
        if "elastic constant" in low or "stiffness" in low:
            found_header = True
            start_idx = idx
            if "kbar" in low:
                unit_scale = 0.1
            elif "gpa" in low:
                unit_scale = 1.0

    search_lines = lines[start_idx:] if found_header else lines
    rows: list[list[float]] = []
    for line in search_lines:
        low = line.lower()
        if "kbar" in low and not rows:
            unit_scale = 0.1
        elif "gpa" in low and not rows:
            unit_scale = 1.0

        vals = _extract_floats(line)
        # thermo_pw sometimes prints row index i followed by 6 values (7 numbers) or 6 values
        if len(vals) == 6:
            rows.append(vals)
        elif len(vals) == 7 and abs(vals[0] - (len(rows) + 1)) < 1e-6:
            rows.append(vals[1:])
        elif rows and len(rows) < 6:
            rows.clear()

        if len(rows) == 6:
            return np.asarray(rows, dtype=np.float64) * unit_scale

    # Fallback to named constants C(i,j) = val or Cij = val
    kv_matches = re.findall(
        r"[cC]\(?([1-6])\s*,?\s*([1-6])\)?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)",
        text,
    )
    if kv_matches:
        C = np.zeros((6, 6), dtype=np.float64)
        for i_s, j_s, v_s in kv_matches:
            i, j = int(i_s) - 1, int(j_s) - 1
            val = float(v_s.replace("D", "e").replace("d", "e")) * unit_scale
            C[i, j] = C[j, i] = val
        return C

    raise ValueError("Could not parse 6x6 elastic tensor from Quantum ESPRESSO thermo_pw output.")


def parse_elastic_tensor(
    path: Union[str, Path],
    fmt: str = "auto",
    sym_tol: float = 0.5,
) -> np.ndarray:
    """Parse and validate a 6x6 elastic stiffness tensor (in GPa).

    Supported formats: ``'json'``, ``'vasp'``, ``'wien2k'``, ``'qe'``, or ``'auto'``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Elastic tensor file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")
    fmt_clean = fmt.strip().lower()

    if fmt_clean == "auto":
        if file_path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
            fmt_clean = "json"
        elif "TOTAL ELASTIC MODULI" in text.upper():
            fmt_clean = "vasp"
        elif "thermo_pw" in text.lower() or "kbar" in text.lower():
            fmt_clean = "qe"
        else:
            fmt_clean = "wien2k"

    if fmt_clean == "json":
        C_raw = _parse_json_elastic(text)
    elif fmt_clean == "vasp":
        C_raw = _parse_vasp_outcar(text)
    elif fmt_clean == "wien2k":
        C_raw = _parse_wien2k_elastic(text)
    elif fmt_clean == "qe":
        C_raw = _parse_qe_thermo_pw(text)
    else:
        raise ValueError(
            f"Unsupported elastic format '{fmt}'; choose from 'auto', 'json', 'vasp', 'wien2k', 'qe'."
        )

    return validate_elastic_tensor(C_raw, sym_tol=sym_tol, sym_rtol=5e-3)
