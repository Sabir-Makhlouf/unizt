"""Extensible central parser registry for DFT / Boltzmann transport codes."""

from __future__ import annotations

from typing import Any, Callable

from unizt.parsers.boltztrap2 import parse_boltztrap2_trace
from unizt.parsers.generic import parse_generic_transport

ParserFunc = Callable[..., dict[str, Any]]

PARSER_REGISTRY: dict[str, ParserFunc] = {
    "boltztrap2": parse_boltztrap2_trace,
    "boltztrap": parse_boltztrap2_trace,
    "generic": parse_generic_transport,
}


def register_parser(name: str, func: ParserFunc, overwrite: bool = True) -> None:
    """Register a custom transport file parser under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Parser name must be a non-empty string.")
    if not callable(func):
        raise TypeError("Registered parser must be callable.")
    if not overwrite and key in PARSER_REGISTRY:
        raise KeyError(f"Parser '{key}' is already registered.")
    PARSER_REGISTRY[key] = func


def get_parser(name: str) -> ParserFunc:
    """Retrieve a registered transport parser by name."""
    key = name.strip().lower()
    if key not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY))
        raise KeyError(f"Unknown transport parser '{name}'. Available: {available}")
    return PARSER_REGISTRY[key]


def list_parsers() -> list[str]:
    """Return sorted list of registered parser names."""
    return sorted(PARSER_REGISTRY)
