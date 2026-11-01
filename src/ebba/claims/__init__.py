"""Claim registry public API.

Implementations are imported lazily by ``run`` so that formula/table audits do
not require torch.  Notebooks should only use :func:`run`, :func:`run_many`
and :func:`summary_table`.
"""

from . import tables  # noqa: F401  (pure data, safe to import everywhere)
from .model import (
    ClaimResult,
    Verdict,
    claim,
    load_registry,
    run,
    run_many,
    summary_table,
    to_jsonable,
)

__all__ = [
    "ClaimResult",
    "Verdict",
    "claim",
    "load_registry",
    "run",
    "run_many",
    "summary_table",
    "to_jsonable",
]
