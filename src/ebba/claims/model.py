"""Claim registry, result types and JSON recording.

A claim is one falsifiable statement extracted from the paper.  Each claim
function returns a ``ClaimResult``; the registry supplies the metadata
(title, paper location, notebook, tolerance).  Results are written to
``reports/claims/<id>.json`` and later aggregated into the HTML report.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import config

def _module_for(claim_id: str) -> str:
    """Route each claim to its implementation module (lazy import).

    Keeping the import lazy lets the exact-only notebooks run without torch,
    and avoids loading the empirical stack when only table arithmetic is run.
    """
    if claim_id in {"A01", "A02", "A03", "A04", "A05", "A06", "A17"}:
        return "formulas"
    if claim_id in {"A07", "A08", "A09", "A10"}:
        return "theory"
    if claim_id.startswith("B"):
        return "empirical"
    if claim_id.startswith("C"):
        return "limits"
    return "audit"


def _ensure_loaded(claim_id: str) -> None:
    if claim_id in _FUNCS:
        return
    module = _module_for(claim_id)
    importlib.import_module(f"{__package__}.{module}")

try:
    import yaml
except ImportError:  # pragma: no cover - only for minimal local tooling
    yaml = None


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    PARTIAL = "PARTIAL"
    CONTRADICTED = "CONTRADICTED"
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
    ERROR = "ERROR"


@dataclass
class ClaimResult:
    verdict: Verdict
    measured: dict[str, Any] = field(default_factory=dict)
    notes_zh: str = ""
    tolerance: dict[str, Any] | None = None


_FUNCS: dict[str, Callable[[], ClaimResult]] = {}


def claim(claim_id: str):
    def decorator(fn: Callable[[], ClaimResult]):
        _FUNCS[claim_id] = fn
        return fn

    return decorator


def registry_path() -> Path:
    return Path(__file__).with_name("registry.yaml")


def load_registry() -> dict[str, dict]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the claim registry")
    payload = yaml.safe_load(registry_path().read_text(encoding="utf-8"))
    return payload["claims"]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Verdict):
        return value.value
    return value


def run(claim_id: str, write: bool = True) -> ClaimResult:
    spec = load_registry().get(claim_id)
    if spec is None:
        raise KeyError(f"claim {claim_id} missing from registry.yaml")
    _ensure_loaded(claim_id)
    fn = _FUNCS.get(claim_id)
    if fn is None:
        raise KeyError(f"claim {claim_id} has no implementation")
    try:
        result = fn()
    except Exception as exc:  # surface as a failed run, never silently pass
        result = ClaimResult(
            verdict=Verdict.ERROR,
            measured={"exception": f"{type(exc).__name__}: {exc}"},
            notes_zh="执行异常，见 measured.exception；这是运行错误，不是论文结论。",
        )
    if write:
        record(claim_id, spec, result)
    line = f"{claim_id:>4}  {result.verdict.value:<17} {spec['title_zh']}"
    print(line)
    return result


def run_many(claim_ids: list[str], write: bool = True) -> dict[str, ClaimResult]:
    results = {}
    for cid in claim_ids:
        results[cid] = run(cid, write=write)
    return results


def record(claim_id: str, spec: dict, result: ClaimResult) -> Path:
    out_dir = Path(config.REPORTS_DIR) / "claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": claim_id,
        "tier": spec.get("tier"),
        "scope": spec.get("scope"),
        "title_zh": spec.get("title_zh"),
        "paper_ref": spec.get("paper_ref"),
        "notebook": spec.get("notebook"),
        "verdict": result.verdict.value,
        "measured": to_jsonable(result.measured),
        "notes_zh": result.notes_zh,
        "tolerance": to_jsonable(result.tolerance if result.tolerance is not None else spec.get("tolerance", {})),
    }
    path = out_dir / f"{claim_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def summary_table(claim_ids: list[str] | None = None) -> list[dict]:
    """Read recorded results for display in notebooks / reports."""
    out_dir = Path(config.REPORTS_DIR) / "claims"
    registry = load_registry()
    ids = claim_ids or list(registry.keys())
    rows = []
    for cid in ids:
        path = out_dir / f"{cid}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "ID": payload["id"],
            "Tier": payload["tier"],
            "Verdict": payload["verdict"],
            "Claim": payload["title_zh"],
            "Paper": payload["paper_ref"],
        })
    return rows
