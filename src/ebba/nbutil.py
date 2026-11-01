"""Small display helpers shared by the notebooks.

Everything here is optional for the verification logic: the claim functions
write JSON independently.  These helpers only format results so a notebook
reader can see verdict, measured values and tolerance in one place.
"""

from __future__ import annotations

import html
import json
from typing import Iterable

import numpy as np

from . import config
from .claims import run
from .claims.model import ClaimResult, Verdict

VERDICT_COLOR = {
    "CONFIRMED": "#1b7f3a",
    "PARTIAL": "#9a6700",
    "CONTRADICTED": "#b42318",
    "NOT_REPRODUCIBLE": "#5f6b7a",
    "ERROR": "#7a1fa2",
}

VERDICT_LABEL_ZH = {
    "CONFIRMED": "已确认",
    "PARTIAL": "部分成立 / 代理支持",
    "CONTRADICTED": "被否定",
    "NOT_REPRODUCIBLE": "不可复现",
    "ERROR": "运行错误",
}


def _display(obj) -> None:
    try:
        from IPython.display import display  # noqa: WPS433

        display(obj)
    except Exception:
        print(obj)


def _html(markup: str) -> None:
    try:
        from IPython.display import HTML  # noqa: WPS433

        _display(HTML(markup))
    except Exception:
        print(markup)


def badge(verdict: str) -> str:
    color = VERDICT_COLOR.get(verdict, "#5f6b7a")
    label = VERDICT_LABEL_ZH.get(verdict, verdict)
    return (
        f'<span style="background:{color};color:#fff;border-radius:999px;'
        f'padding:2px 10px;font-weight:600;font-size:12px">{verdict} · {label}</span>'
    )


def show(
    claim_id: str,
    result: ClaimResult | None = None,
    *,
    max_measured_chars: int | None = 4000,
    run_claim: bool = True,
) -> ClaimResult:
    """Run (or accept) a claim and render a compact result card.

    Parameters
    ----------
    claim_id : str
    result : ClaimResult | None
        Pass an already-computed result to avoid rerunning.
    run_claim : bool
        When False and ``result`` is None, only read the recorded JSON.
    """
    if result is None:
        if run_claim:
            result = run(claim_id)
        else:
            path = config.REPORTS_DIR / "claims" / f"{claim_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = ClaimResult(
                verdict=Verdict(payload["verdict"]),
                measured=payload.get("measured", {}),
                notes_zh=payload.get("notes_zh", ""),
                tolerance=payload.get("tolerance"),
            )

    from .claims.model import to_jsonable

    measured = json.dumps(to_jsonable(result.measured), indent=2, ensure_ascii=False)
    tolerance = json.dumps(to_jsonable(result.tolerance or {}), ensure_ascii=False)
    if max_measured_chars and len(measured) > max_measured_chars:
        measured = measured[:max_measured_chars] + "\n... (完整值见 reports/claims/%s.json)" % claim_id

    _html(
        f"""
<div style="border:1px solid #d8dee4;border-left:5px solid {VERDICT_COLOR.get(result.verdict.value, '#5f6b7a')};
            border-radius:10px;padding:12px 16px;margin:10px 0;background:#fff">
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <b style="font-family:monospace">{html.escape(claim_id)}</b>
    {badge(result.verdict.value)}
    <span style="color:#57606a;font-size:12px">tolerance: {html.escape(tolerance)}</span>
  </div>
  <p style="margin:8px 0">{"<br>".join(html.escape(line) for line in result.notes_zh.splitlines())}</p>
  <details><summary style="cursor:pointer;color:#0969da;font-size:13px">实测值</summary>
  <pre style="background:#f6f8fa;padding:10px;border-radius:8px;overflow-x:auto;font-size:12px">{html.escape(measured)}</pre>
  </details>
</div>"""
    )
    return result


def show_many(claim_ids: Iterable[str], *, max_measured_chars: int | None = 1200) -> dict[str, ClaimResult]:
    out = {}
    for cid in claim_ids:
        out[cid] = show(cid, max_measured_chars=max_measured_chars)
    return out


def verdict_table(claim_ids: Iterable[str] | None = None) -> list[dict]:
    """Read recorded JSON and return rows sorted by claim id."""
    from .claims.model import load_registry

    registry = load_registry()
    ids = list(claim_ids) if claim_ids is not None else list(registry.keys())
    rows = []
    for cid in ids:
        path = config.REPORTS_DIR / "claims" / f"{cid}.json"
        if not path.exists():
            rows.append({"ID": cid, "Tier": registry[cid]["tier"], "Verdict": "PENDING",
                         "Claim": registry[cid]["title_zh"]})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "ID": cid,
            "Tier": payload.get("tier"),
            "Verdict": payload.get("verdict"),
            "Claim": payload.get("title_zh"),
        })
    return rows


def show_table(claim_ids: Iterable[str] | None = None) -> None:
    rows = verdict_table(claim_ids)
    head = "<tr><th>ID</th><th>Tier</th><th>Verdict</th><th>声明</th></tr>"
    body = "".join(
        f"<tr><td style='font-family:monospace'>{html.escape(str(r['ID']))}</td>"
        f"<td>{html.escape(str(r['Tier']))}</td>"
        f"<td>{badge(str(r['Verdict']))}</td>"
        f"<td>{html.escape(str(r['Claim']))}</td></tr>"
        for r in rows
    )
    _html(f"<table style='border-collapse:collapse'>{head}{body}</table>")


def env_fingerprint() -> dict:
    """Versions and device, printed at the top of the notebooks."""
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "root": str(config.PROJECT_ROOT),
        "artifacts": str(config.ARTIFACTS_DIR),
        "reports": str(config.REPORTS_DIR),
        "fast": config.FAST,
        "device_env": __import__("os").environ.get("EBBA_DEVICE", "auto"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:  # torch is optional for exact-tier notebooks
        info["torch"] = f"not available ({type(exc).__name__})"
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def binom_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval, used to avoid over-reading small proxy samples."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
