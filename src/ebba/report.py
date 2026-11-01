"""Render recorded claim results into verdicts.json, Markdown and a standalone HTML page.

Inputs are only the JSON files written by ``claims.record`` plus the artifact
manifest.  The report is intentionally self-contained (no CDN, no external
fonts) so it can be opened offline from the mounted ``reports`` directory.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from collections import Counter
from pathlib import Path

from . import config
from .claims.model import load_registry

VERDICT_ORDER = ("CONFIRMED", "PARTIAL", "CONTRADICTED", "NOT_REPRODUCIBLE", "ERROR")
VERDICT_LABEL_ZH = {
    "CONFIRMED": "已确认",
    "PARTIAL": "部分成立",
    "CONTRADICTED": "被否定",
    "NOT_REPRODUCIBLE": "不可复现",
    "ERROR": "运行错误",
}
VERDICT_COLOR = {
    "CONFIRMED": "#1b7f3a",
    "PARTIAL": "#9a6700",
    "CONTRADICTED": "#b42318",
    "NOT_REPRODUCIBLE": "#5f6b7a",
    "ERROR": "#7a1fa2",
}


def collect(claims_dir: Path | None = None) -> list[dict]:
    claims_dir = Path(claims_dir or Path(config.REPORTS_DIR) / "claims")
    registry = load_registry()
    rows = []
    for cid, spec in registry.items():
        path = claims_dir / f"{cid}.json"
        if not path.exists():
            rows.append({
                "id": cid, "tier": spec.get("tier"), "scope": spec.get("scope"),
                "title_zh": spec.get("title_zh"), "paper_ref": spec.get("paper_ref"),
                "notebook": spec.get("notebook"), "verdict": "PENDING",
                "measured": {}, "notes_zh": "尚未运行", "tolerance": spec.get("tolerance", {}),
            })
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def counts(rows: list[dict]) -> dict[str, int]:
    c = Counter(r["verdict"] for r in rows)
    return {k: c.get(k, 0) for k in (*VERDICT_ORDER, "PENDING")}


def write_json(rows: list[dict], manifest: dict | None, out_path: Path) -> Path:
    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "counts": counts(rows),
        "manifest": manifest or {},
        "claims": {r["id"]: r for r in rows},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_markdown(rows: list[dict], out_path: Path, manifest: dict | None = None) -> Path:
    c = counts(rows)
    lines = [
        "# EBBA / EBBA+ 验证报告",
        "",
        "论文：Gao, Chen, Sun, Li, Li, Shao. *Energy-based Backdoor Defense without "
        "Task-Specific Samples and Model Retraining*. ICML 2024, PMLR 235.",
        "",
        "## 判定统计",
        "",
        "| 判定 | 数量 |",
        "|---|---|",
    ]
    for key in (*VERDICT_ORDER, "PENDING"):
        lines.append(f"| {key} | {c[key]} |")
    lines += ["", "## 逐条结果", "", "| ID | 层级 | 判定 | 声明 | 位置 |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["id"]):
        lines.append(
            f"| {r['id']} | {r['tier']} | {r['verdict']} | {r['title_zh']} | {r['paper_ref']} |"
        )
    lines += ["", "## 被否定的声明", ""]
    for r in sorted(rows, key=lambda x: x["id"]):
        if r["verdict"] == "CONTRADICTED":
            lines.append(f"### {r['id']} {r['title_zh']}\n\n{r['notes_zh']}\n")
    if manifest:
        lines += ["## 产物清单", "", "```json", json.dumps(manifest, indent=2, ensure_ascii=False), "```"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _badge(verdict: str) -> str:
    color = VERDICT_COLOR.get(verdict, "#5f6b7a")
    return (
        f'<span class="badge" style="background:{color}">'
        f'{verdict} · {VERDICT_LABEL_ZH.get(verdict, verdict)}</span>'
    )


def write_html(rows: list[dict], out_path: Path, manifest: dict | None = None) -> Path:
    c = counts(rows)
    total = len(rows)
    contradictions = [r for r in rows if r["verdict"] == "CONTRADICTED"]
    partials = [r for r in rows if r["verdict"] == "PARTIAL"]
    rows_sorted = sorted(rows, key=lambda x: x["id"])

    def esc(value) -> str:
        return html.escape(str(value))

    cards = []
    for r in rows_sorted:
        measured = json.dumps(r.get("measured", {}), indent=2, ensure_ascii=False)
        cards.append(
            f"""
<article class="claim v-{esc(r['verdict'])}">
  <header>
    <span class="cid">{esc(r['id'])}</span>
    {_badge(r['verdict'])}
    <span class="tier">Tier {esc(r['tier'])} · {esc(r['scope'])}</span>
  </header>
  <h3>{esc(r['title_zh'])}</h3>
  <p class="ref">{esc(r['paper_ref'])} · notebook: {esc(r['notebook'])}</p>
  <p class="notes">{esc(r['notes_zh'])}</p>
  <details><summary>实测值与容差</summary>
    <pre>{esc(measured)}</pre>
    <pre>{esc(json.dumps(r.get('tolerance', {}), ensure_ascii=False))}</pre>
  </details>
</article>"""
        )

    summary_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{c[k]}</td></tr>" for k in (*VERDICT_ORDER, "PENDING")
    )
    findings = "".join(
        f"<li><b>{esc(r['id'])}</b> {esc(r['title_zh'])}<br><span>{esc(r['notes_zh'])}</span></li>"
        for r in contradictions
    )
    partial_items = "".join(
        f"<li><b>{esc(r['id'])}</b> {esc(r['title_zh'])}</li>" for r in partials
    )
    manifest_json = esc(json.dumps(manifest or {}, indent=2, ensure_ascii=False))
    generated = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EBBA / EBBA+ 可复现验证报告</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
         margin: 0; background: #f6f7f9; color: #1f2328; line-height: 1.55; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 80px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }}
  h2 {{ font-size: 20px; margin-top: 34px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; }}
  .sub {{ color: #57606a; margin: 0 0 18px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }}
  .stat {{ background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 12px 16px; min-width: 120px; }}
  .stat b {{ display: block; font-size: 24px; }}
  table {{ border-collapse: collapse; background: #fff; }}
  td, th {{ border: 1px solid #d8dee4; padding: 6px 12px; text-align: left; }}
  .badge {{ color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; }}
  .claim {{ background: #fff; border: 1px solid #d8dee4; border-left-width: 5px; border-radius: 10px;
            padding: 14px 18px; margin: 14px 0; }}
  .v-CONFIRMED {{ border-left-color: #1b7f3a; }}
  .v-PARTIAL {{ border-left-color: #9a6700; }}
  .v-CONTRADICTED {{ border-left-color: #b42318; }}
  .v-NOT_REPRODUCIBLE {{ border-left-color: #5f6b7a; }}
  .v-ERROR, .v-PENDING {{ border-left-color: #7a1fa2; }}
  .claim header {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .cid {{ font-family: ui-monospace, Consolas, monospace; font-weight: 700; }}
  .tier, .ref {{ color: #57606a; font-size: 12px; }}
  .claim h3 {{ margin: 10px 0 4px; font-size: 16px; }}
  .notes {{ margin: 8px 0; }}
  details summary {{ cursor: pointer; color: #0969da; font-size: 13px; }}
  pre {{ background: #f6f8fa; padding: 10px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
  ul.findings li {{ margin-bottom: 10px; }}
  ul.findings span {{ color: #57606a; font-size: 13px; }}
  .callout {{ background: #fff8e6; border: 1px solid #f0d58c; border-radius: 10px; padding: 12px 16px; }}
  .lim {{ color: #57606a; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>EBBA / EBBA+ 可复现验证报告</h1>
  <p class="sub">Gao et al., <i>Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining</i>,
  ICML 2024 (PMLR 235) · 生成于 {esc(generated)}</p>

  <div class="callout">
    <b>怎么读这份报告。</b> Tier A 是可直接核对的公式、表格算术与证明步骤；
    Tier B 是缩比模型上的机制验证，按项目规则<b>永远不会标为 CONFIRMED</b>；
    Tier C 是需要论文级算力、容器内无法复现的条目。CONTRADICTED 表示论文的某个可核对断言不成立，
    不表示方法整体无效。
  </div>

  <div class="stats">
    <div class="stat"><b>{total}</b>条声明</div>
"""
    for key in VERDICT_ORDER:
        page += f'    <div class="stat" style="border-top:4px solid {VERDICT_COLOR[key]}"><b>{c[key]}</b>{key}</div>\n'
    page += f"""  </div>

  <h2>判定统计</h2>
  <table><tr><th>判定</th><th>数量</th></tr>{summary_rows}</table>

  <h2>被否定的声明（{len(contradictions)} 条）</h2>
  <ul class="findings">{findings or "<li>无</li>"}</ul>

  <h2>部分成立（{len(partials)} 条）</h2>
  <ul>{partial_items or "<li>无</li>"}</ul>

  <h2>逐条结果</h2>
  {''.join(cards)}

  <h2>代理实验与产物</h2>
  <p class="lim">
    Tier B 使用 CIFAR-10 子集（20,000 训练 / 2,000 测试）、SmallCNN、BadNets 与 Blend，
    任务无关图像用 CIFAR-100 测试集代理。它检验方法的机制方向，不复现论文的 ResNet18、
    200 epochs、三个数据集与五种攻击。所有 proxy 结论不应被引用为论文的精确数值。
  </p>
  <details><summary>manifest.json</summary><pre>{manifest_json}</pre></details>
</main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path


def build(out_dir: Path | None = None, claims_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    rows = collect(claims_dir)
    manifest_path = Path(config.ARTIFACTS_DIR) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    return {
        "verdicts": write_json(rows, manifest, out_dir / "verdicts.json"),
        "markdown": write_markdown(rows, out_dir / "report.md", manifest),
        "html": write_html(rows, out_dir / "report.html", manifest),
    }
