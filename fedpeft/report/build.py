"""
Render the interactive HTML report.

    collect(cfg)  ->  one dict with every KPI  ->  JSON embedded in template.html
                                               +  Plotly.js embedded (works offline)
    ->  reports/FedPEFT-IMU_report_<profile>.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..utils import get_logger, save_json
from .collect import collect

log = get_logger("report")


def _plotly_js() -> str:
    """The Plotly.js bundle shipped with the `plotly` Python package (no internet needed)."""
    from plotly.offline import get_plotlyjs

    # The bundle contains one literal U+FFFD inside a regular expression; the escaped
    # form is identical for JavaScript and keeps the file free of replacement characters.
    return get_plotlyjs().replace("\ufffd", "\\uFFFD")


def _clean(obj):
    """Replace NaN / inf by None so the page receives valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def build_report(cfg) -> Path:
    data = _clean(collect(cfg))
    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(data, reports_dir / f"report_data_{cfg.profile}.json")   # the numbers behind the page

    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=False)
    template = env.get_template("template.html")
    # "</" must not appear inside an inline <script> block.
    data_json = json.dumps(data, separators=(",", ":"), allow_nan=False, default=str).replace("</", "<\\/")
    plotly_js = _plotly_js()
    # 1) Stand-alone file: a complete HTML document you can open from disk or e-mail.
    out = reports_dir / f"FedPEFT-IMU_report_{cfg.profile}.html"
    out.write_text(template.render(plotly_js=plotly_js, data_json=data_json, standalone=True), encoding="utf-8")
    # 2) Web version: page content only (a hosting page supplies <html>/<head>/<body>).
    web = reports_dir / "web" / f"FedPEFT-IMU_report_{cfg.profile}.html"
    web.parent.mkdir(parents=True, exist_ok=True)
    web.write_text(template.render(plotly_js=plotly_js, data_json=data_json, standalone=False), encoding="utf-8")
    log.info(f"report written to {out} ({out.stat().st_size / 1e6:.1f} MB); web version in {web.parent}")
    return out
