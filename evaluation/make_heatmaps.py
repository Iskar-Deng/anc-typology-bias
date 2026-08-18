#!/usr/bin/env python3

# evaluation/make_heatmaps.py

"""
Generate accuracy heatmaps from three-seed mean scores and BAD parse rates.

Usage:
python -m evaluation.make_heatmaps
"""

from __future__ import annotations

import csv
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "evaluation"
PHENOMENA_MANIFEST = ROOT / "evaluation" / "phenomena_manifest.tsv"

MEAN_SCORES = RESULTS / "mean_scores.tsv"
BAD_PARSE = RESULTS / "bad_parse.tsv"

CLAUSE_ORDER = ["sov", "svo", "vos"]
NP_ORDER = ["gn", "ng"]
ALIGN_ORDER = ["nom-acc", "erg-abs"]
COMP_ORDER = ["balancing", "deranking"]
STRATEGY_ORDER = ["sent", "poss-acc", "erg-poss", "nomn"]

WO_LABEL = {"sov": "SOV", "svo": "SVO", "vos": "VOS"}
NP_LABEL = {"gn": "GN", "ng": "NG"}
ALIGN_LABEL = {"nom-acc": "NOM-ACC", "erg-abs": "ERG-ABS"}
STRATEGY_LABEL = {
    "sent": "SENT",
    "poss-acc": "POSS-ACC",
    "erg-poss": "ERG-POSS",
    "nomn": "NOMN",
}
COMP_LABEL = {"balancing": "Balancing", "deranking": "Deranking"}

W = 880
H_OVERALL_WITH_LEGEND = 652
H_PHENOMENON_WITH_LEGEND = 556
H_NO_LEGEND = 486
GX = 116
CW = 88
CH_OVERALL_WITH_LEGEND = 42
CH_PHENOMENON_WITH_LEGEND = 34
CH_NO_LEGEND = 34
NCOLS = 8
NROWS = 12
GRID_W = CW * NCOLS
GREEN = (31, 122, 62)
WHITE = (255, 255, 255)
ORANGE = "#d95f02"
OVERALL_SCALE = (0.70, 1.00)
PHENOMENON_SCALE = (0.50, 1.00)

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def read_phenomena_manifest(path: Path) -> list[tuple[str, str]]:
    rows = read_tsv(path)
    required = {"phenomenon", "figure_stem"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return [(row["phenomenon"], row["figure_stem"]) for row in rows]

def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(round(v)))) for v in rgb
    )

def cell_color(acc: float, scale: tuple[float, float]) -> str:
    vmin, vmax = scale
    t = max(0.0, min(1.0, (acc - vmin) / (vmax - vmin)))
    return rgb_to_hex(tuple(WHITE[i] * (1 - t) + GREEN[i] * t for i in range(3)))

def luminance(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    r, g, b = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]

    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [linearize(c) for c in (r, g, b)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str | None = None,
    stroke: str | None = None,
    sw: float | None = None,
    rx: float | None = None,
    opacity: float | None = None,
) -> str:
    attrs = [f'x="{x:g}"', f'y="{y:g}"', f'width="{w:g}"', f'height="{h:g}"']
    if rx is not None:
        attrs.append(f'rx="{rx:g}"')
    attrs.append(f'fill="{fill if fill else "none"}"')
    if stroke is not None:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{sw if sw is not None else 1:g}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:g}"')
    return "<rect " + " ".join(attrs) + "/>"

def text(
    cls: str,
    value: str,
    x: float,
    y: float,
    anchor: str | None = None,
    fill: str | None = None,
) -> str:
    attrs = [f'class="{cls}"', f'x="{x:g}"', f'y="{y:g}"']
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    return "<text " + " ".join(attrs) + f">{esc(value)}</text>"

def add_headers(parts: list[str], gy: float) -> None:
    for j, (comp, strategy) in enumerate(
        (comp, strategy) for comp in COMP_ORDER for strategy in STRATEGY_ORDER
    ):
        cx = GX + j * CW + CW / 2
        parts.append(text("lab", STRATEGY_LABEL[strategy], cx, gy - 18, "middle"))
        parts.append(text("comp", COMP_LABEL[comp], cx, gy - 5, "middle"))

def add_cells(
    parts: list[str],
    values: dict[tuple[str, str, str, str, str], tuple[float, float]],
    *,
    gy: float,
    ch: float,
    integer_values: bool,
    scale: tuple[float, float],
) -> None:
    row_i = 0
    cols = [(comp, strategy) for comp in COMP_ORDER for strategy in STRATEGY_ORDER]
    for alignment in ALIGN_ORDER:
        for clause_wo in CLAUSE_ORDER:
            for np_wo in NP_ORDER:
                y = gy + row_i * ch
                parts.append(
                    text("lab", f"{WO_LABEL[clause_wo]}/{NP_LABEL[np_wo]}", 104, y + ch * 0.38, "end")
                )
                parts.append(
                    text("lab", ALIGN_LABEL[alignment], 104, y + ch * 0.82, "end")
                )
                for j, (comp, strategy) in enumerate(cols):
                    x = GX + j * CW
                    acc, bad_parse = values[(clause_wo, np_wo, alignment, comp, strategy)]
                    fill = cell_color(acc, scale)
                    parts.append(rect(x, y, CW, ch, fill=fill, stroke="#fff", sw=1.2))
                    fg = "#111" if luminance(fill) > 0.42 else "#fff"
                    label = f"{acc * 100:.0f}" if integer_values else f"{acc * 100:.1f}"
                    parts.append(text("num", label, x + CW / 2, y + ch * 0.62, "middle", fg))
                    if bad_parse > 0:
                        bar_w = max(2.0, min(75.0, bad_parse * 75.0))
                        parts.append(
                            rect(
                                x + 6.5,
                                y + ch - 5.5,
                                bar_w,
                                3 if ch >= 40 else 2.6,
                                fill=ORANGE,
                                rx=1.5,
                                opacity=0.82,
                            )
                        )
                row_i += 1

def add_legend(parts: list[str], gy: float, grid_h: float, scale: tuple[float, float]) -> None:
    vmin, vmax = scale
    bar_w = 220
    bar_h = 11
    bar_y = gy + grid_h + 18
    left_x = GX
    right_x = GX + GRID_W - bar_w
    steps = 80

    for i in range(steps):
        acc = vmin + (i / (steps - 1)) * (vmax - vmin)
        x = left_x + i * bar_w / steps
        parts.append(rect(x, bar_y, bar_w / steps + 0.7, bar_h, fill=cell_color(acc, scale)))
    parts.append(rect(left_x, bar_y, bar_w, bar_h, stroke="#888", sw=0.6))
    parts.append(text("small", f"{vmin * 100:.0f}", left_x, bar_y + 28))
    parts.append(text("small", f"{vmax * 100:.0f}", left_x + bar_w, bar_y + 28, "end"))
    parts.append(text("small", "Accuracy", left_x + bar_w / 2, bar_y + 28, "middle"))

    parts.append(rect(right_x, bar_y + 4, bar_w, 3.2, fill=ORANGE, rx=1.6, opacity=0.82))
    parts.append(text("small", "0", right_x, bar_y + 28))
    parts.append(text("small", "100", right_x + bar_w, bar_y + 28, "end"))
    parts.append(text("small", "BAD parse rate", right_x + bar_w / 2, bar_y + 28, "middle"))

def write_heatmap(
    stem: str,
    values: dict[tuple[str, str, str, str, str], tuple[float, float]],
    *,
    legend: bool,
    integer_values: bool,
    layout: str,
) -> None:
    if layout == "overall":
        ch = CH_OVERALL_WITH_LEGEND if legend else CH_NO_LEGEND
        h = H_OVERALL_WITH_LEGEND if legend else H_NO_LEGEND
        gy = 92 if legend else 70
        scale = OVERALL_SCALE
    elif layout == "phenomenon":
        ch = CH_PHENOMENON_WITH_LEGEND if legend else CH_NO_LEGEND
        h = H_PHENOMENON_WITH_LEGEND if legend else H_NO_LEGEND
        gy = 70
        scale = PHENOMENON_SCALE
    else:
        raise ValueError(f"unknown heatmap layout: {layout}")
    grid_h = ch * NROWS

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}">']
    parts.append(
        '<defs><style>text{font-family:Arial,Helvetica,sans-serif}'
        '.lab{font-size:11px;fill:#111}.num{font-size:13px;font-weight:700}'
        '.small{font-size:12px;fill:#555}.comp{font-size:11px;fill:#555}</style></defs>'
    )
    parts.append(rect(0, 0, W, h, fill="#fff"))
    add_headers(parts, gy)
    add_cells(parts, values, gy=gy, ch=ch, integer_values=integer_values, scale=scale)
    parts.append(rect(GX, gy, GRID_W, grid_h, stroke="#333", sw=1))
    if legend:
        add_legend(parts, gy, grid_h, scale)
    parts.append("</svg>")

    svg_path = OUTDIR / f"{stem}.svg"
    pdf_path = OUTDIR / f"{stem}.pdf"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    print(svg_path.relative_to(ROOT))
    print(pdf_path.relative_to(ROOT))

def key_for(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["clause_wo"],
        row["np_wo"],
        row["alignment"],
        row["comp_system"],
        row["strategy"],
    )

def bad_parse_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {
        (row["phenomenon"], row["language"]): float(row["bad_parse_rate"])
        for row in rows
    }

def grouped_values(
    rows: list[dict[str, str]],
    bad_rates: dict[tuple[str, str], float],
    *,
    phenomena_prefixes: tuple[str, ...] | None = None,
) -> dict[tuple[str, str, str, str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for row in rows:
        if phenomena_prefixes and not row["phenomenon"].startswith(phenomena_prefixes):
            continue
        key = key_for(row)
        n_pairs = int(row["n_pairs"])
        bucket = grouped.setdefault(key, {"n": 0.0, "acc": 0.0, "bad": 0.0})
        bucket["n"] += n_pairs
        bucket["acc"] += float(row["accuracy"]) * n_pairs
        bucket["bad"] += bad_rates[(row["phenomenon"], row["language"])] * n_pairs
    return {key: (val["acc"] / val["n"], val["bad"] / val["n"]) for key, val in grouped.items()}

def phenomenon_values(
    rows: list[dict[str, str]],
    bad_rates: dict[tuple[str, str], float],
    phenomenon: str,
) -> dict[tuple[str, str, str, str, str], tuple[float, float]]:
    values = {}
    for row in rows:
        if row["phenomenon"] != phenomenon:
            continue
        values[key_for(row)] = (
            float(row["accuracy"]),
            bad_rates[(row["phenomenon"], row["language"])],
        )
    return values

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mean_rows = read_tsv(MEAN_SCORES)
    bad_rows = read_tsv(BAD_PARSE)
    bad_rates = bad_parse_lookup(bad_rows)
    phenomena = read_phenomena_manifest(PHENOMENA_MANIFEST)

    write_heatmap(
        "00_overall_accuracy_heatmap",
        grouped_values(mean_rows, bad_rates),
        legend=True,
        integer_values=False,
        layout="overall",
    )
    write_heatmap(
        "00_anc_only_accuracy_heatmap",
        grouped_values(mean_rows, bad_rates, phenomena_prefixes=("4_", "5_", "6_")),
        legend=True,
        integer_values=False,
        layout="overall",
    )
    for phenomenon, stem in phenomena:
        write_heatmap(
            stem,
            phenomenon_values(mean_rows, bad_rates, phenomenon),
            legend=True,
            integer_values=False,
            layout="phenomenon",
        )

if __name__ == "__main__":
    main()
