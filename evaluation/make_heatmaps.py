# evaluation/make_heatmaps.py

"""Generate the paper's overall, ANC-related, and appendix heatmaps.

Run from the repository root: python -m evaluation.make_heatmaps
An optional --output-dir redirects all figures without changing input data.
Outputs are PDF, SVG, and PNG. Input accuracies are already averaged over seeds.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "evaluation"
MEAN_SCORES = RESULTS / "mean_scores.tsv"
BAD_PARSE = RESULTS / "bad_parse.tsv"
PHENOMENA_MANIFEST = ROOT / "evaluation" / "phenomena_manifest.tsv"

CLAUSE_ORDER = ["sov", "svo", "vos"]
NP_ORDER = ["gn", "ng"]
ALIGN_ORDER = ["nom-acc", "erg-abs"]
COMP_ORDER = ["balancing", "deranking"]
STRATEGY_ORDER = ["sent", "poss-acc", "erg-poss", "nomn"]
ALIGN_LABEL = {"nom-acc": "NOM-ACC", "erg-abs": "ERG-ABS"}
COMP_LABEL = {"balancing": "Balancing", "deranking": "Deranking"}
GREEN = (31, 122, 62)
WHITE = (255, 255, 255)
OVERALL_SCALE = (0.70, 1.00)
PHENOMENON_SCALE = (0.50, 1.00)

CELL_W = 76
CELL_H = 34
GRID_X = 132
GRID_Y = 108
GRID_W = CELL_W * 8
GRID_H = CELL_H * 12
PANEL_W = GRID_X + GRID_W
HEIGHT = GRID_Y + GRID_H + 6
SVG_STYLE = (
    '<defs><style>text{font-family:"Liberation Serif","Times New Roman",serif}'
    '.title{font-size:26px;font-weight:bold}.group{font-size:23px}'
    '.strategy{font-size:21px}.row{font-size:22px}'
    '.alignment{font-size:23px}.number{font-size:23px;font-weight:bold}'
    '</style></defs>'
)

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

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

def aggregate(rows, prefixes=None):
    grouped = defaultdict(list)
    for row in rows:
        if prefixes is None or row["phenomenon"].startswith(prefixes):
            grouped[key_for(row)].append(row)
    expected = 34 if prefixes is None else 15
    if len(grouped) != 96:
        raise ValueError("Expected 96 grammar configurations")
    result = {}
    for key, items in grouped.items():
        if len(items) != expected or len({r["phenomenon"] for r in items}) != expected:
            raise ValueError(f"Unexpected phenomenon coverage for {key}")
        if any(int(r["n_pairs"]) != 100 for r in items):
            raise ValueError(f"Expected 100 pairs per phenomenon for {key}")
        result[key] = sum(float(r["accuracy"]) * int(r["n_pairs"]) for r in items) / sum(
            int(r["n_pairs"]) for r in items
        )
    return result

def panel(parts, offset, title, values, *, scale=OVERALL_SCALE, bad_rates=None):
    parts.append(f'<g transform="translate({offset},0)">')
    if title:
        parts.append(text("title", title, GRID_X + GRID_W / 2, 28, "middle"))
    for ci, comp in enumerate(COMP_ORDER):
        start = GRID_X + ci * CELL_W * 4
        parts.append(text("group", COMP_LABEL[comp], start + CELL_W * 2, 57, "middle"))
        parts.append(rect(start + 3, 64, CELL_W * 4 - 6, 0.8, fill="#888"))
        for si, strategy in enumerate(STRATEGY_ORDER):
            x = start + (si + 0.5) * CELL_W
            labels = strategy.upper().split("-")
            if len(labels) == 2:
                parts.append(text("strategy", labels[0] + "-", x, 84, "middle"))
                parts.append(text("strategy", labels[1], x, 103, "middle"))
            else:
                parts.append(text("strategy", labels[0], x, 94, "middle"))

    row_index = 0
    for alignment in ALIGN_ORDER:
        center_y = GRID_Y + (row_index + 3) * CELL_H
        parts.append(
            f'<g transform="translate(19,{center_y}) rotate(-90)">'
            + text("alignment", ALIGN_LABEL[alignment], 0, 0, "middle") + "</g>"
        )
        for clause in CLAUSE_ORDER:
            for np_order in NP_ORDER:
                y = GRID_Y + row_index * CELL_H
                parts.append(text("row", f"{clause.upper()}/{np_order.upper()}", GRID_X - 8, y + 24, "end"))
                for ci, comp in enumerate(COMP_ORDER):
                    for si, strategy in enumerate(STRATEGY_ORDER):
                        x = GRID_X + (ci * 4 + si) * CELL_W
                        accuracy = values[(clause, np_order, alignment, comp, strategy)]
                        color = cell_color(accuracy, scale)
                        foreground = "#111" if luminance(color) > 0.42 else "#fff"
                        parts.append(rect(x, y, CELL_W, CELL_H, fill=color, stroke="#fff", sw=1))
                        baseline = y + (22 if bad_rates is not None else 24)
                        parts.append(text("number", f"{accuracy * 100:.1f}", x + CELL_W / 2, baseline, "middle", foreground))
                        if bad_rates is not None:
                            rate = bad_rates[(clause, np_order, alignment, comp, strategy)]
                            if not 0 <= rate <= 1:
                                raise ValueError(f"Invalid BAD parse rate: {rate}")
                            track_width = CELL_W - 12
                            parts.append(rect(x + 6, y + 27, track_width, 4, fill="#fff", stroke="#999", sw=0.5))
                            if rate > 0:
                                parts.append(rect(x + 6, y + 27, track_width * rate, 4, fill="#d95f02"))
                row_index += 1
    parts.append(rect(GRID_X, GRID_Y, GRID_W, GRID_H, stroke="#555", sw=0.8))
    parts.append(rect(GRID_X, GRID_Y + 6 * CELL_H - 1, GRID_W, 2, fill="#555"))
    parts.append(rect(GRID_X + 4 * CELL_W - 1, GRID_Y, 2, GRID_H, fill="#555"))
    parts.append("</g>")

def validate_data(rows, bad_rows, manifest):
    """Reject incomplete grids or invalid rates before producing figures."""
    names = [row["phenomenon"] for row in manifest]
    stems = [row["figure_stem"] for row in manifest]
    if len(names) != 34 or len(set(names)) != 34 or len(set(stems)) != 34:
        raise ValueError("Expected 34 unique phenomena and figure stems")
    expected_keys = set(product(CLAUSE_ORDER, NP_ORDER, ALIGN_ORDER,
                                COMP_ORDER, STRATEGY_ORDER))
    if set(row["phenomenon"] for row in rows) != set(names):
        raise ValueError("Scores do not match the phenomenon manifest")
    for name in names:
        selected = [r for r in rows if r["phenomenon"] == name]
        if len(selected) != 96 or {key_for(r) for r in selected} != expected_keys:
            raise ValueError(f"Expected 96 unique configurations for {name}")
    score_ids = [(r["phenomenon"], r["language"]) for r in rows]
    bad_ids = [(r["phenomenon"], r["language"]) for r in bad_rows]
    if (len(set(score_ids)) != len(score_ids)
            or len(set(bad_ids)) != len(bad_ids)
            or set(score_ids) != set(bad_ids)):
        raise ValueError("Score and BAD parse rows must match without duplicates")
    for row in rows:
        if not 0 <= float(row["accuracy"]) <= 1 or int(row["n_pairs"]) != 100:
            raise ValueError("Expected valid accuracies and 100 pairs per phenomenon")
    for row in bad_rows:
        if not 0 <= float(row["bad_parse_rate"]) <= 1:
            raise ValueError("BAD parse rates must be between zero and one")


def export_figure(path, description, values, *, title=None,
                  scale=OVERALL_SCALE, bad_rates=None):
    # Overall panels have no titles: the paper supplies LaTeX subcaptions.
    top = 36 if title is None else 0
    width = PANEL_W + 2 if title is None else PANEL_W
    height = HEIGHT - top
    details = (
        'Three-seed mean accuracies across 96 grammatical configurations. '
        'The green color scale runs from 70 to 100 percent.'
        if bad_rates is None else
        'Three-seed mean accuracy in percent, with a green scale of 50--100 percent. '
        'Values below 50 use white. Orange bar length represents BAD parse rate, '
        'with a full track corresponding to 100 percent.'
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 {top} {width} {height}">',
        f'<title>{esc(description)}</title>',
        f'<desc>{details}</desc>',
        SVG_STYLE,
        rect(0, top, width, height, fill="#fff"),
    ]
    panel(parts, 0, title, values, scale=scale, bad_rates=bad_rates)
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts), encoding='utf-8')
    cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix('.png')), scale=1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTDIR,
                        help='Output directory (default: figures/evaluation)')
    output = parser.parse_args().output_dir
    rows = read_tsv(MEAN_SCORES)
    bad_rows = read_tsv(BAD_PARSE)
    manifest = read_tsv(PHENOMENA_MANIFEST)
    validate_data(rows, bad_rows, manifest)
    bad = bad_parse_lookup(bad_rows)

    export_figure(output / 'paper_accuracy_overall.svg',
                  'Accuracy across all 34 phenomena', aggregate(rows))
    export_figure(output / 'paper_accuracy_anc.svg',
                  'Accuracy across the 15 ANC-related phenomena',
                  aggregate(rows, ('4_', '5_', '6_')))
    for phenomenon in manifest:
        name = phenomenon['phenomenon']
        selected = [r for r in rows if r['phenomenon'] == name]
        values = {key_for(r): float(r['accuracy']) for r in selected}
        rates = {key_for(r): bad[(name, r['language'])] for r in selected}
        export_figure(
            output / 'appendix' / (phenomenon['figure_stem'] + '.svg'),
            phenomenon['title'], values,
            title=phenomenon['title'].replace('--', '\u2013'),
            scale=PHENOMENON_SCALE, bad_rates=rates,
        )
    print(f'Exported 2 overview panels and {len(manifest)} appendix figures '
          f'(PDF, SVG, PNG) to {output}')


if __name__ == '__main__':
    main()
