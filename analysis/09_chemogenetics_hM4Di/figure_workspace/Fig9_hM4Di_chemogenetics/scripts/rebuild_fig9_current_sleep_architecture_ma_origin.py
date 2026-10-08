from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd

from rebuild_hm4di_latest_spindle_origin import (
    BLACK, EDGES, FILLS, FONT, GROUPS, LINE_WIDTH, PAIR_LINE_WIDTH, POINT_EDGE_PERCENT,
POINT_SIZE, WHITE, XPOS, add_bracket, add_label, add_line, combine_svgs,
    dodge, normalize_svg, rgb,
)

PAIR_COLORS = [(154, 202, 216), (238, 169, 137)]

SPECS = [
    {"metric": "nrem_percent_of_valid", "stem": "NREM", "title": "NREM", "ylabel": "NREM (%)", "ymax": 60.0, "ystep": 10.0},
    {"metric": "rem_percent_of_valid", "stem": "REM", "title": "REM", "ylabel": "REM (%)", "ymax": 8.0, "ystep": 1.0},
    {"metric": "wake_percent_of_valid", "stem": "WAKE", "title": "WAKE", "ylabel": "WAKE (%)", "ymax": 95.0, "ystep": 20.0},
    {"metric": "ma_per_nrem_h", "stem": "MAs", "title": "MAs", "ylabel": "MAs / NREM\n(h⁻¹)", "ymax": 50.0, "ystep": 10.0},
]


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def export_graph(graph, panels: Path, stem: str):
    png, svg, pdf = [panels / f"{stem}.{ext}" for ext in ("png", "svg", "pdf")]
    graph.save_fig(str(png), type="png", width=2400)
    graph.save_fig(str(svg), type="svg")
    graph.save_fig(str(pdf), type="pdf")
    normalize_svg(svg)
    shutil.copy2(svg, panels / f"{stem}_AI_clean.svg")
    shutil.copy2(pdf, panels / f"{stem}_AI_clean.pdf")


def build_graph(frame, summary, stats, spec):
    metric = spec["metric"]
    raw, animals = [], []
    for group in GROUPS:
        sub = frame.loc[frame.group == group].sort_values("animal")
        raw.append(sub[metric].to_numpy(float))
        animals.append(sub.animal.astype(str).tolist())
    means = [mean_sem(v)[0] for v in raw]
    sems = [mean_sem(v)[1] for v in raw]
    wks = op.new_sheet("w", lname=f"{spec['title']} raw and summary", hidden=False)
    wks.name = ("W" + spec["stem"])[:12]
    wks.cols = 47
    for i, (group, values, names) in enumerate(zip(GROUPS, raw, animals)):
        wks.from_list(i, values.tolist(), lname=f"{group} raw", axis="Y")
        wks.from_list(4 + i, names, lname=f"{group} animal", axis="N")
    wks.from_list(8, XPOS, lname="Bar X", axis="X")
    wks.from_list(9, means, lname="Arithmetic mean", axis="Y")
    wks.from_list(10, sems, lname="Sample SEM", axis="E")
    for i, (center, values) in enumerate(zip(XPOS, raw)):
        wks.from_list(11 + 2 * i, dodge(values, center).tolist(), lname=f"{GROUPS[i]} point X", axis="X")
        wks.from_list(12 + 2 * i, values.tolist(), lname=f"{GROUPS[i]} points", axis="Y")
    for condition_index, (left_group, right_group, left_x, right_x) in enumerate((
        ("baseline+SAL", "baseline+CNO", XPOS[0], XPOS[1]),
        ("POCD+SAL", "POCD+CNO", XPOS[2], XPOS[3]),
    )):
        left = frame.loc[frame.group == left_group].set_index("animal")[metric].sort_index()
        right = frame.loc[frame.group == right_group].set_index("animal")[metric].sort_index()
        if left.index.tolist() != right.index.tolist():
            raise RuntimeError(f"Pairing mismatch: {left_group}/{right_group}")
        for pair_index, animal in enumerate(left.index):
            col = 19 + 2 * (condition_index * 7 + pair_index)
            wks.from_list(col, [left_x, right_x], lname=f"{animal} pair X", axis="X")
            wks.from_list(col + 1, [float(left[animal]), float(right[animal])], lname=f"{animal} pair Y", axis="Y")

    graph = op.new_graph(lname=spec["title"], template="scatter")
    graph.name = ("G" + spec["stem"])[:12]
    layer = graph[0]
    bar = layer.add_plot(wks, colx=8, coly=9, type=203)
    bi = bar.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb -vg 45; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;")
    for idx, (fill, edge) in enumerate(zip(FILLS, EDGES), start=1):
        op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb {idx} -pbc {rgb(edge)}; set rb {idx} -pfb {rgb(fill)}; set rb {idx} -pfc {rgb(fill)};")
    for condition_index in range(2):
        for pair_index in range(7):
            col = 19 + 2 * (condition_index * 7 + pair_index)
            pair = layer.add_plot(wks, colx=col, coly=col + 1, type="line")
            pair.color = PAIR_COLORS[condition_index]
            pair.set_float("line.width", PAIR_LINE_WIDTH)
    for i, edge in enumerate(EDGES):
        plot = layer.add_plot(wks, colx=11 + 2 * i, coly=12 + 2 * i, type="s")
        pi = plot.index() + 1
        op.lt_exec(f"range rp=[{graph.name}]1!{pi}; set rp -k 2; set rp -z {POINT_SIZE}; set rp -kf 1; set rp -kh {POINT_EDGE_PERCENT}; set rp -cse {rgb(edge)}; set rp -csf {rgb(WHITE)};")
    carrier = layer.add_plot(wks, colx=8, coly=9, colyerr=10, type="s")
    ci = carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{ci}; set rc -k 0; set rc -z 0;")
    ei = layer.plot_list()[-1].index() + 1
    op.lt_exec(f"range re=[{graph.name}]1!{ei}; set re -c {rgb(BLACK)}; set re -erw {LINE_WIDTH}; set re -erwc 15; set re -erdy 0;")

    xmin, xmax, ymax, ystep = .1, 3.9, spec["ymax"], spec["ystep"]
    layer.axis("x").set_limits(xmin, xmax, 1)
    layer.axis("y").set_limits(0, ymax, ystep)
    layer.axis("x").title = ""
    layer.axis("y").title = spec["ylabel"]
    op.lt_exec("layer.border=0; layer.x.showAxes=0; layer.y.showAxes=1; layer.x2.showAxes=0; layer.y2.showAxes=0; layer.x.showLabels=0; layer.y.showLabels=1; layer.x.showGrids=0; layer.y.showGrids=0; layer.x.ticks=0; layer.y.ticks=0; layer.x.thickness=0; layer.y.thickness=0; layer.y.label.fsize=21.5; layer.y.label.bold=0; layer.y.label.font=font(Arial);")
    legend = layer.label("legend")
    if legend:
        legend.remove()
    ylabel = layer.label("YL")
    if ylabel:
        ylabel.color = BLACK
        ylabel.set_float("fsize", FONT)
        ylabel.set_int("fontbold", 0)
        ylabel.set_int("font", 4)
    add_line(layer, xmin, 0, xmax, 0)
    add_line(layer, xmin, 0, xmin, ymax)
    for y in np.arange(0, ymax + .01, ystep):
        add_line(layer, xmin, float(y), xmin - .065, float(y))
    for x, text in zip(XPOS, ["SAL", "CNO", "SAL", "CNO"]):
        add_line(layer, x, 0, x, -ymax * .018)
        add_label(layer, text, x + .02, -ymax * .065)
    add_label(layer, "baseline", 1.02, -ymax * .145)
    add_label(layer, "POCD", 3.02, -ymax * .145)
    add_label(layer, spec["title"], 2.02, ymax * 1.05, size=FONT)
    row = stats.loc[(stats.metric == metric) & (stats.comparison == "POCD SAL vs CNO")].iloc[0]
    if isinstance(row.stars, str) and row.stars:
        top = max(float(np.max(raw[2])), float(np.max(raw[3])))
        add_bracket(layer, XPOS[2], XPOS[3], top + ymax * .055, ymax * .02, row.stars)
    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    for key, value in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(key, value)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    panels, editable, data, scripts, docs = [package / x for x in ("panels", "editable", "data", "scripts", "documentation")]
    source = data / "sleep_architecture_ma_current_source_data.csv"
    summary_path = data / "sleep_architecture_ma_current_group_summary.csv"
    stats_path = data / "sleep_architecture_ma_current_statistics.csv"
    frame, summary, stats = pd.read_csv(source), pd.read_csv(summary_path), pd.read_csv(stats_path, keep_default_na=False)
    counts = frame.groupby("group").size().reindex(GROUPS).astype(int).to_dict()
    if counts != {group: 7 for group in GROUPS}:
        raise RuntimeError(counts)
    op.attach()
    op.set_show(True)
    op.new(asksave=False)
    try:
        graphs = [(build_graph(frame, summary, stats, spec), spec) for spec in SPECS]
        project = editable / "Fig9_hM4Di_sleep_architecture_MA_current.opju"
        op.save(str(project))
        for graph, spec in graphs:
            export_graph(graph, panels, spec["stem"])
        op.save(str(project))
    finally:
        op.detach()
    combined = package / "Fig9_hM4Di_sleep_architecture_MA_4panels.svg"
    combine_svgs([panels / f"{s['stem']}.svg" for s in SPECS], combined)
    shutil.copy2(combined, package / "Fig9_hM4Di_sleep_architecture_MA_4panels_AI_clean.svg")
    import cairosvg
    cairosvg.svg2png(bytestring=combined.read_bytes(), write_to=str(package / "Fig9_hM4Di_sleep_architecture_MA_4panels.png"), output_width=6400)
    cairosvg.svg2pdf(bytestring=combined.read_bytes(), write_to=str(package / "Fig9_hM4Di_sleep_architecture_MA_4panels.pdf"))
    shutil.copy2(package / "Fig9_hM4Di_sleep_architecture_MA_4panels.pdf", package / "Fig9_hM4Di_sleep_architecture_MA_4panels_AI_clean.pdf")
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    (docs / "sleep_architecture_ma_current_manifest.json").write_text(json.dumps({
        "source": str(source), "statistics": str(stats_path), "project": str(project),
        "group_n": counts, "MA_definition": "NREM -> Wake <=15 s -> NREM; events per NREM hour",
        "display": "paired NO1-NO7 SAL-to-CNO lines; mean +/- sample SEM; hollow animal points; no animal IDs; MAs is a prespecified standalone primary endpoint and uses its exact paired Wilcoxon p value; other rows retain the recorded conservative Holm sensitivity values",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(project)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
