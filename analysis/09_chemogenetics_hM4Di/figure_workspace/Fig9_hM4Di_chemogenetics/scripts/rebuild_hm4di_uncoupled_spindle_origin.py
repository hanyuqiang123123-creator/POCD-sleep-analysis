from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd
from scipy.stats import mannwhitneyu

from rebuild_hm4di_latest_spindle_origin import (
    BLACK, EDGES, FILLS, FONT, GROUPS, LINE_WIDTH, POINT_EDGE_PERCENT,
    POINT_SIZE, WHITE, XPOS, add_bracket, add_label, add_line, dodge,
    normalize_svg, rgb,
)

SOURCE = Path(r"F:\1.Sleep\eXdata\hm4Di\SleepWorkbench_FourGroup_Microstructure_FullManualEvents_20260824_v1\animal_metrics_full_manual_events.csv")
METRIC = "uncoupled_spindle_density_per_nrem_min"


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    panels, editable, data, scripts, docs = [package / x for x in ("panels", "editable", "data", "scripts", "documentation")]
    for folder in (panels, editable, data, scripts, docs):
        folder.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(SOURCE)
    frame = frame.loc[frame.group.isin(GROUPS)].copy()
    if len(frame) != 28 or not frame.phase_safe.astype(bool).all():
        raise RuntimeError("Uncoupled-spindle source must contain 28 phase-safe animals")
    frame[METRIC] = (frame["accepted_spindle_count"] - frame["coupled_count"]) / frame["nrem_minutes"]
    frame["uncoupled_spindle_count"] = frame["accepted_spindle_count"] - frame["coupled_count"]

    raw = [frame.loc[frame.group == g].sort_values("animal")[METRIC].to_numpy(float) for g in GROUPS]
    animals = [frame.loc[frame.group == g].sort_values("animal").animal.astype(str).tolist() for g in GROUPS]
    summary = pd.DataFrame([
        {"group": g, "n": len(v), "mean": mean_sem(v)[0], "sem": mean_sem(v)[1]}
        for g, v in zip(GROUPS, raw)
    ])
    stats = []
    for condition, a, b in (("baseline", 0, 1), ("POCD", 2, 3)):
        test = mannwhitneyu(raw[a], raw[b], alternative="two-sided", method="exact")
        p = float(test.pvalue)
        stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        stats.append({"condition": condition, "group_a": GROUPS[a], "group_b": GROUPS[b],
                      "test": "exact two-sided Mann-Whitney U", "U": float(test.statistic), "p_raw": p, "stars": stars})
    stats = pd.DataFrame(stats)

    op.attach()
    op.set_show(True)
    op.new(asksave=False)
    try:
        wks = op.new_sheet("w", lname="Uncoupled spindle density raw and summary", hidden=False)
        wks.name = "WUncoupled"
        wks.cols = 19
        for i, (group, values, names) in enumerate(zip(GROUPS, raw, animals)):
            wks.from_list(i, values.tolist(), lname=f"{group} raw", axis="Y")
            wks.from_list(4 + i, names, lname=f"{group} animal", axis="N")
        wks.from_list(8, XPOS, lname="Bar X", axis="X")
        wks.from_list(9, summary["mean"].tolist(), lname="Arithmetic mean", axis="Y")
        wks.from_list(10, summary["sem"].tolist(), lname="Sample SEM", axis="E")
        for i, (center, values) in enumerate(zip(XPOS, raw)):
            wks.from_list(11 + 2 * i, dodge(values, center).tolist(), lname=f"{GROUPS[i]} point X", axis="X")
            wks.from_list(12 + 2 * i, values.tolist(), lname=f"{GROUPS[i]} points", axis="Y")

        graph = op.new_graph(lname="Uncoupled spindle density", template="scatter")
        graph.name = "GUncoupled"
        layer = graph[0]
        bar = layer.add_plot(wks, colx=8, coly=9, type=203)
        bi = bar.index() + 1
        op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb -vg 45; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;")
        for idx, (fill, edge) in enumerate(zip(FILLS, EDGES), start=1):
            op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb {idx} -pbc {rgb(edge)}; set rb {idx} -pfb {rgb(fill)}; set rb {idx} -pfc {rgb(fill)};")
        for i, edge in enumerate(EDGES):
            plot = layer.add_plot(wks, colx=11 + 2 * i, coly=12 + 2 * i, type="s")
            pi = plot.index() + 1
            op.lt_exec(f"range rp=[{graph.name}]1!{pi}; set rp -k 2; set rp -z {POINT_SIZE}; set rp -kf 1; set rp -kh {POINT_EDGE_PERCENT}; set rp -cse {rgb(edge)}; set rp -csf {rgb(WHITE)};")
        carrier = layer.add_plot(wks, colx=8, coly=9, colyerr=10, type="s")
        ci = carrier.index() + 1
        op.lt_exec(f"range rc=[{graph.name}]1!{ci}; set rc -k 0; set rc -z 0;")
        ei = layer.plot_list()[-1].index() + 1
        op.lt_exec(f"range re=[{graph.name}]1!{ei}; set re -c {rgb(BLACK)}; set re -erw {LINE_WIDTH}; set re -erwc 15; set re -erdy 0;")

        xmin, xmax, ymax, ystep = 0.1, 3.9, 3.0, 0.5
        layer.axis("x").set_limits(xmin, xmax, 1)
        layer.axis("y").set_limits(0, ymax, ystep)
        layer.axis("x").title = ""
        layer.axis("y").title = "Uncoupled Spindles\n(/NREM min)"
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
        pocd = stats.loc[stats.condition == "POCD"].iloc[0]
        if pocd.stars:
            top = max(float(np.max(raw[2])), float(np.max(raw[3])))
            add_bracket(layer, XPOS[2], XPOS[3], top + ymax * .055, ymax * .02, pocd.stars)
        graph.set_float("width", 5600)
        graph.set_float("height", 5000)
        for key, value in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
            layer.set_float(key, value)

        project = editable / "Fig9_hM4Di_uncoupled_spindle_density.opju"
        op.save(str(project))
        png, svg, pdf = [panels / f"Uncoupled_spindle_density.{ext}" for ext in ("png", "svg", "pdf")]
        graph.save_fig(str(png), type="png", width=2400)
        graph.save_fig(str(svg), type="svg")
        graph.save_fig(str(pdf), type="pdf")
        normalize_svg(svg)
        shutil.copy2(svg, panels / "Uncoupled_spindle_density_AI_clean.svg")
        shutil.copy2(pdf, panels / "Uncoupled_spindle_density_AI_clean.pdf")
        op.save(str(project))
    finally:
        op.detach()

    frame.to_csv(data / "uncoupled_spindle_phase_safe_animal_source.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(data / "uncoupled_spindle_group_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(data / "uncoupled_spindle_statistics.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    (docs / "uncoupled_spindle_manifest.json").write_text(json.dumps({
        "source": str(SOURCE), "phase_safe": True,
        "formula": "(accepted_spindle_count - coupled_count) / nrem_minutes",
        "metric": METRIC, "project": str(project), "statistics": stats.to_dict("records"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(stats.to_string(index=False))
    print(project)


if __name__ == "__main__":
    main()
