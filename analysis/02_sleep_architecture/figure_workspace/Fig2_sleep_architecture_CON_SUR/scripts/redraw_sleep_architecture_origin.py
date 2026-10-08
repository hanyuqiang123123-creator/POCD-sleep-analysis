"""Rebuild the CON/SUR sleep-architecture panels in OriginPro.

The input is the animal-level AutoFQ summary.  Each graph page contains linked
raw values, means, SEMs, point coordinates, and the displayed statistics.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd
from scipy import stats


SOURCE_CSV = Path(r"F:\Sleep\eXdata\EEG\FQ\EEG\FQ_summary_CON_SUR_rebuilt.csv")
OUTPUT_DIR = Path(r"F:\Sleep\Figure\Fig2_sleep_architecture_CON_SUR_20260904")
PROJECT_PATH = OUTPUT_DIR / "Fig2_sleep_architecture_CON_SUR.opju"
SOURCE_DATA_CSV = OUTPUT_DIR / "Fig2_sleep_architecture_source_data.csv"
STATS_CSV = OUTPUT_DIR / "Fig2_sleep_architecture_stats.csv"

CON_FILL = (130, 201, 232)
CON_EDGE = (46, 117, 145)
SUR_FILL = (240, 111, 116)
SUR_EDGE = (176, 55, 64)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

AXIS_WIDTH = 2.5
TICK_WIDTH = 2.5
BAR_EDGE_WIDTH = 2.5
ERROR_WIDTH = 2.5
BRACKET_WIDTH = 2.5
POINT_SIZE = 14.0
POINT_EDGE_WIDTH = 2.5
POINT_EDGE_PERCENT = 100 * POINT_EDGE_WIDTH / (POINT_SIZE / 2)
POINT_X_STEP = 0.045
ORIGIN_BAR_GAP = 50

DAYS = ["BL", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
BAR_X = [value for center in range(1, 9) for value in (center - 0.16, center + 0.16)]
GROUPS = [group for _day in DAYS for group in ("CON", "SUR")]


@dataclass(frozen=True)
class Metric:
    key: str
    graph_name: str
    title: str
    phase: str
    y_title: str
    y_max: float
    y_step: float
    source_kind: str
    source_phase: str


METRICS = [
    Metric("nrem_light_time", "NREM_Light_Time", "NREM sleep time - light phase", "Light phase", "NREM sleep time (h)", 8.5, 2.0, "time", "part1_NREM"),
    Metric("nrem_dark_time", "NREM_Dark_Time", "NREM sleep time - dark phase", "Dark phase", "NREM sleep time (h)", 8.5, 2.0, "time", "part2_NREM"),
    Metric("nrem_light_duration", "NREM_Light_Duration", "NREM episode duration - light phase", "Light phase", "Duration of NREM episodes (min)", 3.4, 1.0, "duration", "part1_NREM"),
    Metric("nrem_dark_duration", "NREM_Dark_Duration", "NREM episode duration - dark phase", "Dark phase", "Duration of NREM episodes (min)", 3.4, 1.0, "duration", "part2_NREM"),
    Metric("rem_light_time", "REM_Light_Time", "REM sleep time - light phase", "Light phase", "REM sleep time (h)", 1.8, 0.5, "time", "part1_REM"),
    Metric("rem_dark_time", "REM_Dark_Time", "REM sleep time - dark phase", "Dark phase", "REM sleep time (h)", 0.65, 0.2, "time", "part2_REM"),
    Metric("rem_light_duration", "REM_Light_Duration", "REM episode duration - light phase", "Light phase", "Duration of REM episodes (min)", 1.8, 0.5, "duration", "part1_REM"),
    Metric("rem_dark_duration", "REM_Dark_Duration", "REM episode duration - dark phase", "Dark phase", "Duration of REM episodes (min)", 1.8, 0.5, "duration", "part2_REM"),
]


def rgb_expr(rgb: tuple[int, int, int]) -> str:
    return f"color({rgb[0]},{rgb[1]},{rgb[2]})"


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def star_for_p(p_value: float) -> str:
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def normalize_svg(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(
        r"#([0-9A-Fa-f]{6})",
        lambda match: "rgb(" + ",".join(
            str(int(match.group(1)[index:index + 2], 16)) for index in (0, 2, 4)
        ) + ")",
        text,
    )
    text = re.sub(r"(?<=[=\"' :;])black(?=[;\"' ])", "rgb(0,0,0)", text)
    text = re.sub(r"(?<=[=\"' :;])white(?=[;\"' ])", "rgb(255,255,255)", text)
    svg_path.write_text(text, encoding="utf-8", newline="\n")


def load_long_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_CSV)
    day_token = data["Filename"].str.extract(r"(Pre1|P[1-7])")[0]
    mapping = {"Pre1": "BL", **{f"P{index}": f"D{index}" for index in range(1, 8)}}
    data["Day"] = day_token.replace(mapping)
    data["Subject"] = data["Group"] + "_" + data["Folder"].astype(str)
    if data["Day"].isna().any():
        raise ValueError("Could not derive BL/D1-D7 from every Filename")
    if set(data["Group"]) != {"CON", "SUR"}:
        raise ValueError(f"Unexpected groups: {sorted(data['Group'].unique())}")

    for metric in METRICS:
        percentage = data[f"{metric.source_phase}_pct"].astype(float)
        if metric.source_kind == "time":
            data[metric.key] = percentage * 12.0 / 100.0
        else:
            count = data[f"{metric.source_phase}_count"].astype(float)
            data[metric.key] = percentage * 7.2 / count
    return data


def collision_aware_x(center: float, values: list[float], y_max: float) -> list[float]:
    result = [center] * len(values)
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    threshold = y_max * 0.024
    clusters: list[list[int]] = []
    for index in order:
        if not clusters or values[index] - values[clusters[-1][-1]] > threshold:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    for cluster in clusters:
        midpoint = (len(cluster) - 1) / 2
        for position, index in enumerate(cluster):
            result[index] = center + (position - midpoint) * POINT_X_STEP
    return result


def set_label_style(label, size: float, *, bold: bool = False) -> None:
    if label is None:
        return
    label.color = BLACK
    label.set_float("fsize", size)
    label.set_int("fontbold", int(bold))


def add_rectangle(
    layer,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
):
    rectangle = layer.obj.GraphObjects.Add(8)
    if not rectangle:
        raise RuntimeError("Origin failed to create a legend rectangle")
    rectangle.SetNumProp("attach", 2)
    rectangle.SetNumProp("x1", x1)
    rectangle.SetNumProp("y1", y1)
    rectangle.SetNumProp("x2", x2)
    rectangle.SetNumProp("y2", y2)
    rectangle.SetNumProp("fillcolor", op.utils.ocolor(fill))
    rectangle.SetNumProp("color", op.utils.ocolor(edge))
    rectangle.SetNumProp("fillpattern", 0)
    rectangle.SetNumProp("linewidth", 0.7)
    return rectangle


def add_bracket(layer, x1: float, x2: float, y0: float, y1: float, stars: str) -> None:
    for line in (
        layer.add_line(x1, y0, x1, y1),
        layer.add_line(x1, y1, x2, y1),
        layer.add_line(x2, y1, x2, y0),
    ):
        line.color = BLACK
        line.width = BRACKET_WIDTH
    label = layer.add_label(stars, (x1 + x2) / 2, y1 + (y1 - y0) * 0.35)
    set_label_style(label, 20, bold=True)
    label.set_int("attach", 2)
    label.set_int("anchor", 8)


def metric_cells(data: pd.DataFrame, metric: Metric) -> list[list[float]]:
    cells: list[list[float]] = []
    for day, group in zip([day for day in DAYS for _ in (0, 1)], GROUPS, strict=True):
        values = (
            data.loc[(data["Day"] == day) & (data["Group"] == group), metric.key]
            .astype(float)
            .tolist()
        )
        if len(values) != 6:
            raise ValueError(f"{metric.key} {day} {group}: expected n=6, got n={len(values)}")
        cells.append(values)
    return cells


def make_source_and_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    for metric in METRICS:
        for day in DAYS:
            group_values: dict[str, np.ndarray] = {}
            for group in ("CON", "SUR"):
                subset = data.loc[(data["Day"] == day) & (data["Group"] == group)]
                values = subset[metric.key].to_numpy(dtype=float)
                group_values[group] = values
                for (_, row), value in zip(subset.iterrows(), values, strict=True):
                    source_rows.append(
                        {
                            "metric": metric.key,
                            "phase": metric.phase,
                            "day": day,
                            "group": group,
                            "subject": row["Subject"],
                            "folder": row["Folder"],
                            "filename": row["Filename"],
                            "value": value,
                            "unit": "h" if metric.source_kind == "time" else "min",
                        }
                    )
            con = group_values["CON"]
            sur = group_values["SUR"]
            p_value = float(stats.ttest_ind(con, sur, equal_var=True).pvalue)
            con_mean, con_sem = mean_sem(con.tolist())
            sur_mean, sur_sem = mean_sem(sur.tolist())
            stats_rows.append(
                {
                    "metric": metric.key,
                    "phase": metric.phase,
                    "day": day,
                    "test": "two-sided unpaired Student t-test",
                    "n_CON": len(con),
                    "n_SUR": len(sur),
                    "mean_CON": con_mean,
                    "SEM_CON": con_sem,
                    "mean_SUR": sur_mean,
                    "SEM_SUR": sur_sem,
                    "t_statistic": float(stats.ttest_ind(con, sur, equal_var=True).statistic),
                    "df": len(con) + len(sur) - 2,
                    "p_value": p_value,
                    "stars": star_for_p(p_value),
                }
            )
    source = pd.DataFrame(source_rows)
    stats_table = pd.DataFrame(stats_rows)
    source.to_csv(SOURCE_DATA_CSV, index=False, encoding="utf-8-sig")
    stats_table.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
    return source, stats_table


def build_origin_panel(data: pd.DataFrame, stats_table: pd.DataFrame, metric: Metric):
    values_by_cell = metric_cells(data, metric)
    summaries = [mean_sem(values) for values in values_by_cell]

    wks = op.new_sheet("w", lname=f"Data | {metric.title}", hidden=False)
    wks.name = f"D_{metric.graph_name[:15]}"
    wks.cols = 51

    for index, (values, day, group) in enumerate(
        zip(values_by_cell, [day for day in DAYS for _ in (0, 1)], GROUPS, strict=True)
    ):
        unit = "h" if metric.source_kind == "time" else "min"
        wks.from_list(index, values, lname=f"{day} {group} raw", units=unit, axis="Y")

    wks.from_list(16, BAR_X, lname="Bar X", axis="X")
    wks.from_list(17, [item[0] for item in summaries], lname="Mean from raw", axis="Y")
    wks.from_list(18, [item[1] for item in summaries], lname="SEM from raw", axis="E")

    for index, (center, values, day, group) in enumerate(
        zip(BAR_X, values_by_cell, [day for day in DAYS for _ in (0, 1)], GROUPS, strict=True)
    ):
        point_base = 19 + index * 2
        wks.from_list(
            point_base,
            collision_aware_x(center, values, metric.y_max),
            lname=f"{day} {group} point X",
            axis="X",
        )
        wks.from_list(point_base + 1, values, lname=f"{day} {group} points", axis="Y")

    graph = op.new_graph(lname=metric.title, template="scatter")
    graph.name = metric.graph_name
    layer = graph[0]

    bar_plot = layer.add_plot(wks, colx=16, coly=17, type="c")
    bar_plot.color = CON_FILL
    bar_index = bar_plot.index() + 1
    op.lt_exec(
        f"range rb=[{metric.graph_name}]1!{bar_index}; "
        f"set rb -vg {ORIGIN_BAR_GAP}; set rb -pbw {BAR_EDGE_WIDTH}; set rb -pfp 0;"
    )
    for point_index, group in enumerate(GROUPS, start=1):
        fill, edge = (CON_FILL, CON_EDGE) if group == "CON" else (SUR_FILL, SUR_EDGE)
        op.lt_exec(
            f"range rb=[{metric.graph_name}]1!{bar_index}; "
            f"set rb {point_index} -pbc {rgb_expr(edge)}; "
            f"set rb {point_index} -pfb {rgb_expr(fill)}; "
            f"set rb {point_index} -pfc {rgb_expr(fill)};"
        )

    for index, group in enumerate(GROUPS):
        edge = CON_EDGE if group == "CON" else SUR_EDGE
        point_base = 19 + index * 2
        plot = layer.add_plot(wks, colx=point_base, coly=point_base + 1, type="s")
        plot.color = edge
        plot.set_int("symbol.kind", 2)
        plot.set_float("symbol.size", POINT_SIZE)
        plot.set_int("symbol.edgecolor", op.utils.ocolor(edge))
        plot.set_int("symbol.fillcolor", op.utils.ocolor(WHITE))
        plot_index = plot.index() + 1
        op.lt_exec(
            f"range rr=[{metric.graph_name}]1!{plot_index}; "
            f"set rr -k 2; set rr -z {POINT_SIZE}; set rr -kf 1; "
            f"set rr -kh {POINT_EDGE_PERCENT}; set rr -cse {rgb_expr(edge)}; "
            f"set rr -csf {rgb_expr(WHITE)};"
        )

    error_carrier = layer.add_plot(wks, colx=16, coly=17, colyerr=18, type="s")
    carrier_index = error_carrier.index() + 1
    op.lt_exec(f"range rc=[{metric.graph_name}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_plot = layer.plot_list()[-1]
    error_plot.color = BLACK
    error_index = error_plot.index() + 1
    op.lt_exec(
        f"range re=[{metric.graph_name}]1!{error_index}; set re -c color(0,0,0); "
        f"set re -erw {ERROR_WIDTH}; set re -erwc 15; set re -erdy 0;"
    )

    x_end = 8.48
    layer.axis("x").set_limits(0.52, x_end, 1.0)
    layer.axis("y").set_limits(0, metric.y_max, metric.y_step)
    layer.axis("x").title = ""
    layer.axis("y").title = metric.y_title
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=1; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; "
        "layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={AXIS_WIDTH}; layer.y.thickness={AXIS_WIDTH}; "
        "layer.x.ticks=0; layer.y.ticks=1;"
    )

    bottom = layer.add_line(0.52, 0, x_end, 0)
    left = layer.add_line(0.52, 0, 0.52, metric.y_max)
    for line in (bottom, left):
        line.color = BLACK
        line.width = AXIS_WIDTH

    y_value = 0.0
    while y_value <= metric.y_max + 1e-9:
        tick = layer.add_line(0.52, y_value, 0.47, y_value)
        tick.color = BLACK
        tick.width = TICK_WIDTH
        y_value += metric.y_step

    for center, day in enumerate(DAYS, start=1):
        tick = layer.add_line(center, 0, center, -metric.y_max * 0.025)
        tick.color = BLACK
        tick.width = TICK_WIDTH
        label = layer.add_label(day, center + 0.025, -metric.y_max * 0.075)
        set_label_style(label, 20)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)

    title = layer.add_label(metric.phase, 4.5, metric.y_max * 0.955)
    set_label_style(title, 21.5)
    title.set_int("attach", 2)
    title.set_int("anchor", 8)

    y_title = layer.label("YL")
    set_label_style(y_title, 21.5)
    op.lt_exec("layer.y.label.fsize=21.5; layer.y.tickLabels.fsize=20;")

    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    for color, y_position in (
        (CON_FILL, metric.y_max * 0.88),
        (SUR_FILL, metric.y_max * 0.81),
    ):
        swatch = layer.add_line(8.55, y_position, 8.70, y_position)
        swatch.color = color
        swatch.width = 10
    for text, y_position in (("CON", metric.y_max * 0.88), ("SUR", metric.y_max * 0.81)):
        label = layer.add_label(text, 8.78, y_position)
        set_label_style(label, 19)
        label.set_int("attach", 2)
        label.set_int("anchor", 4)

    if metric.graph_name.endswith(("Duration", "Time")):
        for obj in layer.obj.GraphObjects:
            text = obj.GetText()
            x1, x2 = obj.GetNumProp("x1"), obj.GetNumProp("x2")
            y1 = obj.GetNumProp("y1")
            if abs(x1 - 8.55) < .01 and abs(x2 - 8.70) < .01:
                con = abs(y1 - metric.y_max * .88) < .01
                obj.SetNumProp("x1", 6.0 if con else 7.15)
                obj.SetNumProp("x2", 6.18 if con else 7.33)
                obj.SetNumProp("y1", metric.y_max * .965)
                obj.SetNumProp("y2", metric.y_max * .965)
            if text in ("CON", "SUR"):
                obj.SetNumProp("x1", 6.30 if text == "CON" else 7.45)
                obj.SetNumProp("y1", metric.y_max * .965)

    metric_stats = stats_table[stats_table["metric"] == metric.key].set_index("day")
    for day_index, day in enumerate(DAYS):
        stars = str(metric_stats.loc[day, "stars"])
        if not stars:
            continue
        left_x, right_x = BAR_X[day_index * 2], BAR_X[day_index * 2 + 1]
        cell_max = max(max(values_by_cell[day_index * 2]), max(values_by_cell[day_index * 2 + 1]))
        y0 = min(cell_max + metric.y_max * 0.045, metric.y_max * 0.86)
        y1 = min(y0 + metric.y_max * 0.03, metric.y_max * 0.91)
        add_bracket(layer, left_x, right_x, y0, y1, stars)

    graph.set_float("width", 9000)
    graph.set_float("height", 5000)
    layer.set_float("left", 12)
    layer.set_float("top", 10)
    layer.set_float("width", 76)
    layer.set_float("height", 74)

    stem = metric.graph_name
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    graph.save_fig(str(png_path), type="png", width=2400)
    graph.save_fig(str(pdf_path), type="pdf")
    graph.save_fig(str(svg_path), type="svg")
    normalize_svg(svg_path)
    shutil.copy2(svg_path, OUTPUT_DIR / f"{stem}_AI_clean.svg")
    shutil.copy2(pdf_path, OUTPUT_DIR / f"{stem}_AI_clean.pdf")
    return graph


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_long_data()
    source, stats_table = make_source_and_stats(data)
    print(f"source rows={len(source)} stats rows={len(stats_table)}")

    op.attach()
    op.set_show(True)
    op.new()
    for metric in METRICS:
        build_origin_panel(data, stats_table, metric)
        significant = stats_table.loc[
            (stats_table["metric"] == metric.key) & (stats_table["stars"] != ""),
            ["day", "p_value", "stars"],
        ]
        print(metric.key, significant.to_dict("records"))
    op.save(str(PROJECT_PATH))
    print(f"project={PROJECT_PATH}")
    print(f"stats={STATS_CSV}")
    op.detach()


if __name__ == "__main__":
    main()
