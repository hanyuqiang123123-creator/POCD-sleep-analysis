from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import originpro as op


OUTPUT_DIR = Path(r"F:\Sleep\Figure\Sur_behavior_Day3_Day7_Origin_Nature_style_20260904\03_NOR")
JSON_PATH = OUTPUT_DIR / "nor_di_raw.json"
PROJECT_PATH = OUTPUT_DIR / "NOR_DI_Day3_Day7.opju"

CON_FILL = (130, 201, 232)
CON_EDGE = (46, 117, 145)
SUR_FILL = (240, 111, 116)
SUR_EDGE = (176, 55, 64)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BAR_X = [0.84, 1.16]
POINT_SIZE = 14.0
POINT_EDGE_WIDTH = 2.5
POINT_EDGE_PERCENT = 100 * POINT_EDGE_WIDTH / (POINT_SIZE / 2)
POINT_X_STEP = 0.045
POINT_COLLISION_Y = 0.045
LINE_WIDTH = 2.5
MAIN_FONT_SIZE = 21.5
LEGEND_FONT_SIZE = 20.0


def collision_aware_x(center: float, values: list[float]) -> list[float]:
    """Deterministically dodge only vertically colliding markers."""
    result = [center] * len(values)
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    clusters: list[list[int]] = []
    for index in order:
        if not clusters or values[index] - values[clusters[-1][-1]] > POINT_COLLISION_Y:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    for cluster in clusters:
        midpoint = (len(cluster) - 1) / 2
        for position, index in enumerate(cluster):
            result[index] = center + (position - midpoint) * POINT_X_STEP
    return result


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def rgb_expr(rgb: tuple[int, int, int]) -> str:
    return f"color({rgb[0]},{rgb[1]},{rgb[2]})"


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(?<=[=\"\' :;])black(?=[;\"\' ])', "rgb(0,0,0)", text)
    text = re.sub(r'(?<=[=\"\' :;])white(?=[;\"\' ])', "rgb(255,255,255)", text)
    text = re.sub(
        r"#([0-9A-Fa-f]{6})",
        lambda match: "rgb(" + ",".join(str(int(match.group(1)[i:i + 2], 16)) for i in (0, 2, 4)) + ")",
        text,
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def style_label(label, size: float, color: tuple[int, int, int] = BLACK) -> None:
    if label is not None:
        label.color = color
        label.set_float("fsize", size)


def add_panel(day: str, groups: dict[str, list[dict[str, float | int]]], show_legend: bool):
    values = [[float(row["di"]) for row in groups[group]] for group in ("CON", "SUR")]
    stats = [mean_sem(group_values) for group_values in values]

    wks = op.new_sheet("w", lname=f"NORDIData{day}", hidden=False)
    wks.cols = 9
    wks.from_list(0, values[0], lname="CON raw DI", axis="Y")
    wks.from_list(1, values[1], lname="SUR raw DI", axis="Y")
    wks.from_list(2, BAR_X, lname="Bar X", axis="X")
    wks.from_list(3, [item[0] for item in stats], lname="Mean from raw", axis="Y")
    wks.from_list(4, [item[1] for item in stats], lname="SEM from raw", axis="E")
    for index, group in enumerate(("CON", "SUR")):
        base = 5 + index * 2
        wks.from_list(base, collision_aware_x(BAR_X[index], values[index]), lname=f"{group} point X", axis="X")
        wks.from_list(base + 1, values[index], lname=f"{group} points", axis="Y")

    graph = op.new_graph(lname=f"NOR discrimination index | {day}", template="scatter")
    graph.name = f"NORDIGraph{day}"
    layer = graph[0]

    bar_plot = layer.add_plot(wks, colx=2, coly=3, type="c")
    bar_index = bar_plot.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bar_index}; set rb -vg 40; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;")
    for point_index, (fill, edge) in enumerate(((CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE)), start=1):
        op.lt_exec(
            f"range rb=[{graph.name}]1!{bar_index}; set rb {point_index} -pbc {rgb_expr(edge)}; "
            f"set rb {point_index} -pfb {rgb_expr(fill)}; set rb {point_index} -pfc {rgb_expr(fill)};"
        )

    for index, edge in enumerate((CON_EDGE, SUR_EDGE)):
        base = 5 + index * 2
        outer = layer.add_plot(wks, colx=base, coly=base + 1, type="s")
        outer_index = outer.index() + 1
        op.lt_exec(
            f"range rr=[{graph.name}]1!{outer_index}; set rr -k 2; set rr -z {POINT_SIZE}; "
            f"set rr -kf 1; set rr -kh {POINT_EDGE_PERCENT}; "
            f"set rr -cse {rgb_expr(edge)}; set rr -csf {rgb_expr(WHITE)};"
        )

    error_carrier = layer.add_plot(wks, colx=2, coly=3, colyerr=4, type="s")
    carrier_index = error_carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_plot = layer.plot_list()[-1]
    error_index = error_plot.index() + 1
    op.lt_exec(
        f"range re=[{graph.name}]1!{error_index}; set re -c {rgb_expr(BLACK)}; "
        f"set re -erw {LINE_WIDTH}; set re -erwc 15; set re -erdy 0;"
    )

    layer.axis("x").set_limits(0.55, 1.45, 0.5)
    layer.axis("y").set_limits(-0.8, 1.0, 0.5)
    layer.axis("x").title = ""
    layer.axis("y").title = "Discrimination index"
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=1; layer.y.showAxes=1; layer.x2.showAxes=0; layer.y2.showAxes=0; "
        "layer.x.showLabels=0; layer.y.showLabels=1; layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={LINE_WIDTH}; layer.y.thickness={LINE_WIDTH}; layer.x.ticks=0; layer.y.ticks=1; "
        f"layer.y.label.fsize={MAIN_FONT_SIZE}; layer.y.tickLabels.fsize={MAIN_FONT_SIZE};"
    )

    zero = layer.add_line(0.55, 0, 1.45, 0)
    zero.color = BLACK
    zero.width = LINE_WIDTH

    # Origin's exported native spine is not reliable with borderless scatter
    # templates. Draw one explicit, scale-attached spine set and outward ticks.
    for axis_line in (
        layer.add_line(0.55, -0.8, 1.45, -0.8),
        layer.add_line(0.55, -0.8, 0.55, 1.0),
    ):
        axis_line.color = BLACK
        axis_line.width = LINE_WIDTH
    for y in (-0.5, 0.0, 0.5, 1.0):
        tick = layer.add_line(0.55, y, 0.532, y)
        tick.color = BLACK
        tick.width = LINE_WIDTH

    for text, x in (("CON", BAR_X[0]), ("SUR", BAR_X[1])):
        # Origin positions text by its bounding box; the small positive offset
        # places the visual text center exactly below the manual group tick.
        label = layer.add_label(text, x + 0.05, -0.94)
        style_label(label, MAIN_FONT_SIZE)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)
        tick = layer.add_line(x, -0.8, x, -0.83)
        tick.color = BLACK
        tick.width = LINE_WIDTH

    title = layer.add_label(day.replace("Day", "Day "), 1.0, 0.94)
    style_label(title, MAIN_FONT_SIZE)
    title.set_int("attach", 2)
    title.set_int("anchor", 8)
    style_label(layer.label("YL"), MAIN_FONT_SIZE)

    # Welch tests were reproduced from the earlier Python figure: Day 3
    # p=0.044785..., Day 7 p=0.018212..., so both panels carry one star.
    for x0, y0, x1, y1 in ((BAR_X[0], 0.52, BAR_X[0], 0.60), (BAR_X[0], 0.60, BAR_X[1], 0.60), (BAR_X[1], 0.60, BAR_X[1], 0.52)):
        bracket = layer.add_line(x0, y0, x1, y1)
        bracket.color = BLACK
        bracket.width = LINE_WIDTH
    star = layer.add_label("*", 1.0, 0.66)
    style_label(star, MAIN_FONT_SIZE)
    star.set_int("attach", 2)
    star.set_int("anchor", 8)

    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    if show_legend:
        for color, y in ((CON_FILL, 0.91), (SUR_FILL, 0.75)):
            label = layer.add_label("■", 1.26, y)
            style_label(label, MAIN_FONT_SIZE, color)
            label.set_int("attach", 2)
            label.set_int("anchor", 4)
        for text, y in (("CON", 0.91), ("SUR", 0.75)):
            label = layer.add_label(text, 1.34, y)
            style_label(label, LEGEND_FONT_SIZE)
            label.set_int("attach", 2)
            label.set_int("anchor", 4)

    graph.set_float("width", 4200)
    graph.set_float("height", 5000)
    layer.set_float("left", 21)
    layer.set_float("top", 11)
    layer.set_float("width", 68)
    layer.set_float("height", 72)
    return graph, stats


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    op.attach()
    op.set_show(True)
    op.new(asksave=False)
    for day, show_legend in (("Day3", False), ("Day7", True)):
        graph, stats = add_panel(day, payload[day], show_legend)
        stem = OUTPUT_DIR / f"NOR_DI_{day}"
        graph.save_fig(str(stem.with_suffix(".png")), type="png", width=2400)
        graph.save_fig(str(stem.with_suffix(".pdf")), type="pdf")
        svg_path = stem.with_suffix(".svg")
        graph.save_fig(str(svg_path), type="svg")
        normalize_svg(svg_path)
        shutil.copy2(svg_path, OUTPUT_DIR / f"{stem.name}_AI_clean.svg")
        print(day, stats)
    op.save(str(PROJECT_PATH))
    print(PROJECT_PATH)
    op.detach()


if __name__ == "__main__":
    main()
