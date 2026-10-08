from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import originpro as op


OUTPUT_DIR = Path(r"F:\Sleep\Figure\Sur_behavior_Day3_Day7_Origin_Nature_style_20260904\02_OF_zone_time")
JSON_PATH = OUTPUT_DIR / "of_zone_time_raw.json"
PROJECT_PATH = OUTPUT_DIR / "SurOF_zone_time_Day3_Day7.opju"

CON_FILL = (130, 201, 232)
CON_EDGE = (46, 117, 145)
SUR_FILL = (240, 111, 116)
SUR_EDGE = (176, 55, 64)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BAR_X = [0.84, 1.16, 1.84, 2.16]
REGIONS = ["Center", "Center", "Border", "Border"]
GROUPS = ["CON", "SUR", "CON", "SUR"]
POINT_SIZE = 14.0
POINT_EDGE_WIDTH = 2.5
POINT_EDGE_PERCENT = 100 * POINT_EDGE_WIDTH / (POINT_SIZE / 2)
POINT_X_STEP = 0.045
POINT_COLLISION_Y = 8.0
LINE_WIDTH = 2.5
MAIN_FONT = 21.5
LEGEND_FONT = 20.0


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
        lambda match: "rgb(" + ",".join(
            str(int(match.group(1)[index:index + 2], 16)) for index in (0, 2, 4)
        ) + ")",
        text,
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def style_label(label, size: float, color: tuple[int, int, int] = BLACK) -> None:
    if label is not None:
        label.color = color
        label.set_float("fsize", size)


def add_panel(day: str, day_data: dict[str, dict[str, list[float]]], show_legend: bool):
    values_by_cell = [day_data[region][group] for region, group in zip(REGIONS, GROUPS, strict=True)]
    if [len(values) for values in values_by_cell] != [8, 8, 8, 8]:
        raise RuntimeError(f"{day}: expected n=8 in each cell")
    stats = [mean_sem(values) for values in values_by_cell]

    wks = op.new_sheet("w", lname=f"ZoneTime{day}", hidden=False)
    wks.cols = 15
    for index, (values, region, group) in enumerate(zip(values_by_cell, REGIONS, GROUPS, strict=True)):
        wks.from_list(index, values, lname=f"{region} {group} raw", units="s", axis="Y")
    wks.from_list(4, BAR_X, lname="Bar X", axis="X")
    wks.from_list(5, [mean for mean, _ in stats], lname="Mean from raw", units="s", axis="Y")
    wks.from_list(6, [sem for _, sem in stats], lname="SEM from raw", units="s", axis="E")
    for index, (x, values, region, group) in enumerate(zip(BAR_X, values_by_cell, REGIONS, GROUPS, strict=True)):
        base = 7 + index * 2
        wks.from_list(base, collision_aware_x(x, values), lname=f"{region} {group} point X", axis="X")
        wks.from_list(base + 1, values, lname=f"{region} {group} points", units="s", axis="Y")

    graph = op.new_graph(lname=f"OF zone time | {day}", template="scatter")
    graph.name = f"Zone{day}"
    layer = graph[0]

    bar_plot = layer.add_plot(wks, colx=4, coly=5, type="c")
    bar_plot.color = CON_FILL
    bar_index = bar_plot.index() + 1
    op.lt_exec(
        f"range rb=[{graph.name}]1!{bar_index}; "
        f"set rb -vg 50; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;"
    )
    colors = [(CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE), (CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE)]
    for point_index, (fill, edge) in enumerate(colors, start=1):
        op.lt_exec(
            f"range rb=[{graph.name}]1!{bar_index}; "
            f"set rb {point_index} -pbc {rgb_expr(edge)}; "
            f"set rb {point_index} -pfb {rgb_expr(fill)}; "
            f"set rb {point_index} -pfc {rgb_expr(fill)};"
        )

    for index, edge in enumerate((CON_EDGE, SUR_EDGE, CON_EDGE, SUR_EDGE)):
        base = 7 + index * 2
        outer = layer.add_plot(wks, colx=base, coly=base + 1, type="s")
        outer_index = outer.index() + 1
        op.lt_exec(
            f"range rr=[{graph.name}]1!{outer_index}; set rr -k 2; set rr -z {POINT_SIZE}; "
            f"set rr -kf 1; set rr -kh {POINT_EDGE_PERCENT}; "
            f"set rr -cse {rgb_expr(edge)}; set rr -csf {rgb_expr(WHITE)};"
        )

    error_carrier = layer.add_plot(wks, colx=4, coly=5, colyerr=6, type="s")
    carrier_index = error_carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_plot = layer.plot_list()[-1]
    error_plot.color = BLACK
    error_index = error_plot.index() + 1
    op.lt_exec(
        f"range re=[{graph.name}]1!{error_index}; "
        f"set re -c {rgb_expr(BLACK)}; set re -erw {LINE_WIDTH}; "
        "set re -erwc 15; set re -erdy 0;"
    )

    layer.axis("x").set_limits(0.55, 2.45, 0.5)
    layer.axis("y").set_limits(0, 420, 100)
    layer.axis("x").title = ""
    layer.axis("y").title = "Time in specific region (s)"
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=1; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; "
        "layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={LINE_WIDTH}; layer.y.thickness={LINE_WIDTH}; "
        f"layer.x.ticks=0; layer.y.ticks=1; layer.y.label.fsize={MAIN_FONT}; layer.y.tickLabels.fsize={MAIN_FONT};"
    )

    for line in (layer.add_line(0.55, 0, 2.45, 0), layer.add_line(0.55, 0, 0.55, 420)):
        line.color = BLACK
        line.width = LINE_WIDTH
    for y in (0, 100, 200, 300, 400):
        tick = layer.add_line(0.55, y, 0.525, y)
        tick.color = BLACK
        tick.width = LINE_WIDTH

    for text, x in (("Center", 1.0), ("Border", 2.0)):
        label = layer.add_label(text, x + 0.05, -38)
        style_label(label, MAIN_FONT)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)
        tick = layer.add_line(x, 0, x, -12)
        tick.color = BLACK
        tick.width = LINE_WIDTH

    title = layer.add_label(day.replace("Day", "Day "), 1.55, 410)
    style_label(title, MAIN_FONT)
    title.set_int("attach", 2)
    title.set_int("anchor", 8)
    style_label(layer.label("YL"), MAIN_FONT)

    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    if show_legend:
        for square, color, y in (("■", CON_FILL, 400), ("■", SUR_FILL, 365)):
            label = layer.add_label(square, 2.48, y)
            style_label(label, MAIN_FONT, color)
            label.set_int("attach", 2)
            label.set_int("anchor", 4)
        for text, y in (("CON", 400), ("SUR", 365)):
            label = layer.add_label(text, 2.60, y)
            style_label(label, LEGEND_FONT)
            label.set_int("attach", 2)
            label.set_int("anchor", 4)

    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    layer.set_float("left", 18)
    layer.set_float("top", 11)
    layer.set_float("width", 64)
    layer.set_float("height", 72)
    return graph, stats, values_by_cell


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    op.attach()
    op.set_show(True)
    op.new(asksave=False)

    outputs = []
    for day, show_legend in (("Day3", False), ("Day7", True)):
        graph, stats, cells = add_panel(day, payload[day], show_legend)
        stem = OUTPUT_DIR / f"SurOF_zone_time_{day}"
        graph.save_fig(str(stem.with_suffix(".png")), type="png", width=2400)
        graph.save_fig(str(stem.with_suffix(".pdf")), type="pdf")
        svg_path = stem.with_suffix(".svg")
        graph.save_fig(str(svg_path), type="svg")
        normalize_svg(svg_path)
        shutil.copy2(svg_path, OUTPUT_DIR / f"{stem.name}_AI_clean.svg")
        outputs.append((day, stats, [len(values) for values in cells]))

    op.save(str(PROJECT_PATH))
    for day, stats, counts in outputs:
        print(f"{day}: counts={counts}")
        for region, group, (mean, sem) in zip(REGIONS, GROUPS, stats, strict=True):
            print(f"{day} {region} {group}: mean={mean:.8f}, sem={sem:.8f}")
    print(f"project={PROJECT_PATH}")
    op.detach()


if __name__ == "__main__":
    main()
