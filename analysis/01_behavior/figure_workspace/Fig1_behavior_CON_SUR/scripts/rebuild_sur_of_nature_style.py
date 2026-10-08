from __future__ import annotations

import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import originpro as op


SOURCE_PZFX = Path(r"F:\1.Sleep\eXdata\BehaviorTest\SurPostDay3\SurOF\Totaldistance.pzfx")
OUTPUT_DIR = Path(r"F:\Sleep\Figure\Sur_behavior_Day3_Day7_Origin_Nature_style_20260904\01_OF_total_distance")
STEM = "SurOF_total_distance_Nature_style"
PROJECT_PATH = OUTPUT_DIR / f"{STEM}.opju"
GRAPH_NAME = "Graph1"

CON_FILL = (130, 201, 232)
CON_EDGE = (46, 117, 145)
SUR_FILL = (240, 111, 116)
SUR_EDGE = (176, 55, 64)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# User-requested uniform stroke weight for the final Origin revision.
AXIS_WIDTH = 2.5
TICK_WIDTH = 2.5
BAR_EDGE_WIDTH = 2.5
ERROR_WIDTH = 2.5
POINT_SIZE = 14.0
POINT_EDGE_WIDTH = 2.5
POINT_EDGE_PERCENT = 100 * POINT_EDGE_WIDTH / (POINT_SIZE / 2)
POINT_X_STEP = 0.045
POINT_COLLISION_Y = 1.8

# Higher Origin column-gap value produces narrower bars. This is deliberately
# calibrated at the final 89-mm Nature single-column reduction rather than at
# the large editing canvas.
ORIGIN_BAR_GAP = 50

# Revised proximity hierarchy: group centers 1 and 2, offsets +/-0.16.
# Within-day center distance is 0.32, while the Day3-to-Day7 boundary distance
# is 0.68 (2.125x larger), so CON/SUR read as a pair without touching.
BAR_X = [0.84, 1.16, 1.84, 2.16]
BAR_WIDTH = 0.258
GROUPS = ["CON", "SUR", "CON", "SUR"]
DAYS = ["Day3", "Day3", "Day7", "Day7"]


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


def read_values() -> list[list[float]]:
    root = ET.parse(SOURCE_PZFX).getroot()
    ns = {"p": "http://graphpad.com/prism/Prism.htm"}
    table = next(
        item
        for item in root.findall("p:Table", ns)
        if item.findtext("p:Title", default="", namespaces=ns) == "Total distance (m)"
    )
    days = [d.text or "" for d in table.findall("p:RowTitlesColumn/p:Subcolumn/p:d", ns)]
    by_cell: dict[tuple[str, str], list[float]] = {}
    for ycol in table.findall("p:YColumn", ns):
        group = ycol.findtext("p:Title", default="", namespaces=ns)
        for replicate in ycol.findall("p:Subcolumn", ns):
            for day, value in zip(days, replicate.findall("p:d", ns), strict=True):
                by_cell.setdefault((day, group), []).append(float(value.text))
    return [by_cell[(day, group)] for day, group in zip(DAYS, GROUPS, strict=True)]


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def rgb_expr(rgb: tuple[int, int, int]) -> str:
    return f"color({rgb[0]},{rgb[1]},{rgb[2]})"


def normalize_svg(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    replacements = {
        "#82C9E8": "rgb(130,201,232)",
        "#3F98B9": "rgb(63,152,185)",
        "#F06F74": "rgb(240,111,116)",
        "#C34B50": "rgb(195,75,80)",
        "#000000": "rgb(0,0,0)",
        "#FFFFFF": "rgb(255,255,255)",
    }
    for source, target in replacements.items():
        text = text.replace(source, target).replace(source.lower(), target)
    text = re.sub(r'(?<=[=\"\' :;])black(?=[;\"\' ])', "rgb(0,0,0)", text)
    text = re.sub(r'(?<=[=\"\' :;])white(?=[;\"\' ])', "rgb(255,255,255)", text)
    text = re.sub(
        r"#([0-9A-Fa-f]{6})",
        lambda match: "rgb(" + ",".join(
            str(int(match.group(1)[i:i + 2], 16)) for i in (0, 2, 4)
        ) + ")",
        text,
    )
    svg_path.write_text(text, encoding="utf-8", newline="\n")


def set_label_style(label, size: float, *, bold: bool = False) -> None:
    if label is None:
        return
    label.color = BLACK
    label.set_float("fsize", size)
    label.set_int("fontbold", int(bold))


def add_rectangle(layer, x1: float, y1: float, x2: float, y2: float,
                  fill: tuple[int, int, int], edge: tuple[int, int, int]):
    obj = layer.obj.GraphObjects.Add(8)
    if not obj:
        raise RuntimeError("Origin failed to create a rectangle graph object")
    obj.SetNumProp("attach", 2)
    obj.SetNumProp("x1", x1)
    obj.SetNumProp("y1", y1)
    obj.SetNumProp("x2", x2)
    obj.SetNumProp("y2", y2)
    obj.SetNumProp("fillcolor", op.utils.ocolor(fill))
    obj.SetNumProp("color", op.utils.ocolor(edge))
    obj.SetNumProp("fillpattern", 0)
    obj.SetNumProp("linewidth", 0.7)
    return obj


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values_by_cell = read_values()
    stats = [mean_sem(values) for values in values_by_cell]

    # Preserve one recoverable copy before replacing this revision.
    if PROJECT_PATH.exists():
        archive = OUTPUT_DIR / "archive_previous_revision"
        archive.mkdir(parents=True, exist_ok=True)
        backup = archive / f"{STEM}_before_font_calibration.opju"
        if not backup.exists():
            shutil.copy2(PROJECT_PATH, backup)

    op.attach()
    op.set_show(True)
    op.open(str(PROJECT_PATH))

    old_graph = op.find_graph(GRAPH_NAME)
    if old_graph is not None:
        old_graph.destroy()

    own_books = []
    for book in op.pages("w"):
        if book.name.startswith("OriginStyleD"):
            own_books.append(book)
    for book in own_books:
        book.destroy()

    wks = op.new_sheet("w", lname="OriginStyleData", hidden=False)
    # A-D preserve the four original animal-level columns exactly as read from
    # the Prism PZFX. E-G are the directly derived bar X/mean/SEM columns, and
    # H-O are collision-aware, deterministic raw-point X/Y pairs used by the graph.
    wks.cols = 15
    for i, (values, day, group) in enumerate(zip(values_by_cell, DAYS, GROUPS, strict=True)):
        wks.from_list(i, values, lname=f"{day} {group} raw", units="m", axis="Y")
    wks.from_list(4, BAR_X, lname="Bar X", axis="X")
    wks.from_list(5, [mean for mean, _sem in stats], lname="Mean from raw", units="m", axis="Y")
    wks.from_list(6, [sem for _mean, sem in stats], lname="SEM from raw", units="m", axis="E")
    for i, (x, (mean, sem), values, day, group) in enumerate(
        zip(BAR_X, stats, values_by_cell, DAYS, GROUPS, strict=True)
    ):
        point_base = 7 + i * 2
        wks.from_list(point_base, collision_aware_x(x, values), lname=f"{day} {group} point X", axis="X")
        wks.from_list(point_base + 1, values, lname=f"{day} {group} raw", units="m", axis="Y")

    graph = op.new_graph(lname="SurOF total distance | Python reference match", template="scatter")
    graph.name = GRAPH_NAME
    layer = graph[0]

    # Bars remain at the back. The worksheet-linked SEM plot is added again
    # after all points so its stem and cap are always rendered on top.
    colors = [(CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE),
              (CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE)]
    bar_plot = layer.add_plot(wks, colx=4, coly=5, type="c")
    bar_plot.color = CON_FILL
    bar_index = bar_plot.index() + 1
    op.lt_exec(
        f"range rb=[{GRAPH_NAME}]1!{bar_index}; "
        f"set rb -vg {ORIGIN_BAR_GAP}; set rb -pbw {BAR_EDGE_WIDTH}; set rb -pfp 0;"
    )
    for point_index, (fill, edge) in enumerate(colors, start=1):
        op.lt_exec(
            f"range rb=[{GRAPH_NAME}]1!{bar_index}; "
            f"set rb {point_index} -pbc {rgb_expr(edge)}; "
            f"set rb {point_index} -pfb {rgb_expr(fill)}; "
            f"set rb {point_index} -pfc {rgb_expr(fill)};"
        )

    for i, edge in enumerate([CON_EDGE, SUR_EDGE, CON_EDGE, SUR_EDGE]):
        point_base = 7 + i * 2
        outer = layer.add_plot(wks, colx=point_base, coly=point_base + 1, type="s")
        outer_index = outer.index() + 1
        op.lt_exec(
            f"range rr=[{GRAPH_NAME}]1!{outer_index}; set rr -k 2; set rr -z {POINT_SIZE}; "
            f"set rr -kf 1; set rr -kh {POINT_EDGE_PERCENT}; "
            f"set rr -cse {rgb_expr(edge)}; set rr -csf {rgb_expr(WHITE)};"
        )

    error_carrier = layer.add_plot(wks, colx=4, coly=5, colyerr=6, type="s")
    carrier_index = error_carrier.index() + 1
    op.lt_exec(f"range rc=[{GRAPH_NAME}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_plot = layer.plot_list()[-1]
    error_plot.color = BLACK
    error_index = error_plot.index() + 1
    op.lt_exec(
        f"range re=[{GRAPH_NAME}]1!{error_index}; "
        f"set re -c color(0,0,0); set re -erw {ERROR_WIDTH}; "
        "set re -erwc 15; set re -erdy 0;"
    )

    layer.axis("x").set_limits(0.55, 2.45, 0.5)
    layer.axis("y").set_limits(0, 75, 20)
    layer.axis("y").title = "Total distance (m)"
    layer.axis("x").title = ""
    op.lt_exec(
        "layer.border=0; "
        "layer.x.showAxes=1; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; "
        "layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={AXIS_WIDTH}; layer.y.thickness={AXIS_WIDTH}; "
        "layer.x.ticks=0; layer.y.ticks=1;"
    )

    # Explicit left/bottom axes guarantee the no-frame appearance independent
    # of the locally installed Origin template defaults.
    for line in (
        layer.add_line(0.55, 0, 2.45, 0),
        layer.add_line(0.55, 0, 0.55, 75),
    ):
        line.color = BLACK
        line.width = AXIS_WIDTH
    for y in (0, 20, 40, 60):
        tick = layer.add_line(0.55, y, 0.525, y)
        tick.color = BLACK
        tick.width = TICK_WIDTH

    # Custom one-level categorical labels, matching the Python panel.
    for text, x in (("Day3", 1.0), ("Day7", 2.0)):
        label = layer.add_label(text, x + 0.05, -7.0)
        set_label_style(label, 21.5)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)
        tick = layer.add_line(x, 0, x, -2.0)
        tick.color = BLACK
        tick.width = AXIS_WIDTH

    y_title = layer.label("YL")
    set_label_style(y_title, 24)
    op.lt_exec("layer.y.label.fsize=21.5; layer.y.tickLabels.fsize=21.5;")

    # Manual legend avoids template-dependent grouping and uses exact RGBs.
    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    for square, color, y in (("■", CON_FILL, 70.0), ("■", SUR_FILL, 63.8)):
        label = layer.add_label(square, 2.02, y)
        set_label_style(label, 21.5)
        label.color = color
        label.set_int("attach", 2)
        label.set_int("anchor", 4)
    for text, y in (("CON", 70.0), ("SUR", 63.8)):
        label = layer.add_label(text, 2.13, y)
        set_label_style(label, 20)
        label.set_int("attach", 2)
        label.set_int("anchor", 4)

    # Compact, portrait-like panel close to the supplied crop.
    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    layer.set_float("left", 18)
    layer.set_float("top", 11)
    layer.set_float("width", 70)
    layer.set_float("height", 72)

    op.save(str(PROJECT_PATH))
    png_path = OUTPUT_DIR / f"{STEM}.png"
    pdf_path = OUTPUT_DIR / f"{STEM}.pdf"
    svg_path = OUTPUT_DIR / f"{STEM}.svg"
    graph.save_fig(str(png_path), type="png", width=2400)
    graph.save_fig(str(pdf_path), type="pdf")
    graph.save_fig(str(svg_path), type="svg")
    normalize_svg(svg_path)
    shutil.copy2(svg_path, OUTPUT_DIR / f"{STEM}_AI_clean.svg")

    for i, ((mean, sem), values) in enumerate(zip(stats, values_by_cell, strict=True)):
        print(f"cell={i + 1} n={len(values)} mean={mean:.8f} sem={sem:.8f}")
    print(f"project={PROJECT_PATH}")
    print(f"png={png_path}")
    op.detach()


if __name__ == "__main__":
    main()
