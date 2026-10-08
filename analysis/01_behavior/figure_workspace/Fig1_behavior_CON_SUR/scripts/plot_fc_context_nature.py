from __future__ import annotations

import csv
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import originpro as op


SOURCE_PZFX = Path(r"F:\1.Sleep\eXdata\BehaviorTest\SurPostDay3\FC\FC.pzfx")
OUTPUT_DIR = Path(r"F:\Sleep\Figure\Sur_behavior_Day3_Day7_Origin_Nature_style_20260904\05_FC_context")
STEM = "Sur_FC_context_freezing_Day3_Day7"
PROJECT_PATH = OUTPUT_DIR / f"{STEM}.opju"
GRAPH_NAME = "FCGraph"

CON_FILL = (130, 201, 232)
CON_EDGE = (46, 117, 145)
SUR_FILL = (240, 111, 116)
SUR_EDGE = (176, 55, 64)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

LINE_WIDTH = 2.5
POINT_SIZE = 14.0
POINT_EDGE_WIDTH = 2.5
POINT_EDGE_PERCENT = 100 * POINT_EDGE_WIDTH / (POINT_SIZE / 2)
POINT_X_STEP = 0.045
POINT_COLLISION_Y = 2.5
MAIN_FONT = 21.5
LEGEND_FONT = 20.0
BAR_X = [0.84, 1.16, 1.84, 2.16]
DAYS = ["Day3", "Day3", "Day7", "Day7"]
GROUPS = ["CON", "SUR", "CON", "SUR"]


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


def title_text(node: ET.Element, ns: dict[str, str]) -> str:
    return "".join(node.itertext()).strip()


def read_context_values() -> list[list[float]]:
    root = ET.parse(SOURCE_PZFX).getroot()
    ns = {"p": "http://graphpad.com/prism/Prism.htm"}
    cells: list[list[float]] = []
    for table_title in ("DAY3CONTEXT", "DAY7CONTEXT"):
        table = next(
            table for table in root.findall("p:Table", ns)
            if table.findtext("p:Title", default="", namespaces=ns) == table_title
        )
        by_group: dict[str, list[float]] = {}
        for column in table.findall("p:YColumn", ns):
            group = title_text(column.find("p:Title", ns), ns)
            by_group[group] = [float(node.text) for node in column.findall("p:Subcolumn/p:d", ns)]
        for group in ("CON", "SUR"):
            values = by_group[group]
            if len(values) != 6:
                raise RuntimeError(f"{table_title} {group}: expected n=6")
            cells.append(values)
    return cells


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise RuntimeError("Incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x in (0.0, 1.0):
        return x
    log_term = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(log_term + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b


def welch_test(first: list[float], second: list[float]) -> tuple[float, float, float]:
    mean1, mean2 = sum(first) / len(first), sum(second) / len(second)
    var1 = sum((value - mean1) ** 2 for value in first) / (len(first) - 1)
    var2 = sum((value - mean2) ** 2 for value in second) / (len(second) - 1)
    term1, term2 = var1 / len(first), var2 / len(second)
    t_value = (mean1 - mean2) / math.sqrt(term1 + term2)
    degrees = (term1 + term2) ** 2 / (term1 ** 2 / (len(first) - 1) + term2 ** 2 / (len(second) - 1))
    p_value = regularized_incomplete_beta(degrees / 2.0, 0.5, degrees / (degrees + t_value ** 2))
    return t_value, degrees, p_value


def rgb_expr(rgb: tuple[int, int, int]) -> str:
    return f"color({rgb[0]},{rgb[1]},{rgb[2]})"


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
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
    text = re.sub(r'(?<=[="\' :;])black(?=[;"\' ])', "rgb(0,0,0)", text)
    text = re.sub(r'(?<=[="\' :;])white(?=[;"\' ])', "rgb(255,255,255)", text)
    text = re.sub(
        r"#([0-9A-Fa-f]{6})",
        lambda match: "rgb(" + ",".join(
            str(int(match.group(1)[index:index + 2], 16)) for index in (0, 2, 4)
        ) + ")",
        text,
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def style_label(label, size: float, color: tuple[int, int, int] = BLACK, *, bold: bool = False) -> None:
    if label is None:
        return
    label.color = color
    label.set_float("fsize", size)
    label.set_int("fontbold", int(bold))


def add_line(layer, x1: float, y1: float, x2: float, y2: float):
    line = layer.add_line(x1, y1, x2, y2)
    line.color = BLACK
    line.width = LINE_WIDTH
    return line


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = read_context_values()
    day3 = {"CON": cells[0], "SUR": cells[1]}
    day7 = {"CON": cells[2], "SUR": cells[3]}
    identifiers = [
        [f"{day} {group} animal {index}" for index in range(1, 7)]
        for day, group in zip(DAYS, GROUPS, strict=True)
    ]
    stats = [mean_sem(values) for values in cells]
    tests = {"Day3": welch_test(day3["CON"], day3["SUR"]),
             "Day7": welch_test(day7["CON"], day7["SUR"])}

    source_csv = OUTPUT_DIR / "fc_context_freezing_source_data.csv"
    with source_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "group", "subject_index", "context_freezing_percent"])
        for day, group, names, values in zip(DAYS, GROUPS, identifiers, cells, strict=True):
            for index, value in enumerate(values, start=1):
                writer.writerow([day, group, index, f"{value:.8f}"])

    stats_csv = OUTPUT_DIR / "fc_context_freezing_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "comparison", "test", "n_CON", "n_SUR", "mean_CON", "sem_CON",
                         "mean_SUR", "sem_SUR", "t", "df", "p_two_sided", "symbol"])
        for day, indices in (("Day3", (0, 1)), ("Day7", (2, 3))):
            t_value, degrees, p_value = tests[day]
            con_mean, con_sem = stats[indices[0]]
            sur_mean, sur_sem = stats[indices[1]]
            writer.writerow([day, "CON vs SUR", "Welch independent-samples t-test", 6, 6,
                             f"{con_mean:.8f}", f"{con_sem:.8f}", f"{sur_mean:.8f}", f"{sur_sem:.8f}",
                             f"{t_value:.8f}", f"{degrees:.8f}", f"{p_value:.10f}",
                             "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."])

    op.attach()
    op.set_show(True)
    op.new(asksave=False)

    wks = op.new_sheet("w", lname="FCRawAndSummary", hidden=False)
    wks.cols = 19
    for index, (values, day, group) in enumerate(zip(cells, DAYS, GROUPS, strict=True)):
        wks.from_list(index, values, lname=f"{day} {group} raw", units="%", axis="Y")
    wks.from_list(4, BAR_X, lname="Bar X", axis="X")
    wks.from_list(5, [mean for mean, _ in stats], lname="Mean from raw", units="%", axis="Y")
    wks.from_list(6, [sem for _, sem in stats], lname="SEM from raw", units="%", axis="E")
    for index, (x_value, values, day, group) in enumerate(zip(BAR_X, cells, DAYS, GROUPS, strict=True)):
        base = 7 + index * 2
        wks.from_list(base, collision_aware_x(x_value, values), lname=f"{day} {group} point X", axis="X")
        wks.from_list(base + 1, values, lname=f"{day} {group} points", units="%", axis="Y")
    for index, (names, day, group) in enumerate(zip(identifiers, DAYS, GROUPS, strict=True), start=15):
        wks.from_list(index, names, lname=f"{day} {group} animal", axis="N")

    graph = op.new_graph(lname="FC context freezing | Day 3 and Day 7", template="scatter")
    graph.name = GRAPH_NAME
    layer = graph[0]

    bar_plot = layer.add_plot(wks, colx=4, coly=5, type="c")
    bar_plot.color = CON_FILL
    bar_index = bar_plot.index() + 1
    op.lt_exec(f"range rb=[{GRAPH_NAME}]1!{bar_index}; set rb -vg 50; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;")
    colors = [(CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE), (CON_FILL, CON_EDGE), (SUR_FILL, SUR_EDGE)]
    for point_index, (fill, edge) in enumerate(colors, start=1):
        op.lt_exec(
            f"range rb=[{GRAPH_NAME}]1!{bar_index}; "
            f"set rb {point_index} -pbc {rgb_expr(edge)}; "
            f"set rb {point_index} -pfb {rgb_expr(fill)}; "
            f"set rb {point_index} -pfc {rgb_expr(fill)};"
        )

    for index, edge in enumerate((CON_EDGE, SUR_EDGE, CON_EDGE, SUR_EDGE)):
        base = 7 + index * 2
        outer = layer.add_plot(wks, colx=base, coly=base + 1, type="s")
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
        f"range re=[{GRAPH_NAME}]1!{error_index}; set re -c {rgb_expr(BLACK)}; "
        f"set re -erw {LINE_WIDTH}; set re -erwc 15; set re -erdy 0;"
    )

    layer.axis("x").set_limits(0.55, 2.45, 0.5)
    layer.axis("y").set_limits(0, 110, 20)
    layer.axis("x").title = ""
    layer.axis("y").title = "Freezing time (%)"
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=1; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={LINE_WIDTH}; layer.y.thickness={LINE_WIDTH}; "
        f"layer.x.ticks=0; layer.y.ticks=1; layer.y.label.fsize={MAIN_FONT}; layer.y.tickLabels.fsize={MAIN_FONT};"
    )

    add_line(layer, 0.55, 0, 2.45, 0)
    add_line(layer, 0.55, 0, 0.55, 110)
    for y_value in (0, 20, 40, 60, 80, 100):
        add_line(layer, 0.55, y_value, 0.525, y_value)

    for text, x_value in (("DAY3", 1.0), ("DAY7", 2.0)):
        label = layer.add_label(text, x_value + 0.05, -10.0)
        style_label(label, MAIN_FONT)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)
        add_line(layer, x_value, 0, x_value, -3.0)

    style_label(layer.label("YL"), MAIN_FONT)

    for day, x1, x2 in (("Day3", 0.78, 1.22), ("Day7", 1.78, 2.22)):
        add_line(layer, x1, 97, x1, 101)
        add_line(layer, x1, 101, x2, 101)
        add_line(layer, x2, 101, x2, 97)
        p_value = tests[day][2]
        symbol = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        star = layer.add_label(symbol, (x1 + x2) / 2.0 + 0.05, 106)
        style_label(star, MAIN_FONT, bold=True)
        star.set_int("attach", 2)
        star.set_int("anchor", 8)

    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    for color, y_value in ((CON_FILL, 104), (SUR_FILL, 95)):
        label = layer.add_label("■", 2.48, y_value)
        style_label(label, MAIN_FONT, color)
        label.set_int("attach", 2)
        label.set_int("anchor", 4)
    for text, y_value in (("CON", 104), ("SUR", 95)):
        label = layer.add_label(text, 2.60, y_value)
        style_label(label, LEGEND_FONT)
        label.set_int("attach", 2)
        label.set_int("anchor", 4)

    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    layer.set_float("left", 18)
    layer.set_float("top", 11)
    layer.set_float("width", 64)
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

    for index, ((mean, sem), values) in enumerate(zip(stats, cells, strict=True), start=1):
        print(f"cell={index} n={len(values)} mean={mean:.8f} sem={sem:.8f}")
    for day, (t_value, degrees, p_value) in tests.items():
        print(f"{day} Welch_t={t_value:.8f} df={degrees:.8f} p={p_value:.10f}")
    print(f"project={PROJECT_PATH}")
    print(f"png={png_path}")
    op.detach()


if __name__ == "__main__":
    main()
