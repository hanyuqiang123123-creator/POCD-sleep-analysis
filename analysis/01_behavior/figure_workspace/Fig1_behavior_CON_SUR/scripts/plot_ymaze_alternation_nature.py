from __future__ import annotations

import csv
import math
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import originpro as op


DAY3_XLSX = Path(r"F:\1.Sleep\eXdata\BehaviorTest\SurPostDay3\Ymaze3\SUR3.xlsx")
DAY7_XLSX = Path(r"F:\1.Sleep\eXdata\BehaviorTest\SurPostDay7\YMAZE7\ymaze7.xlsx")
OUTPUT_DIR = Path(r"F:\Sleep\Figure\Sur_behavior_Day3_Day7_Origin_Nature_style_20260904\04_Ymaze")
STEM = "Sur_Ymaze_spontaneous_alternation_Day3_Day7"
PROJECT_PATH = OUTPUT_DIR / f"{STEM}.opju"
GRAPH_NAME = "YmazeGraph"

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


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_xlsx_rows(path: Path) -> list[dict[str, object]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find("m:sheets/m:sheet", ns)
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(rel.attrib["Target"] for rel in relationships.findall("r:Relationship", rel_ns)
                      if rel.attrib["Id"] == rel_id)
        sheet_path = "xl/" + target.lstrip("/")
        sheet = ET.fromstring(archive.read(sheet_path))

        matrix: list[list[object]] = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            cells: dict[int, object] = {}
            for cell in row.findall("m:c", ns):
                idx = column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iterfind(".//m:t", ns))
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = float(value_node.text)
                cells[idx] = value
            width = max(cells, default=-1) + 1
            matrix.append([cells.get(i, "") for i in range(width)])

    headers = [str(value) for value in matrix[0]]
    return [dict(zip(headers, row, strict=False)) for row in matrix[1:]]


def read_day(path: Path) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    values = {"CON": [], "SUR": []}
    identifiers = {"CON": [], "SUR": []}
    for row in read_xlsx_rows(path):
        animal = str(row["文件名"])
        match = re.match(r"^(CON|SUR)", animal)
        if match is None:
            raise RuntimeError(f"Unrecognized group in {path}: {animal}")
        group = match.group(1)
        rate = float(row["正确率"]) * 100.0
        identifiers[group].append(animal)
        values[group].append(rate)
    if [len(values[group]) for group in ("CON", "SUR")] != [8, 8]:
        raise RuntimeError(f"Expected n=8/group in {path}")
    return values, identifiers


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


def pooled_factorial_analysis(cells: list[list[float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Balanced 2 x 2 independent-groups ANOVA and Holm-adjusted simple effects."""
    cell_means = [sum(values) / len(values) for values in cells]
    residual_ss = sum(
        sum((value - cell_mean) ** 2 for value in values)
        for values, cell_mean in zip(cells, cell_means, strict=True)
    )
    residual_df = sum(len(values) for values in cells) - len(cells)
    mse = residual_ss / residual_df

    grand_mean = sum(sum(values) for values in cells) / sum(len(values) for values in cells)
    day_means = [(cell_means[0] + cell_means[1]) / 2.0,
                 (cell_means[2] + cell_means[3]) / 2.0]
    group_means = [(cell_means[0] + cell_means[2]) / 2.0,
                   (cell_means[1] + cell_means[3]) / 2.0]
    n_cell = len(cells[0])
    ss_day = 2 * n_cell * sum((value - grand_mean) ** 2 for value in day_means)
    ss_group = 2 * n_cell * sum((value - grand_mean) ** 2 for value in group_means)
    ss_interaction = n_cell * sum(
        (cell_means[row * 2 + column] - day_means[row] - group_means[column] + grand_mean) ** 2
        for row in range(2) for column in range(2)
    )
    anova = []
    for effect, ss in (("Day", ss_day), ("Group", ss_group), ("Day x Group", ss_interaction)):
        f_value = ss / mse
        p_value = regularized_incomplete_beta(residual_df / 2.0, 0.5,
                                              residual_df / (residual_df + f_value))
        anova.append({"effect": effect, "ss": ss, "df": 1.0, "f": f_value, "p": p_value})
    anova.append({"effect": "Residual", "ss": residual_ss, "df": float(residual_df),
                  "f": math.nan, "p": math.nan})

    contrast_se = math.sqrt(mse * (1.0 / n_cell + 1.0 / n_cell))
    contrasts = []
    for day, con_index, sur_index in (("Day3", 0, 1), ("Day7", 2, 3)):
        difference = cell_means[con_index] - cell_means[sur_index]
        t_value = difference / contrast_se
        p_value = regularized_incomplete_beta(residual_df / 2.0, 0.5,
                                              residual_df / (residual_df + t_value ** 2))
        contrasts.append({"day": day, "difference": difference, "se": contrast_se,
                          "t": t_value, "df": float(residual_df), "p_raw": p_value})

    order = sorted(range(len(contrasts)), key=lambda index: contrasts[index]["p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(contrasts) - rank) * contrasts[index]["p_raw"])
        running = max(running, adjusted)
        contrasts[index]["p_holm"] = running
    return anova, contrasts


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
    day3, ids3 = read_day(DAY3_XLSX)
    day7, ids7 = read_day(DAY7_XLSX)
    cells = [day3["CON"], day3["SUR"], day7["CON"], day7["SUR"]]
    identifiers = [ids3["CON"], ids3["SUR"], ids7["CON"], ids7["SUR"]]
    stats = [mean_sem(values) for values in cells]
    sensitivity_tests = {"Day3": welch_test(day3["CON"], day3["SUR"]),
                         "Day7": welch_test(day7["CON"], day7["SUR"])}
    anova, contrasts = pooled_factorial_analysis(cells)

    source_csv = OUTPUT_DIR / "ymaze_alternation_source_data.csv"
    with source_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "group", "animal_file", "spontaneous_alternation_percent"])
        for day, group, names, values in zip(DAYS, GROUPS, identifiers, cells, strict=True):
            for name, value in zip(names, values, strict=True):
                writer.writerow([day, group, name, f"{value:.8f}"])

    stats_csv = OUTPUT_DIR / "ymaze_alternation_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "comparison", "primary_test", "n_CON", "n_SUR", "mean_CON", "sem_CON",
                         "mean_SUR", "sem_SUR", "Welch_t", "Welch_df", "Welch_p_two_sided", "symbol",
                         "factorial_simple_difference", "factorial_pooled_SE", "factorial_simple_t",
                         "factorial_residual_df", "factorial_p_raw", "factorial_p_holm"])
        for contrast, indices in zip(contrasts, ((0, 1), (2, 3)), strict=True):
            day = str(contrast["day"])
            con_mean, con_sem = stats[indices[0]]
            sur_mean, sur_sem = stats[indices[1]]
            welch_t, welch_df, welch_p = sensitivity_tests[day]
            symbol = "***" if welch_p < 0.001 else "**" if welch_p < 0.01 else "*" if welch_p < 0.05 else "n.s."
            writer.writerow([day, "CON vs SUR", "two-sided Welch independent-samples t-test", 8, 8,
                             f"{con_mean:.8f}", f"{con_sem:.8f}", f"{sur_mean:.8f}", f"{sur_sem:.8f}",
                             f"{welch_t:.8f}", f"{welch_df:.8f}", f"{welch_p:.10f}", symbol,
                             f"{contrast['difference']:.8f}", f"{contrast['se']:.8f}",
                             f"{contrast['t']:.8f}", f"{contrast['df']:.0f}",
                             f"{contrast['p_raw']:.10f}", f"{contrast['p_holm']:.10f}"])

    anova_csv = OUTPUT_DIR / "ymaze_factorial_anova.csv"
    with anova_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["effect", "sum_squares", "df", "mean_square", "F", "p"])
        residual_row = anova[-1]
        residual_mse = residual_row["ss"] / residual_row["df"]
        for row in anova:
            writer.writerow([row["effect"], f"{row['ss']:.8f}", f"{row['df']:.0f}",
                             f"{(row['ss'] / row['df']):.8f}",
                             "" if math.isnan(row["f"]) else f"{row['f']:.8f}",
                             "" if math.isnan(row["p"]) else f"{row['p']:.10f}"])

    op.attach()
    op.set_show(True)
    op.new(asksave=False)

    wks = op.new_sheet("w", lname="YmazeRawAndSummary", hidden=False)
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

    graph = op.new_graph(lname="Y-maze spontaneous alternation | Day 3 and Day 7", template="scatter")
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
    layer.axis("y").set_limits(0, 105, 20)
    layer.axis("x").title = ""
    layer.axis("y").title = "Spontaneous alternation (%)"
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=1; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        f"layer.x.thickness={LINE_WIDTH}; layer.y.thickness={LINE_WIDTH}; "
        f"layer.x.ticks=0; layer.y.ticks=1; layer.y.label.fsize={MAIN_FONT}; layer.y.tickLabels.fsize={MAIN_FONT};"
    )

    add_line(layer, 0.55, 0, 2.45, 0)
    add_line(layer, 0.55, 0, 0.55, 105)
    for y_value in (0, 20, 40, 60, 80, 100):
        add_line(layer, 0.55, y_value, 0.525, y_value)

    for text, x_value in (("DAY3", 1.0), ("DAY7", 2.0)):
        label = layer.add_label(text, x_value + 0.05, -10.0)
        style_label(label, MAIN_FONT)
        label.set_int("attach", 2)
        label.set_int("anchor", 8)
        add_line(layer, x_value, 0, x_value, -3.0)

    style_label(layer.label("YL"), MAIN_FONT)

    significant_days = {day for day, (_, _, p_value) in sensitivity_tests.items() if p_value < 0.05}
    for day, x1, x2 in (("Day3", 0.78, 1.22), ("Day7", 1.78, 2.22)):
        if day not in significant_days:
            continue
        add_line(layer, x1, 88, x1, 92)
        add_line(layer, x1, 92, x2, 92)
        add_line(layer, x2, 92, x2, 88)
        star = layer.add_label("*", (x1 + x2) / 2.0 + 0.05, 96)
        style_label(star, MAIN_FONT, bold=True)
        star.set_int("attach", 2)
        star.set_int("anchor", 8)

    legend = layer.label("legend")
    if legend is not None:
        legend.remove()
    for color, y_value in ((CON_FILL, 99), (SUR_FILL, 90)):
        label = layer.add_label("■", 2.48, y_value)
        style_label(label, MAIN_FONT, color)
        label.set_int("attach", 2)
        label.set_int("anchor", 4)
    for text, y_value in (("CON", 99), ("SUR", 90)):
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
    for row in contrasts:
        print(f"{row['day']} factorial_simple_t={row['t']:.8f} df={row['df']:.0f} "
              f"p_raw={row['p_raw']:.10f} p_holm={row['p_holm']:.10f}")
    print(f"project={PROJECT_PATH}")
    print(f"png={png_path}")
    op.detach()


if __name__ == "__main__":
    main()
