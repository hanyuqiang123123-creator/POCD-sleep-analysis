from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd


GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
XPOS = [0.72, 1.28, 2.72, 3.28]
FILLS = [(169, 220, 233), (102, 182, 205), (244, 190, 166), (238, 138, 85)]
EDGES = [(74, 159, 186), (39, 125, 152), (228, 120, 72), (217, 84, 32)]
PAIR_COLORS = [(154, 202, 216), (238, 169, 137)]
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LINE_WIDTH = 3.0
PAIR_LINE_WIDTH = 1.8
POINT_SIZE = 14.0
# Origin stores marker-edge thickness as a percentage of the 7-pt symbol
# radius. 42.857% therefore gives the same 3.0-pt authoring stroke used by
# bars, axes, error bars and significance brackets.
POINT_EDGE_PERCENT = 42.857
FONT = 21.5
STAR_FONT = 25.0
ANIMAL_LABEL_FONT = 12.0

SPECS = [
    {
        "metric": "spindle_density_per_nrem_min",
        "stem": "Spindle_density",
        "ylabel": "Spindle density\n(/NREM min)",
        "ymax": 3.0,
        "ystep": 1.0,
    },
    {
        "metric": "mean_event_spindle_power_db_vs_nonspindle_nrem_sigma",
        "stem": "Normalized_spindle_power",
        "ylabel": "Spindle power / non-Spindle\nNREM background (dB)",
        "ymax": 8.0,
        "ystep": 2.0,
    },
    {
        "metric": "so_density_per_nrem_min",
        "stem": "SO_density",
        "ylabel": "SO density\n(/NREM min)",
        "ymax": 15.0,
        "ystep": 5.0,
    },
]


def rgb(color: tuple[int, int, int]) -> str:
    return f"color({color[0]},{color[1]},{color[2]})"


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"#([0-9A-Fa-f]{6})",
        lambda m: "rgb(" + ",".join(str(int(m.group(1)[i:i + 2], 16)) for i in (0, 2, 4)) + ")",
        text,
    )
    text = re.sub(r"(?i)\bblack\b", "rgb(0,0,0)", text)
    text = re.sub(r"(?i)\bwhite\b", "rgb(255,255,255)", text)
    path.write_text(text, encoding="utf-8", newline="\n")


def style_label(label, size=FONT, bold=False, color=BLACK) -> None:
    label.color = color
    label.set_float("fsize", float(size))
    label.set_int("fontbold", int(bool(bold)))
    label.set_int("font", 4)


def add_line(layer, x1, y1, x2, y2, width=LINE_WIDTH):
    line = layer.add_line(float(x1), float(y1), float(x2), float(y2))
    line.color = BLACK
    line.width = float(width)
    return line


def add_label(layer, text, x, y, size=FONT, bold=False):
    label = layer.add_label(text, float(x), float(y))
    style_label(label, size=size, bold=bold)
    label.set_int("attach", 2)
    label.set_int("anchor", 8)
    return label


def spread_label_y(values: np.ndarray, ymin: float, ymax: float, min_gap: float) -> np.ndarray:
    """Separate dense labels downward so they do not invade significance brackets."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values)
    placed = values[order].copy()
    for i in range(1, len(placed)):
        placed[i] = min(placed[i], placed[i - 1] - min_gap)
    if len(placed) and placed[0] > ymax:
        placed -= placed[0] - ymax
    if len(placed) and placed[-1] < ymin:
        placed += ymin - placed[-1]
    result = np.empty_like(placed)
    result[order] = placed
    return result


def add_animal_labels(layer, raw_values, raw_animals, ymax):
    """Add readable Origin text labels derived from the worksheet animal-ID columns."""
    for i, (center, values, animals) in enumerate(zip(XPOS, raw_values, raw_animals)):
        label_y = spread_label_y(values, ymax * 0.025, ymax * 0.92, ymax * 0.024)
        side = -1 if i % 2 == 0 else 1
        text_x = center + side * 0.22
        for value, placed_y, animal in zip(values, label_y, animals):
            add_line(layer, center + side * 0.035, float(value), text_x - side * 0.055, float(placed_y), width=0.8)
            add_label(layer, str(animal), text_x, float(placed_y), size=ANIMAL_LABEL_FONT)


def dodge(values: np.ndarray, center: float, step=0.045) -> np.ndarray:
    """Only separate exact display collisions; keep the mean X at the bar center."""
    values = np.asarray(values, dtype=float)
    x = np.full(len(values), center, dtype=float)
    rounded = np.round(values, 4)
    for value in np.unique(rounded):
        idx = np.where(rounded == value)[0]
        if len(idx) > 1:
            offsets = (np.arange(len(idx)) - (len(idx) - 1) / 2.0) * step
            x[idx] += offsets
    return x


def add_bracket(layer, x1, x2, y, height, label):
    add_line(layer, x1, y, x1, y + height)
    add_line(layer, x1, y + height, x2, y + height)
    add_line(layer, x2, y + height, x2, y)
    add_label(layer, label, (x1 + x2) / 2 + 0.02, y + height * 1.75, size=STAR_FONT)


def prepare_sheet(frame: pd.DataFrame, metric: str, summary: pd.DataFrame):
    worksheet = op.new_sheet("w", lname=f"{metric} raw and summary", hidden=False)
    max_n = max(int((frame.group == group).sum()) for group in GROUPS)
    # 19 display/summary columns plus two X/Y columns for every valid animal pair.
    # baseline is intentionally NO1–NO6 because baseline+CNO/NO7 has no current
    # spindle/SO analysis output; POCD is the complete NO1–NO7 cohort.
    expected_pairs = {
        "baseline": [f"NO{i}" for i in range(1, 7)],
        "POCD": [f"NO{i}" for i in range(1, 8)],
    }
    worksheet.cols = 19 + 2 * sum(len(ids) for ids in expected_pairs.values())
    raw_values: list[np.ndarray] = []
    raw_animals: list[list[str]] = []
    point_x_by_group: dict[str, dict[str, float]] = {}
    for i, group in enumerate(GROUPS):
        sub = frame.loc[frame.group == group].sort_values("animal")
        values = sub[metric].to_numpy(float)
        animals = sub["animal"].astype(str).tolist()
        raw_values.append(values)
        raw_animals.append(animals)
        worksheet.from_list(i, values.tolist(), lname=f"{group} raw", axis="Y")
        worksheet.from_list(4 + i, animals, lname=f"{group} animal", axis="N")
    metric_summary = summary.loc[summary.metric == metric].set_index("group").loc[GROUPS]
    worksheet.from_list(8, XPOS, lname="Bar X", axis="X")
    worksheet.from_list(9, metric_summary["mean"].to_numpy(float).tolist(), lname="Arithmetic mean", axis="Y")
    worksheet.from_list(10, metric_summary["sem"].to_numpy(float).tolist(), lname="Sample SEM", axis="E")
    for i, (center, values) in enumerate(zip(XPOS, raw_values)):
        point_x = dodge(values, center)
        point_x_by_group[GROUPS[i]] = dict(zip(raw_animals[i], point_x.tolist()))
        worksheet.from_list(11 + 2 * i, point_x.tolist(), lname=f"{GROUPS[i]} point X", axis="X")
        worksheet.from_list(12 + 2 * i, values.tolist(), lname=f"{GROUPS[i]} points", axis="Y")
    pair_specs = []
    pair_col = 19
    for condition, left_group, right_group, left_x, right_x in (
        ("baseline", "baseline+SAL", "baseline+CNO", XPOS[0], XPOS[1]),
        ("POCD", "POCD+SAL", "POCD+CNO", XPOS[2], XPOS[3]),
    ):
        left = frame.loc[frame.group == left_group].set_index("animal")
        right = frame.loc[frame.group == right_group].set_index("animal")
        actual = [animal for animal in expected_pairs[condition] if animal in left.index and animal in right.index]
        if actual != expected_pairs[condition]:
            raise RuntimeError(
                f"Pairing mismatch for {metric}/{condition}: expected {expected_pairs[condition]}, got {actual}"
            )
        color_index = 0 if condition == "baseline" else 1
        for animal in actual:
            worksheet.from_list(
                pair_col,
                [point_x_by_group[left_group][animal], point_x_by_group[right_group][animal]],
                lname=f"{condition} {animal} pair X",
                axis="X",
            )
            worksheet.from_list(
                pair_col + 1,
                [float(left.at[animal, metric]), float(right.at[animal, metric])],
                lname=f"{condition} {animal} pair Y",
                axis="Y",
            )
            pair_specs.append((pair_col, pair_col + 1, color_index, animal))
            pair_col += 2
    # Explicit validation of the derived values placed in the worksheet.
    for i, values in enumerate(raw_values):
        mean, sem = mean_sem(values)
        if not np.isclose(mean, float(metric_summary.iloc[i]["mean"]), atol=1e-12):
            raise RuntimeError(f"Mean mismatch for {metric}/{GROUPS[i]}")
        if not np.isclose(sem, float(metric_summary.iloc[i]["sem"]), atol=1e-12):
            raise RuntimeError(f"SEM mismatch for {metric}/{GROUPS[i]}")
    return worksheet, raw_values, raw_animals, pair_specs


def build_graph(worksheet, raw_values, raw_animals, pair_specs, spec, planned: pd.DataFrame):
    graph = op.new_graph(lname=f"{spec['stem']} latest manual Spindle review", template="scatter")
    graph.name = ("G" + spec["stem"].replace("_", ""))[:12]
    layer = graph[0]
    bar = layer.add_plot(worksheet, colx=8, coly=9, type=203)
    bar_index = bar.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bar_index}; set rb -vg 45; set rb -pbw {LINE_WIDTH}; set rb -pfp 0;")
    for index, (fill, edge) in enumerate(zip(FILLS, EDGES), start=1):
        op.lt_exec(
            f"range rb=[{graph.name}]1!{bar_index}; set rb {index} -pbc {rgb(edge)}; "
            f"set rb {index} -pfb {rgb(fill)}; set rb {index} -pfc {rgb(fill)};"
        )
    # Pair trajectories are plotted before the hollow points so the points remain
    # fully legible and every line terminates at the correct animal's marker.
    for colx, coly, color_index, _animal in pair_specs:
        pair_plot = layer.add_plot(worksheet, colx=colx, coly=coly, type="l")
        pair_index = pair_plot.index() + 1
        op.lt_exec(
            f"range rl=[{graph.name}]1!{pair_index}; set rl -c {rgb(PAIR_COLORS[color_index])};"
        )
        # LabTalk's set -w path rounds decimal values in Origin 2023b. Use the
        # float plot property so 1.8 pt survives into SVG/PDF/PNG exports.
        pair_plot.set_float("line.width", PAIR_LINE_WIDTH)
    for i, edge in enumerate(EDGES):
        plot = layer.add_plot(worksheet, colx=11 + 2 * i, coly=12 + 2 * i, type="s")
        plot_index = plot.index() + 1
        op.lt_exec(
            f"range rp=[{graph.name}]1!{plot_index}; set rp -k 2; set rp -z {POINT_SIZE}; "
            f"set rp -kf 1; set rp -kh {POINT_EDGE_PERCENT}; set rp -cse {rgb(edge)}; "
            f"set rp -csf {rgb(WHITE)};"
        )
    carrier = layer.add_plot(worksheet, colx=8, coly=9, colyerr=10, type="s")
    carrier_index = carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_index = layer.plot_list()[-1].index() + 1
    op.lt_exec(
        f"range re=[{graph.name}]1!{error_index}; set re -c {rgb(BLACK)}; "
        f"set re -erw {LINE_WIDTH}; set re -erwc 15; set re -erdy 0;"
    )

    xmin, xmax, ymax, ystep = 0.1, 3.9, float(spec["ymax"]), float(spec["ystep"])
    layer.axis("x").set_limits(xmin, xmax, 1)
    layer.axis("y").set_limits(0, ymax, ystep)
    layer.axis("x").title = ""
    layer.axis("y").title = spec["ylabel"]
    op.lt_exec(
        "layer.border=0; layer.x.showAxes=0; layer.y.showAxes=1; "
        "layer.x2.showAxes=0; layer.y2.showAxes=0; layer.x.showLabels=0; layer.y.showLabels=1; "
        "layer.x.showGrids=0; layer.y.showGrids=0; layer.x.ticks=0; layer.y.ticks=0; "
        "layer.x.thickness=0; layer.y.thickness=0; layer.y.label.fsize=21.5; "
        "layer.y.label.bold=0; layer.y.label.font=font(Arial);"
    )
    legend = layer.label("legend")
    if legend:
        legend.remove()
    ylabel = layer.label("YL")
    if ylabel:
        style_label(ylabel)

    add_line(layer, xmin, 0, xmax, 0)
    add_line(layer, xmin, 0, xmin, ymax)
    for y in np.arange(0, ymax + ystep * 0.1, ystep):
        add_line(layer, xmin, float(y), xmin - 0.065, float(y))
    for x, text in zip(XPOS, ["SAL", "CNO", "SAL", "CNO"]):
        add_line(layer, x, 0, x, -ymax * 0.018)
        add_label(layer, text, x + 0.02, -ymax * 0.065)
    add_label(layer, "baseline", 1.02, -ymax * 0.145)
    add_label(layer, "POCD", 3.02, -ymax * 0.145)

    rows = planned.loc[(planned.metric == spec["metric"]) & (planned.stars.fillna("") != "")]
    for row in rows.itertuples(index=False):
        if row.group_a.startswith("baseline"):
            a, b = 0, 1
        else:
            a, b = 2, 3
        pair_top = max(float(np.max(raw_values[a])), float(np.max(raw_values[b])))
        y = pair_top + ymax * 0.055
        add_bracket(layer, XPOS[a], XPOS[b], y, ymax * 0.02, row.stars)

    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    for key, value in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(key, value)
    return graph


def export_graph(graph, panel_dir: Path, stem: str) -> list[Path]:
    png = panel_dir / f"{stem}.png"
    svg = panel_dir / f"{stem}.svg"
    pdf = panel_dir / f"{stem}.pdf"
    graph.save_fig(str(png), type="png", width=2400)
    graph.save_fig(str(svg), type="svg")
    graph.save_fig(str(pdf), type="pdf")
    normalize_svg(svg)
    clean_svg = panel_dir / f"{stem}_AI_clean.svg"
    clean_pdf = panel_dir / f"{stem}_AI_clean.pdf"
    shutil.copy2(svg, clean_svg)
    shutil.copy2(pdf, clean_pdf)
    return [png, svg, pdf, clean_svg, clean_pdf]


def prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    replacements = {}
    for element in root.iter():
        old = element.attrib.get("id")
        if old:
            new = prefix + old
            replacements[old] = new
            element.set("id", new)
    if not replacements:
        return
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for old, new in replacements.items():
                value = value.replace(f"url(#{old})", f"url(#{new})")
                value = value.replace(f"#{old}", f"#{new}")
            element.set(key, value)


def combine_svgs(svgs: list[Path], output: Path) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ns = "http://www.w3.org/2000/svg"
    total_width = 1000 * len(svgs)
    outer = ET.Element(f"{{{ns}}}svg", {"viewBox": f"0 0 {total_width} 1000", "width": str(total_width), "height": "1000"})
    ET.SubElement(outer, f"{{{ns}}}rect", {
        "x": "0", "y": "0", "width": str(total_width), "height": "1000", "fill": "rgb(255,255,255)"
    })
    for i, source in enumerate(svgs):
        root = ET.parse(source).getroot()
        prefix_svg_ids(root, f"p{i}_")
        nested = ET.SubElement(outer, f"{{{ns}}}svg", {
            "x": str(i * 1000), "y": "0", "width": "1000", "height": "1000",
            "viewBox": root.attrib.get("viewBox", "0 0 1000 1000"),
            "preserveAspectRatio": "xMidYMid meet",
        })
        for child in list(root):
            nested.append(child)
    ET.ElementTree(outer).write(output, encoding="utf-8", xml_declaration=True)
    normalize_svg(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    package = args.package.resolve()
    panel_dir = package / "panels"
    editable_dir = package / "editable"
    data_dir = package / "data"
    scripts_dir = package / "scripts"
    docs_dir = package / "documentation"
    for directory in (panel_dir, editable_dir, data_dir, scripts_dir, docs_dir, package / "archive"):
        directory.mkdir(parents=True, exist_ok=True)

    for source in analysis.iterdir():
        if source.is_file():
            shutil.copy2(source, data_dir / source.name)
    frame = pd.read_csv(analysis / "hm4di_latest_manual_spindle_source_data.csv")
    summary = pd.read_csv(analysis / "hm4di_latest_manual_spindle_group_summary.csv")
    planned = pd.read_csv(analysis / "hm4di_latest_manual_spindle_statistics.csv")

    op.attach()
    op.set_show(True)
    op.new(asksave=False)
    graphs = []
    outputs: list[Path] = []
    try:
        for spec in SPECS:
            worksheet, raw_values, raw_animals, pair_specs = prepare_sheet(frame, spec["metric"], summary)
            worksheet.name = ("W" + spec["stem"].replace("_", ""))[:12]
            graph = build_graph(worksheet, raw_values, raw_animals, pair_specs, spec, planned)
            graphs.append((graph, spec["stem"]))
        project = editable_dir / "Fig9_hM4Di_latest_manual_spindle.opju"
        op.save(str(project))
        for graph, stem in graphs:
            outputs.extend(export_graph(graph, panel_dir, stem))
        op.save(str(project))
    finally:
        op.detach()

    combined_svg = package / "Fig9_hM4Di_three_spindle_panels.svg"
    combine_svgs([panel_dir / f"{spec['stem']}.svg" for spec in SPECS], combined_svg)
    shutil.copy2(combined_svg, package / "Fig9_hM4Di_three_spindle_panels_AI_clean.svg")
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=combined_svg.read_bytes(), write_to=str(package / "Fig9_hM4Di_three_spindle_panels.png"), output_width=4800)
        cairosvg.svg2pdf(bytestring=combined_svg.read_bytes(), write_to=str(package / "Fig9_hM4Di_three_spindle_panels.pdf"))
        shutil.copy2(package / "Fig9_hM4Di_three_spindle_panels.pdf", package / "Fig9_hM4Di_three_spindle_panels_AI_clean.pdf")
    except Exception as exc:
        (docs_dir / "combined_export_warning.txt").write_text(str(exc) + "\n", encoding="utf-8")

    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)
    manifest = {
        "backend": "OriginPro 2023b / originpro",
        "project": str(editable_dir / "Fig9_hM4Di_latest_manual_spindle.opju"),
        "group_order": GROUPS,
        "group_n": frame.groupby("group").size().reindex(GROUPS).astype(int).to_dict(),
        "bar_statistic": "arithmetic mean",
        "error": "sample SEM",
        "raw_points": "native hollow circles; exact bar centers except deterministic symmetric dodge for exact display collisions",
        "pairing": "same-animal SAL-to-CNO trajectories: baseline NO1-NO6 (baseline+CNO/NO7 unavailable), POCD NO1-NO7; lines are behind points; no cross-animal connections",
        "style": "formal manuscript Sleep Origin grouped-bar contract with user-requested Fig9 stroke override; 3.0-pt main authoring strokes; 1.8-pt paired trajectories; 14-pt hollow points with 3.0-pt edges; RGB-only; no animal-ID labels",
        "outputs": [str(path) for path in outputs],
    }
    (docs_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PROJECT={manifest['project']}")
    print(f"PACKAGE={package}")
    print(f"GRAPHS={len(graphs)}")
    print(f"OUTPUTS={len(outputs)}")


if __name__ == "__main__":
    main()
