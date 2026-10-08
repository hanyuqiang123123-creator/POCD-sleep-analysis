from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import numpy as np
import originpro as op
import pandas as pd


GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
XPOS4 = [0.72, 1.28, 2.72, 3.28]
FILLS4 = [(169, 220, 233), (102, 182, 205), (244, 190, 166), (238, 138, 85)]
EDGES4 = [(74, 159, 186), (39, 125, 152), (228, 120, 72), (217, 84, 32)]
PSD_LINES = [(237, 145, 103), (217, 84, 32)]
PAIR_COLORS = [(154, 202, 216), (238, 169, 137)]
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
LW, FONT, STAR_FONT = 2.5, 21.5, 25.0


def rgb(c):
    return f"color({c[0]},{c[1]},{c[2]})"


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"#([0-9A-Fa-f]{6})", lambda m: "rgb(" + ",".join(str(int(m.group(1)[i:i+2], 16)) for i in (0, 2, 4)) + ")", text)
    text = re.sub(r"(?i)\bblack\b", "rgb(0,0,0)", text)
    text = re.sub(r"(?i)\bwhite\b", "rgb(255,255,255)", text)
    path.write_text(text, encoding="utf-8", newline="\n")


def style_label(obj, size=FONT, bold=False, color=BLACK):
    obj.color = color
    obj.set_float("fsize", float(size))
    obj.set_int("fontbold", int(bool(bold)))
    obj.set_int("font", 4)


def add_line(layer, x1, y1, x2, y2, width=LW, color=BLACK):
    obj = layer.add_line(float(x1), float(y1), float(x2), float(y2))
    obj.color, obj.width = color, float(width)
    return obj


def add_label(layer, text, x, y, size=FONT, bold=False, color=BLACK):
    obj = layer.add_label(str(text), float(x), float(y))
    style_label(obj, size, bold, color)
    obj.set_int("attach", 2)
    obj.set_int("anchor", 8)
    obj.set_float("x", float(x))
    obj.set_float("y", float(y))
    return obj


def add_bracket(layer, x1, x2, y, ymax, stars):
    h = ymax * 0.02
    add_line(layer, x1, y, x1, y + h)
    add_line(layer, x1, y + h, x2, y + h)
    add_line(layer, x2, y + h, x2, y)
    add_label(layer, stars, (x1 + x2) / 2 + 0.02, y + h * 1.8, STAR_FONT)


def set_axes(layer, ylabel, xmin, xmax, ymax, ystep):
    layer.axis("x").set_limits(xmin, xmax, 1)
    layer.axis("y").set_limits(0, ymax, ystep)
    layer.axis("x").title, layer.axis("y").title = "", ylabel
    op.lt_exec("layer.border=0; layer.x.showAxes=0; layer.y.showAxes=1; layer.x2.showAxes=0; layer.y2.showAxes=0; layer.x.showLabels=0; layer.y.showLabels=1; layer.x.showGrids=0; layer.y.showGrids=0; layer.x.ticks=0; layer.y.ticks=0; layer.x.thickness=0; layer.y.thickness=0; layer.y.label.fsize=21.5; layer.y.label.bold=0; layer.y.label.font=font(Arial);")
    legend = layer.label("legend")
    if legend:
        legend.remove()
    yl = layer.label("YL")
    if yl:
        style_label(yl)
    add_line(layer, xmin, 0, xmax, 0)
    add_line(layer, xmin, 0, xmin, ymax)
    tick = (xmax - xmin) * 0.015
    for y in np.arange(0, ymax + ystep * 0.1, ystep):
        add_line(layer, xmin, float(y), xmin - tick, float(y))


def dodge(values, center, step=0.045):
    values = np.asarray(values, float)
    x = np.full(len(values), center)
    rounded = np.round(values, 4)
    for value in np.unique(rounded):
        idx = np.where(rounded == value)[0]
        if len(idx) > 1:
            x[idx] += (np.arange(len(idx)) - (len(idx) - 1) / 2) * step
    return x


def export_graph(graph, panels: Path, stem: str):
    graph.save_fig(str(panels / f"{stem}.png"), type="png", width=2400)
    graph.save_fig(str(panels / f"{stem}.svg"), type="svg")
    graph.save_fig(str(panels / f"{stem}.pdf"), type="pdf")
    normalize_svg(panels / f"{stem}.svg")
    shutil.copy2(panels / f"{stem}.svg", panels / f"{stem}_AI_clean.svg")
    shutil.copy2(panels / f"{stem}.pdf", panels / f"{stem}_AI_clean.pdf")


def build_duration(data: Path):
    df = pd.read_csv(data / "NREM_bout_duration_Origin_table.csv")
    stats = pd.read_csv(data / "nrem_bout_duration_statistics.csv", keep_default_na=False)
    ws = op.new_sheet("w", lname="NREM bout duration source", hidden=False)
    ws.from_df(df)
    ws.cols = 47
    for condition_index, (left_col, right_col, left_x, right_x) in enumerate((("Raw_0", "Raw_1", XPOS4[0], XPOS4[1]), ("Raw_2", "Raw_3", XPOS4[2], XPOS4[3]))):
        left = df[left_col].dropna().to_numpy(float)
        right = df[right_col].dropna().to_numpy(float)
        if len(left) != 7 or len(right) != 7:
            raise RuntimeError(f"Expected seven pairs for {left_col}/{right_col}")
        for pair_index, (a, b) in enumerate(zip(left, right)):
            col = 19 + 2 * (condition_index * 7 + pair_index)
            ws.from_list(col, [left_x, right_x], lname=f"pair {condition_index}-{pair_index} X", axis="X")
            ws.from_list(col + 1, [float(a), float(b)], lname=f"pair {condition_index}-{pair_index} Y", axis="Y")
    graph = op.new_graph(lname="NREM_bout_duration", template="scatter")
    graph.name = "GNREMBOUT"
    layer = graph[0]
    bar = layer.add_plot(ws, colx=4, coly=5, type=203)
    bi = bar.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb -vg 45; set rb -pbw {LW}; set rb -pfp 0;")
    for i, (fill, edge) in enumerate(zip(FILLS4, EDGES4), 1):
        op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb {i} -pbc {rgb(edge)}; set rb {i} -pfb {rgb(fill)}; set rb {i} -pfc {rgb(fill)};")
    for condition_index in range(2):
        for pair_index in range(7):
            col = 19 + 2 * (condition_index * 7 + pair_index)
            pair = layer.add_plot(ws, colx=col, coly=col + 1, type="line")
            pair.color = PAIR_COLORS[condition_index]
            pair.set_float("line.width", 1.2)
    for i, edge in enumerate(EDGES4):
        p = layer.add_plot(ws, colx=7 + 3 * i, coly=8 + 3 * i, type="s")
        pi = p.index() + 1
        op.lt_exec(f"range rp=[{graph.name}]1!{pi}; set rp -k 2; set rp -z 14; set rp -kf 1; set rp -kh 35.714; set rp -cse {rgb(edge)}; set rp -csf {rgb(WHITE)};")
    carrier = layer.add_plot(ws, colx=4, coly=5, colyerr=6, type="s")
    ci = carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{ci}; set rc -k 0; set rc -z 0;")
    ei = layer.plot_list()[-1].index() + 1
    op.lt_exec(f"range re=[{graph.name}]1!{ei}; set re -c {rgb(BLACK)}; set re -erw {LW}; set re -erwc 15; set re -erdy 0;")
    ymax = 300.0
    set_axes(layer, "NREM bout duration (s)", 0.1, 3.9, ymax, 50.0)
    for x, text in zip(XPOS4, ["SAL", "CNO", "SAL", "CNO"]):
        add_line(layer, x, 0, x, -ymax * 0.018)
        add_label(layer, text, x + 0.02, -ymax * 0.065)
    add_label(layer, "baseline", 1.02, -ymax * 0.145)
    add_label(layer, "POCD", 3.02, -ymax * 0.145)
    add_label(layer, "NREM", 2.0, ymax * 1.06)
    row = stats.loc[stats.condition == "POCD"].iloc[0]
    if row.stars_holm:
        top = np.nanmax(df[["Raw_2", "Raw_3"]].to_numpy(float))
        add_bracket(layer, XPOS4[2], XPOS4[3], max(top + 15, 250), ymax, row.stars_holm)
    graph.set_float("width", 5600); graph.set_float("height", 5000)
    for k, v in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(k, v)
    return graph


def spectral_summary(spectra: pd.DataFrame) -> pd.DataFrame:
    p = spectra.pivot_table(index="frequency_hz", columns="group", values="relative_psd_percent_per_hz", aggfunc=["mean", "sem"]).reset_index()
    p.columns = ["frequency_hz", "POCD_SAL_mean", "POCD_CNO_mean", "POCD_SAL_sem", "POCD_CNO_sem"]
    return p


def build_psd(data: Path):
    spectra = pd.read_csv(data / "animal_level_nrem_spectra.csv")
    summary = spectral_summary(spectra)
    summary.to_csv(data / "NREM_relative_PSD_Origin_table.csv", index=False)
    ws = op.new_sheet("w", lname="NREM relative PSD source", hidden=False)
    ws.from_df(summary)
    animals = op.new_sheet("w", lname="NREM individual spectra", hidden=False)
    animals.from_df(spectra)
    graph = op.new_graph(lname="NREM_relative_PSD", template="scatter")
    graph.name = "GNREMPSD"
    layer = graph[0]
    for col, color in ((1, PSD_LINES[0]), (2, PSD_LINES[1])):
        p = layer.add_plot(ws, colx=0, coly=col, type="line")
        p.color = color; p.set_float("line.width", LW)
    ymax = 25.0
    set_axes(layer, "Relative PSD (%/Hz)", 0.5, 25.0, ymax, 5.0)
    for x in (0.5, 5, 10, 15, 20, 25):
        add_line(layer, x, 0, x, -ymax * 0.02)
        add_label(layer, str(x), x, -ymax * 0.065, 20)
    add_label(layer, "Frequency (Hz)", 12.75, -ymax * 0.16)
    add_label(layer, "NREM", 12.75, ymax * 1.06)
    for j, (text, color) in enumerate((("POCD + SAL (n=7)", PSD_LINES[0]), ("POCD + CNO (n=7)", PSD_LINES[1]))):
        y = 6.0 - j * 2.5
        add_line(layer, 10.8, y, 12.3, y, color=color)
        add_label(layer, text, 16.0, y, 19)
    graph.set_float("width", 5600); graph.set_float("height", 5000)
    for k, v in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(k, v)
    return graph


def make_band_table(data: Path) -> pd.DataFrame:
    raw = pd.read_csv(data / "animal_level_nrem_band_power.csv")
    rows = []
    centers = {"Delta": 1.0, "Theta": 2.0, "Sigma": 3.0}
    for band in ("Delta", "Theta", "Sigma"):
        for j, group in enumerate(("POCD+SAL", "POCD+CNO")):
            v = raw.loc[(raw.band == band) & (raw.group == group)].sort_values("animal")["relative_mean_percent"].to_numpy(float)
            rows.append({"Bar_X": centers[band] + (-0.22 if j == 0 else 0.22), "Mean": v.mean(), "SEM": v.std(ddof=1) / math.sqrt(len(v)), "Band": band, "Group": group, "Raw": v})
    nmax = max(len(r["Raw"]) for r in rows)
    out = pd.DataFrame(index=range(nmax))
    for name, values in {
        "Bar_X": [r["Bar_X"] for r in rows], "Mean": [r["Mean"] for r in rows],
        "SEM": [r["SEM"] for r in rows], "Band": [r["Band"] for r in rows],
        "Group": [r["Group"] for r in rows],
    }.items():
        out[name] = pd.Series(values)
    for i, row in enumerate(rows):
        vals = list(row["Raw"]) + [np.nan] * (nmax - len(row["Raw"]))
        out[f"Raw_{i}"] = vals
        out[f"PointX_{i}"] = [row["Bar_X"]] * nmax
        out[f"PointY_{i}"] = vals
    out.to_csv(data / "NREM_Delta_Theta_Sigma_Origin_table.csv", index=False)
    return out


def build_bands(data: Path):
    df = make_band_table(data)
    stats = pd.read_csv(data / "nrem_band_power_statistics.csv", keep_default_na=False)
    ws = op.new_sheet("w", lname="NREM band power source", hidden=False)
    ws.from_df(df)
    ws.cols = 65
    for band_index in range(3):
        sal = df[f"Raw_{2 * band_index}"].dropna().to_numpy(float)
        cno = df[f"Raw_{2 * band_index + 1}"].dropna().to_numpy(float)
        if len(sal) != 7 or len(cno) != 7:
            raise RuntimeError(f"Expected seven spectral pairs for band {band_index}")
        for pair_index, (a, b) in enumerate(zip(sal, cno)):
            col = 23 + 2 * (band_index * 7 + pair_index)
            center = float(band_index + 1)
            ws.from_list(col, [center - 0.22, center + 0.22], lname=f"band {band_index} pair {pair_index} X", axis="X")
            ws.from_list(col + 1, [float(a), float(b)], lname=f"band {band_index} pair {pair_index} Y", axis="Y")
    graph = op.new_graph(lname="NREM_Delta_Theta_Sigma", template="scatter")
    graph.name = "GNREMBANDS"
    layer = graph[0]
    bar = layer.add_plot(ws, colx=0, coly=1, type=203)
    bi = bar.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb -vg 22; set rb -pbw {LW}; set rb -pfp 0;")
    for i in range(6):
        c = 2 if i % 2 == 0 else 3
        op.lt_exec(f"range rb=[{graph.name}]1!{bi}; set rb {i+1} -pbc {rgb(EDGES4[c])}; set rb {i+1} -pfb {rgb(FILLS4[c])}; set rb {i+1} -pfc {rgb(FILLS4[c])};")
    for band_index in range(3):
        for pair_index in range(7):
            col = 23 + 2 * (band_index * 7 + pair_index)
            pair = layer.add_plot(ws, colx=col, coly=col + 1, type="line")
            pair.color = PAIR_COLORS[1]
            pair.set_float("line.width", 1.2)
    for i in range(6):
        c = 2 if i % 2 == 0 else 3
        p = layer.add_plot(ws, colx=6 + 3 * i, coly=7 + 3 * i, type="s")
        pi = p.index() + 1
        op.lt_exec(f"range rp=[{graph.name}]1!{pi}; set rp -k 2; set rp -z 10; set rp -kf 1; set rp -kh 35.714; set rp -cse {rgb(EDGES4[c])}; set rp -csf {rgb(WHITE)};")
    carrier = layer.add_plot(ws, colx=0, coly=1, colyerr=2, type="s")
    ci = carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{ci}; set rc -k 0; set rc -z 0;")
    ei = layer.plot_list()[-1].index() + 1
    op.lt_exec(f"range re=[{graph.name}]1!{ei}; set re -c {rgb(BLACK)}; set re -erw {LW}; set re -erwc 15; set re -erdy 0;")
    ymax = 50.0
    set_axes(layer, "Relative band power (%)", 0.45, 4.35, ymax, 10.0)
    for x, text in zip((1, 2, 3), ("Delta", "Theta", "Sigma")):
        add_line(layer, x, 0, x, -ymax * 0.018)
        add_label(layer, text, x, -ymax * 0.068)
    add_label(layer, "NREM", 2.0, ymax * 1.06)
    for j, (text, edge) in enumerate((("POCD + SAL (n=7)", EDGES4[2]), ("POCD + CNO (n=7)", EDGES4[3]))):
        y = 46 - j * 5
        add_line(layer, 3.30, y, 3.48, y, width=6, color=edge)
        add_label(layer, text, 3.88, y, 17)
    for center, band in zip((1.0, 2.0, 3.0), ("Delta", "Theta", "Sigma")):
        row = stats.loc[stats.band == band].iloc[0]
        if row.stars_holm:
            top = float(df.loc[df.Band == band, "Mean"].max() + df.loc[df.Band == band, "SEM"].max())
            add_bracket(layer, center - 0.22, center + 0.22, top + 2.5, ymax, row.stars_holm)
    graph.set_float("width", 5600); graph.set_float("height", 5000)
    for k, v in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(k, v)
    return graph


def shade_psd(svg: Path):
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    root = ET.parse(svg).getroot()
    tag = lambda x: f"{{{ns}}}{x}"
    left, top, width, height = 588.0, 275.0, 2286.0, 1800.0
    bg, labels = ET.Element(tag("g"), {"id": "frequency_band_backgrounds"}), ET.Element(tag("g"), {"id": "frequency_band_labels"})
    for name, low, high, fill, ink in (("Delta", .5, 4, "rgb(220,234,247)", "rgb(69,107,139)"), ("Theta", 4, 8, "rgb(224,239,229)", "rgb(73,116,91)"), ("Sigma", 10, 15, "rgb(235,226,243)", "rgb(119,91,145)")):
        x = left + (low - .5) / 24.5 * width; w = (high - low) / 24.5 * width
        ET.SubElement(bg, tag("rect"), {"x": str(x), "y": str(top), "width": str(w), "height": str(height), "fill": fill})
        t = ET.SubElement(labels, tag("text"), {"x": str(x + w / 2), "y": "390", "text-anchor": "middle", "font-family": "Arial", "font-size": "78", "fill": ink}); t.text = name
    root.insert(0, bg); root.append(labels)
    ET.ElementTree(root).write(svg, encoding="utf-8", xml_declaration=True)
    normalize_svg(svg)


def prefix_ids(root, prefix):
    rep = {}
    for e in root.iter():
        if e.attrib.get("id"):
            rep[e.attrib["id"]] = prefix + e.attrib["id"]; e.set("id", prefix + e.attrib["id"])
    for e in root.iter():
        for k, value in list(e.attrib.items()):
            for old, new in rep.items():
                value = value.replace(f"url(#{old})", f"url(#{new})").replace(f"#{old}", f"#{new}")
            e.set(k, value)


def combine(svgs, output):
    ns = "http://www.w3.org/2000/svg"; ET.register_namespace("", ns); tag = lambda x: f"{{{ns}}}{x}"
    root = ET.Element(tag("svg"), {"viewBox": "0 0 3000 1000", "width": "3000", "height": "1000"})
    ET.SubElement(root, tag("rect"), {"x": "0", "y": "0", "width": "3000", "height": "1000", "fill": "rgb(255,255,255)"})
    for i, source in enumerate(svgs):
        src = ET.parse(source).getroot(); prefix_ids(src, f"p{i}_")
        nested = ET.SubElement(root, tag("svg"), {"x": str(i * 1000), "y": "0", "width": "1000", "height": "1000", "viewBox": src.attrib.get("viewBox", "0 0 1000 1000"), "preserveAspectRatio": "xMidYMid meet"})
        for child in list(src): nested.append(child)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True); normalize_svg(output)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--package", required=True, type=Path); args = ap.parse_args()
    package = args.package.resolve(); data = package / "data" / "missing_C_H_metrics"; panels = package / "panels"; editable = package / "editable"; scripts = package / "scripts"; docs = package / "documentation"
    for d in (panels, editable, scripts, docs): d.mkdir(parents=True, exist_ok=True)
    op.attach(); op.set_show(True); op.new(asksave=False)
    try:
        graphs = [("NREM_bout_duration", build_duration(data)), ("NREM_relative_PSD", build_psd(data)), ("NREM_Delta_Theta_Sigma", build_bands(data))]
        project = editable / "Fig9_hM4Di_missing_C_H_metrics.opju"; op.save(str(project))
        for stem, graph in graphs: export_graph(graph, panels, stem)
        op.save(str(project))
    finally:
        op.detach()
    psd_svg = panels / "NREM_relative_PSD.svg"; shade_psd(psd_svg)
    cairosvg.svg2png(bytestring=psd_svg.read_bytes(), write_to=str(panels / "NREM_relative_PSD.png"), output_width=2400, background_color="white")
    cairosvg.svg2pdf(bytestring=psd_svg.read_bytes(), write_to=str(panels / "NREM_relative_PSD.pdf"))
    shutil.copy2(psd_svg, panels / "NREM_relative_PSD_AI_clean.svg"); shutil.copy2(panels / "NREM_relative_PSD.pdf", panels / "NREM_relative_PSD_AI_clean.pdf")
    combined = package / "Fig9_hM4Di_missing_three_panels.svg"
    combine([panels / f"{s}.svg" for s, _ in graphs], combined)
    shutil.copy2(combined, package / "Fig9_hM4Di_missing_three_panels_AI_clean.svg")
    cairosvg.svg2png(bytestring=combined.read_bytes(), write_to=str(package / "Fig9_hM4Di_missing_three_panels.png"), output_width=6000, background_color="white")
    cairosvg.svg2pdf(bytestring=combined.read_bytes(), write_to=str(package / "Fig9_hM4Di_missing_three_panels.pdf"))
    shutil.copy2(package / "Fig9_hM4Di_missing_three_panels.pdf", package / "Fig9_hM4Di_missing_three_panels_AI_clean.pdf")
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    (docs / "missing_C_H_metrics_origin_manifest.json").write_text(json.dumps({"project": str(project), "data": str(data), "panels": [s for s, _ in graphs], "display": "mean +/- sample SEM; hollow animal points; no animal IDs; star-only Holm-adjusted annotations"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(project); print(combined)


if __name__ == "__main__":
    main()
