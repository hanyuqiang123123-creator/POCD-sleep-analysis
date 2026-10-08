from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(r"F:\1.Sleep\PHD稿件")
WORK = ROOT / "Figure_Workspace" / "Fig9_hM4Di_chemogenetics"
FORMAL = ROOT / "Figures" / "Fig9_hM4Di_chemogenetics"
SOURCE = FORMAL / "data" / "ma_bout_review_current" / "sleep_architecture_ma_current_source_data.csv"
DATA_NAME = "rem_wake_spectral_sd_current"
PROJECT_NAME = "Fig9_REM_WAKE_spectral_SD.opju"
GROUPS = ("POCD+SAL", "POCD+CNO")
ANIMALS = tuple(f"NO{i}" for i in range(1, 8))
STAGES = {"REM": 3, "WAKE": 1}
BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Sigma": (10.0, 15.0)}
FILLS = ((244, 190, 166), (238, 138, 85))
EDGES = ((228, 120, 72), (217, 84, 32))
LINES = ((237, 145, 103), (217, 84, 32))
RIBBONS = ((248, 211, 193), (239, 182, 155))
PAIR_COLOR = (238, 169, 137)
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
LW, PAIR_LW, FONT, STAR_FONT = 2.5, 1.2, 21.5, 25.0
POINT_SIZE, POINT_EDGE = 10.0, 35.714
BAND_OFFSET = 0.16
PSD_STEMS = {"REM": "Fig9_14_REM_relative_PSD", "WAKE": "Fig9_16_WAKE_relative_PSD"}
BAND_STEMS = {"REM": "Fig9_15_REM_Delta_Theta_Sigma", "WAKE": "Fig9_17_WAKE_Delta_Theta_Sigma"}
ALIASES = {
    "REM": ("REM_relative_PSD", "REM_Delta_Theta_Sigma"),
    "WAKE": ("WAKE_relative_PSD", "WAKE_Delta_Theta_Sigma"),
}

SCRIPT_PATH = Path(__file__).resolve()
STYLE_CANDIDATES = [SCRIPT_PATH.parents[1] / "origin_scripts", SCRIPT_PATH.parent / "origin_scripts"]
STYLE_DIR = next((path for path in STYLE_CANDIDATES if path.exists()), STYLE_CANDIDATES[0])
sys.path.insert(0, str(STYLE_DIR))
import build_fig9_missing_metrics_origin as style  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rgb(color: tuple[int, int, int]) -> str:
    return f"color({color[0]},{color[1]},{color[2]})"


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    adjusted = np.empty_like(p)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select start_time_seconds, score from sleep_scores_table "
            "order by start_time_seconds, start_time_sub_seconds"
        ).fetchall()
    values = np.asarray(rows)
    times = values[:, 0].astype(np.int64)
    scores = values[:, 1].astype(np.int16)
    if len(scores) != 17280 or not np.all(np.diff(times) == 5):
        raise RuntimeError(f"Expected 17,280 consecutive 5-s epochs: {path}")
    if not set(np.unique(scores)).issubset({0, 1, 2, 3}):
        raise RuntimeError(f"Unexpected scores in {path}: {np.unique(scores)}")
    return times, scores


def read_eeg(path: Path) -> tuple[np.ndarray, float, str, str]:
    with path.open("rb") as stream:
        header = stream.read(256)
        channels = int(header[252:256])
        signal_header = stream.read(256 * channels)
        position, fields = 0, []
        for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
            fields.append([
                signal_header[position + i * width:position + (i + 1) * width]
                .decode("latin1").strip()
                for i in range(channels)
            ])
            position += channels * width
        samples_per_record = list(map(int, fields[8]))
        records = int(header[236:244])
        duration = float(header[244:252])
        stream.seek(int(header[184:192]))
        raw = np.fromfile(stream, dtype="<i2").reshape(records, sum(samples_per_record))
        digital = raw[:, :samples_per_record[0]].reshape(-1).astype(float)
        pmin, pmax, dmin, dmax = [float(fields[i][0]) for i in (3, 4, 5, 6)]
        eeg = (digital - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
        return eeg.astype(np.float32), samples_per_record[0] / duration, fields[0][0], fields[2][0]


def interp_psd(psd: np.ndarray, frequencies: np.ndarray, target: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(frequencies, target), 1, len(frequencies) - 1)
    alpha = (target - frequencies[index - 1]) / (frequencies[index] - frequencies[index - 1])
    return psd[:, index - 1] * (1 - alpha) + psd[:, index] * alpha


def integrate(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    target = np.r_[low, frequencies[(frequencies > low) & (frequencies < high)], high]
    return np.trapezoid(interp_psd(psd, frequencies, target), target, axis=1)


def artifact_mask(folder: Path, scores: np.ndarray) -> tuple[np.ndarray, str, str]:
    mask = scores == 0
    artifact_path = folder / "export_SO200Hz.artifacts.json"
    status = "score0_only"
    if artifact_path.is_file():
        obj = json.loads(artifact_path.read_text(encoding="utf-8"))
        record = obj.get("records", {}).get("eeg0_emg1")
        if record:
            for start, end in record.get("final_artifact_ranges", []):
                start = max(0, int(start))
                end = min(len(scores) - 1, int(end))
                if end >= start:
                    mask[start:end + 1] = True
            status = "verified_sidecar_plus_score0"
    return mask, status, str(artifact_path) if artifact_path.is_file() else ""


def validate_source(source: pd.DataFrame) -> pd.DataFrame:
    subset = source[source.group.isin(GROUPS)].copy()
    counts = subset.groupby("group").size().to_dict()
    if counts != {group: 7 for group in GROUPS}:
        raise RuntimeError(f"Unexpected group counts: {counts}")
    for group in GROUPS:
        ids = tuple(subset.loc[subset.group == group, "animal"].sort_values())
        if ids != ANIMALS:
            raise RuntimeError(f"Unexpected animal IDs for {group}: {ids}")
    return subset


def analyze(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spectra, band_rows, inventories = [], [], []
    samples_per_epoch, fs_expected = 1000, 200.0
    tapers = np.sqrt(2 / 1001) * np.sin(
        np.pi * np.arange(1, 6)[:, None] * np.arange(1, 1001)[None, :] / 1001
    )
    frequencies = np.fft.rfftfreq(samples_per_epoch, 1 / fs_expected)
    grid = np.r_[0.5, frequencies[(frequencies > 0.5) & (frequencies <= 25.0)]]
    for record_index, row in enumerate(source.sort_values(["group", "animal"]).itertuples(), start=1):
        db = Path(row.scores_db_path)
        folder = db.parent
        edf = folder / "export_SO200Hz.edf"
        times, scores = read_scores(db)
        artifacts, artifact_status, artifact_path = artifact_mask(folder, scores)
        eeg, fs, label, unit = read_eeg(edf)
        if not np.isclose(fs, fs_expected):
            raise RuntimeError(f"Unexpected sampling rate {fs}: {edf}")
        required = len(scores) * samples_per_epoch
        if len(eeg) < required:
            raise RuntimeError(f"EEG shorter than scoring: {edf}")
        epochs = eeg[:required].reshape(len(scores), samples_per_epoch)
        stage_counts = {}
        for stage, score_code in STAGES.items():
            valid = (scores == score_code) & ~artifacts
            stage_counts[stage] = int(valid.sum())
            if valid.sum() < 2:
                raise RuntimeError(f"Insufficient {stage} epochs: {row.group}/{row.animal}")
            values = epochs[valid].astype(float)
            band_chunks, spectrum_chunks = [], []
            for start in range(0, len(values), 256):
                chunk = values[start:start + 256]
                psd = sum(
                    np.abs(np.fft.rfft(chunk * taper, axis=1)) ** 2
                    / (fs * np.sum(tapers[0] ** 2))
                    for taper in tapers
                ) / len(tapers)
                psd[:, 1:-1] *= 2
                denominator = integrate(psd, frequencies, 0.5, 25.0)
                band_power = np.column_stack([
                    integrate(psd, frequencies, low, high) for low, high in BANDS.values()
                ])
                band_chunks.append(band_power / denominator[:, None] * 100)
                spectrum_chunks.append(interp_psd(psd, frequencies, grid) / denominator[:, None] * 100)
            relative_bands = np.vstack(band_chunks)
            animal_spectrum = np.vstack(spectrum_chunks).mean(axis=0)
            for band_index, (band, (low, high)) in enumerate(BANDS.items()):
                band_rows.append({
                    "group": row.group, "animal": row.animal, "stage": stage,
                    "band": band, "low_hz": low, "high_hz": high,
                    "valid_epochs": int(valid.sum()),
                    "relative_mean_percent": float(relative_bands[:, band_index].mean()),
                })
            spectra.append(pd.DataFrame({
                "group": row.group, "animal": row.animal, "stage": stage,
                "frequency_hz": grid,
                "relative_psd_percent_per_hz": animal_spectrum,
            }))
        inventories.append({
            "group": row.group, "animal": row.animal,
            "edf": str(edf), "db3": str(db),
            "edf_sha256": sha256(edf), "db3_sha256": sha256(db),
            "artifact_path": artifact_path, "artifact_status": artifact_status,
            "REM_valid_epochs": stage_counts["REM"], "WAKE_valid_epochs": stage_counts["WAKE"],
            "score_epochs": len(scores), "score_start_unix": int(times[0]),
            "score_end_unix": int(times[-1] + 5), "eeg_label": label, "eeg_unit": unit,
        })
        print(
            f"[{record_index:02d}/14] {row.group}/{row.animal}: "
            f"REM={stage_counts['REM']} WAKE={stage_counts['WAKE']}",
            flush=True,
        )
    bands = pd.DataFrame(band_rows)
    animal_spectra = pd.concat(spectra, ignore_index=True)
    stats_rows = []
    for stage in STAGES:
        stage_rows = []
        for band in BANDS:
            sal = (
                bands[(bands.stage == stage) & (bands.group == "POCD+SAL") & (bands.band == band)]
                .set_index("animal").relative_mean_percent.sort_index()
            )
            cno = (
                bands[(bands.stage == stage) & (bands.group == "POCD+CNO") & (bands.band == band)]
                .set_index("animal").relative_mean_percent.sort_index()
            )
            if tuple(sal.index) != ANIMALS or tuple(cno.index) != ANIMALS:
                raise RuntimeError(f"Pairing mismatch for {stage}/{band}")
            test = wilcoxon(sal, cno, alternative="two-sided", method="exact")
            stage_rows.append({
                "stage": stage, "band": band,
                "group_a": "POCD+SAL", "group_b": "POCD+CNO",
                "n_pairs": 7, "paired_ids": "|".join(ANIMALS),
                "mean_a": float(sal.mean()), "sd_a": float(sal.std(ddof=1)),
                "mean_b": float(cno.mean()), "sd_b": float(cno.std(ddof=1)),
                "test": "exact two-sided paired Wilcoxon",
                "wilcoxon_w": float(test.statistic), "p_raw": float(test.pvalue),
            })
        adjusted = holm([row["p_raw"] for row in stage_rows])
        for row, p_value in zip(stage_rows, adjusted):
            row["p_holm_across_3_bands_within_stage"] = p_value
            row["stars_holm"] = stars(p_value)
        stats_rows.extend(stage_rows)
    return animal_spectra, bands, pd.DataFrame(stats_rows), pd.DataFrame(inventories)


def spectrum_summary(spectra: pd.DataFrame, stage: str) -> pd.DataFrame:
    result = pd.DataFrame({"frequency_hz": sorted(spectra.frequency_hz.unique())})
    for prefix, group in (("SAL", "POCD+SAL"), ("CNO", "POCD+CNO")):
        pivot = (
            spectra[(spectra.stage == stage) & (spectra.group == group)]
            .pivot(index="frequency_hz", columns="animal", values="relative_psd_percent_per_hz")
            .sort_index()
        )
        if tuple(pivot.columns) != ANIMALS or pivot.isna().any().any():
            raise RuntimeError(f"Incomplete {stage} spectrum for {group}")
        mean = pivot.mean(axis=1)
        sd = pivot.std(axis=1, ddof=1)
        result[f"{prefix}_mean"] = mean.to_numpy()
        result[f"{prefix}_lower"] = (mean - sd).to_numpy()
        result[f"{prefix}_upper"] = (mean + sd).to_numpy()
        result[f"{prefix}_SD"] = sd.to_numpy()
    return result


def add_frequency_backgrounds(layer, sheet, ymax: float) -> None:
    definitions = (
        ("Delta", 0.5, 4.0, (220, 234, 247), (69, 107, 139)),
        ("Theta", 4.0, 8.0, (224, 239, 229), (73, 116, 91)),
        ("Sigma", 10.0, 15.0, (235, 226, 243), (119, 91, 145)),
    )
    for index, (name, low, high, color, ink) in enumerate(definitions):
        sheet.from_list(3 * index, [low, high], lname=f"{name} X", axis="X")
        sheet.from_list(3 * index + 1, [ymax, ymax], lname=f"{name} upper")
        sheet.from_list(3 * index + 2, [0, 0], lname=f"{name} lower")
        upper = layer.add_plot(sheet, colx=3 * index, coly=3 * index + 1, type="line")
        lower = layer.add_plot(sheet, colx=3 * index, coly=3 * index + 2, type="line")
        for plot in (upper, lower):
            plot.color = color
            plot.set_float("line.width", 0)
        upper.set_fill_area(op.ocolor(color), 9, op.ocolor(color))
        style.add_label(layer, name, (low + high) / 2, ymax * 0.945, 18, color=ink)


def build_psd_graph(spectra: pd.DataFrame, stage: str, data_dir: Path):
    summary = spectrum_summary(spectra, stage)
    summary.to_csv(data_dir / f"{stage}_relative_PSD_SD_Origin_table.csv", index=False, encoding="utf-8-sig")
    sheet = op.new_sheet("w", lname=f"Fig9 {stage} PSD mean and SD", hidden=False)
    sheet.name = f"W{stage}PSD"[:12]
    sheet.from_df(summary)
    raw_sheet = op.new_sheet("w", lname=f"Fig9 {stage} animal spectra", hidden=False)
    raw_sheet.from_df(spectra[spectra.stage == stage])
    background_sheet = op.new_sheet("w", lname=f"Fig9 {stage} frequency bands", hidden=False)
    graph = op.new_graph(lname=f"Fig9 {stage} relative PSD mean +/- SD", template="scatter")
    graph.name = f"G{stage}PSD"[:12]
    layer = graph[0]
    ymax = 25.0
    add_frequency_backgrounds(layer, background_sheet, ymax)
    for lower_col, upper_col, color in ((2, 3, RIBBONS[0]), (6, 7, RIBBONS[1])):
        upper = layer.add_plot(sheet, colx=0, coly=upper_col, type="line")
        lower = layer.add_plot(sheet, colx=0, coly=lower_col, type="line")
        for plot in (upper, lower):
            plot.color = color
            plot.set_float("line.width", 0)
        upper.set_fill_area(op.ocolor(color), 9, op.ocolor(color))
    for col, color in ((1, LINES[0]), (5, LINES[1])):
        plot = layer.add_plot(sheet, colx=0, coly=col, type="line")
        plot.color = color
        plot.set_float("line.width", LW)
    style.set_axes(layer, "Relative PSD (%/Hz)", 0.5, 25.0, ymax, 5.0)
    op.lt_exec("layer.x.showLabels=0; layer.x.showAxes=0; layer.y.ticks=0; layer.y.minorTicks=0;")
    for x in (0.5, 5, 10, 15, 20, 25):
        style.add_line(layer, x, 0, x, -ymax * 0.02)
        style.add_label(layer, str(x), x, -ymax * 0.065, 20)
    style.add_label(layer, "Frequency (Hz)", 12.75, -ymax * 0.16)
    style.add_label(layer, stage, 12.75, ymax * 1.06)
    for index, (text, color) in enumerate((("POCD + SAL (n=7)", LINES[0]), ("POCD + CNO (n=7)", LINES[1]))):
        y = ymax * (0.16 - index * 0.08)
        style.add_line(layer, 2.0, y, 3.8, y, color=color)
        label = layer.add_label(text, 4.15, y)
        style.style_label(label, 16, False, BLACK)
        label.set_int("attach", 2)
        label.set_int("anchor", 6)
    graph.set_float("width", 5600)
    graph.set_float("height", 5000)
    for key, value in (("left", 18), ("top", 11), ("width", 70), ("height", 72)):
        layer.set_float(key, value)
    return graph


def band_table(bands: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for center, band in enumerate(BANDS, start=1):
        for group_index, group in enumerate(GROUPS):
            cell = (
                bands[(bands.stage == stage) & (bands.band == band) & (bands.group == group)]
                .sort_values("animal")
            )
            if tuple(cell.animal) != ANIMALS:
                raise RuntimeError(f"Incomplete {stage}/{band}/{group}")
            values = cell.relative_mean_percent.to_numpy(float)
            rows.append({
                "Bar_X": center + (-BAND_OFFSET if group_index == 0 else BAND_OFFSET),
                "Mean": float(values.mean()), "SD": float(values.std(ddof=1)),
                "Band": band, "Group": group, "Raw": values,
            })
    output = pd.DataFrame(index=range(7))
    for name in ("Bar_X", "Mean", "SD", "Band", "Group"):
        output[name] = pd.Series([row[name] for row in rows])
    for index, row in enumerate(rows):
        output[f"Raw_{index}"] = row["Raw"]
        output[f"PointX_{index}"] = [row["Bar_X"]] * 7
        output[f"PointY_{index}"] = row["Raw"]
        output[f"ID_{index}"] = list(ANIMALS)
    return output


def build_band_graph(bands: pd.DataFrame, stats_frame: pd.DataFrame, stage: str, data_dir: Path):
    table = band_table(bands, stage)
    table.to_csv(data_dir / f"{stage}_Delta_Theta_Sigma_SD_Origin_table.csv", index=False, encoding="utf-8-sig")
    sheet = op.new_sheet("w", lname=f"Fig9 {stage} band power mean and SD", hidden=False)
    sheet.name = f"W{stage}Band"[:12]
    sheet.from_df(table)
    sheet.cols = 75
    for band_index in range(3):
        sal = table[f"Raw_{2 * band_index}"].to_numpy(float)
        cno = table[f"Raw_{2 * band_index + 1}"].to_numpy(float)
        for animal_index, (sal_value, cno_value) in enumerate(zip(sal, cno)):
            column = 29 + 2 * (band_index * 7 + animal_index)
            center = band_index + 1.0
            sheet.from_list(column, [center - BAND_OFFSET, center + BAND_OFFSET], lname=f"{stage} {band_index} pair X", axis="X")
            sheet.from_list(column + 1, [float(sal_value), float(cno_value)], lname=f"{stage} {band_index} pair Y", axis="Y")
    graph = op.new_graph(lname=f"Fig9 {stage} relative band power mean +/- SD", template="scatter")
    graph.name = f"G{stage}Band"[:12]
    layer = graph[0]
    bar = layer.add_plot(sheet, colx=0, coly=1, type=203)
    bar_index = bar.index() + 1
    op.lt_exec(f"range rb=[{graph.name}]1!{bar_index}; set rb -vg 50; set rb -pbw {LW}; set rb -pfp 0;")
    for item_index in range(6):
        group_index = item_index % 2
        op.lt_exec(
            f"range rb=[{graph.name}]1!{bar_index}; set rb {item_index + 1} -pbc {rgb(EDGES[group_index])}; "
            f"set rb {item_index + 1} -pfb {rgb(FILLS[group_index])}; set rb {item_index + 1} -pfc {rgb(FILLS[group_index])};"
        )
    for band_index in range(3):
        for animal_index in range(7):
            column = 29 + 2 * (band_index * 7 + animal_index)
            pair = layer.add_plot(sheet, colx=column, coly=column + 1, type="line")
            pair.color = PAIR_COLOR
            pair.set_float("line.width", PAIR_LW)
    for item_index in range(6):
        group_index = item_index % 2
        point = layer.add_plot(sheet, colx=6 + 4 * item_index, coly=7 + 4 * item_index, type="s")
        point_index = point.index() + 1
        op.lt_exec(
            f"range rp=[{graph.name}]1!{point_index}; set rp -k 2; set rp -z {POINT_SIZE}; set rp -kf 1; "
            f"set rp -kh {POINT_EDGE}; set rp -cse {rgb(EDGES[group_index])}; set rp -csf {rgb(WHITE)};"
        )
    carrier = layer.add_plot(sheet, colx=0, coly=1, colyerr=2, type="s")
    carrier_index = carrier.index() + 1
    op.lt_exec(f"range rc=[{graph.name}]1!{carrier_index}; set rc -k 0; set rc -z 0;")
    error_index = layer.plot_list()[-1].index() + 1
    op.lt_exec(f"range re=[{graph.name}]1!{error_index}; set re -c {rgb(BLACK)}; set re -erw {LW}; set re -erwc 15; set re -erdy 0;")
    ymax = 70.0
    style.set_axes(layer, "Relative band power (%)", 0.45, 3.65, ymax, 10.0)
    for x, label in zip((1, 2, 3), BANDS):
        style.add_line(layer, x, 0, x, -ymax * 0.018)
        style.add_label(layer, label, x + 0.03, -ymax * 0.068)
    style.add_label(layer, stage, 2.0, ymax * 1.06)
    for index, (text, edge) in enumerate((("POCD + SAL (n=7)", EDGES[0]), ("POCD + CNO (n=7)", EDGES[1]))):
        y = ymax * (0.91 - index * 0.08)
        style.add_line(layer, 1.82, y, 2.02, y, width=6, color=edge)
        label = layer.add_label(text, 2.16, y)
        style.style_label(label, 17, False, BLACK)
        label.set_int("attach", 2)
        label.set_int("anchor", 6)
    for center, band in enumerate(BANDS, start=1):
        stat = stats_frame[(stats_frame.stage == stage) & (stats_frame.band == band)].iloc[0]
        if isinstance(stat.stars_holm, str) and stat.stars_holm:
            top = max(
                float(table.loc[2 * (center - 1), "Mean"] + table.loc[2 * (center - 1), "SD"]),
                float(table.loc[2 * (center - 1) + 1, "Mean"] + table.loc[2 * (center - 1) + 1, "SD"]),
                float(table[f"Raw_{2 * (center - 1)}"].max()),
                float(table[f"Raw_{2 * (center - 1) + 1}"].max()),
            )
            style.add_bracket(layer, center - BAND_OFFSET, center + BAND_OFFSET, min(top + 3.0, ymax * 0.88), ymax, stat.stars_holm)
    graph.set_float("width", 6000)
    graph.set_float("height", 4800)
    for key, value in (("left", 15), ("top", 11), ("width", 78), ("height", 72)):
        layer.set_float(key, value)
    return graph


def archive_current(stamp: str) -> None:
    all_stems = list(PSD_STEMS.values()) + list(BAND_STEMS.values()) + [alias for pair in ALIASES.values() for alias in pair]
    for root in (WORK, FORMAL):
        archive = root / "archive" / stamp / "rem_wake_spectral_four_panels"
        for stem in all_stems:
            for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
                source = root / "panels" / f"{stem}{extension}"
                if source.exists():
                    destination = archive / "panels" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        for relative in (Path("data") / DATA_NAME, Path("editable") / PROJECT_NAME):
            source = root / relative
            if source.exists():
                destination = archive / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)


def copy_panel_set(stage_dir: Path, root: Path, stage: str) -> None:
    pairs = (
        (PSD_STEMS[stage], PSD_STEMS[stage]),
        (PSD_STEMS[stage], ALIASES[stage][0]),
        (BAND_STEMS[stage], BAND_STEMS[stage]),
        (BAND_STEMS[stage], ALIASES[stage][1]),
    )
    for source_stem, destination_stem in pairs:
        for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
            destination = root / "panels" / f"{destination_stem}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage_dir / f"{source_stem}{extension}", destination)


def update_manifest() -> None:
    path = FORMAL / "documentation" / "outputs_manifest.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    target_stems = set(PSD_STEMS.values()) | set(BAND_STEMS.values())
    rows = [row for row in rows if not any(stem in row.get("path", "") for stem in target_stems)]
    for stage in STAGES:
        for stem, description in (
            (PSD_STEMS[stage], f"{stage} relative PSD mean +/- SD"),
            (BAND_STEMS[stage], f"{stage} Delta/Theta/Sigma mean +/- SD"),
        ):
            rows.extend([
                {"type": "preview_png", "path": str(FORMAL / "panels" / f"{stem}.png"), "description": description, "status": "current"},
                {"type": "panel_pdf", "path": str(FORMAL / "panels" / f"{stem}.pdf"), "description": description, "status": "current"},
                {"type": "panel_svg", "path": str(FORMAL / "panels" / f"{stem}.svg"), "description": description, "status": "current"},
            ])
    rows.extend([
        {"type": "origin_project", "path": str(FORMAL / "editable" / PROJECT_NAME), "description": "Editable Origin project for REM/Wake spectral panels", "status": "current"},
        {"type": "source_data", "path": str(FORMAL / "data" / DATA_NAME / "animal_level_rem_wake_spectra.csv"), "description": "Latest animal-level REM/Wake spectra", "status": "current"},
        {"type": "stats", "path": str(FORMAL / "data" / DATA_NAME / "rem_wake_band_power_statistics.csv"), "description": "Paired Wilcoxon with Holm correction across three bands within stage", "status": "current"},
    ])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("type", "path", "description", "status"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = validate_source(pd.read_csv(SOURCE))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_current(stamp)
    data_dir = WORK / "data" / DATA_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    if "--cached" in sys.argv:
        spectra = pd.read_csv(data_dir / "animal_level_rem_wake_spectra.csv")
        bands = pd.read_csv(data_dir / "animal_level_rem_wake_band_power.csv")
        statistics = pd.read_csv(data_dir / "rem_wake_band_power_statistics.csv", keep_default_na=False)
        inventory = pd.read_csv(data_dir / "spectral_input_inventory.csv")
    else:
        spectra, bands, statistics, inventory = analyze(source)
    outputs = {
        "animal_level_rem_wake_spectra.csv": spectra,
        "animal_level_rem_wake_band_power.csv": bands,
        "rem_wake_band_power_statistics.csv": statistics,
        "spectral_input_inventory.csv": inventory,
    }
    for name, frame in outputs.items():
        frame.to_csv(data_dir / name, index=False, encoding="utf-8-sig")
    (data_dir / "analysis_parameters.json").write_text(json.dumps({
        "stages": STAGES,
        "sampling_rate_hz": 200,
        "epoch_seconds": 5,
        "reference_hz": [0.5, 25.0],
        "bands_hz": BANDS,
        "spectral_estimator": "five orthogonal sine tapers; epoch-level PSD; animal mean",
        "artifact_rule": "score 0 plus final artifact ranges from export_SO200Hz.artifacts.json",
        "display_error": "sample SD across seven paired animals",
        "statistics": "exact two-sided paired Wilcoxon; Holm across Delta/Theta/Sigma within each stage",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stage_dir = WORK / "panels" / "rem_wake_spectral_latest"
    stage_dir.mkdir(parents=True, exist_ok=True)
    graphs = []
    op.attach()
    try:
        for stage in STAGES:
            graph = build_psd_graph(spectra, stage, data_dir)
            style.export_graph(graph, stage_dir, PSD_STEMS[stage])
            graphs.append(graph)
            graph = build_band_graph(bands, statistics, stage, data_dir)
            style.export_graph(graph, stage_dir, BAND_STEMS[stage])
            graphs.append(graph)
        project = WORK / "editable" / PROJECT_NAME
        graphs[-1].activate()
        op.save(str(project))
    finally:
        op.detach()

    global_provenance = Path(r"F:\Sleep\outputs\figure_provenance")
    global_provenance.mkdir(parents=True, exist_ok=True)
    for root in (WORK, FORMAL):
        copy_panel_set(stage_dir, root, "REM")
        copy_panel_set(stage_dir, root, "WAKE")
        target_data = root / "data" / DATA_NAME
        if target_data != data_dir:
            shutil.copytree(data_dir, target_data, dirs_exist_ok=True)
        target_project = root / "editable" / PROJECT_NAME
        if target_project != project:
            target_project.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project, target_project)
        scripts_dir = root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT_PATH, scripts_dir / SCRIPT_PATH.name)
        dependency_dir = scripts_dir / "origin_scripts"
        dependency_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(style.__file__).resolve(), dependency_dir / Path(style.__file__).name)
        for stage in STAGES:
            metric_stats = statistics[statistics.stage == stage]
            note = f"""# Fig9 {stage} relative PSD and Delta/Theta/Sigma band power

- generated_at: {stamp}
- status: approved formal single panels; combined Fig9 not regenerated
- groups: POCD+SAL and POCD+CNO, paired NO1-NO7; biological replicate is one animal
- sources: data/{DATA_NAME}/spectral_input_inventory.csv contains current EDF/DB3 paths and SHA256 hashes
- analysis: current scores.db3; 5-s {stage} epochs; score-0 and artifact-sidecar exclusions; first EEG channel at 200 Hz; five orthogonal sine tapers; epoch PSD normalized by 0.5-25 Hz power, then averaged within animal
- bands: Delta 0.5-4 Hz; Theta 4-8 Hz; Sigma 10-15 Hz
- display: mean +/- sample SD across animals; PSD SD ribbon; band-power hollow points and same-animal SAL-CNO connections; no animal IDs
- inference: exact two-sided paired Wilcoxon; Holm adjustment across Delta/Theta/Sigma within {stage}; non-significant brackets omitted
- editable: editable/{PROJECT_NAME}
- outputs: panels/{PSD_STEMS[stage]}.* and panels/{BAND_STEMS[stage]}.* plus descriptive aliases
- script: scripts/{SCRIPT_PATH.name}

{metric_stats.to_csv(index=False)}
"""
            docs = root / "documentation"
            docs.mkdir(parents=True, exist_ok=True)
            (docs / f"fig9_{stage.lower()}_spectral_sd_provenance.md").write_text(note, encoding="utf-8")
            (global_provenance / f"fig9_{stage.lower()}_spectral_sd_provenance.md").write_text(note, encoding="utf-8")
            metric_stats.to_csv(global_provenance / f"fig9_{stage.lower()}_spectral_sd_stats.csv", index=False, encoding="utf-8-sig")
            bands[bands.stage == stage].to_csv(global_provenance / f"fig9_{stage.lower()}_spectral_sd_source_data.csv", index=False, encoding="utf-8-sig")
    update_manifest()
    print(statistics.to_string(index=False))
    print(FORMAL / "panels")


if __name__ == "__main__":
    main()
