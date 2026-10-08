from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PACKAGE = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig9_hM4Di_chemogenetics")
SOURCE = PACKAGE / "data" / "sleep_architecture_ma_current_source_data.csv"
OUT = PACKAGE / "data" / "missing_C_H_metrics"
GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
EXPECTED = {group: 7 for group in GROUPS}
XPOS = [0.72, 1.28, 2.72, 3.28]
BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Sigma": (10.0, 15.0)}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select start_time_seconds, score from sleep_scores_table "
            "order by start_time_seconds, start_time_sub_seconds"
        ).fetchall()
    values = np.asarray(rows)
    times = values[:, 0].astype(np.int64)
    scores = values[:, 1].astype(np.int16)
    if len(scores) < 2 or not np.all(np.diff(times) == 5):
        raise RuntimeError(f"Non-consecutive 5-s scores: {path}")
    if not set(np.unique(scores)).issubset({0, 1, 2, 3}):
        raise RuntimeError(f"Unexpected scores in {path}: {np.unique(scores)}")
    return times, scores


def nrem_bouts_seconds(scores: np.ndarray) -> np.ndarray:
    mask = scores == 2
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(float) * 5.0


def read_eeg(path: Path) -> tuple[np.ndarray, float, str, str]:
    with path.open("rb") as stream:
        header = stream.read(256)
        channels = int(header[252:256])
        signal_header = stream.read(256 * channels)
        position = 0
        fields = []
        for width in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]:
            fields.append([
                signal_header[position + i * width: position + (i + 1) * width]
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
        pmin, pmax, dmin, dmax = [float(fields[i][0]) for i in [3, 4, 5, 6]]
        eeg = (digital - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
        return eeg.astype(np.float32), samples_per_record[0] / duration, fields[0][0], fields[2][0]


def interp_psd(psd: np.ndarray, frequencies: np.ndarray, x: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(frequencies, x), 1, len(frequencies) - 1)
    alpha = (x - frequencies[index - 1]) / (frequencies[index] - frequencies[index - 1])
    return psd[:, index - 1] * (1 - alpha) + psd[:, index] * alpha


def integrate(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.r_[low, frequencies[(frequencies > low) & (frequencies < high)], high]
    return np.trapezoid(interp_psd(psd, frequencies, x), x, axis=1)


def deterministic_x(values: np.ndarray, center: float) -> np.ndarray:
    x = np.full(len(values), center, dtype=float)
    rounded = np.round(values, 4)
    for value in np.unique(rounded):
        idx = np.flatnonzero(rounded == value)
        if len(idx) > 1:
            x[idx] += (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.045
    return x


def duration_origin_table(frame: pd.DataFrame) -> pd.DataFrame:
    cells = [frame.loc[frame.group == group].sort_values("animal") for group in GROUPS]
    columns: dict[str, pd.Series] = {}
    for i, cell in enumerate(cells):
        values = cell.nrem_mean_bout_seconds.to_numpy(float)
        columns[f"Raw_{i}"] = pd.Series(values)
    columns["Bar_X"] = pd.Series(XPOS)
    columns["Mean"] = pd.Series([cell.nrem_mean_bout_seconds.mean() for cell in cells])
    columns["SEM"] = pd.Series([
        cell.nrem_mean_bout_seconds.std(ddof=1) / np.sqrt(len(cell)) for cell in cells
    ])
    for i, (center, cell) in enumerate(zip(XPOS, cells)):
        values = cell.nrem_mean_bout_seconds.to_numpy(float)
        columns[f"PointX_{i}"] = pd.Series(deterministic_x(values, center))
        columns[f"PointY_{i}"] = pd.Series(values)
        columns[f"ID_{i}"] = pd.Series(cell.animal.tolist())
    return pd.DataFrame(columns)


def analyze_duration(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in source.itertuples():
        db = Path(row.scores_db_path)
        _, scores = read_scores(db)
        bouts = nrem_bouts_seconds(scores)
        rows.append({
            "group": row.group,
            "animal": row.animal,
            "scores_db_path": str(db),
            "scores_db_sha256": sha256(db),
            "epoch_seconds": 5.0,
            "nrem_bout_count": len(bouts),
            "nrem_total_minutes": float(np.sum(bouts) / 60.0),
            "nrem_mean_bout_seconds": float(np.mean(bouts)),
            "nrem_median_bout_seconds": float(np.median(bouts)),
        })
    frame = pd.DataFrame(rows)
    if frame.groupby("group").size().to_dict() != EXPECTED:
        raise RuntimeError(frame.groupby("group").size().to_dict())
    stats_rows = []
    for condition, a_group, b_group in (
        ("baseline", "baseline+SAL", "baseline+CNO"),
        ("POCD", "POCD+SAL", "POCD+CNO"),
    ):
        a = frame.loc[frame.group == a_group].set_index("animal")["nrem_mean_bout_seconds"].sort_index()
        b = frame.loc[frame.group == b_group].set_index("animal")["nrem_mean_bout_seconds"].sort_index()
        if a.index.tolist() != b.index.tolist() or a.index.tolist() != [f"NO{i}" for i in range(1, 8)]:
            raise RuntimeError(f"Pairing mismatch: {a_group}/{b_group}")
        test = wilcoxon(a.to_numpy(float), b.to_numpy(float), alternative="two-sided", method="exact")
        stats_rows.append({
            "metric": "nrem_mean_bout_seconds", "condition": condition,
            "group_a": a_group, "group_b": b_group, "n_pairs": len(a),
            "paired_ids": "|".join(a.index),
            "mean_a": a.mean(), "sem_a": a.std(ddof=1) / np.sqrt(len(a)),
            "mean_b": b.mean(), "sem_b": b.std(ddof=1) / np.sqrt(len(b)),
            "test": "exact two-sided paired Wilcoxon", "wilcoxon_w": float(test.statistic),
            "p_raw": float(test.pvalue),
        })
    stats = pd.DataFrame(stats_rows)
    stats["p_holm_across_2_conditions"] = holm(stats.p_raw.tolist())
    stats["stars_holm"] = stats.p_holm_across_2_conditions.map(stars)
    return frame, stats


def analyze_spectral(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = source[source.group.isin(["POCD+SAL", "POCD+CNO"])].copy()
    counts = subset.groupby("group").size().to_dict()
    if counts != {"POCD+SAL": 7, "POCD+CNO": 7}:
        raise RuntimeError(counts)
    spectra, band_rows, inventories = [], [], []
    for row in subset.itertuples():
        db = Path(row.scores_db_path)
        folder = db.parent
        edf = folder / "export_SO200Hz.edf"
        artifact_path = folder / "export_SO200Hz.artifacts.json"
        times, scores = read_scores(db)
        artifact_mask = scores == 0
        artifact_status = "score0_only"
        if artifact_path.is_file():
            obj = json.loads(artifact_path.read_text(encoding="utf-8"))
            record = obj.get("records", {}).get("eeg0_emg1")
            if record:
                for start, end in record.get("final_artifact_ranges", []):
                    start = max(0, int(start)); end = min(len(scores) - 1, int(end))
                    if end >= start:
                        artifact_mask[start:end + 1] = True
                artifact_status = "verified_sidecar_plus_score0"
        eeg, fs, label, unit = read_eeg(edf)
        if not np.isclose(fs, 200.0):
            raise RuntimeError(f"Unexpected sampling rate {fs}: {edf}")
        samples_per_epoch = 1000
        required = len(scores) * samples_per_epoch
        if len(eeg) < required:
            raise RuntimeError(f"EEG shorter than scoring: {edf}")
        epochs = eeg[:required].reshape(len(scores), samples_per_epoch)
        valid = (scores == 2) & ~artifact_mask
        values = epochs[valid].astype(float)
        tapers = np.sqrt(2 / 1001) * np.sin(
            np.pi * np.arange(1, 6)[:, None] * np.arange(1, 1001)[None, :] / 1001
        )
        frequencies = np.fft.rfftfreq(samples_per_epoch, 1 / fs)
        grid = np.r_[0.5, frequencies[(frequencies > 0.5) & (frequencies <= 25.0)]]
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
        for index, (band, (low, high)) in enumerate(BANDS.items()):
            band_rows.append({
                "group": row.group, "animal": row.animal, "stage": "NREM",
                "band": band, "low_hz": low, "high_hz": high,
                "valid_epochs": int(valid.sum()),
                "relative_mean_percent": float(relative_bands[:, index].mean()),
            })
        spectra.append(pd.DataFrame({
            "group": row.group, "animal": row.animal, "stage": "NREM",
            "frequency_hz": grid, "relative_psd_percent_per_hz": animal_spectrum,
        }))
        inventories.append({
            "group": row.group, "animal": row.animal, "edf": str(edf), "db3": str(db),
            "edf_sha256": sha256(edf), "db3_sha256": sha256(db),
            "artifact_path": str(artifact_path) if artifact_path.is_file() else "",
            "artifact_status": artifact_status, "NREM_valid_epochs": int(valid.sum()),
            "score_epochs": len(scores), "score_start_unix": int(times[0]),
            "score_end_unix": int(times[-1] + 5), "eeg_label": label, "eeg_unit": unit,
        })
        print(row.group, row.animal, "NREM epochs", int(valid.sum()), flush=True)
    bands = pd.DataFrame(band_rows)
    animal_spectra = pd.concat(spectra, ignore_index=True)
    stats_rows = []
    for band in BANDS:
        a = bands[(bands.group == "POCD+SAL") & (bands.band == band)].set_index("animal").relative_mean_percent.sort_index()
        b = bands[(bands.group == "POCD+CNO") & (bands.band == band)].set_index("animal").relative_mean_percent.sort_index()
        if a.index.tolist() != b.index.tolist() or a.index.tolist() != [f"NO{i}" for i in range(1, 8)]:
            raise RuntimeError(f"Spectral pairing mismatch for {band}")
        test = wilcoxon(a.to_numpy(float), b.to_numpy(float), alternative="two-sided", method="exact")
        stats_rows.append({
            "stage": "NREM", "band": band, "group_a": "POCD+SAL", "group_b": "POCD+CNO",
            "n_pairs": len(a), "paired_ids": "|".join(a.index),
            "mean_a": a.mean(), "sem_a": a.std(ddof=1) / np.sqrt(len(a)),
            "mean_b": b.mean(), "sem_b": b.std(ddof=1) / np.sqrt(len(b)),
            "test": "exact two-sided paired Wilcoxon", "wilcoxon_w": float(test.statistic),
            "p_raw": float(test.pvalue),
        })
    stats = pd.DataFrame(stats_rows)
    stats["p_holm_across_3_bands"] = holm(stats.p_raw.tolist())
    stats["stars_holm"] = stats.p_holm_across_3_bands.map(stars)
    return animal_spectra, bands, stats, pd.DataFrame(inventories)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    duration, duration_stats = analyze_duration(source)
    duration.to_csv(OUT / "animal_level_nrem_bout_duration.csv", index=False, encoding="utf-8-sig")
    duration_stats.to_csv(OUT / "nrem_bout_duration_statistics.csv", index=False, encoding="utf-8-sig")
    duration_origin_table(duration).to_csv(
        OUT / "NREM_bout_duration_Origin_table.csv", index=False, encoding="utf-8-sig"
    )
    duration.groupby("group").nrem_mean_bout_seconds.agg(
        n="count", mean="mean", sd="std", median="median"
    ).assign(sem=lambda x: x.sd / np.sqrt(x.n)).reset_index().to_csv(
        OUT / "nrem_bout_duration_group_summary.csv", index=False, encoding="utf-8-sig"
    )

    spectra, bands, spectral_stats, inventory = analyze_spectral(source)
    spectra.to_csv(OUT / "animal_level_nrem_spectra.csv", index=False, encoding="utf-8-sig")
    bands.to_csv(OUT / "animal_level_nrem_band_power.csv", index=False, encoding="utf-8-sig")
    spectral_stats.to_csv(OUT / "nrem_band_power_statistics.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(OUT / "spectral_input_inventory.csv", index=False, encoding="utf-8-sig")
    (OUT / "analysis_parameters.json").write_text(json.dumps({
        "nrem_bout_definition": "maximal consecutive score==2 run; 5-s epochs",
        "spectral_stage": "NREM", "sampling_rate_hz": 200, "epoch_seconds": 5,
        "reference_hz": [0.5, 25.0], "bands_hz": BANDS,
        "spectral_estimator": "five orthogonal sine tapers; epoch-level PSD; animal mean",
        "artifact_rule": "score 0 plus final artifact ranges from export_SO200Hz.artifacts.json",
        "statistical_unit": "animal",
        "duration_test": "exact two-sided paired Wilcoxon; Holm across baseline and POCD contrasts",
        "spectral_test": "exact two-sided paired Wilcoxon; Holm across Delta/Theta/Sigma",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nNREM BOUT DURATION\n", duration_stats.to_string(index=False))
    print("\nNREM BANDS\n", spectral_stats.to_string(index=False))


if __name__ == "__main__":
    main()
