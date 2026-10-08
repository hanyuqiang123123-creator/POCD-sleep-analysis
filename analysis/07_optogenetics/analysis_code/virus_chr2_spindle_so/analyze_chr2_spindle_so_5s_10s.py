from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt


ROOT = Path(r"F:\1.Sleep")
RAW_05 = ROOT / "eXdata" / "VirusCHR2" / "Time" / "Raw" / "05sec"
RAW_10 = ROOT / "eXdata" / "VirusCHR2" / "Time" / "Raw" / "10sec"
OUT = ROOT / "outputs" / "virus_chr2_time" / "spindle_so_5s_10s"
FIG_DIR = ROOT / "Figure" / "VirusCHR2_spindle_so_5s_10s_20260716"
PROV_DIR = ROOT / "outputs" / "figure_provenance"

FS_TARGET = 200.0
EPOCH_SEC = 5.0
PERI_START = -100
PERI_END = 100
BIN_SEC = 5
BASELINE = (-100, -10)
POST = (0, 100)
MIN_NREM_SEC_WINDOW = 20.0

STATE_WAKE = 1
STATE_NREM = 2
STATE_REM = 3
STATE_MA = 4
STATE_UNSCORED = 255


@dataclass
class EdfData:
    path: Path
    fs: float
    labels: list[str]
    signals: dict[str, np.ndarray]
    start_datetime: datetime | None
    duration_sec: float


def parse_edf_start(date_s: str, time_s: str) -> datetime | None:
    date_s = date_s.strip()
    time_s = time_s.strip()
    for fmt in ("%d.%m.%y %H.%M.%S", "%d.%m.%y %H:%M:%S", "%d.%m.%Y %H.%M.%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt)
        except ValueError:
            pass
    return None


def read_edf(path: Path) -> EdfData:
    with path.open("rb") as f:
        fixed = f.read(256)
        start_date = fixed[168:176].decode("latin1", errors="ignore")
        start_time = fixed[176:184].decode("latin1", errors="ignore")
        header_bytes = int(fixed[184:192].decode("latin1", errors="ignore").strip())
        n_records = int(float(fixed[236:244].decode("latin1", errors="ignore").strip()))
        record_duration = float(fixed[244:252].decode("latin1", errors="ignore").strip())
        n_signals = int(fixed[252:256].decode("latin1", errors="ignore").strip())
        rest = f.read(header_bytes - 256)

        def field(offset: int, width: int) -> list[str]:
            out = []
            for i in range(n_signals):
                s = rest[offset + i * width : offset + (i + 1) * width]
                out.append(s.decode("latin1", errors="ignore").strip())
            return out

        offset = 0
        labels = field(offset, 16); offset += 16 * n_signals
        offset += 80 * n_signals  # transducer
        phys_dim = field(offset, 8); offset += 8 * n_signals
        phys_min = np.array([float(x or 0) for x in field(offset, 8)]); offset += 8 * n_signals
        phys_max = np.array([float(x or 0) for x in field(offset, 8)]); offset += 8 * n_signals
        dig_min = np.array([float(x or 0) for x in field(offset, 8)]); offset += 8 * n_signals
        dig_max = np.array([float(x or 0) for x in field(offset, 8)]); offset += 8 * n_signals
        offset += 80 * n_signals  # prefiltering
        samples_per_record = np.array([int(float(x or 0)) for x in field(offset, 8)])

        raw = np.fromfile(f, dtype="<i2")

    total_per_record = int(samples_per_record.sum())
    expected = n_records * total_per_record
    if raw.size < expected:
        n_records = raw.size // total_per_record
        raw = raw[: n_records * total_per_record]
    else:
        raw = raw[:expected]

    chunks = raw.reshape(n_records, total_per_record)
    signals: dict[str, np.ndarray] = {}
    start = 0
    for i, label in enumerate(labels):
        n = samples_per_record[i]
        dig = chunks[:, start : start + n].reshape(-1).astype(float)
        start += n
        scale = (phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i]) if dig_max[i] != dig_min[i] else 1.0
        sig = (dig - dig_min[i]) * scale + phys_min[i]
        signals[label or f"ch{i + 1}"] = sig

    fs = float(samples_per_record[0] / record_duration)
    duration_sec = n_records * record_duration
    return EdfData(path, fs, labels, signals, parse_edf_start(start_date, start_time), duration_sec)


def choose_channels(edf: EdfData) -> tuple[str, str]:
    labels = list(edf.signals.keys())
    lower = [x.lower() for x in labels]
    eeg_candidates = [i for i, x in enumerate(lower) if "eeg" in x or "ch1" in x or "ch 1" in x]
    emg_candidates = [i for i, x in enumerate(lower) if "emg" in x or "ch2" in x or "ch 2" in x]
    eeg_idx = eeg_candidates[0] if eeg_candidates else (1 if len(labels) > 1 else 0)
    emg_idx = emg_candidates[0] if emg_candidates else (2 if len(labels) > 2 else min(1, len(labels) - 1))
    return labels[eeg_idx], labels[emg_idx]


def parse_scores_tsv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Date\tTime\tTime Stamp\tTime from Start"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find score table header in {path}")
    header = lines[header_idx].split("\t")
    try:
        time_idx = header.index("Time from Start")
    except ValueError as exc:
        raise ValueError(f"Cannot find Time from Start in {path}") from exc
    numeric_idx = [i for i, col in enumerate(header) if col.strip().lower().endswith("numeric")]
    score_idx = numeric_idx[-1] if numeric_idx else len(header) - 1
    rows = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) <= max(time_idx, score_idx):
            continue
        rows.append({"time_s": parts[time_idx], "state": parts[score_idx]})
    df = pd.DataFrame(rows)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["state"] = pd.to_numeric(df["state"], errors="coerce").fillna(STATE_UNSCORED).astype(int)
    return df.dropna(subset=["time_s"]).reset_index(drop=True)


def parse_scores_db3(path: Path) -> pd.DataFrame:
    con = sqlite3.connect(path)
    rows = con.execute(
        """
        select start_time_seconds, start_time_sub_seconds,
               end_time_seconds, end_time_sub_seconds, score
        from sleep_scores_table
        order by start_time_seconds, start_time_sub_seconds
        """
    ).fetchall()
    con.close()
    if not rows:
        raise ValueError(f"No score rows in {path}")
    raw_start = np.array([float(r[0]) + float(r[1]) for r in rows], dtype=float)
    raw_end = np.array([float(r[2]) + float(r[3]) for r in rows], dtype=float)
    t0 = raw_start[0]
    df = pd.DataFrame(
        {
            "time_s": raw_start - t0,
            "end_s": raw_end - t0,
            "state": [int(r[4]) for r in rows],
        }
    )
    return recode_ma_scores(df)


def find_score_source(folder: Path, fallback_tsv: str = "scores.tsv") -> Path:
    candidates = [
        folder / "scores0.1sec.db3",
        folder / "Scoring_DB3_TSV" / "scores0.1sec.db3",
        folder / fallback_tsv,
        folder / "scores.tsv",
        folder / "Scoring_DB3_TSV" / fallback_tsv,
        folder / "Scoring_DB3_TSV" / "scores.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No scores0.1sec.db3 or TSV found in {folder}")


def load_scores(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".db3":
        return parse_scores_db3(path)
    return recode_ma_scores(parse_scores_tsv(path))


def recode_ma_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if (out["state"] == STATE_MA).any() or len(out) == 0:
        return out
    states = out["state"].to_numpy(dtype=np.int16).copy()
    wake = states == STATE_WAKE
    changes = np.diff(np.r_[0, wake.astype(np.int8), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        left_nrem = start > 0 and states[start - 1] == STATE_NREM
        right_nrem = end < len(states) and states[end] == STATE_NREM
        if not (left_nrem and right_nrem):
            continue
        if "end_s" in out.columns:
            duration = float(out.iloc[end - 1]["end_s"] - out.iloc[start]["time_s"])
        else:
            times = out["time_s"].to_numpy(dtype=float)
            step = float(np.nanmedian(np.diff(times))) if len(times) > 1 else EPOCH_SEC
            duration = (end - start) * step
        if duration <= 20.0:
            states[start:end] = STATE_MA
    out["state"] = states
    return out


def scores_start_datetime(path: Path) -> datetime | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[:8]:
        if line.startswith("Start:"):
            parts = line.split("\t")
            if len(parts) >= 3:
                raw = parts[-1].strip()
                for fmt in ("%m/%d/%Y %H:%M:%S", "%d.%m.%y %H.%M.%S", "%d.%m.%Y %H.%M.%S"):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        pass
            if len(parts) >= 2:
                raw = parts[1].strip()
                for fmt in ("%d.%m.%y %H.%M.%S", "%d.%m.%Y %H.%M.%S"):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        pass
    return None


def stage_per_sample(scores: pd.DataFrame, n_samples: int, fs: float) -> np.ndarray:
    starts = scores["time_s"].to_numpy(dtype=float)
    states = scores["state"].to_numpy(dtype=np.int16)
    if "end_s" in scores.columns:
        ends = scores["end_s"].to_numpy(dtype=float)
    else:
        step = float(np.nanmedian(np.diff(starts))) if len(starts) > 1 else EPOCH_SEC
        ends = starts + step
    sample_t = np.arange(n_samples, dtype=float) / fs
    idx = np.searchsorted(starts, sample_t, side="right") - 1
    stage = np.full(n_samples, STATE_UNSCORED, dtype=np.int16)
    valid = (idx >= 0) & (idx < len(states)) & (sample_t < ends[np.clip(idx, 0, len(states) - 1)])
    stage[valid] = states[idx[valid]]
    return stage


def butter_filter(x: np.ndarray, fs: float, low: float | None, high: float | None, order: int = 3) -> np.ndarray:
    if low is not None and high is not None:
        sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    elif low is not None:
        sos = signal.butter(order, low, btype="highpass", fs=fs, output="sos")
    elif high is not None:
        sos = signal.butter(order, high, btype="lowpass", fs=fs, output="sos")
    else:
        return x.copy()
    return signal.sosfiltfilt(sos, x)


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def contiguous_true(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def detect_spindles(eeg: np.ndarray, fs: float, stage: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sigma = butter_filter(eeg, fs, 10, 16)
    env = np.abs(signal.hilbert(sigma))
    smooth = moving_mean(env, int(round(fs)))
    nrem = stage == STATE_NREM
    if nrem.sum() < fs * 10:
        return pd.DataFrame(), sigma, smooth
    thr = float(np.nanmean(smooth[nrem]) + 1.5 * np.nanstd(smooth[nrem]))
    cand = (smooth > thr) & nrem
    rows = []
    for start, end in contiguous_true(cand):
        dur = (end - start) / fs
        if 0.4 <= dur <= 2.0:
            seg_env = smooth[start:end]
            peak_rel = int(np.argmax(seg_env)) if seg_env.size else 0
            trough_rel = int(np.argmin(sigma[start:end])) if end > start else 0
            rows.append({
                "onset_s": start / fs,
                "offset_s": end / fs,
                "duration_s": dur,
                "peak_time_s": (start + peak_rel) / fs,
                "peak_envelope": float(seg_env[peak_rel]) if seg_env.size else np.nan,
                "trough_time_s": (start + trough_rel) / fs,
                "trough_uV": float(sigma[start + trough_rel]) if end > start else np.nan,
                "threshold": thr,
            })
    return pd.DataFrame(rows), sigma, smooth


def detect_so(eeg: np.ndarray, fs: float, stage: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    so = butter_filter(eeg, fs, 0.3, 4.5)
    y = so.copy()
    y[stage != STATE_NREM] = 0
    crossings = np.where((y[:-1] >= 0) & (y[1:] < 0))[0] + 1
    rows0 = []
    min_len = int(round(0.5 * fs))
    max_len = int(round(2.0 * fs))
    for a, b in zip(crossings[:-1], crossings[1:]):
        if min_len <= (b - a) <= max_len and np.all(stage[a:b] == STATE_NREM):
            seg = so[a:b]
            neg_rel = int(np.argmin(seg))
            pos_rel = int(np.argmax(seg))
            neg = float(seg[neg_rel])
            pos = float(seg[pos_rel])
            rows0.append((a, b, neg_rel, pos_rel, neg, pos, pos - neg))
    if not rows0:
        phase = np.angle(signal.hilbert(butter_filter(eeg, fs, 0.5, 1.25)))
        return pd.DataFrame(), so, phase
    p2p = np.array([r[6] for r in rows0])
    negs = np.array([r[4] for r in rows0])
    thr_p2p = 0.66 * float(np.mean(p2p))
    thr_neg = 0.66 * float(np.mean(negs))
    phase_sig = butter_filter(eeg, fs, 0.5, 1.25)
    phase = np.angle(signal.hilbert(phase_sig))
    rows = []
    for a, b, neg_rel, pos_rel, neg, pos, amp in rows0:
        if neg < thr_neg and amp > thr_p2p:
            rows.append({
                "onset_s": a / fs,
                "offset_s": b / fs,
                "duration_s": (b - a) / fs,
                "neg_peak_time_s": (a + neg_rel) / fs,
                "pos_peak_time_s": (a + pos_rel) / fs,
                "neg_peak_uV": neg,
                "pos_peak_uV": pos,
                "p2p_uV": amp,
                "frequency_hz": fs / (b - a),
                "threshold_neg_uV": thr_neg,
                "threshold_p2p_uV": thr_p2p,
            })
    return pd.DataFrame(rows), so, phase


def detect_coupling(spindles: pd.DataFrame, sos: pd.DataFrame, phase: np.ndarray, fs: float) -> pd.DataFrame:
    if spindles.empty or sos.empty:
        return pd.DataFrame()
    rows = []
    for si, sp in spindles.iterrows():
        peak = float(sp["peak_time_s"])
        hit = sos[(sos["onset_s"] <= peak) & (sos["offset_s"] >= peak)].copy()
        if hit.empty:
            continue
        hit["dist"] = np.abs(hit["neg_peak_time_s"] - peak)
        so_idx = hit["dist"].idxmin()
        so_row = sos.loc[so_idx]
        sample = int(np.clip(round(peak * fs), 0, len(phase) - 1))
        rows.append({
            "spindle_index": int(si),
            "so_index": int(so_idx),
            "spindle_peak_time_s": peak,
            "spindle_onset_s": float(sp["onset_s"]),
            "spindle_offset_s": float(sp["offset_s"]),
            "so_onset_s": float(so_row["onset_s"]),
            "so_offset_s": float(so_row["offset_s"]),
            "so_neg_peak_time_s": float(so_row["neg_peak_time_s"]),
            "so_phase_at_spindle_peak_rad": float(phase[sample]),
        })
    return pd.DataFrame(rows)


def load_5s_stim_table() -> pd.DataFrame:
    path = RAW_05 / "Group_Stimulation_MA_Results_07_08-5_09_11_12" / "all_animals_stim_MA_summary.csv"
    df = pd.read_csv(path)
    df["mouse"] = df["animal"].astype(str).str.replace("-5", "", regex=False).str.zfill(2)
    df["stim_duration_s"] = 5
    df["condition"] = "5s"
    df = df.rename(columns={"stim_s_from_recording": "stim_s"})
    return df[["mouse", "condition", "stim_index", "clock_time", "stim_s", "stim_duration_s", "schedule_note"]]


def make_10s_stim_table(records: list[dict]) -> pd.DataFrame:
    starts = {
        "07": "16:17",
        "08": "14:26",
        "09": "15:01",
        "10": "08:36",
        "11": "22:19",
        "12": "12:26",
    }
    rows = []
    for rec in records:
        mouse = rec["mouse"]
        score_start = rec.get("edf_start_datetime") or scores_start_datetime(rec["scores"])
        edf_duration = rec["edf_duration_s"]
        hh, mm = [int(x) for x in starts[mouse].split(":")]
        if score_start is None:
            first_s = np.nan
            note = "score start unavailable"
        else:
            stim_dt = score_start.replace(hour=hh, minute=mm, second=0, microsecond=0)
            first_s = (stim_dt - score_start).total_seconds()
            while first_s < 0:
                first_s += 300.0
            note = f"10s schedule from user clock start {starts[mouse]}, every 5 min; first in-recording pulse used"
        stim_index = 1
        stim_s = first_s
        while np.isfinite(stim_s) and stim_s <= edf_duration - 10:
            rows.append({
                "mouse": mouse,
                "condition": "10s",
                "stim_index": stim_index,
                "clock_time": starts[mouse],
                "stim_s": float(stim_s),
                "stim_duration_s": 10,
                "schedule_note": note,
            })
            stim_index += 1
            stim_s += 300.0
    return pd.DataFrame(rows)


def discover_records() -> list[dict]:
    records = []
    for mouse in ["07", "08", "09", "10", "11", "12"]:
        folder = RAW_05 / mouse
        edf = folder / "export_200Hz.edf"
        if mouse == "12" and not edf.exists():
            edf = folder / "export_200Hz_emg_preserve_test.edf"
        try:
            score = find_score_source(folder, "export_scores.tsv" if mouse == "07" else "scores.tsv")
        except FileNotFoundError:
            continue
        if edf.exists() and score.exists():
            records.append({"mouse": mouse, "condition": "5s", "folder": folder, "edf": edf, "scores": score})
    for mouse in ["07", "08", "09", "10", "11", "12"]:
        folder = RAW_10 / str(int(mouse))
        edf = folder / "export_200Hz.edf"
        try:
            score = find_score_source(folder)
        except FileNotFoundError:
            continue
        if edf.exists() and score.exists():
            records.append({"mouse": mouse, "condition": "10s", "folder": folder, "edf": edf, "scores": score})
    return records


def nrem_seconds(stage: np.ndarray, fs: float, start_s: float, end_s: float) -> float:
    a = max(0, int(round(start_s * fs)))
    b = min(len(stage), int(round(end_s * fs)))
    if b <= a:
        return 0.0
    return float(np.isin(stage[a:b], [STATE_NREM, STATE_MA]).sum() / fs)


def count_events(df: pd.DataFrame, time_col: str, start_s: float, end_s: float) -> int:
    if df.empty:
        return 0
    t = df[time_col].to_numpy()
    return int(((t >= start_s) & (t < end_s)).sum())


def analyze_record(rec: dict, stim_table: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict]:
    edf = read_edf(rec["edf"])
    eeg_label, emg_label = choose_channels(edf)
    eeg = edf.signals[eeg_label].astype(float)
    if abs(edf.fs - FS_TARGET) > 1:
        raise ValueError(f"Unexpected sampling rate {edf.fs} in {edf.path}")
    scores = load_scores(rec["scores"])
    score_end = scores["end_s"].max() if "end_s" in scores.columns else scores["time_s"].max() + EPOCH_SEC
    n = min(len(eeg), int(score_end * edf.fs), len(eeg))
    eeg = eeg[:n]
    stage = stage_per_sample(scores, n, edf.fs)
    sp, sigma, sigma_env = detect_spindles(eeg, edf.fs, stage)
    so, so_sig, phase = detect_so(eeg, edf.fs, stage)
    coup = detect_coupling(sp, so, phase, edf.fs)

    event_dir = OUT / "event_tables"
    event_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"mouse{rec['mouse']}_{rec['condition']}"
    sp.to_csv(event_dir / f"{prefix}_spindles.csv", index=False)
    so.to_csv(event_dir / f"{prefix}_so.csv", index=False)
    coup.to_csv(event_dir / f"{prefix}_coupling.csv", index=False)

    stims = stim_table[(stim_table["mouse"] == rec["mouse"]) & (stim_table["condition"] == rec["condition"])].copy()
    bins = np.arange(PERI_START, PERI_END + BIN_SEC, BIN_SEC)
    bin_rows = []
    win_rows = []
    for _, stim in stims.iterrows():
        stim_s = float(stim["stim_s"])
        if stim_s + PERI_START < 0 or stim_s + PERI_END > n / edf.fs:
            continue
        for rel in bins[:-1]:
            a = stim_s + rel
            b = stim_s + rel + BIN_SEC
            nrem_sec = nrem_seconds(stage, edf.fs, a, b)
            row = {
                "mouse": rec["mouse"],
                "condition": rec["condition"],
                "stim_index": int(stim["stim_index"]),
                "rel_time_s": rel,
                "nrem_sec": nrem_sec,
                "wake_pct": 100 * float((stage[int(a * edf.fs): int(b * edf.fs)] == STATE_WAKE).mean()),
                "nrem_pct": 100 * float(np.isin(stage[int(a * edf.fs): int(b * edf.fs)], [STATE_NREM, STATE_MA]).mean()),
                "ma_pct": 100 * float((stage[int(a * edf.fs): int(b * edf.fs)] == STATE_MA).mean()),
                "rem_pct": 100 * float((stage[int(a * edf.fs): int(b * edf.fs)] == STATE_REM).mean()),
            }
            sp_count = count_events(sp, "peak_time_s", a, b)
            so_count = count_events(so, "neg_peak_time_s", a, b)
            cp_count = count_events(coup, "spindle_peak_time_s", a, b)
            row.update({
                "spindle_count": sp_count,
                "so_count": so_count,
                "coupling_count": cp_count,
                "spindle_per_nrem_min": sp_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
                "so_per_nrem_min": so_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
                "coupling_per_nrem_min": cp_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
            })
            bin_rows.append(row)
        for window_name, (lo, hi) in {"baseline": BASELINE, "post_0_100": POST, "early_0_30": (0, 30), "late_30_100": (30, 100)}.items():
            a = stim_s + lo
            b = stim_s + hi
            nrem_sec = nrem_seconds(stage, edf.fs, a, b)
            sp_count = count_events(sp, "peak_time_s", a, b)
            so_count = count_events(so, "neg_peak_time_s", a, b)
            cp_count = count_events(coup, "spindle_peak_time_s", a, b)
            win_rows.append({
                "mouse": rec["mouse"],
                "condition": rec["condition"],
                "stim_index": int(stim["stim_index"]),
                "window": window_name,
                "nrem_sec": nrem_sec,
                "valid_nrem_window": nrem_sec >= MIN_NREM_SEC_WINDOW,
                "spindle_count": sp_count,
                "so_count": so_count,
                "coupling_count": cp_count,
                "spindle_per_nrem_min": sp_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
                "so_per_nrem_min": so_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
                "coupling_per_nrem_min": cp_count / (nrem_sec / 60) if nrem_sec > 0 else np.nan,
                "pct_spindles_coupled": 100 * cp_count / sp_count if sp_count > 0 else np.nan,
                "pct_so_coupled": 100 * cp_count / so_count if so_count > 0 else np.nan,
            })

    qc = {
        "mouse": rec["mouse"],
        "condition": rec["condition"],
        "edf": str(rec["edf"]),
        "scores": str(rec["scores"]),
        "edf_fs": edf.fs,
        "edf_duration_s": edf.duration_sec,
        "used_duration_s": n / edf.fs,
        "eeg_label": eeg_label,
        "emg_label": emg_label,
        "score_epochs": len(scores),
        "stim_trials_available": len(stims),
        "spindles_detected": len(sp),
        "so_detected": len(so),
        "couplings_detected": len(coup),
        "nrem_min_stage2": float((stage == STATE_NREM).sum() / edf.fs / 60),
    }
    return [pd.DataFrame(bin_rows)], [pd.DataFrame(win_rows)], qc


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def make_stats(mouse_window: pd.DataFrame) -> pd.DataFrame:
    metrics = ["spindle_per_nrem_min", "so_per_nrem_min", "coupling_per_nrem_min", "pct_spindles_coupled", "pct_so_coupled"]
    rows = []
    for cond in ["5s", "10s"]:
        for metric in metrics:
            sub = mouse_window[(mouse_window["condition"] == cond) & (mouse_window["window"].isin(["baseline", "post_0_100"]))]
            piv = sub.pivot(index="mouse", columns="window", values=metric).dropna()
            if {"baseline", "post_0_100"}.issubset(piv.columns) and len(piv) >= 2:
                diff = piv["post_0_100"] - piv["baseline"]
                try:
                    w_p = stats.wilcoxon(piv["baseline"], piv["post_0_100"]).pvalue if len(diff) >= 3 and np.any(diff != 0) else np.nan
                except ValueError:
                    w_p = np.nan
                t_p = stats.ttest_rel(piv["baseline"], piv["post_0_100"], nan_policy="omit").pvalue if len(diff) >= 2 else np.nan
                rows.append({
                    "comparison": f"{cond} post_0_100 vs baseline",
                    "metric": metric,
                    "n_mice": len(piv),
                    "baseline_mean": piv["baseline"].mean(),
                    "post_mean": piv["post_0_100"].mean(),
                    "delta_mean": diff.mean(),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                })
    for metric in metrics:
        deltas = []
        for cond in ["5s", "10s"]:
            sub = mouse_window[(mouse_window["condition"] == cond) & (mouse_window["window"].isin(["baseline", "post_0_100"]))]
            piv = sub.pivot(index="mouse", columns="window", values=metric).dropna()
            if not {"baseline", "post_0_100"}.issubset(piv.columns):
                continue
            piv[f"delta_{cond}"] = piv["post_0_100"] - piv["baseline"]
            deltas.append(piv[[f"delta_{cond}"]])
        if len(deltas) < 2:
            continue
        merged = deltas[0].join(deltas[1], how="inner").dropna()
        if len(merged) >= 2:
            diff = merged["delta_10s"] - merged["delta_5s"]
            rows.append({
                "comparison": "delta_10s vs delta_5s",
                "metric": metric,
                "n_mice": len(merged),
                "baseline_mean": np.nan,
                "post_mean": np.nan,
                "delta_mean": diff.mean(),
                "paired_t_p": stats.ttest_rel(merged["delta_5s"], merged["delta_10s"], nan_policy="omit").pvalue,
                "wilcoxon_p": stats.wilcoxon(merged["delta_5s"], merged["delta_10s"]).pvalue if len(merged) >= 3 and np.any(diff != 0) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_peri(mouse_bin: pd.DataFrame, fig_dir: Path) -> list[Path]:
    metrics = [
        ("spindle_per_nrem_min", "Spindles / NREM min"),
        ("so_per_nrem_min", "SO / NREM min"),
        ("coupling_per_nrem_min", "SO-spindle / NREM min"),
        ("nrem_pct", "NREM + MA (%)"),
        ("wake_pct", "Wake (%)"),
    ]
    colors = {"5s": "#4C9BE8", "10s": "#7A4BB2"}
    fig, axes = plt.subplots(len(metrics), 1, figsize=(4.3, 7.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for cond in ["5s", "10s"]:
            sub = mouse_bin[mouse_bin["condition"] == cond]
            g = sub.groupby("rel_time_s")[metric].agg(["mean", sem, "count"]).reset_index()
            ax.plot(g["rel_time_s"], g["mean"], color=colors[cond], lw=1.5, label=cond)
            ax.fill_between(g["rel_time_s"].to_numpy(), (g["mean"] - g["sem"]).to_numpy(), (g["mean"] + g["sem"]).to_numpy(), color=colors[cond], alpha=0.18, lw=0)
        ax.axvspan(0, 5, color="#4C9BE8", alpha=0.13, lw=0)
        ax.axvspan(0, 10, color="#7A4BB2", alpha=0.08, lw=0)
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time from stimulation onset (s)", fontsize=9)
    axes[-1].set_xlim(PERI_START, PERI_END)
    fig.tight_layout()
    paths = []
    for ext in ["png", "pdf", "svg"]:
        p = fig_dir / f"virus_chr2_spindle_so_5s_10s_peristimulus.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_peri_separate(mouse_bin: pd.DataFrame, fig_dir: Path) -> list[Path]:
    metrics = [
        ("spindle_per_nrem_min", "Spindles / NREM min"),
        ("so_per_nrem_min", "SO / NREM min"),
        ("coupling_per_nrem_min", "SO-spindle / NREM min"),
    ]
    metric_names = {
        "spindle_per_nrem_min": "spindle",
        "so_per_nrem_min": "so",
        "coupling_per_nrem_min": "so_spindle_coupling",
    }
    colors = {"5s": "#4C9BE8", "10s": "#7A4BB2"}
    laser_dur = {"5s": 5, "10s": 10}
    paths = []
    for cond in ["5s", "10s"]:
        sub_cond = mouse_bin[mouse_bin["condition"] == cond]
        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(2.75, 2.0))
            g = sub_cond.groupby("rel_time_s")[metric].agg(["mean", sem, "count"]).reset_index()
            x = g["rel_time_s"].to_numpy(dtype=float)
            mean = g["mean"].to_numpy(dtype=float)
            err = g["sem"].to_numpy(dtype=float)
            ax.plot(x, mean, color=colors[cond], lw=1.5)
            ax.fill_between(x, mean - err, mean + err, color=colors[cond], alpha=0.18, lw=0)
            ax.axvspan(0, laser_dur[cond], color=colors[cond], alpha=0.16, lw=0)
            ax.axvline(0, color="0.2", lw=0.8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_title(cond.replace("s", " s"), fontsize=10, pad=2)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_xlim(PERI_START, PERI_END)
            fig.tight_layout(pad=0.7)
            fig.subplots_adjust(left=0.34, bottom=0.24, top=0.82)
            for ext in ["png", "pdf", "svg"]:
                p = fig_dir / f"virus_chr2_{cond}_{metric_names[metric]}_peristimulus_separate.{ext}"
                fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
                paths.append(p)
            plt.close(fig)
    return paths


def plot_paired(mouse_window: pd.DataFrame, fig_dir: Path) -> list[Path]:
    metrics = [
        ("spindle_per_nrem_min", "Spindles / NREM min"),
        ("so_per_nrem_min", "SO / NREM min"),
        ("coupling_per_nrem_min", "SO-spindle / NREM min"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    colors = {"5s": "#4C9BE8", "10s": "#7A4BB2"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        xpos = {"5s_baseline": 0, "5s_post_0_100": 1, "10s_baseline": 3, "10s_post_0_100": 4}
        for cond in ["5s", "10s"]:
            sub = mouse_window[(mouse_window["condition"] == cond) & (mouse_window["window"].isin(["baseline", "post_0_100"]))]
            piv = sub.pivot(index="mouse", columns="window", values=metric)
            if not {"baseline", "post_0_100"}.issubset(piv.columns):
                continue
            for mouse, row in piv.iterrows():
                if np.isfinite(row.get("baseline", np.nan)) and np.isfinite(row.get("post_0_100", np.nan)):
                    xs = [xpos[f"{cond}_baseline"], xpos[f"{cond}_post_0_100"]]
                    ys = [row["baseline"], row["post_0_100"]]
                    ax.plot(xs, ys, color=colors[cond], alpha=0.35, lw=0.8)
                    ax.scatter(xs, ys, s=13, facecolor="white", edgecolor=colors[cond], lw=0.8, zorder=3)
            means = piv[["baseline", "post_0_100"]].mean()
            sems = piv[["baseline", "post_0_100"]].sem()
            ax.errorbar([xpos[f"{cond}_baseline"], xpos[f"{cond}_post_0_100"]], means, yerr=sems, color=colors[cond], lw=1.6, capsize=2, marker="o", ms=3)
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels(["5s\nbase", "5s\npost", "10s\nbase", "10s\npost"], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    paths = []
    for ext in ["png", "pdf", "svg"]:
        p = fig_dir / f"virus_chr2_spindle_so_5s_10s_paired_summary.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_paired_separate(mouse_window: pd.DataFrame, fig_dir: Path) -> list[Path]:
    metrics = [
        ("spindle_per_nrem_min", "Spindles / NREM min"),
        ("so_per_nrem_min", "SO / NREM min"),
        ("coupling_per_nrem_min", "SO-spindle / NREM min"),
    ]
    metric_names = {
        "spindle_per_nrem_min": "spindle",
        "so_per_nrem_min": "so",
        "coupling_per_nrem_min": "so_spindle_coupling",
    }
    colors = {"5s": "#4C9BE8", "10s": "#7A4BB2"}
    paths = []
    for cond in ["5s", "10s"]:
        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(2.05, 2.05))
            sub = mouse_window[(mouse_window["condition"] == cond) & (mouse_window["window"].isin(["baseline", "post_0_100"]))]
            piv = sub.pivot(index="mouse", columns="window", values=metric)
            if {"baseline", "post_0_100"}.issubset(piv.columns):
                for _, row in piv.iterrows():
                    if np.isfinite(row.get("baseline", np.nan)) and np.isfinite(row.get("post_0_100", np.nan)):
                        ax.plot([0, 1], [row["baseline"], row["post_0_100"]], color=colors[cond], alpha=0.35, lw=0.8)
                        ax.scatter([0, 1], [row["baseline"], row["post_0_100"]], s=13, facecolor="white", edgecolor=colors[cond], lw=0.8, zorder=3)
                means = piv[["baseline", "post_0_100"]].mean()
                sems = piv[["baseline", "post_0_100"]].sem()
                ax.errorbar([0, 1], means, yerr=sems, color=colors[cond], lw=1.6, capsize=2, marker="o", ms=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["base", "post"], fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_title(cond.replace("s", " s"), fontsize=10, pad=2)
            fig.tight_layout(pad=0.7)
            fig.subplots_adjust(left=0.38, bottom=0.22, top=0.82)
            for ext in ["png", "pdf", "svg"]:
                p = fig_dir / f"virus_chr2_{cond}_{metric_names[metric]}_paired_separate.{ext}"
                fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
                paths.append(p)
            plt.close(fig)
    return paths


def write_metadata(paths: list[Path], qc: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    source_rows = []
    for _, row in qc.iterrows():
        source_rows.append({"role": f"{row['condition']} mouse {row['mouse']} EDF", "path": row["edf"], "description": row["eeg_label"]})
        source_rows.append({"role": f"{row['condition']} mouse {row['mouse']} scores", "path": row["scores"], "description": "0.1-s sleep scoring DB3 when available; TSV fallback otherwise"})
    pd.DataFrame(source_rows).to_csv(FIG_DIR / "source_paths.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"path": str(p), "format": p.suffix.lstrip("."), "note": "generated by analyze_chr2_spindle_so_5s_10s.py"} for p in paths]).to_csv(FIG_DIR / "outputs_manifest.csv", index=False, encoding="utf-8-sig")
    readme = f"""# VirusCHR2 Spindle/SO 5s vs 10s

Purpose: test whether 5 s and 10 s blue-light stimulation reduce NREM spindle, SO, and SO-spindle coupling metrics in VirusCHR2 Time recordings.

Analysis script: `F:\\1.Sleep\\analysis_code\\virus_chr2_spindle_so\\analyze_chr2_spindle_so_5s_10s.py`

Detection summary:
- EDF: `export_200Hz.edf` or documented 200 Hz replacement.
- Scoring: current `scores0.1sec.db3` when available, with TSV fallback only if a 0.1-s DB3 is absent.
- Spindles: NREM stage 2, 10-16 Hz, Hilbert envelope smoothed with 1 s moving average, threshold NREM mean + 1.5 SD, duration 0.4-2.0 s.
- SO: NREM stage 2, 0.3-4.5 Hz, positive-to-negative zero-crossing cycle, duration 0.5-2.0 s, loose amplitude thresholds from project MATLAB pipeline.
- Coupling: spindle envelope peak within detected SO cycle; SO phase at spindle peak also exported.
- Event densities are normalized by NREM+MA seconds in each window, following the project convention that MA epochs are retained in NREM-normalized event denominators.

Windows:
- Peri-stimulus curve: {PERI_START} to {PERI_END} s in {BIN_SEC}-s bins.
- Main summary: baseline {BASELINE[0]} to {BASELINE[1]} s; post {POST[0]} to {POST[1]} s.
- Trial windows with < {MIN_NREM_SEC_WINDOW} s NREM+MA are flagged and excluded from mouse-level window averages.

Caution:
- NREM-normalized post-stimulation metrics are only averaged for trial windows with at least {MIN_NREM_SEC_WINDOW} s NREM+MA. In the 10-s condition, many post windows are excluded because stimulation strongly drives sustained wakefulness and leaves little/no NREM.
- Coupling currently uses detected SO zero-crossing cycle containment; exported phase values allow later replacement with exact MATLAB 90-degree phase-cycle boundaries.

Outputs:
{chr(10).join('- ' + str(p) for p in paths)}
"""
    (FIG_DIR / "README_analysis.md").write_text(readme, encoding="utf-8")
    shutil.copy2(ROOT / "analysis_code" / "virus_chr2_spindle_so" / "analyze_chr2_spindle_so_5s_10s.py", FIG_DIR / "analyze_chr2_spindle_so_5s_10s.py")

    PROV_DIR.mkdir(parents=True, exist_ok=True)
    prov = PROV_DIR / "virus_chr2_spindle_so_5s_10s_provenance.md"
    prov.write_text(readme.replace("# VirusCHR2 Spindle/SO 5s vs 10s", "# Provenance: VirusCHR2 spindle/SO 5s vs 10s"), encoding="utf-8")
    stats_df.to_csv(PROV_DIR / "virus_chr2_spindle_so_5s_10s_stats.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = discover_records()

    # Read EDF durations once for the 10-s schedule.
    rec_meta = []
    for rec in records:
        edf = read_edf(rec["edf"])
        rec["edf_duration_s"] = edf.duration_sec
        rec["edf_start_datetime"] = edf.start_datetime
        rec_meta.append({k: str(v) if isinstance(v, Path) else v for k, v in rec.items()})
    stim_5 = load_5s_stim_table()
    stim_10 = make_10s_stim_table([r for r in records if r["condition"] == "10s"])
    stim_table = pd.concat([stim_5, stim_10], ignore_index=True)
    stim_table.to_csv(OUT / "stimulation_schedule_5s_10s.csv", index=False, encoding="utf-8-sig")

    bin_all = []
    win_all = []
    qcs = []
    for rec in records:
        bins, wins, qc = analyze_record(rec, stim_table)
        bin_all.extend(bins)
        win_all.extend(wins)
        qcs.append(qc)
        print(f"done mouse {rec['mouse']} {rec['condition']}: sp={qc['spindles_detected']} so={qc['so_detected']} coupling={qc['couplings_detected']}")

    trial_bin = pd.concat([x for x in bin_all if not x.empty], ignore_index=True)
    trial_win = pd.concat([x for x in win_all if not x.empty], ignore_index=True)
    qc_df = pd.DataFrame(qcs)
    qc_df.to_csv(OUT / "qc_record_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rec_meta).to_csv(OUT / "input_records.csv", index=False, encoding="utf-8-sig")
    trial_bin.to_csv(OUT / "peri_stimulus_by_trial_5s_10s.csv", index=False, encoding="utf-8-sig")
    trial_win.to_csv(OUT / "window_metrics_by_trial_5s_10s.csv", index=False, encoding="utf-8-sig")

    valid_win = trial_win[trial_win["valid_nrem_window"]].copy()
    metric_cols = ["nrem_sec", "spindle_per_nrem_min", "so_per_nrem_min", "coupling_per_nrem_min", "pct_spindles_coupled", "pct_so_coupled"]
    mouse_win = valid_win.groupby(["mouse", "condition", "window"], as_index=False)[metric_cols].mean()
    mouse_bin = trial_bin.groupby(["mouse", "condition", "rel_time_s"], as_index=False)[["wake_pct", "nrem_pct", "ma_pct", "rem_pct", "spindle_per_nrem_min", "so_per_nrem_min", "coupling_per_nrem_min"]].mean()
    mouse_win.to_csv(OUT / "window_metrics_by_mouse_5s_10s.csv", index=False, encoding="utf-8-sig")
    mouse_bin.to_csv(OUT / "peri_stimulus_by_mouse_5s_10s.csv", index=False, encoding="utf-8-sig")

    stats_df = make_stats(mouse_win)
    stats_df.to_csv(OUT / "stats_5s_10s.csv", index=False, encoding="utf-8-sig")
    fig_paths = plot_peri(mouse_bin, FIG_DIR)
    fig_paths.extend(plot_peri_separate(mouse_bin, FIG_DIR))
    fig_paths.extend(plot_paired(mouse_win, FIG_DIR))
    fig_paths.extend(plot_paired_separate(mouse_win, FIG_DIR))
    for p in fig_paths:
        shutil.copy2(p, OUT / p.name)
    write_metadata(fig_paths, qc_df, stats_df)
    print(json.dumps({"out": str(OUT), "figure_dir": str(FIG_DIR), "figures": [str(p) for p in fig_paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
