from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon


WORK = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig8_hM3Dq_chemogenetics")
FORMAL = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig8_hM3Dq_chemogenetics")
GLOBAL_PROV = Path(r"F:\Sleep\outputs\figure_provenance")
SOURCE = WORK / "data" / "Latest_C_H_reanalysis" / "analysis" / "animal_level_all_metrics_normalized.tsv"
SCRIPT = Path(__file__).resolve()
GROUPS = ["mCherry+SAL", "mCherry+CNO", "hmdq3+SAL", "hmdq3+CNO"]
STATES = {"WAKE": 1, "NREM": 2, "REM": 3}
STATE_METRIC = {"NREM": "nrem_percent", "REM": "rem_percent", "WAKE": "wake_percent"}
COLORS = {
    "mCherry+SAL": (0.82, 0.82, 0.82),
    "mCherry+CNO": (0.62, 0.62, 0.62),
    "hmdq3+SAL": (0.95, 0.70, 0.58),
    "hmdq3+CNO": (0.93, 0.48, 0.25),
}
EDGES = {
    "mCherry+SAL": "#666666", "mCherry+CNO": "#666666",
    "hmdq3+SAL": "#d85827", "hmdq3+CNO": "#d85827",
}
DRUG_COLORS = {
    "mCherry": {"SAL": "#bdbdbd", "CNO": "#666666"},
    "hM3Dq": {"SAL": "#f2a181", "CNO": "#e6552d"},
}
POS4 = [0.65, 1.35, 2.65, 3.35]
LABEL4 = ["SAL", "CNO", "SAL", "CNO"]
ANALYSIS_EPOCHS = 17220
EPOCH_SECONDS = 5
MA_MAX_SECONDS = 20
SUSTAINED_WAKE_MIN_SECONDS_EXCLUSIVE = 20
TIME_X = np.array([0, 6, 12, 18], dtype=float)
BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Sigma": (10.0, 15.0)}

PANEL_SPECS = [
    ("A_NREM", "architecture", "NREM"),
    ("B_REM", "architecture", "REM"),
    ("C_Wake", "architecture", "WAKE"),
    ("Fig8_NREM_Delta_Theta_Sigma", "band", "NREM"),
    ("Fig8_REM_Delta_Theta_Sigma", "band", "REM"),
    ("Fig8_Wake_Delta_Theta_Sigma", "band", "WAKE"),
    ("Fig8_bout_rem_duration", "bout", ("REM", "duration")),
    ("Fig8_bout_rem_frequency", "bout", ("REM", "frequency")),
    ("Fig8_bout_wake_duration", "bout", ("WAKE", "duration")),
    ("Fig8_bout_wake_frequency", "bout", ("WAKE", "frequency")),
    ("Fig8_bout_nrem_mcherry_6h_timecourse", "time", ("NREM", "mCherry")),
    ("Fig8_bout_nrem_hm3dq_6h_timecourse", "time", ("NREM", "hM3Dq")),
    ("Fig8_bout_rem_mcherry_6h_timecourse", "time", ("REM", "mCherry")),
    ("Fig8_bout_rem_hm3dq_6h_timecourse", "time", ("REM", "hM3Dq")),
    ("Fig8_bout_wake_mcherry_6h_timecourse", "time", ("WAKE", "mCherry")),
    ("Fig8_bout_wake_hm3dq_6h_timecourse", "time", ("WAKE", "hM3Dq")),
]


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "pdf.compression": 0,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sem(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(a.std(ddof=1) / np.sqrt(len(a)))


def holm(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.maximum.accumulate(np.minimum(1.0, p[order] * np.arange(len(p), 0, -1)))
    return out


def p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                "select start_time_seconds, score from sleep_scores_table "
                "order by start_time_seconds, start_time_sub_seconds"
            ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Unable to read score database {path}: {exc}") from exc
    times = np.asarray([r[0] for r in rows], dtype=float)
    scores = np.asarray([r[1] for r in rows], dtype=int)
    if len(times) > 1 and not np.allclose(np.diff(times), EPOCH_SECONDS):
        raise RuntimeError(f"Non-contiguous 5-s scores: {path}")
    return times, scores


def run_boundaries(scores: np.ndarray, code: int) -> tuple[np.ndarray, np.ndarray]:
    hit = scores == code
    padded = np.r_[False, hit, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    return starts, ends


def bouts(scores: np.ndarray, code: int) -> np.ndarray:
    starts, ends = run_boundaries(scores, code)
    return (ends - starts) * EPOCH_SECONDS


def prepare_score_metrics(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, timeseries, audit = [], [], []
    for r in inventory.itertuples(index=False):
        db = Path(r.scores_db3)
        t, scores_all = read_scores(db)
        if len(scores_all) < ANALYSIS_EPOCHS:
            raise RuntimeError(f"{db} has only {len(scores_all)} epochs")
        scores = scores_all[:ANALYSIS_EPOCHS]
        valid = np.isin(scores, [1, 2, 3])
        valid_hours = valid.sum() * EPOCH_SECONDS / 3600
        base = {
            "group": r.group, "virus": r.virus, "drug": r.drug,
            "subject": r.subject_for_sensitivity, "animal_folder": r.animal,
            "scores_db3": str(db), "analysis_epochs": ANALYSIS_EPOCHS,
            "analysis_hours": ANALYSIS_EPOCHS * EPOCH_SECONDS / 3600,
            "valid_hours": valid_hours,
        }
        nrem_hours = np.sum(scores == STATES["NREM"]) * EPOCH_SECONDS / 3600
        for state, code in STATES.items():
            state_bouts = bouts(scores, code)
            # Macro-architecture retains every WAKE epoch.  Bout analyses use
            # sustained WAKE only, keeping brief arousals separate.
            if state == "WAKE":
                state_bouts = state_bouts[state_bouts > SUSTAINED_WAKE_MIN_SECONDS_EXCLUSIVE]
            base[f"{state.lower()}_percent"] = 100 * np.sum(scores == code) / valid.sum()
            base[f"{state.lower()}_mean_bout_seconds"] = float(np.mean(state_bouts))
            base[f"{state.lower()}_bout_frequency_per_valid_h"] = float(len(state_bouts) / valid_hours)
        wake_starts, wake_ends = run_boundaries(scores, STATES["WAKE"])
        wake_durations = (wake_ends - wake_starts) * EPOCH_SECONDS
        embedded_in_nrem = (
            (wake_starts > 0)
            & (wake_ends < len(scores))
            & (scores[np.maximum(wake_starts - 1, 0)] == STATES["NREM"])
            & (scores[np.minimum(wake_ends, len(scores) - 1)] == STATES["NREM"])
        )
        ma_mask = (wake_durations <= MA_MAX_SECONDS) & embedded_in_nrem
        ma_durations = wake_durations[ma_mask]
        base["ma_count_5_to_20s_nrem_flanked"] = int(ma_mask.sum())
        base["ma_frequency_per_nrem_h"] = float(ma_mask.sum() / nrem_hours) if nrem_hours > 0 else np.nan
        base["ma_mean_duration_seconds"] = float(np.mean(ma_durations)) if len(ma_durations) else np.nan
        base["brief_wake_not_nrem_flanked_count"] = int(((wake_durations <= MA_MAX_SECONDS) & ~embedded_in_nrem).sum())
        base["sustained_wake_definition_seconds"] = ">20"
        base["ma_definition_seconds"] = "5-20, NREM-flanked"
        rows.append(base)
        edges = [0, 4320, 8640, 12960, ANALYSIS_EPOCHS]
        for bin_i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            chunk = scores[a:b]
            ok = np.isin(chunk, [1, 2, 3])
            for state, code in STATES.items():
                timeseries.append({
                    "group": r.group, "virus": r.virus, "drug": r.drug,
                    "subject": r.subject_for_sensitivity, "state": state,
                    "bin_index": bin_i, "zeitgeber_start_h": TIME_X[bin_i],
                    "bin_start_epoch0": a, "bin_end_epoch0_exclusive": b,
                    "percent": 100 * np.sum(chunk == code) / ok.sum(),
                })
        audit.append({
            "group": r.group, "subject": r.subject_for_sensitivity,
            "scores_db3": str(db), "exists": db.exists(), "sha256": sha256(db),
            "modified": datetime.fromtimestamp(db.stat().st_mtime).isoformat(),
            "available_epochs": len(scores_all), "used_epochs": ANALYSIS_EPOCHS,
            "score_start_seconds": float(t[0]), "score_end_seconds": float(t[ANALYSIS_EPOCHS - 1] + EPOCH_SECONDS),
        })
    full = pd.DataFrame(rows)
    time = pd.DataFrame(timeseries)
    return full, time, pd.DataFrame(audit)


def read_edf_first_signal(path: Path) -> tuple[np.ndarray, float]:
    with path.open("rb") as f:
        header = f.read(256)
        ns = int(header[252:256])
        sig = f.read(256 * ns)
        pos, fields = 0, []
        for width in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]:
            fields.append([sig[pos + i * width:pos + (i + 1) * width].decode("latin1").strip() for i in range(ns)])
            pos += ns * width
        spr = list(map(int, fields[8]))
        n_records = int(header[236:244])
        record_duration = float(header[244:252])
        f.seek(int(header[184:192]))
        raw = np.fromfile(f, dtype="<i2").reshape(n_records, sum(spr))[:, :spr[0]].reshape(-1).astype(np.float32)
        physical_min, physical_max, digital_min, digital_max = [float(fields[i][0]) for i in [3, 4, 5, 6]]
        raw = (raw - digital_min) * (physical_max - physical_min) / (digital_max - digital_min) + physical_min
    return raw.astype(np.float32), spr[0] / record_duration


def integrate(psd: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.r_[lo, f[(f > lo) & (f < hi)], hi]
    ix = np.clip(np.searchsorted(f, x), 1, len(f) - 1)
    alpha = (x - f[ix - 1]) / (f[ix] - f[ix - 1])
    y = psd[:, ix - 1] * (1 - alpha) + psd[:, ix] * alpha
    return np.trapezoid(y, x, axis=1)


def artifact_mask(row, scores: np.ndarray) -> tuple[np.ndarray, str]:
    mask = scores == 0
    sidecar = Path(row.edf_path).with_name(Path(row.edf_path).stem + ".artifacts.json")
    if sidecar.exists():
        obj = json.loads(sidecar.read_text(encoding="utf-8"))
        records = obj.get("records", {})
        rec = records.get("eeg0_emg1") or next(iter(records.values()), {})
        for a, b in rec.get("final_artifact_ranges", []):
            a = max(0, int(a)); b = min(len(mask) - 1, int(b))
            if b >= a:
                mask[a:b + 1] = True
        return mask, str(sidecar)
    return mask, ""


def prepare_band_metrics(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = inventory[inventory.virus == "hM3Dq"].copy()
    rows, audit = [], []
    tapers = np.sqrt(2 / 1001) * np.sin(np.pi * np.arange(1, 6)[:, None] * np.arange(1, 1001)[None, :] / 1001)
    for r in subset.itertuples(index=False):
        edf = Path(r.edf_path); db = Path(r.scores_db3)
        _, scores_all = read_scores(db)
        scores = scores_all[:ANALYSIS_EPOCHS]
        mask, artifact_path = artifact_mask(r, scores)
        eeg, fs = read_edf_first_signal(edf)
        if not np.isclose(fs, 200):
            raise RuntimeError(f"Unexpected sampling rate {fs}: {edf}")
        samples_per_epoch = int(round(fs * EPOCH_SECONDS))
        if len(eeg) < ANALYSIS_EPOCHS * samples_per_epoch:
            raise RuntimeError(f"EDF too short: {edf}")
        epochs = eeg[:ANALYSIS_EPOCHS * samples_per_epoch].reshape(ANALYSIS_EPOCHS, samples_per_epoch)
        freq = np.fft.rfftfreq(samples_per_epoch, 1 / fs)
        for state, code in STATES.items():
            idx = np.flatnonzero((scores == code) & ~mask)
            totals = np.zeros(len(BANDS), dtype=float)
            count = 0
            for start in range(0, len(idx), 192):
                block = epochs[idx[start:start + 192]].astype(float)
                psd = sum(
                    np.abs(np.fft.rfft(block * taper, axis=1)) ** 2 / (fs * np.sum(taper ** 2))
                    for taper in tapers
                ) / len(tapers)
                psd[:, 1:-1] *= 2
                den = integrate(psd, freq, 0.5, 25.0)
                vals = np.column_stack([integrate(psd, freq, lo, hi) / den * 100 for lo, hi in BANDS.values()])
                totals += vals.sum(axis=0); count += len(vals)
            for i, (band, (lo, hi)) in enumerate(BANDS.items()):
                rows.append({
                    "group": r.group, "virus": r.virus, "drug": r.drug,
                    "subject": r.subject_for_sensitivity, "animal_folder": r.animal,
                    "state": state, "band": band, "low_hz": lo, "high_hz": hi,
                    "relative_power_percent": totals[i] / count, "valid_epochs": count,
                })
        audit.append({
            "group": r.group, "subject": r.subject_for_sensitivity,
            "edf_path": str(edf), "edf_sha256": sha256(edf),
            "scores_db3": str(db), "scores_sha256": sha256(db),
            "artifact_path": artifact_path,
            "artifact_sha256": sha256(Path(artifact_path)) if artifact_path else "",
            "sampling_rate_hz": fs, "epoch_seconds": EPOCH_SECONDS,
        })
        print(f"spectral recalculated: {r.group} {r.subject_for_sensitivity}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def paired_stats(table: pd.DataFrame, value: str, family: str, panel: str, state: str = "", band: str = "") -> list[dict]:
    output = []
    for virus, g1, g2 in [("mCherry", "mCherry+SAL", "mCherry+CNO"), ("hM3Dq", "hmdq3+SAL", "hmdq3+CNO")]:
        a = table[table.group == g1].set_index("subject")[value].sort_index()
        b = table[table.group == g2].set_index("subject")[value].sort_index()
        if len(a) == 0 and len(b) == 0:
            continue
        if list(a.index) != list(b.index) or len(a) != 6:
            raise RuntimeError(f"Pair mismatch: {panel} {virus}")
        test = wilcoxon(a, b, alternative="two-sided", method="exact")
        output.append({
            "family": family, "panel": panel, "state": state, "band": band,
            "virus": virus, "comparison": "SAL vs CNO", "test": "exact paired Wilcoxon, two-sided",
            "n_pairs": len(a), "statistic": float(test.statistic), "p_raw": float(test.pvalue),
            "mean_sal": float(a.mean()), "sem_sal": sem(a), "mean_cno": float(b.mean()), "sem_cno": sem(b),
        })
    return output


def build_stats(full: pd.DataFrame, time: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ["NREM", "REM", "WAKE"]:
        rows += paired_stats(full, f"{state.lower()}_percent", "architecture", f"{state} percentage", state=state)
    arch = pd.DataFrame(rows)
    arch["p_adjusted"] = arch.groupby("virus")["p_raw"].transform(holm)
    arch["correction_family"] = "Holm across NREM, REM and WAKE within virus"

    band_rows = []
    for state in ["NREM", "REM", "WAKE"]:
        for band in BANDS:
            sub = bands[(bands.state == state) & (bands.band == band)]
            band_rows += paired_stats(sub, "relative_power_percent", "band_power", f"{state} {band}", state=state, band=band)
    band_df = pd.DataFrame(band_rows)
    band_df = band_df[band_df.virus == "hM3Dq"].copy()
    band_df["p_adjusted"] = band_df.groupby("state")["p_raw"].transform(holm)
    band_df["correction_family"] = "Holm across Delta, Theta and Sigma within state"

    bout_rows = []
    for state in ["NREM", "REM", "WAKE"]:
        for kind, suffix in [("duration", "mean_bout_seconds"), ("frequency", "bout_frequency_per_valid_h")]:
            metric = f"{state.lower()}_{suffix}"
            bout_rows += paired_stats(full, metric, "bout", f"{state} {kind}", state=state, band=kind)
    bout_df = pd.DataFrame(bout_rows)
    bout_df["p_adjusted"] = bout_df.groupby("virus")["p_raw"].transform(holm)
    bout_df["correction_family"] = "Holm across six bout outcomes within virus"
    bout_df["metric_definition"] = np.where(
        bout_df.state == "WAKE",
        "sustained WAKE runs >20 s; 5-20 s NREM-flanked events excluded as MA",
        "all contiguous same-state runs",
    )

    time_rows = []
    try:
        from statsmodels.stats.anova import AnovaRM
        for state in ["NREM", "REM", "WAKE"]:
            for virus in ["mCherry", "hM3Dq"]:
                sub = time[(time.state == state) & (time.virus == virus)].copy()
                fit = AnovaRM(sub, "percent", "subject", within=["drug", "bin_index"]).fit()
                for effect, record in fit.anova_table.iterrows():
                    time_rows.append({
                        "family": "timecourse", "panel": f"{state} {virus}", "state": state,
                        "band": "", "virus": virus, "comparison": effect,
                        "test": "two-way repeated-measures ANOVA", "n_pairs": 6,
                        "statistic": float(record["F Value"]), "p_raw": float(record["Pr > F"]),
                        "p_adjusted": np.nan, "correction_family": "model-level test; no pointwise stars displayed",
                        "mean_sal": np.nan, "sem_sal": np.nan, "mean_cno": np.nan, "sem_cno": np.nan,
                    })
    except Exception as exc:
        time_rows.append({"family": "timecourse", "panel": "model_error", "test": repr(exc)})
    stats = pd.concat([arch, band_df, bout_df, pd.DataFrame(time_rows)], ignore_index=True, sort=False)
    # Preserve the established Fig8 convention used by the supplied panels and
    # the current SO panel: stars show the prespecified raw paired comparison.
    # Multiplicity-adjusted values remain explicit sensitivity results.
    stats["display_p"] = stats.p_raw
    stats["display_annotation"] = stats.display_p.apply(lambda x: p_to_star(x) if pd.notna(x) else "")
    return stats


def style_axis(ax):
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", pad=1.5, labelsize=6.5)
    ax.grid(False)


def add_bracket(ax, x1, x2, y, star, height=None):
    if not star:
        return
    if height is None:
        height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
    ax.plot([x1, x1, x2, x2], [y - height, y, y, y - height], color="black", lw=0.85, clip_on=False)
    ax.text((x1 + x2) / 2, y + height * 0.15, star, ha="center", va="bottom", fontsize=8, fontweight="bold")


def plot_four_group(ax, full: pd.DataFrame, value: str, title: str, ylabel: str, ylim, stats: pd.DataFrame, stat_panel: str):
    width = 0.40
    offsets = dict(zip([f"NO{i}" for i in range(1, 7)], np.linspace(-0.035, 0.035, 6)))
    for pair_i, (g1, g2) in enumerate([(GROUPS[0], GROUPS[1]), (GROUPS[2], GROUPS[3])]):
        p1, p2 = POS4[2 * pair_i:2 * pair_i + 2]
        a = full[full.group == g1].set_index("subject")[value].sort_index()
        b = full[full.group == g2].set_index("subject")[value].sort_index()
        for subject in a.index:
            j = offsets[str(subject)]
            ax.plot([p1 + j, p2 + j], [a.loc[subject], b.loc[subject]], color=EDGES[g1], alpha=0.42, lw=0.65, zorder=2)
    for pos, group in zip(POS4, GROUPS):
        sub = full[full.group == group].sort_values("subject")
        vals = sub[value].to_numpy(float)
        ax.bar(pos, vals.mean(), width=width, color=COLORS[group], edgecolor=EDGES[group], linewidth=0.85, zorder=1)
        ax.errorbar(pos, vals.mean(), yerr=sem(vals), fmt="none", ecolor="black", elinewidth=0.85, capsize=2.2, capthick=0.85, zorder=5)
        xx = np.array([pos + offsets[str(s)] for s in sub.subject])
        ax.scatter(xx, vals, s=18, facecolors="white", edgecolors=EDGES[group], linewidths=0.8, zorder=4)
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks(POS4, LABEL4)
    ax.set_xlim(0.25, 3.75); ax.set_ylim(*ylim)
    ax.text(1.0, -0.18, "mCherry", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6.5)
    ax.text(3.0, -0.18, "hM3Dq", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6.5)
    for virus, x1, x2 in [("mCherry", POS4[0], POS4[1]), ("hM3Dq", POS4[2], POS4[3])]:
        row = stats[(stats.panel == stat_panel) & (stats.virus == virus)]
        if len(row):
            star = p_to_star(float(row.iloc[0].p_raw)) if row.iloc[0].family == "architecture" else str(row.iloc[0].display_annotation)
            if star:
                sub = full[full.group.isin([GROUPS[0 if virus == 'mCherry' else 2], GROUPS[1 if virus == 'mCherry' else 3]])]
                y = min(ylim[1] * 0.94, sub[value].max() + (ylim[1] - ylim[0]) * 0.10)
                add_bracket(ax, x1, x2, y, star)
    style_axis(ax)


def plot_band(ax, bands: pd.DataFrame, state: str, stats: pd.DataFrame):
    sub = bands[bands.state == state]
    centers = np.arange(3)
    width = 0.30
    offsets = dict(zip([f"NO{i}" for i in range(1, 7)], np.linspace(-0.025, 0.025, 6)))
    for i, band in enumerate(BANDS):
        a = sub[(sub.band == band) & (sub.drug == "SAL")].set_index("subject")["relative_power_percent"].sort_index()
        b = sub[(sub.band == band) & (sub.drug == "CNO")].set_index("subject")["relative_power_percent"].sort_index()
        for subject in a.index:
            j = offsets[str(subject)]
            ax.plot([i - width / 2 + j, i + width / 2 + j], [a.loc[subject], b.loc[subject]], color="#df7a55", alpha=0.45, lw=0.6, zorder=2)
        for drug, x, color in [("SAL", i - width / 2, COLORS["hmdq3+SAL"]), ("CNO", i + width / 2, COLORS["hmdq3+CNO"])]:
            d = sub[(sub.band == band) & (sub.drug == drug)].sort_values("subject")
            vals = d.relative_power_percent.to_numpy(float)
            ax.bar(x, vals.mean(), width=width * 0.90, color=color, edgecolor="#d85827", linewidth=0.85, zorder=1)
            ax.errorbar(x, vals.mean(), yerr=sem(vals), fmt="none", ecolor="black", elinewidth=0.85, capsize=2.2, capthick=0.85, zorder=5)
            xx = np.array([x + offsets[str(s)] for s in d.subject])
            ax.scatter(xx, vals, s=17, facecolors="white", edgecolors="#d85827", linewidths=0.8, zorder=4)
        row = stats[(stats.family == "band_power") & (stats.state == state) & (stats.band == band)]
        if len(row) and row.iloc[0].display_annotation:
            y = min(66, max(a.max(), b.max()) + 7)
            add_bracket(ax, i - width / 2, i + width / 2, y, str(row.iloc[0].display_annotation), height=1.3)
    ax.set_title(state, fontsize=7.5, pad=4)
    ax.set_ylabel("Relative band power (%)", fontsize=7)
    ax.set_xticks(centers, list(BANDS))
    ax.set_ylim(0, 70)
    ax.plot([], [], color=COLORS["hmdq3+SAL"], lw=4, label="hM3Dq + SAL (n=6)")
    ax.plot([], [], color=COLORS["hmdq3+CNO"], lw=4, label="hM3Dq + CNO (n=6)")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.5)
    style_axis(ax)


def plot_bout(ax, full: pd.DataFrame, state: str, kind: str, stats: pd.DataFrame):
    if kind == "duration":
        value = f"{state.lower()}_mean_bout_seconds"; ylabel = "Duration (s)"
    else:
        value = f"{state.lower()}_bout_frequency_per_valid_h"; ylabel = "Frequency / h"
    vals = full[value].to_numpy(float)
    upper = max(vals) * 1.30
    if kind == "duration":
        step = 20 if upper < 120 else 50
    else:
        step = 0.5 if upper < 5 else 5
    ylim = (0, math.ceil(upper / step) * step)
    title = "Sustained WAKE (>20 s)" if state == "WAKE" else state
    plot_four_group(ax, full, value, title, ylabel, ylim, stats, f"{state} {kind}")


def plot_time(ax, time: pd.DataFrame, state: str, virus: str):
    sub = time[(time.state == state) & (time.virus == virus)]
    for drug in ["SAL", "CNO"]:
        color = DRUG_COLORS[virus][drug]
        pivot = sub[sub.drug == drug].pivot(index="subject", columns="bin_index", values="percent").sort_index()
        mean = pivot.mean(axis=0).to_numpy(); err = pivot.sem(axis=0).to_numpy()
        ax.errorbar(TIME_X, mean, yerr=err, color=color, marker="o", mfc="white", mec=color,
                    ms=3.5, lw=1.1, elinewidth=0.8, capsize=2.0, label=drug)
    ylim = {"NREM": (0, 80), "REM": (0, 20), "WAKE": (0, 100)}[state]
    ax.set_xlim(0, 24); ax.set_ylim(*ylim)
    ax.set_xticks([0, 6, 12, 18, 24], ["ZT0", "ZT6", "ZT12", "ZT18", "ZT24"])
    ax.set_ylabel(f"{state} (%)", fontsize=7)
    ax.set_xlabel("Zeitgeber time", fontsize=7)
    ax.set_title(state, fontsize=7.5, pad=9)
    ax.text(0.98, 0.92, virus, ha="right", va="top", transform=ax.transAxes, fontsize=6.5)
    ax.legend(loc="lower left", fontsize=5.5, handlelength=1.5)
    ybar = 1.025
    ax.plot([0, 0.5], [ybar, ybar], transform=ax.transAxes, color="black", lw=3, solid_capstyle="butt", clip_on=False)
    ax.plot([0.5, 1], [ybar, ybar], transform=ax.transAxes, color="black", lw=3, solid_capstyle="butt", clip_on=False)
    ax.plot([0, 0.5], [ybar, ybar], transform=ax.transAxes, color="white", lw=1.8, solid_capstyle="butt", clip_on=False)
    ax.text(0.25, 1.045, "Light", ha="center", va="bottom", transform=ax.transAxes, fontsize=5.5)
    ax.text(0.75, 1.045, "Dark", ha="center", va="bottom", transform=ax.transAxes, fontsize=5.5)
    style_axis(ax)


def render_panel(ax, kind: str, arg, full, time, bands, stats):
    if kind == "architecture":
        ylims = {"NREM": (0, 60), "REM": (0, 8), "WAKE": (0, 100)}
        plot_four_group(ax, full, f"{arg.lower()}_percent", arg, f"{arg} (%)", ylims[arg], stats, f"{arg} percentage")
    elif kind == "band":
        plot_band(ax, bands, arg, stats)
    elif kind == "bout":
        plot_bout(ax, full, arg[0], arg[1], stats)
    elif kind == "time":
        plot_time(ax, time, arg[0], arg[1])


def save_figure(fig, base: Path, tight=True):
    base.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white", **kw)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", **kw)
    fig.savefig(base.with_suffix(".svg"), facecolor="white", **kw)
    fig.savefig(base.parent / f"{base.name}_AI_clean.pdf", facecolor="white", **kw)
    fig.savefig(base.parent / f"{base.name}_AI_clean.svg", facecolor="white", **kw)


def archive_targets(root: Path, timestamp: str) -> Path:
    archive = root / "archive" / f"before_requested_16panel_redraw_{timestamp}"
    targets = []
    for stem, _, _ in PANEL_SPECS:
        targets.extend((root / "panels").glob(stem + ".*"))
        targets.extend((root / "panels").glob(stem + "_AI_clean.*"))
    targets.extend(root.glob("Fig8_requested_16panel_redraw_latest.*"))
    # Preserve the exact pre-update numeric/statistical record as well as the
    # rendered panels.  This is essential when an animal-day input is replaced.
    targets.extend((root / "data").glob("Fig8_requested_*"))
    targets.extend((root / "documentation").glob("Fig8_requested_16panel_redraw*"))
    targets.extend((root / "scripts").glob(SCRIPT.name))
    targets.extend(root.glob("Fig8_requested_16panel_redraw_latest_AI_clean.*"))
    for src in targets:
        if src.is_file():
            dst = archive / src.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return archive


def update_manifest(root: Path, paths: list[Path]):
    manifest = root / "documentation" / "outputs_manifest.csv"
    old = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame(columns=["type", "path", "description", "status"])
    rel = [str(p.relative_to(root)).replace("/", "\\") for p in paths if p.exists() and root in p.parents]
    if "path" in old:
        old = old[~old.path.astype(str).isin(rel)]
    new = pd.DataFrame([{"type": "figure_output", "path": x, "description": "Fig8 requested 16-panel redraw from current scoring", "status": "current"} for x in rel])
    pd.concat([old, new], ignore_index=True).to_csv(manifest, index=False, encoding="utf-8-sig")


def main():
    inventory = pd.read_csv(SOURCE, sep="\t")
    inventory = inventory[inventory.group.isin(GROUPS)].copy()
    # The hM3Dq+CNO folders were normalized from legacy Dq3_1...Dq3_6 names
    # to NO1...NO6.  Always resolve missing legacy paths to the current,
    # subject-coded folder so a redraw uses today's actual inputs.
    for idx, row in inventory.iterrows():
        db = Path(str(row.scores_db3))
        edf = Path(str(row.edf_path))
        if (not db.exists() or not edf.exists()) and row.group == "hmdq3+CNO":
            current = Path(r"F:\1.Sleep\eXdata\hm3Dq\CNO\hmdq3+CNO") / str(row.subject_for_sensitivity)
            current_db = current / "scores.db3"
            current_edf = current / "export_SO200Hz.edf"
            if current_db.exists() and current_edf.exists():
                inventory.at[idx, "animal"] = str(row.subject_for_sensitivity)
                inventory.at[idx, "scores_db3"] = str(current_db)
                inventory.at[idx, "edf_path"] = str(current_edf)
    if inventory.groupby("group").size().to_dict() != {g: 6 for g in GROUPS}:
        raise RuntimeError(inventory.groupby("group").size().to_dict())
    for group, sub in inventory.groupby("group"):
        if set(sub.subject_for_sensitivity.astype(str)) != {f"NO{i}" for i in range(1, 7)}:
            raise RuntimeError(f"Pairing mismatch: {group}")

    full, time, score_audit = prepare_score_metrics(inventory)
    bands, spectral_audit = prepare_band_metrics(inventory)
    stats = build_stats(full, time, bands)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    temp = SCRIPT.parent / "_fig8_16panel_temp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    for stem, kind, arg in PANEL_SPECS:
        fig, ax = plt.subplots(figsize=(2.10, 1.82))
        render_panel(ax, kind, arg, full, time, bands, stats)
        fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.28)
        save_figure(fig, temp / stem)
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(8.0, 7.5))
    for ax, (_, kind, arg) in zip(axes.flat, PANEL_SPECS):
        render_panel(ax, kind, arg, full, time, bands, stats)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.07, wspace=0.52, hspace=0.78)
    save_figure(fig, temp / "Fig8_requested_16panel_redraw_latest", tight=False)
    plt.close(fig)

    source_long = []
    for r in full.itertuples(index=False):
        for col in [c for c in full.columns if c.endswith("_percent") or c.endswith("_seconds") or c.endswith("_per_valid_h")]:
            source_long.append({"data_family": "full_record", "group": r.group, "virus": r.virus, "drug": r.drug, "subject": r.subject, "metric": col, "state": "", "band_or_bin": "", "value": getattr(r, col)})
    for r in time.itertuples(index=False):
        source_long.append({"data_family": "timecourse", "group": r.group, "virus": r.virus, "drug": r.drug, "subject": r.subject, "metric": "state_percent", "state": r.state, "band_or_bin": f"ZT{int(r.zeitgeber_start_h)}", "value": r.percent})
    for r in bands.itertuples(index=False):
        source_long.append({"data_family": "band_power", "group": r.group, "virus": r.virus, "drug": r.drug, "subject": r.subject, "metric": "relative_power_percent", "state": r.state, "band_or_bin": r.band, "value": r.relative_power_percent})
    source_long = pd.DataFrame(source_long)

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "source": str(SOURCE), "source_sha256": sha256(SOURCE), "archives": {}, "panels": [x[0] for x in PANEL_SPECS]}
    for root in [WORK, FORMAL]:
        archive = archive_targets(root, timestamp)
        report["archives"][str(root)] = str(archive)
        for d in [root / "panels", root / "data", root / "scripts", root / "documentation"]:
            d.mkdir(parents=True, exist_ok=True)
        generated = []
        for stem, _, _ in PANEL_SPECS:
            for src in temp.glob(stem + "*"):
                dst = root / "panels" / src.name
                shutil.copy2(src, dst); generated.append(dst)
        for src in temp.glob("Fig8_requested_16panel_redraw_latest*"):
            dst = root / src.name
            shutil.copy2(src, dst); generated.append(dst)
        tables = {
            "Fig8_requested_16panel_source_data.csv": source_long,
            "Fig8_requested_full_record_metrics.csv": full,
            "Fig8_requested_6h_timecourse_metrics.csv": time,
            "Fig8_requested_band_power_metrics.csv": bands,
            "Fig8_requested_16panel_statistics.csv": stats,
            "Fig8_requested_score_input_audit.csv": score_audit,
            "Fig8_requested_spectral_input_audit.csv": spectral_audit,
            "Fig8_requested_microarousal_classification_metrics.csv": full[[
                "group", "virus", "drug", "subject", "scores_db3", "valid_hours",
                "ma_count_5_to_20s_nrem_flanked", "ma_frequency_per_nrem_h",
                "ma_mean_duration_seconds", "brief_wake_not_nrem_flanked_count",
                "sustained_wake_definition_seconds", "ma_definition_seconds",
            ]].copy(),
        }
        for name, frame in tables.items():
            dst = root / "data" / name
            frame.to_csv(dst, index=False, encoding="utf-8-sig"); generated.append(dst)
        script_dst = root / "scripts" / SCRIPT.name
        shutil.copy2(SCRIPT, script_dst); generated.append(script_dst)
        prov = root / "documentation" / "Fig8_requested_16panel_redraw_provenance.md"
        prov.write_text(
            "# Fig8 requested 16-panel redraw provenance\n\n"
            f"- Generated: {report['generated_at']}\n"
            f"- Current inventory source: `{SOURCE}`; SHA-256 `{report['source_sha256']}`.\n"
            "- Sample: mCherry+SAL, mCherry+CNO, hM3Dq+SAL and hM3Dq+CNO; six confirmed paired animals per condition.\n"
            "- Score-derived panels were recalculated directly from the current 24 scores.db3 files using the first 17,220 contiguous 5-s epochs (23.9167 h).\n"
            "- WAKE percentage retains every WAKE-coded 5-s epoch. WAKE bout duration/frequency use sustained WAKE runs >20 s only (minimum five consecutive epochs); their frequency denominator is valid scored hours.\n"
            "- MA is kept separate: a 5-20 s WAKE run flanked immediately by NREM on both sides. MA frequency uses NREM hours as denominator and is recorded in the dedicated classification table; it is not included in sustained-WAKE bout panels.\n"
            "- REM/NREM bout duration is the animal-level arithmetic mean of all contiguous same-state runs.\n"
            "- Time course uses four bins: ZT0-6, ZT6-12, ZT12-18 and ZT18-23.9167; points are displayed at ZT0, ZT6, ZT12 and ZT18 with the axis extending to ZT24.\n"
            "- Band power was freshly recalculated from the current hM3Dq SAL/CNO EDF and score files: 200 Hz, 5-s epochs, five orthogonal sine tapers, relative to 0.5-25 Hz, artifact epochs excluded.\n"
            "- Bars show mean ± SEM, hollow animal points and confirmed same-animal SAL-CNO traces.\n"
            "- Figure stars use raw exact paired Wilcoxon p values, matching the established Fig8/SO display convention. Holm sensitivity values (three bands per state; six bout outcomes per virus; three architecture states per virus) remain explicit in the stats table. Time courses show no pointwise stars; repeated-measures ANOVA is recorded.\n"
            "- All panels were authored and QA-rendered in Python; standalone PDF/SVG remain editable.\n"
            f"- Superseded live outputs were archived under `{archive}`.\n",
            encoding="utf-8",
        )
        generated.append(prov)
        update_manifest(root, generated)

    GLOBAL_PROV.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        (FORMAL / "documentation" / "Fig8_requested_16panel_redraw_provenance.md", GLOBAL_PROV / "Fig8_requested_16panel_redraw_provenance.md"),
        (FORMAL / "data" / "Fig8_requested_16panel_source_data.csv", GLOBAL_PROV / "Fig8_requested_16panel_source_data.csv"),
        (FORMAL / "data" / "Fig8_requested_16panel_statistics.csv", GLOBAL_PROV / "Fig8_requested_16panel_stats.csv"),
    ]:
        shutil.copy2(src_name, dst_name)

    qa = []
    for path in (FORMAL / "panels").glob("*.png"):
        if path.stem in {x[0] for x in PANEL_SPECS}:
            with Image.open(path) as im:
                qa.append({"path": str(path), "width": im.width, "height": im.height, "mode": im.mode, "nonempty": path.stat().st_size > 1000})
    report["qa"] = qa
    report["statistics"] = stats.to_dict(orient="records")
    (FORMAL / "documentation" / "Fig8_requested_16panel_redraw_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated_at": report["generated_at"], "panels": len(qa), "archives": report["archives"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
