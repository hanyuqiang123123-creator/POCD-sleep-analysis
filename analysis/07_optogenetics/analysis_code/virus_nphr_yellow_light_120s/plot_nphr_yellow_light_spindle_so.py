from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RAW_ROOT = Path(r"F:\1.Sleep\eXdata\VirusNPHR\NPHR3.0\PerStim120")
FIG_DIR = Path(r"F:\Sleep\Figure\VirusNPHR_yellow_light_spindle_so_20260718")
PROV_DIR = Path(r"F:\Sleep\outputs\figure_provenance")
DETECTOR_SCRIPT = Path(r"F:\Sleep\analysis_code\virus_chr2_spindle_so\analyze_chr2_spindle_so_5s_10s.py")
SCRIPT_PATH = Path(__file__).resolve()
FIGURE_ID = "virus_nphr_yellow_light_spindle_so"

LIGHT_ONSET_S = 300.0
LIGHT_OFFSET_S = 420.0
WINDOWS = {
    "No light": (180.0, 300.0),
    "Yellow light": (300.0, 420.0),
    "Post light": (420.0, 540.0),
}

CURVE_START_S = -120.0
CURVE_END_S = 240.0
CURVE_STEP_S = 5.0
CURVE_WINDOW_S = 30.0
MIN_NREM_CURVE_S = 10.0
MIN_NREM_SUMMARY_S = 20.0

METRICS = [
    ("spindle_per_nrem_min", "Spindles / NREM min", "spindle", "peak_time_s"),
    ("so_per_nrem_min", "SO / NREM min", "so", "neg_peak_time_s"),
    ("coupling_per_nrem_min", "SO-spindle / NREM min", "coupling", "spindle_peak_time_s"),
]

LINE_COLOR = "#4C9BE8"
FILL_COLOR = "#A9D3FF"
LIGHT_COLOR = "#FFE36E"
LIGHT_EDGE = "#C99800"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.compression": 0,
        "font.size": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "legend.frameon": False,
    }
)


def load_detector_module():
    if not DETECTOR_SCRIPT.exists():
        raise FileNotFoundError(f"Project detector not found: {DETECTOR_SCRIPT}")
    spec = importlib.util.spec_from_file_location("virus_chr2_spindle_so_detector", DETECTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


detector = load_detector_module()


def animal_sort_key(animal: str) -> int:
    return int(animal[1:])


def sem(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan


def discover_records() -> list[dict]:
    records = []
    for score_file in RAW_ROOT.glob("N*/*/export_scores.tsv"):
        animal, trial = score_file.relative_to(RAW_ROOT).parts[:2]
        edf_file = score_file.with_name("export_200.edf")
        if not edf_file.exists():
            raise FileNotFoundError(f"Missing paired EDF: {edf_file}")
        records.append({"animal": animal, "trial": trial, "edf": edf_file, "scores": score_file})
    return sorted(records, key=lambda r: (animal_sort_key(r["animal"]), int(r["trial"])))


def safe_event_table(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    if not df.empty:
        return df.copy()
    return pd.DataFrame(columns=expected_columns)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    if not len(valid_idx):
        return pd.Series(out, index=p_values.index)
    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return pd.Series(out, index=p_values.index)


def analyze_records(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_root = FIG_DIR / "event_tables"
    event_root.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict] = []
    window_rows: list[dict] = []
    qc_rows: list[dict] = []
    source_rows: list[dict] = [
        {
            "role": "project_detector",
            "path": str(DETECTOR_SCRIPT),
            "description": "Established Python detector previously used for VirusCHR2 spindle/SO/coupling figures.",
        }
    ]
    all_events: list[pd.DataFrame] = []

    centers = np.arange(CURVE_START_S, CURVE_END_S + CURVE_STEP_S / 2, CURVE_STEP_S)
    half_window = CURVE_WINDOW_S / 2.0

    for record in records:
        animal, trial = record["animal"], record["trial"]
        edf = detector.read_edf(record["edf"])
        eeg_label, emg_label = detector.choose_channels(edf)
        if abs(edf.fs - 200.0) > 1e-6:
            raise ValueError(f"Expected 200 Hz EDF: {record['edf']} ({edf.fs} Hz)")

        scores = detector.load_scores(record["scores"])
        score_end = float(scores["time_s"].max() + 5.0)
        n_samples = min(len(edf.signals[eeg_label]), int(round(score_end * edf.fs)))
        eeg = edf.signals[eeg_label][:n_samples].astype(float)
        stage = detector.stage_per_sample(scores, n_samples, edf.fs)

        spindles, _, _ = detector.detect_spindles(eeg, edf.fs, stage)
        slow_oscillations, _, phase = detector.detect_so(eeg, edf.fs, stage)
        coupling = detector.detect_coupling(spindles, slow_oscillations, phase, edf.fs)

        spindles = safe_event_table(
            spindles,
            ["onset_s", "offset_s", "duration_s", "peak_time_s", "peak_envelope", "trough_time_s", "trough_uV", "threshold"],
        )
        slow_oscillations = safe_event_table(
            slow_oscillations,
            [
                "onset_s",
                "offset_s",
                "duration_s",
                "neg_peak_time_s",
                "pos_peak_time_s",
                "neg_peak_uV",
                "pos_peak_uV",
                "p2p_uV",
                "frequency_hz",
                "threshold_neg_uV",
                "threshold_p2p_uV",
            ],
        )
        coupling = safe_event_table(
            coupling,
            [
                "spindle_index",
                "so_index",
                "spindle_peak_time_s",
                "spindle_onset_s",
                "spindle_offset_s",
                "so_onset_s",
                "so_offset_s",
                "so_neg_peak_time_s",
                "so_phase_at_spindle_peak_rad",
            ],
        )

        tables = {"spindle": spindles, "so": slow_oscillations, "coupling": coupling}
        for event_type, table in tables.items():
            tagged = table.copy()
            tagged.insert(0, "event_type", event_type)
            tagged.insert(0, "trial", trial)
            tagged.insert(0, "animal", animal)
            event_path = event_root / f"{animal}_trial{trial}_{event_type}_events.csv"
            tagged.to_csv(event_path, index=False, encoding="utf-8-sig")
            all_events.append(tagged)
            source_rows.append(
                {
                    "role": f"derived_{animal}_trial{trial}_{event_type}",
                    "path": str(event_path),
                    "description": "Event-level table generated by this analysis.",
                }
            )

        for rel_center in centers:
            start_s = LIGHT_ONSET_S + rel_center - half_window
            end_s = LIGHT_ONSET_S + rel_center + half_window
            nrem_s = detector.nrem_seconds(stage, edf.fs, start_s, end_s)
            row = {
                "animal": animal,
                "trial": trial,
                "rel_time_s": float(rel_center),
                "window_start_s": float(start_s),
                "window_end_s": float(end_s),
                "nrem_plus_ma_s": nrem_s,
                "valid_nrem": bool(nrem_s >= MIN_NREM_CURVE_S),
            }
            for metric, _, event_type, time_col in METRICS:
                count = detector.count_events(tables[event_type], time_col, start_s, end_s)
                row[metric.replace("_per_nrem_min", "_count")] = count
                row[metric] = count / (nrem_s / 60.0) if nrem_s >= MIN_NREM_CURVE_S else np.nan
            curve_rows.append(row)

        for window_name, (start_s, end_s) in WINDOWS.items():
            nrem_s = detector.nrem_seconds(stage, edf.fs, start_s, end_s)
            row = {
                "animal": animal,
                "trial": trial,
                "window": window_name,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "nrem_plus_ma_s": nrem_s,
                "valid_nrem": bool(nrem_s >= MIN_NREM_SUMMARY_S),
            }
            for metric, _, event_type, time_col in METRICS:
                count = detector.count_events(tables[event_type], time_col, start_s, end_s)
                row[metric.replace("_per_nrem_min", "_count")] = count
                row[metric] = count / (nrem_s / 60.0) if nrem_s >= MIN_NREM_SUMMARY_S else np.nan
            window_rows.append(row)

        qc_rows.append(
            {
                "animal": animal,
                "trial": trial,
                "edf": str(record["edf"]),
                "scores": str(record["scores"]),
                "edf_fs": edf.fs,
                "edf_duration_s": edf.duration_sec,
                "used_duration_s": n_samples / edf.fs,
                "eeg_label": eeg_label,
                "emg_label": emg_label,
                "score_epochs": len(scores),
                "nrem_stage2_min": float(np.sum(stage == detector.STATE_NREM) / edf.fs / 60.0),
                "ma_min": float(np.sum(stage == detector.STATE_MA) / edf.fs / 60.0),
                "spindles_detected": len(spindles),
                "so_detected": len(slow_oscillations),
                "couplings_detected": len(coupling),
                "pct_spindles_coupled": 100.0 * len(coupling) / len(spindles) if len(spindles) else np.nan,
            }
        )
        source_rows.extend(
            [
                {
                    "role": f"{animal}_trial{trial}_edf",
                    "path": str(record["edf"]),
                    "description": f"200-Hz EDF, EEG={eeg_label}, EMG={emg_label}.",
                },
                {
                    "role": f"{animal}_trial{trial}_scores",
                    "path": str(record["scores"]),
                    "description": "5-s Sirenia scoring; MA_20s recoded in the project detector.",
                },
            ]
        )

    events = pd.concat(all_events, ignore_index=True, sort=False)
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(window_rows),
        pd.DataFrame(qc_rows),
        pd.DataFrame(source_rows),
        events,
    )


def aggregate_curve(record_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [metric for metric, _, _, _ in METRICS]
    animal_curve = (
        record_curve.groupby(["animal", "rel_time_s"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["animal", "rel_time_s"])
    )
    rows: list[dict] = []
    for rel_time_s, time_df in animal_curve.groupby("rel_time_s", sort=True):
        for metric in metric_cols:
            values = pd.to_numeric(time_df[metric], errors="coerce").dropna()
            rows.append(
                {
                    "rel_time_s": rel_time_s,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sem": sem(values),
                    "n_animals": int(len(values)),
                }
            )
    return animal_curve, pd.DataFrame(rows)


def aggregate_windows(record_windows: pd.DataFrame) -> pd.DataFrame:
    valid = record_windows[record_windows["valid_nrem"]].copy()
    rows: list[dict] = []
    for (animal, window), sub in valid.groupby(["animal", "window"], sort=False):
        total_nrem_s = float(sub["nrem_plus_ma_s"].sum())
        row = {
            "animal": animal,
            "n_trials": int(sub["trial"].nunique()),
            "window": window,
            "nrem_plus_ma_s": total_nrem_s,
        }
        for metric, _, _, _ in METRICS:
            count_col = metric.replace("_per_nrem_min", "_count")
            total_count = int(sub[count_col].sum())
            row[count_col] = total_count
            row[metric] = total_count / (total_nrem_s / 60.0) if total_nrem_s > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["animal", "window"], key=lambda s: s.map(animal_sort_key) if s.name == "animal" else s)


def make_stats(animal_windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison_target in ["Yellow light", "Post light"]:
        for metric, ylabel, _, _ in METRICS:
            piv = animal_windows.pivot(index="animal", columns="window", values=metric)
            pair = piv[["No light", comparison_target]].dropna()
            baseline = pair["No light"].to_numpy(dtype=float)
            target = pair[comparison_target].to_numpy(dtype=float)
            diff = target - baseline
            t_p = stats.ttest_rel(baseline, target).pvalue if len(pair) >= 2 else np.nan
            try:
                w_p = stats.wilcoxon(baseline, target).pvalue if len(pair) >= 3 and np.any(diff != 0) else np.nan
            except ValueError:
                w_p = np.nan
            diff_sd = np.std(diff, ddof=1) if len(diff) > 1 else np.nan
            rows.append(
                {
                    "comparison": f"{comparison_target} vs No light",
                    "metric": metric,
                    "label": ylabel,
                    "n_animals": len(pair),
                    "no_light_mean": float(np.mean(baseline)),
                    "no_light_sem": sem(baseline),
                    "target_mean": float(np.mean(target)),
                    "target_sem": sem(target),
                    "mean_difference": float(np.mean(diff)),
                    "paired_t_p": float(t_p),
                    "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                    "cohen_dz": float(np.mean(diff) / diff_sd) if np.isfinite(diff_sd) and diff_sd > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["paired_t_p_holm"] = np.nan
    out["wilcoxon_p_holm"] = np.nan
    for comparison, idx in out.groupby("comparison").groups.items():
        out.loc[idx, "paired_t_p_holm"] = holm_adjust(out.loc[idx, "paired_t_p"])
        out.loc[idx, "wilcoxon_p_holm"] = holm_adjust(out.loc[idx, "wilcoxon_p"])
    return out


def draw_metric(ax: plt.Axes, group_curve: pd.DataFrame, metric: str, ylabel: str, panel_label: str | None = None) -> None:
    sub = group_curve[group_curve["metric"] == metric].sort_values("rel_time_s")
    x = sub["rel_time_s"].to_numpy(dtype=float)
    mean = sub["mean"].to_numpy(dtype=float)
    err = sub["sem"].to_numpy(dtype=float)
    ax.axvspan(0, 120, color=LIGHT_COLOR, alpha=0.42, lw=0, zorder=0)
    ax.axvline(0, color=LIGHT_EDGE, lw=0.7, alpha=0.9)
    ax.axvline(120, color=LIGHT_EDGE, lw=0.7, alpha=0.9)
    ax.plot(x, mean, color=LINE_COLOR, lw=1.35, zorder=3)
    ax.fill_between(x, np.maximum(0, mean - err), mean + err, color=FILL_COLOR, alpha=0.42, lw=0, zorder=2)
    ax.set_xlim(CURVE_START_S, CURVE_END_S)
    ax.set_xticks([-120, -60, 0, 60, 120, 180, 240])
    finite_top = np.nanmax(mean + np.nan_to_num(err, nan=0.0)) if np.isfinite(mean).any() else 1.0
    ax.set_ylim(0, max(1.0, finite_top * 1.12))
    ax.set_xlabel("Time (s)", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title("120 s", fontsize=9.5, pad=2.5)
    ax.tick_params(labelsize=6.8, pad=1.3)
    if panel_label:
        ax.text(-0.17, 1.04, panel_label, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for ext in ["png", "pdf", "svg", "tiff"]:
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600 if ext in {"png", "tiff"} else None, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    for ext in ["pdf", "svg"]:
        path = FIG_DIR / f"{stem}_ai_clean.{ext}"
        fig.savefig(path, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    return outputs


def make_figures(group_curve: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    for metric, ylabel, event_type, _ in METRICS:
        fig, ax = plt.subplots(figsize=(2.35, 2.0))
        draw_metric(ax, group_curve, metric, ylabel)
        fig.tight_layout(pad=0.4)
        outputs.extend(save_figure(fig, f"virus_nphr_120s_yellow_light_{event_type}_timecourse"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.15))
    for panel, ax, (metric, ylabel, _, _) in zip(["A", "B", "C"], axes, METRICS):
        draw_metric(ax, group_curve, metric, ylabel, panel)
    fig.tight_layout(w_pad=1.05, pad=0.45)
    outputs.extend(save_figure(fig, "virus_nphr_120s_yellow_light_spindle_so_coupling_combined"))
    plt.close(fig)
    return outputs


def write_documentation(
    records: list[dict],
    record_curve: pd.DataFrame,
    animal_curve: pd.DataFrame,
    group_curve: pd.DataFrame,
    record_windows: pd.DataFrame,
    animal_windows: pd.DataFrame,
    qc: pd.DataFrame,
    sources: pd.DataFrame,
    events: pd.DataFrame,
    stats_df: pd.DataFrame,
    figure_outputs: list[Path],
) -> list[Path]:
    tables = {
        "record_curve": FIG_DIR / f"{FIGURE_ID}_30s_curve_by_record.csv",
        "animal_curve": FIG_DIR / f"{FIGURE_ID}_30s_curve_by_animal.csv",
        "group_curve": FIG_DIR / f"{FIGURE_ID}_30s_curve_group_mean_sem.csv",
        "record_windows": FIG_DIR / f"{FIGURE_ID}_window_metrics_by_record.csv",
        "animal_windows": FIG_DIR / f"{FIGURE_ID}_window_metrics_by_animal.csv",
        "events": FIG_DIR / f"{FIGURE_ID}_all_events_long.csv",
        "qc": FIG_DIR / f"{FIGURE_ID}_qc.csv",
        "stats": FIG_DIR / f"{FIGURE_ID}_stats.csv",
    }
    record_curve.to_csv(tables["record_curve"], index=False, encoding="utf-8-sig")
    animal_curve.to_csv(tables["animal_curve"], index=False, encoding="utf-8-sig")
    group_curve.to_csv(tables["group_curve"], index=False, encoding="utf-8-sig")
    record_windows.to_csv(tables["record_windows"], index=False, encoding="utf-8-sig")
    animal_windows.to_csv(tables["animal_windows"], index=False, encoding="utf-8-sig")
    events.to_csv(tables["events"], index=False, encoding="utf-8-sig")
    qc.to_csv(tables["qc"], index=False, encoding="utf-8-sig")
    stats_df.to_csv(tables["stats"], index=False, encoding="utf-8-sig")
    sources.to_csv(FIG_DIR / "source_paths.csv", index=False, encoding="utf-8-sig")

    scripts_dir = FIG_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_copy = scripts_dir / SCRIPT_PATH.name
    detector_copy = scripts_dir / DETECTOR_SCRIPT.name
    shutil.copy2(SCRIPT_PATH, script_copy)
    shutil.copy2(DETECTOR_SCRIPT, detector_copy)

    stat_lines = []
    for row in stats_df.itertuples(index=False):
        stat_lines.append(
            f"- {row.label}, {row.comparison}: n={row.n_animals}; No light {row.no_light_mean:.3f} +/- {row.no_light_sem:.3f}; "
            f"target {row.target_mean:.3f} +/- {row.target_sem:.3f}; paired t p={row.paired_t_p:.6g}, Holm p={row.paired_t_p_holm:.6g}; "
            f"Wilcoxon p={row.wilcoxon_p if pd.notna(row.wilcoxon_p) else 'NA'}."
        )

    readme = f"""# VirusNPHR 120-s yellow-light Spindle/SO/coupling response

## Figure contract

- Core conclusion: test whether 120-s yellow-light optogenetic inhibition changes NREM-normalized Spindle, SO, or Spindle-SO coupling density during stimulation and the first 2 min after stimulation.
- Archetype: quantitative grid.
- Backend: Python/Matplotlib only.
- Panels: A Spindle/NREM min; B SO/NREM min; C coupled Spindle-SO/NREM min.
- Error representation: group mean +/- SEM across animals.

## Sample definition

- Recordings: {len(records)} paired 200-Hz EDF and 5-s score files.
- Animals: {qc['animal'].nunique()} ({', '.join(sorted(qc['animal'].unique(), key=animal_sort_key))}).
- Statistical unit: animal. Repeated recordings were combined within animal before group statistics.
- EEG channel: Ch1 in all records; EMG channel: EMG.

## Event detection

- Spindle: NREM-only 10-16 Hz signal, Hilbert envelope smoothed over 1 s; threshold = NREM mean + 1.5 SD; duration 0.4-2.0 s.
- SO: NREM-only 0.3-4.5 Hz positive-to-negative zero-crossing cycles; duration 0.5-2.0 s; project amplitude thresholds (0.66 x mean negative and peak-to-peak amplitudes).
- Coupling: spindle envelope peak falls inside a detected SO zero-crossing cycle; SO phase at the spindle peak is retained in the event table.
- Density denominator: NREM + MA minutes. MA is Wake <=20 s flanked by NREM, following the project convention.

## Windows and curves

- No light: 180-300 s (minute 3-5; score epochs 37-60).
- Yellow light: 300-420 s (minute 5-7; score epochs 61-84).
- Post light: 420-540 s (minute 7-9; score epochs 85-108).
- Curves: centered 30-s event-density windows stepped every 5 s from -120 to +240 s relative to yellow-light onset.
- Curve bins require >= {MIN_NREM_CURVE_S:g} s NREM+MA; 120-s summary windows require >= {MIN_NREM_SUMMARY_S:g} s NREM+MA.

## Statistics

Paired t-test and Wilcoxon signed-rank test across animal-level densities. Holm correction is applied across the three metrics separately for each comparison.

{chr(10).join(stat_lines)}

## Outputs

- Figures: PNG, PDF, SVG, 600-dpi TIFF, and Illustrator-clean PDF/SVG.
- Event-level tables: `{FIG_DIR / 'event_tables'}` and `{tables['events']}`.
- Animal source data: `{tables['animal_curve']}` and `{tables['animal_windows']}`.
- QC: `{tables['qc']}`.
- Statistics: `{tables['stats']}`.
- Script: `{SCRIPT_PATH}`.

## Review risks

- The supplied historical Spindle result folders did not preserve reliable animal/trial identifiers and contained no SO/coupling tables; this figure therefore re-detects all three event classes directly from the 17 EDF files with the established project detector.
- Coupling is cycle containment, not a phase-window significance test.
- Detection thresholds are calculated separately per recording; manual event review remains advisable before manuscript approval.
"""
    readme_path = FIG_DIR / "README_analysis.md"
    readme_path.write_text(readme, encoding="utf-8")

    PROV_DIR.mkdir(parents=True, exist_ok=True)
    prov_source = PROV_DIR / f"{FIGURE_ID}_source_data.csv"
    prov_stats = PROV_DIR / f"{FIGURE_ID}_stats.csv"
    prov_md = PROV_DIR / f"{FIGURE_ID}_provenance.md"
    animal_windows.to_csv(prov_source, index=False, encoding="utf-8-sig")
    stats_df.to_csv(prov_stats, index=False, encoding="utf-8-sig")
    provenance = f"""# {FIGURE_ID}

- figure_id: `{FIGURE_ID}`
- figure_title: VirusNPHR 120-s yellow-light Spindle, SO, and coupling response
- manuscript_context: Candidate optogenetic inhibition panel testing NREM oscillation changes.
- status: draft
- generated_at: 2026-07-18 Asia/Shanghai
- data_sources: `{RAW_ROOT}`; exact EDF and scoring paths are listed in `{FIG_DIR / 'source_paths.csv'}`.
- sample_definition: 17 recordings from 9 animals; repeated recordings combined within animal; animal is the biological replicate.
- analysis_window: No light 180-300 s; yellow light 300-420 s; post light 420-540 s; curves -120 to +240 s using centered 30-s windows stepped every 5 s.
- scoring: 5-s Sirenia files; Wake=1, NREM=2, REM=3; MA_20s recoding; NREM+MA denominator.
- sampling_rate: 200 Hz.
- channel_selection: Ch1 EEG for all records.
- filters_and_detection: Spindle 10-16 Hz, Hilbert envelope, 1-s smoothing, mean+1.5SD, 0.4-2.0 s; SO 0.3-4.5 Hz, 0.5-2.0 s crossing cycles with amplitude thresholds.
- coupling_definition: spindle envelope peak inside detected SO cycle; phase retained.
- statistics: paired t-test and Wilcoxon across animal-level densities; Holm correction across three metrics per comparison.
- scripts: `{SCRIPT_PATH}`; detector `{DETECTOR_SCRIPT}`.
- source_data: `{prov_source}`.
- stats_file: `{prov_stats}`.
- outputs: `{FIG_DIR / 'outputs_manifest.csv'}`.
- review_notes: draft pending manual event-level review; historical Spindle folders were not used because their animal/trial provenance was incomplete and SO/coupling tables were absent.
"""
    prov_md.write_text(provenance, encoding="utf-8")

    delivered = figure_outputs + list(tables.values()) + [FIG_DIR / "source_paths.csv", readme_path, script_copy, detector_copy, prov_source, prov_stats, prov_md]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path),
                "format": path.suffix.lstrip("."),
                "description": "Generated figure" if path in figure_outputs else "Analysis data, documentation, provenance, or script",
            }
            for path in delivered
        ]
    )
    manifest_path = FIG_DIR / "outputs_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return delivered + [manifest_path]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = discover_records()
    record_curve, record_windows, qc, sources, events = analyze_records(records)
    animal_curve, group_curve = aggregate_curve(record_curve)
    animal_windows = aggregate_windows(record_windows)
    stats_df = make_stats(animal_windows)
    figure_outputs = make_figures(group_curve)
    delivered = write_documentation(
        records,
        record_curve,
        animal_curve,
        group_curve,
        record_windows,
        animal_windows,
        qc,
        sources,
        events,
        stats_df,
        figure_outputs,
    )

    print("QC")
    print(qc[["animal", "trial", "nrem_stage2_min", "spindles_detected", "so_detected", "couplings_detected", "pct_spindles_coupled"]].to_string(index=False))
    print("\nSTATS")
    print(stats_df.to_string(index=False))
    print("\nOUTPUTS")
    for path in delivered:
        print(path)


if __name__ == "__main__":
    main()
