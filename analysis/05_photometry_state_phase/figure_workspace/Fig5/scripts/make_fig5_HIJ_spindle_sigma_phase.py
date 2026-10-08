from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


FIG_DIR = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5")
DATA_DIR = FIG_DIR / "data"
PANEL_DIR = FIG_DIR / "panels"

RAW_EEG_CSV = DATA_DIR / "Fig5H_raw_EEG_6P1_P1_0_30min.csv"
SCORE_CSV = DATA_DIR / "Fig5H_scores_6P1_P1_0_30min.csv"
TRACE_CSV = DATA_DIR / "Fig5H_sigma_trace_6P1_P1_0_30min.csv"
EVENT_CSV = DATA_DIR / "Fig5H_spindle_events_6P1_P1_0_30min.csv"
ALL_BINS_CSV = DATA_DIR / "Fig5I_all_bins_combined.csv"
ALL_METRICS_CSV = DATA_DIR / "Fig5J_all_metrics_combined.csv"
SUMMARY_CSV = DATA_DIR / "Fig5J_group_summary_metrics.csv"

SOURCE_BINS = Path(r"F:\1.Sleep\eXdata\EEG\EEG_CON\321\GroupSummary\all_bins_combined.csv")
SOURCE_METRICS = Path(r"F:\1.Sleep\eXdata\EEG\EEG_CON\321\GroupSummary\all_metrics_combined.csv")
SOURCE_SUMMARY = Path(r"F:\1.Sleep\eXdata\EEG\EEG_CON\321\GroupSummary\group_summary_metrics.csv")

TEAL = "#16888A"
TEAL_LIGHT = "#64B7C3"
BLUE = "#2F6FB3"
ORANGE = "#F28E2B"
GRAY = "#5B5B5B"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def copy_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (SOURCE_BINS, ALL_BINS_CSV),
        (SOURCE_METRICS, ALL_METRICS_CSV),
        (SOURCE_SUMMARY, SUMMARY_CSV),
    ]:
        if not dst.exists() or src.read_bytes() != dst.read_bytes():
            shutil.copy2(src, dst)


def build_sigma_trace() -> None:
    """Recalculate the H-panel trace from raw EEG using Python only."""
    raw = pd.read_csv(RAW_EEG_CSV)
    scores = pd.read_csv(SCORE_CSV)
    fs = float(raw["sampling_rate_hz"].iloc[0])
    eeg = raw["eeg_uv"].to_numpy(float)

    highpass = signal.butter(4, 0.5, btype="highpass", fs=fs, output="sos")
    eeg_hp = signal.sosfiltfilt(highpass, eeg)
    sigma_filter = signal.butter(4, [10.0, 15.0], btype="bandpass", fs=fs, output="sos")
    sigma = signal.sosfiltfilt(sigma_filter, eeg_hp)
    sigma_power = np.abs(signal.hilbert(sigma)) ** 2

    fs_dyn = 10.0
    sigma_10hz = signal.resample_poly(sigma_power, int(fs_dyn), int(round(fs)))
    lowpass = signal.butter(4, 0.025, btype="lowpass", fs=fs_dyn, output="sos")
    dynamics = signal.sosfiltfilt(lowpass, sigma_10hz)

    stage = scores["stage_code"].to_numpy(int)
    nrem_mask = np.repeat(stage == 2, int(5 * fs_dyn))
    n = min(len(dynamics), len(nrem_mask))
    dynamics = dynamics[:n]
    nrem_mask = nrem_mask[:n]
    reference = dynamics[nrem_mask] if np.any(nrem_mask) else dynamics
    scale = np.std(reference, ddof=1)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = np.std(dynamics, ddof=1)
    z = (dynamics - np.mean(reference)) / scale
    trace = pd.DataFrame({"time_min": np.arange(n) / fs_dyn / 60.0, "sigma_power_z": z})
    trace.to_csv(TRACE_CSV, index=False)


def save_panel(fig: mpl.figure.Figure, stem: str) -> list[Path]:
    base = PANEL_DIR / stem
    outputs = []
    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
        ("_AI_clean.pdf", {}),
        ("_AI_clean.svg", {}),
    ]:
        out = Path(str(base) + suffix)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03, **kwargs)
        outputs.append(out)
    return outputs


def panel_h() -> mpl.figure.Figure:
    trace = pd.read_csv(TRACE_CSV)
    events = pd.read_csv(EVENT_CSV)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(75 / 25.4, 36 / 25.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.75], "hspace": 0.08},
    )
    ax, raster = axes
    ax.plot(trace["time_min"], trace["sigma_power_z"], color=TEAL, lw=1.1)
    ax.axhline(0, color="#B0B0B0", lw=0.5, zorder=0)
    ax.set_ylabel("Sigma power\n(z score)")
    ax.set_ylim(-2.5, 4.2)
    ax.set_yticks([-2, 0, 2, 4])

    for row in events.itertuples(index=False):
        raster.plot([row.start_min, row.end_min], [1, 1], color="black", lw=2.0, solid_capstyle="butt")
        raster.vlines(row.center_min, 0, 1, color="black", lw=0.6)
    raster.set_ylim(0, 1.12)
    raster.set_yticks([])
    raster.set_ylabel("Spindle")
    raster.set_xlabel("Time (min)")
    raster.set_xlim(0, 30)
    raster.set_xticks(np.arange(0, 31, 5))
    return fig


def pooled_phase_counts(bins: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    phase_col = "cycle_phase_deg"
    count_col = "spindle_count"
    grouped = bins.groupby(phase_col, as_index=False)[count_col].sum().sort_values(phase_col)
    return grouped[phase_col].to_numpy(float), grouped[count_col].to_numpy(float)


def panel_i() -> mpl.figure.Figure:
    bins = pd.read_csv(ALL_BINS_CSV)
    summary = pd.read_csv(SUMMARY_CSV).iloc[0]
    phase_deg, counts = pooled_phase_counts(bins)
    theta = np.deg2rad(phase_deg)
    width = np.deg2rad(float(np.median(np.diff(phase_deg))))
    fig = plt.figure(figsize=(48 / 25.4, 48 / 25.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(theta, counts, width=width, color=TEAL_LIGHT, edgecolor="none", alpha=0.95)
    mean_deg = float(summary["circular_mean_angle_deg"])
    rmax = max(counts) * 1.05
    ax.plot([np.deg2rad(mean_deg), np.deg2rad(mean_deg)], [0, rmax], color="black", lw=1.5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°\nTrough", "90°", "180°\nPeak", "270°"])
    ax.set_yticklabels([])
    ax.grid(color="#D9D9D9", lw=0.45)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("Pooled spindle phase distribution", pad=8, fontweight="normal")
    return fig


def panel_j() -> mpl.figure.Figure:
    metrics = pd.read_csv(ALL_METRICS_CSV)
    summary = pd.read_csv(SUMMARY_CSV).iloc[0]
    x = metrics["mean_angle_cycle_deg"].to_numpy(float)
    y = metrics["R_cycle"].to_numpy(float)
    mouse = metrics["file_id"].astype(str).str.extract(r"^(\d+)", expand=False)
    palette = {"1": "#4E79A7", "2": "#F28E2B", "3": "#59A14F", "4": "#B07AA1", "6": "#E15759"}

    fig, ax = plt.subplots(figsize=(43 / 25.4, 48 / 25.4))
    for mouse_id in sorted(mouse.dropna().unique(), key=int):
        mask = mouse == mouse_id
        ax.scatter(
            x[mask], y[mask], s=16, facecolor="white", edgecolor=palette.get(mouse_id, GRAY),
            linewidth=0.9, label=f"Mouse {mouse_id}", zorder=3,
        )
    mean_angle = float(summary["circular_mean_angle_deg"])
    mean_r = float(summary["mean_R_cycle"])
    sd_r = float(summary["std_R_cycle"])
    ax.axvline(mean_angle, color=GRAY, lw=1.1, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(mean_angle, mean_r, yerr=sd_r, fmt="o", ms=4.5, mfc=BLUE, mec=BLUE,
                ecolor=BLUE, elinewidth=1.1, capsize=2.5, zorder=5)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0.2, 0.5)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Preferred phase angle (°)")
    ax.set_ylabel("Coupling strength (R)")
    ax.set_title("Preferred phase and coupling strength", pad=5)
    ax.text(0.03, 0.97, "35 recordings\n5 mice × 7 days", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2)
    ax.text(0.97, 0.04, f"R = {mean_r:.3f} ± {sd_r:.3f} SD", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2)
    return fig


def combined_figure() -> mpl.figure.Figure:
    trace = pd.read_csv(TRACE_CSV)
    events = pd.read_csv(EVENT_CSV)
    bins = pd.read_csv(ALL_BINS_CSV)
    metrics = pd.read_csv(ALL_METRICS_CSV)
    summary = pd.read_csv(SUMMARY_CSV).iloc[0]

    fig = plt.figure(figsize=(183 / 25.4, 61 / 25.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.82, 0.82], wspace=0.55)
    left = outer[0].subgridspec(2, 1, height_ratios=[2.2, 0.75], hspace=0.08)
    axh = fig.add_subplot(left[0])
    axr = fig.add_subplot(left[1], sharex=axh)
    axh.plot(trace["time_min"], trace["sigma_power_z"], color=TEAL, lw=1.1)
    axh.axhline(0, color="#B0B0B0", lw=0.5, zorder=0)
    axh.set_ylabel("Sigma power\n(z score)")
    axh.set_ylim(-2.5, 4.2)
    axh.set_yticks([-2, 0, 2, 4])
    axh.tick_params(labelbottom=False)
    for row in events.itertuples(index=False):
        axr.plot([row.start_min, row.end_min], [1, 1], color="black", lw=2.0, solid_capstyle="butt")
        axr.vlines(row.center_min, 0, 1, color="black", lw=0.6)
    axr.set_ylim(0, 1.12)
    axr.set_yticks([])
    axr.set_ylabel("Spindle")
    axr.set_xlabel("Time (min)")
    axr.set_xlim(0, 30)
    axr.set_xticks(np.arange(0, 31, 5))

    axi = fig.add_subplot(outer[1], projection="polar")
    phase_deg, counts = pooled_phase_counts(bins)
    theta = np.deg2rad(phase_deg)
    width = np.deg2rad(float(np.median(np.diff(phase_deg))))
    axi.bar(theta, counts, width=width, color=TEAL_LIGHT, edgecolor="none", alpha=0.95)
    mean_deg = float(summary["circular_mean_angle_deg"])
    axi.plot([np.deg2rad(mean_deg)] * 2, [0, max(counts) * 1.05], color="black", lw=1.5)
    axi.set_theta_zero_location("E")
    axi.set_theta_direction(1)
    axi.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axi.set_xticklabels(["0°\nTrough", "90°", "180°\nPeak", "270°"])
    axi.set_yticklabels([])
    axi.grid(color="#D9D9D9", lw=0.45)
    axi.spines["polar"].set_linewidth(0.8)
    axi.set_title("Pooled spindle phase distribution", pad=8)

    axj = fig.add_subplot(outer[2])
    x = metrics["mean_angle_cycle_deg"].to_numpy(float)
    y = metrics["R_cycle"].to_numpy(float)
    mouse = metrics["file_id"].astype(str).str.extract(r"^(\d+)", expand=False)
    palette = {"1": "#4E79A7", "2": "#F28E2B", "3": "#59A14F", "4": "#B07AA1", "6": "#E15759"}
    for mouse_id in sorted(mouse.dropna().unique(), key=int):
        mask = mouse == mouse_id
        axj.scatter(x[mask], y[mask], s=16, facecolor="white", edgecolor=palette.get(mouse_id, GRAY), linewidth=0.9, zorder=3)
    mean_r = float(summary["mean_R_cycle"])
    sd_r = float(summary["std_R_cycle"])
    axj.axvline(mean_deg, color=GRAY, lw=1.1, ls=(0, (3, 2)), zorder=1)
    axj.errorbar(mean_deg, mean_r, yerr=sd_r, fmt="o", ms=4.5, mfc=BLUE, mec=BLUE,
                 ecolor=BLUE, elinewidth=1.1, capsize=2.5, zorder=5)
    axj.set_xlim(0, 360)
    axj.set_xticks([0, 90, 180, 270, 360])
    axj.set_ylim(0.2, 0.5)
    axj.set_yticks([0.2, 0.3, 0.4, 0.5])
    axj.set_xlabel("Preferred phase angle (°)")
    axj.set_ylabel("Coupling strength (R)")
    axj.set_title("Preferred phase and coupling strength", pad=5)
    axj.text(0.03, 0.97, "35 recordings\n5 mice × 7 days", transform=axj.transAxes,
             ha="left", va="top", fontsize=6.2)
    axj.text(0.97, 0.04, f"R = {mean_r:.3f} ± {sd_r:.3f} SD", transform=axj.transAxes,
             ha="right", va="bottom", fontsize=6.2)

    for label, ax in zip(["H", "I", "J"], [axh, axi, axj]):
        ax.text(-0.15, 1.13, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    return fig


def save_combined(fig: mpl.figure.Figure) -> list[Path]:
    base = FIG_DIR / "Fig5_HIJ_spindle_sigma_phase"
    outputs = []
    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
        ("_AI_clean.pdf", {}),
        ("_AI_clean.svg", {}),
    ]:
        out = Path(str(base) + suffix)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04, **kwargs)
        outputs.append(out)
    return outputs


def main() -> None:
    copy_inputs()
    build_sigma_trace()
    figures = [
        (panel_h(), "Fig5H_sigma_power_spindle_raster"),
        (panel_i(), "Fig5I_pooled_spindle_phase_distribution"),
        (panel_j(), "Fig5J_preferred_phase_coupling_strength"),
    ]
    for fig, stem in figures:
        save_panel(fig, stem)
        plt.close(fig)
    fig = combined_figure()
    save_combined(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
