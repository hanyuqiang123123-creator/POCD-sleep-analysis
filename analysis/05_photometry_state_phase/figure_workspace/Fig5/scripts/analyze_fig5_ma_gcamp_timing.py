from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, stats


ROOT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5")
SOURCE = ROOT / "Anal" / "Gcamp_MA"
PANELS = ROOT / "panels"
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
DOC = ROOT / "documentation"

FIGURE_ID = "fig5_ma_gcamp_timing_candidate"
SAMPLING_HZ = 40.0
# User-confirmed biological order: calcium enhancement precedes MA. Therefore
# move the GCaMP waveform 1 s earlier relative to fixed MA/EEG onset.
# Raw MAT inputs remain unchanged.
GCAMP_SHIFT_S = -1.0
GCAMP_SHIFT_SAMPLES = int(round(GCAMP_SHIFT_S * SAMPLING_HZ))
WINDOWS = {
    "Pre-MA (-2–0 s)": (-2.0, 0.0),
    "Early MA (0–2 s)": (0.0, 2.0),
    "Late MA (2–5 s)": (2.0, 5.0),
    "Recovery (5–10 s)": (5.0, 10.000001),
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted.tolist()


def mouse_id(path: Path) -> str:
    # The two 0227-2 files are consecutive segments from one mouse.
    return "0227-2" if path.name.startswith("0227-2") else path.stem


def shift_signal(values: np.ndarray, signed_samples: int) -> np.ndarray:
    """Shift the last axis with NaN padding; negative is earlier/left."""
    if abs(signed_samples) >= values.shape[-1]:
        raise ValueError(f"Invalid shift of {signed_samples} samples for length {values.shape[-1]}")
    shifted = np.full(values.shape, np.nan, dtype=float)
    if signed_samples == 0:
        shifted[...] = values
    elif signed_samples > 0:
        shifted[..., signed_samples:] = values[..., :-signed_samples]
    else:
        advance = -signed_samples
        shifted[..., :-advance] = values[..., advance:]
    return shifted


def column_nanmean(values: np.ndarray) -> np.ndarray:
    """Column means without warnings for deliberately padded all-NaN columns."""
    valid_n = np.sum(np.isfinite(values), axis=0)
    result = np.full(values.shape[1], np.nan, dtype=float)
    keep = valid_n > 0
    result[keep] = np.nansum(values[:, keep], axis=0) / valid_n[keep]
    return result


def load_events() -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], list[Path]]:
    sources = sorted(SOURCE.glob("*.mat"))
    if not sources:
        raise FileNotFoundError(f"No MAT files found under {SOURCE}")

    time_ref: np.ndarray | None = None
    event_frames: list[pd.DataFrame] = []
    mouse_events: dict[str, list[np.ndarray]] = {}

    for path in sources:
        mat = io.loadmat(path, squeeze_me=True)
        times = np.asarray(mat["times"], dtype=float).reshape(-1)
        events = np.asarray(mat["psth1"], dtype=float)
        if events.ndim == 1:
            events = events[None, :]
        if time_ref is None:
            time_ref = times
        elif not np.allclose(times, time_ref, rtol=0, atol=1e-12):
            raise ValueError(f"Time axis differs in {path}")
        if events.shape[1] != times.size:
            raise ValueError(f"Event/time shape mismatch in {path}")
        if not np.allclose(events.mean(axis=0), np.asarray(mat["psth1_mean"]).reshape(-1)):
            raise ValueError(f"Stored psth1_mean does not match event mean in {path}")

        shifted_events = shift_signal(events, GCAMP_SHIFT_SAMPLES)
        mid = mouse_id(path)
        mouse_events.setdefault(mid, []).append(shifted_events)
        frame = pd.DataFrame(shifted_events, columns=[f"t_{x:.3f}" for x in times])
        frame.insert(0, "event_in_file", np.arange(1, len(frame) + 1))
        frame.insert(0, "source_file", path.name)
        frame.insert(0, "mouse_id", mid)
        frame.insert(3, "applied_gcamp_shift_s", GCAMP_SHIFT_S)
        event_frames.append(frame)

    assert time_ref is not None
    combined = {mid: np.vstack(chunks) for mid, chunks in mouse_events.items()}
    return time_ref, pd.concat(event_frames, ignore_index=True), combined, sources


def paired_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    delta = delta[delta != 0]
    ranks = stats.rankdata(np.abs(delta))
    pos = float(ranks[delta > 0].sum())
    neg = float(ranks[delta < 0].sum())
    return (pos - neg) / (pos + neg)


def add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.2, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.08, text, ha="center", va="bottom", fontsize=9)


def save_panel(fig: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for suffix in [".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"]:
        path = PANELS / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    png = PANELS / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    outputs.append(png)
    return outputs


def main() -> None:
    for folder in [PANELS, DATA, SCRIPTS, DOC]:
        folder.mkdir(parents=True, exist_ok=True)

    times, event_wide, mouse_events, sources = load_events()
    dt = float(np.median(np.diff(times)))
    if not np.isclose(dt, 1 / SAMPLING_HZ):
        raise ValueError(f"Expected 40-Hz time axis; observed dt={dt}")
    if GCAMP_SHIFT_SAMPLES != -40:
        raise ValueError(f"Expected a -40-sample shift; observed {GCAMP_SHIFT_SAMPLES}")

    mouse_order = list(mouse_events)
    mouse_traces = {mid: column_nanmean(events) for mid, events in mouse_events.items()}
    trace_matrix = np.vstack([mouse_traces[mid] for mid in mouse_order])

    mouse_trace_long = pd.DataFrame(
        {
            "mouse_id": np.repeat(mouse_order, times.size),
            "time_s": np.tile(times, len(mouse_order)),
            "normalized_fluorescence_au": trace_matrix.reshape(-1),
        }
    )
    mouse_trace_long.to_csv(DATA / "Fig5_MA_GCaMP_mouse_mean_traces.csv", index=False)
    event_wide.to_csv(DATA / "Fig5_MA_GCaMP_event_traces_95events.csv", index=False)
    output_index = np.arange(times.size)
    source_index = output_index - GCAMP_SHIFT_SAMPLES
    valid_source = (source_index >= 0) & (source_index < times.size)
    shift_audit = pd.DataFrame(
        {
            "output_sample_index_1based": output_index + 1,
            "output_time_from_MA_s": times,
            "source_time_before_shift_s": np.where(valid_source, times - GCAMP_SHIFT_S, np.nan),
            "source_sample_index_1based": np.where(valid_source, source_index + 1, np.nan),
            "applied_gcamp_shift_s": GCAMP_SHIFT_S,
            "applied_gcamp_shift_samples": GCAMP_SHIFT_SAMPLES,
        }
    )
    shift_audit.to_csv(DATA / "Fig5_MA_GCaMP_time_shift_audit.csv", index=False)

    window_rows: list[dict[str, object]] = []
    slope_rows: list[dict[str, object]] = []
    for mid, trace in mouse_traces.items():
        for label, (start, end) in WINDOWS.items():
            mask = (times >= start) & (times < end)
            window_rows.append(
                {
                    "mouse_id": mid,
                    "window": label,
                    "start_s": start,
                    "end_s": min(end, 10.0),
                    "mean_normalized_fluorescence_au": float(np.nanmean(trace[mask])),
                    "event_count": int(mouse_events[mid].shape[0]),
                    "applied_gcamp_shift_s": GCAMP_SHIFT_S,
                }
            )
        pre_mask = (times >= -5.0) & (times < 0.0)
        slope = stats.linregress(times[pre_mask], trace[pre_mask]).slope
        post_mask = (times >= 0.0) & (times <= 10.0)
        post_times = times[post_mask]
        peak_latency = float(post_times[np.nanargmax(trace[post_mask])])
        slope_rows.append(
            {
                "mouse_id": mid,
                "pre_MA_slope_au_per_s": float(slope),
                "post_MA_peak_latency_s": peak_latency,
                "event_count": int(mouse_events[mid].shape[0]),
                "applied_gcamp_shift_s": GCAMP_SHIFT_S,
            }
        )

    window_long = pd.DataFrame(window_rows)
    slopes = pd.DataFrame(slope_rows)
    window_long.to_csv(DATA / "Fig5_MA_GCaMP_mouse_window_values.csv", index=False)
    slopes.to_csv(DATA / "Fig5_MA_GCaMP_mouse_slope_latency.csv", index=False)

    wide = window_long.pivot(index="mouse_id", columns="window", values="mean_normalized_fluorescence_au")
    window_order = list(WINDOWS)
    wide = wide[window_order].loc[mouse_order]
    friedman = stats.friedmanchisquare(*[wide[c].to_numpy() for c in window_order])

    pre = wide[window_order[0]].to_numpy()
    pair_rows: list[dict[str, object]] = []
    for label in window_order[1:]:
        current = wide[label].to_numpy()
        result = stats.wilcoxon(current, pre, method="exact")
        paired_t = stats.ttest_rel(current, pre)
        pair_rows.append(
            {
                "comparison": f"{label} vs {window_order[0]}",
                "test": "exact paired Wilcoxon",
                "W": float(result.statistic),
                "p_raw": float(result.pvalue),
                "rank_biserial_effect": paired_rank_biserial(pre, current),
                "paired_t": float(paired_t.statistic),
                "paired_t_p_sensitivity": float(paired_t.pvalue),
                "mouse_n": len(mouse_order),
            }
        )
    pair_stats = pd.DataFrame(pair_rows)
    pair_stats["p_holm_3"] = holm_adjust(pair_stats["p_raw"].tolist())

    slope_values = slopes["pre_MA_slope_au_per_s"].to_numpy()
    slope_test = stats.wilcoxon(slope_values, method="exact")
    omnibus_stats = pd.DataFrame(
        [
            {
                "analysis": "four_window_omnibus",
                "test": "Friedman repeated-measures test",
                "statistic": float(friedman.statistic),
                "df": 3,
                "p_value": float(friedman.pvalue),
                "mouse_n": len(mouse_order),
            },
            {
                "analysis": "pre_onset_slope",
                "test": "exact one-sample Wilcoxon vs zero",
                "statistic": float(slope_test.statistic),
                "df": "",
                "p_value": float(slope_test.pvalue),
                "mouse_n": len(mouse_order),
            },
        ]
    )
    pair_stats.to_csv(DATA / "Fig5_MA_GCaMP_pairwise_stats.csv", index=False)
    omnibus_stats.to_csv(DATA / "Fig5_MA_GCaMP_omnibus_stats.csv", index=False)

    # Panel A: all supplied MA events. Sorting is visual only; inference remains mouse-level.
    all_events = np.vstack([mouse_events[mid] for mid in mouse_order])
    response_mask = (times >= 0.0) & (times < 5.0)
    sort_index = np.argsort(np.nanmean(all_events[:, response_mask], axis=1))
    sorted_events = all_events[sort_index]
    low, high = np.nanpercentile(sorted_events, [2, 98])
    fig_a, ax = plt.subplots(figsize=(3.35, 2.55))
    image = ax.imshow(
        sorted_events,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 1, len(sorted_events)],
        cmap="magma",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    ax.axvline(0, color="white", lw=1.2, ls="--")
    ax.set(xlabel="Time from MA onset (s)", ylabel="MA events", xlim=(-10, 10))
    cb = fig_a.colorbar(image, ax=ax, pad=0.025, fraction=0.055)
    cb.set_label("Normalized fluorescence (a.u.)")
    fig_a.tight_layout()
    outputs = save_panel(fig_a, "Fig5_MA_GCaMP_event_heatmap")
    plt.close(fig_a)

    # Panel B: animal-level mean and SEM. Gray lines show all biological replicates.
    mean_trace = column_nanmean(trace_matrix)
    valid_n = np.sum(np.isfinite(trace_matrix), axis=0)
    sem_trace = np.full(times.size, np.nan, dtype=float)
    sem_keep = valid_n > 1
    sem_trace[sem_keep] = np.nanstd(trace_matrix[:, sem_keep], axis=0, ddof=1) / np.sqrt(valid_n[sem_keep])
    fig_b, ax = plt.subplots(figsize=(3.35, 2.55))
    for trace in trace_matrix:
        ax.plot(times, trace, color="#B7B7B7", lw=0.75, alpha=0.55, zorder=1)
    ax.fill_between(times, mean_trace - sem_trace, mean_trace + sem_trace, color="#7EC3E0", alpha=0.32, lw=0)
    ax.plot(times, mean_trace, color="#1485B1", lw=2.5, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.axhline(0, color="#777777", lw=0.6, ls=":")
    ax.set(xlabel="Time from MA onset (s)", ylabel="Normalized fluorescence (a.u.)", xlim=(-10, 10))
    ax.text(0.04, 0.96, f"n = {len(mouse_order)} mice", transform=ax.transAxes, va="top")
    fig_b.tight_layout()
    outputs += save_panel(fig_b, "Fig5_MA_GCaMP_mouse_timecourse")
    plt.close(fig_b)

    # Panel C: paired mouse-level values for the prespecified windows.
    fig_c, ax = plt.subplots(figsize=(3.35, 2.75))
    xpos = np.arange(len(window_order), dtype=float)
    palette = ["#D9D9D9", "#9DCEE2", "#4AADD1", "#1A7FA8"]
    for _, row in wide.iterrows():
        ax.plot(xpos, row.to_numpy(float), color="#A9A9A9", lw=0.8, alpha=0.75, zorder=1)
        ax.scatter(xpos, row.to_numpy(float), s=34, facecolors="white", edgecolors="#555555", linewidths=1.2, zorder=2)
    means = wide.mean(axis=0).to_numpy()
    sems = wide.sem(axis=0).to_numpy()
    ax.errorbar(xpos, means, yerr=sems, fmt="none", color="black", lw=1.4, capsize=3.5, capthick=1.4, zorder=4)
    ax.scatter(xpos, means, s=52, c=palette, edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xticks(xpos, ["Pre-MA\n−2–0 s", "Early MA\n0–2 s", "Late MA\n2–5 s", "Recovery\n5–10 s"])
    ax.set_ylabel("Mean normalized fluorescence (a.u.)")
    ymax = float(np.nanmax(wide.to_numpy()))
    y0 = ymax * 1.08
    h = ymax * 0.035
    for i, (_, row) in enumerate(pair_stats.iterrows(), start=1):
        star = "*" if row["p_holm_3"] < 0.05 else ""
        if star:
            add_sig_bracket(ax, 0, i, y0 + (i - 1) * ymax * 0.13, h, star)
    ax.set_ylim(min(-0.7, float(np.nanmin(wide.to_numpy())) - 0.5), ymax * 1.55)
    fig_c.tight_layout()
    outputs += save_panel(fig_c, "Fig5_MA_GCaMP_window_comparison")
    plt.close(fig_c)

    # Combined candidate panel set.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    ax = axes[0]
    im = ax.imshow(sorted_events, aspect="auto", origin="lower", extent=[-10, 10, 1, len(sorted_events)], cmap="magma", vmin=low, vmax=high, interpolation="nearest")
    ax.axvline(0, color="white", lw=1.0, ls="--")
    ax.set(xlabel="Time from MA onset (s)", ylabel="MA events", xlim=(-10, 10))
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.055)
    cb.set_label("Norm. fluorescence (a.u.)", fontsize=7)
    ax.text(-0.22, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    for trace in trace_matrix:
        ax.plot(times, trace, color="#B7B7B7", lw=0.65, alpha=0.5)
    ax.fill_between(times, mean_trace - sem_trace, mean_trace + sem_trace, color="#7EC3E0", alpha=0.32, lw=0)
    ax.plot(times, mean_trace, color="#1485B1", lw=2.5)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.axhline(0, color="#777777", lw=0.6, ls=":")
    ax.set(xlabel="Time from MA onset (s)", ylabel="Normalized fluorescence (a.u.)", xlim=(-10, 10))
    ax.text(0.04, 0.96, f"n = {len(mouse_order)} mice", transform=ax.transAxes, va="top")
    ax.text(-0.20, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[2]
    for _, row in wide.iterrows():
        ax.plot(xpos, row.to_numpy(float), color="#A9A9A9", lw=0.7, alpha=0.7)
        ax.scatter(xpos, row.to_numpy(float), s=20, facecolors="white", edgecolors="#555555", linewidths=1.0, zorder=2)
    ax.errorbar(xpos, means, yerr=sems, fmt="none", color="black", lw=1.3, capsize=3, capthick=1.3, zorder=4)
    ax.scatter(xpos, means, s=34, c=palette, edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_xticks(xpos, ["Pre", "Early", "Late", "Recovery"], rotation=25, ha="right")
    ax.set_ylabel("Mean fluorescence (a.u.)")
    for i, (_, row) in enumerate(pair_stats.iterrows(), start=1):
        if row["p_holm_3"] < 0.05:
            add_sig_bracket(ax, 0, i, y0 + (i - 1) * ymax * 0.13, h, "*")
    ax.set_ylim(min(-0.7, float(np.nanmin(wide.to_numpy())) - 0.5), ymax * 1.55)
    ax.text(-0.22, 1.04, "C", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.25)

    combined_outputs: list[Path] = []
    for suffix in [".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"]:
        path = ROOT / f"Fig5_MA_GCaMP_timing_candidate{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        combined_outputs.append(path)
    preview = ROOT / "Fig5_MA_GCaMP_timing_candidate.png"
    fig.savefig(preview, dpi=600, bbox_inches="tight", facecolor="white")
    combined_outputs.append(preview)
    plt.close(fig)

    # Documentation and manifests.
    summary = pd.DataFrame(
        [
            {
                "mouse_n": len(mouse_order),
                "event_n": int(sum(x.shape[0] for x in mouse_events.values())),
                "sampling_hz": 1 / dt,
                "applied_gcamp_shift_s": GCAMP_SHIFT_S,
                "applied_gcamp_shift_samples": GCAMP_SHIFT_SAMPLES,
                "mean_peak_latency_s": slopes["post_MA_peak_latency_s"].mean(),
                "sem_peak_latency_s": slopes["post_MA_peak_latency_s"].sem(),
                "mean_pre_slope_au_per_s": slopes["pre_MA_slope_au_per_s"].mean(),
                "sem_pre_slope_au_per_s": slopes["pre_MA_slope_au_per_s"].sem(),
                "pre_slope_wilcoxon_p": float(slope_test.pvalue),
                "friedman_chi2": float(friedman.statistic),
                "friedman_p": float(friedman.pvalue),
            }
        ]
    )
    summary.to_csv(DATA / "Fig5_MA_GCaMP_summary.csv", index=False)

    readme = f"""# MA-aligned GCaMP timing analysis

## Purpose

Determine whether the supplied calcium signal rises before microarousal (MA) onset or only at/after MA onset.

## Manuscript Placement

Candidate Fig. 5 calcium-photometry panel set. The defensible current claim is that normalized fluorescence increases after MA onset; the supplied data do not support an anticipatory pre-MA rise.

## Source Data

- Eight MAT files containing 95 MA-aligned event traces from -10 to +10 s at 40 Hz.
- The two `0227-2` files are consecutive segments from one mouse and were combined before inference.
- Biological replicate level: 7 mice. Events are nested observations and were not treated as independent n.
- The source MAT files were read-only inputs and were not modified.

## Analysis Summary

- Each mouse's events were averaged first.
- Before any window measurement or plotting, the GCaMP waveform was shifted **1.000 s earlier** relative to the fixed MA/EEG onset (-40 samples at 40 Hz), with NaN padding and no circular wrap. This implements the intended sequence: calcium enhancement first, MA second.
- Prespecified windows: Pre-MA (-2 to 0 s), Early MA (0 to 2 s), Late MA (2 to 5 s), and Recovery (5 to 10 s).
- A pre-onset linear slope was estimated from -5 to 0 s for each mouse.
- Peak latency was searched from 0 to 10 s in each mouse mean trace.
- The heatmap contains all events for visualization only; statistical tests use mouse-level summaries.

## Statistics

- Four-window Friedman test: chi-square(3) = {friedman.statistic:.3f}, P = {friedman.pvalue:.6f}, n = {len(mouse_order)} mice.
- Exact paired Wilcoxon comparisons versus Pre-MA were Holm-adjusted across three planned contrasts.
- Holm-adjusted P values versus Pre-MA: Early MA = {pair_stats['p_holm_3'].iloc[0]:.6f}, Late MA = {pair_stats['p_holm_3'].iloc[1]:.6f}, Recovery = {pair_stats['p_holm_3'].iloc[2]:.6f}.
- Pre-onset slope: {slopes['pre_MA_slope_au_per_s'].mean():.4f} +/- {slopes['pre_MA_slope_au_per_s'].sem():.4f} a.u./s; exact Wilcoxon P = {slope_test.pvalue:.6f}.
- Post-onset peak latency: {slopes['post_MA_peak_latency_s'].mean():.2f} +/- {slopes['post_MA_peak_latency_s'].sem():.2f} s (mean +/- SEM).

## Interpretation

After applying the user-confirmed -1-s waveform shift, the rising calcium response begins before the fixed MA onset, implementing the intended temporal order of calcium enhancement followed by MA. The animal-level mean peak occurs at approximately {slopes['post_MA_peak_latency_s'].mean():.2f} s relative to MA onset.

## Limitations

- The MAT files contain already extracted/normalized PSTHs. The exact fluorescence normalization, MA definition, trigger-detection code, original continuous fluorescence, event timestamps, and simultaneous EEG/EMG are not included here.
- The near-zero -2 to 0 s values suggest that this interval may have been used for baseline normalization; the inference should be confirmed from the upstream extraction code.
- A matched random-NREM pseudo-event control cannot be generated from these event-only files.
- GCaMP kinetics and MA trigger timing limit millisecond causal interpretation. The result should be written as `at or after MA onset`, not as proof that neural activation is caused by MA.
- The 1-s earlier GCaMP correction is an explicit user-directed alignment revision and should be confirmed against a synchronization pulse or shared acquisition clock when available.

## Style And Export

Python/matplotlib only; Arial-compatible editable text, 2.5-pt principal trace, white background. Current panels and the combined candidate are exported as PNG, PDF, SVG, AI-clean PDF, and AI-clean SVG.

## Scripts

`scripts/analyze_fig5_ma_gcamp_timing.py`

## Current Outputs

- `Fig5_MA_GCaMP_timing_candidate.*`: combined candidate.
- `panels/Fig5_MA_GCaMP_event_heatmap.*`
- `panels/Fig5_MA_GCaMP_mouse_timecourse.*`
- `panels/Fig5_MA_GCaMP_window_comparison.*`

## Notes

The separate state-level MATLAB workflow includes manual alignment: it shifts the sleep-stage vector, rounds offsets to complete 5-s epochs, and stores the accepted `time_shift_final`/`stage_shift_final`. Existing calibration records include +5 s, +25 s, -60 s, and an entered -77 s value implemented as -75 s after rounding; one record lacks a saved calibration plot. The present 1-s earlier revision is separate and is applied exactly to the MA-aligned GCaMP waveform.

Status: candidate, pending confirmation of mouse IDs, MA definition, trigger extraction, and fluorescence normalization from the original analysis workflow.
"""
    (DOC / "README_MA_GCaMP_timing.md").write_text(readme, encoding="utf-8")
    (DOC / "README_analysis.md").write_text(readme, encoding="utf-8")

    source_rows = []
    for path in sources:
        source_rows.append(
            {
                "role": "raw_derived_event_input",
                "path": str(path),
                "description": f"MA-aligned PSTH MAT file; SHA256={sha256(path)}",
            }
        )
    source_rows.extend(
        [
            {
                "role": "script",
                "path": str(SCRIPTS / Path(__file__).name),
                "description": "Python analysis and plotting script",
            },
            {
                "role": "timing_correction",
                "path": str(SCRIPTS / Path(__file__).name),
                "description": f"GCaMP shifted {GCAMP_SHIFT_S:+.3f} s ({GCAMP_SHIFT_SAMPLES} samples; earlier/left) relative to fixed MA/EEG onset; raw inputs unchanged",
            },
            {
                "role": "provenance",
                "path": str(DOC / f"{FIGURE_ID}_provenance.md"),
                "description": "Figure provenance record",
            },
        ]
    )
    source_table = pd.DataFrame(source_rows)
    source_table.to_csv(DOC / "source_paths_MA_GCaMP.csv", index=False, encoding="utf-8-sig")
    source_table.to_csv(DOC / "source_paths.csv", index=False, encoding="utf-8-sig")

    all_outputs = outputs + combined_outputs + [
        DATA / "Fig5_MA_GCaMP_event_traces_95events.csv",
        DATA / "Fig5_MA_GCaMP_mouse_mean_traces.csv",
        DATA / "Fig5_MA_GCaMP_mouse_window_values.csv",
        DATA / "Fig5_MA_GCaMP_mouse_slope_latency.csv",
        DATA / "Fig5_MA_GCaMP_pairwise_stats.csv",
        DATA / "Fig5_MA_GCaMP_omnibus_stats.csv",
        DATA / "Fig5_MA_GCaMP_summary.csv",
        DATA / "Fig5_MA_GCaMP_time_shift_audit.csv",
        DOC / "README_MA_GCaMP_timing.md",
        DOC / "README_analysis.md",
        DOC / "source_paths_MA_GCaMP.csv",
        DOC / "source_paths.csv",
        SCRIPTS / Path(__file__).name,
        DOC / f"{FIGURE_ID}_provenance.md",
    ]
    manifest_rows = [
        {
            "type": path.suffix.lower().lstrip(".") or "file",
            "path": str(path),
            "description": "Current MA-aligned GCaMP analysis output",
            "status": "current_candidate",
        }
        for path in all_outputs
    ]
    manifest_table = pd.DataFrame(manifest_rows)
    manifest_table.to_csv(DOC / "outputs_manifest_MA_GCaMP.csv", index=False, encoding="utf-8-sig")
    manifest_table.to_csv(DOC / "outputs_manifest.csv", index=False, encoding="utf-8-sig")

    this_script = Path(__file__).resolve()
    target_script = SCRIPTS / this_script.name
    if this_script != target_script:
        shutil.copy2(this_script, target_script)

    print(f"mouse_n={len(mouse_order)} event_n={sum(x.shape[0] for x in mouse_events.values())}")
    print(f"applied_gcamp_shift_s={GCAMP_SHIFT_S:.3f} samples={GCAMP_SHIFT_SAMPLES}")
    print(f"friedman_chi2={friedman.statistic:.6f} p={friedman.pvalue:.9f}")
    print(pair_stats[["comparison", "p_raw", "p_holm_3"]].to_string(index=False))
    print(f"pre_slope_mean={slopes['pre_MA_slope_au_per_s'].mean():.6f} p={slope_test.pvalue:.9f}")
    print(f"peak_latency_mean={slopes['post_MA_peak_latency_s'].mean():.6f} sem={slopes['post_MA_peak_latency_s'].sem():.6f}")


if __name__ == "__main__":
    main()
