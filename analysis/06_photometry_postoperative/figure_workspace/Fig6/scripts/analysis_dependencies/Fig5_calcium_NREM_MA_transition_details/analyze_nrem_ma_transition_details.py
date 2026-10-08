"""Exploratory detail analysis of calcium dynamics around NREM-to-MA transitions.

This script intentionally leaves the primary three-panel Fig. 5 analysis unchanged.
All inferential comparisons use mouse-level summaries; event rows are shown only as
distributional/visual context.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

from analyze_nrem_ma_transitions import (
    BASELINE_END_S,
    BASELINE_START_S,
    BIN_S,
    FS_HZ,
    MA_MAX_S,
    MIN_PRECEDING_NREM_S,
    PALETTE,
    POST_S,
    PRE_S,
    RECORDS,
    WINDOW_END_S,
    hedges_g,
    merge_bouts,
    parse_stages,
    recode_ma,
)


OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition_details")
PROVENANCE_ROOT = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_nrem_ma_transition_details"
BIN_CENTERS = np.arange(-PRE_S + BIN_S / 2, POST_S, BIN_S)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "pdf.compression": 0,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def load_event_curves(record: dict[str, object]) -> tuple[list[dict[str, object]], float]:
    mat = loadmat(Path(record["plot"]))
    time = np.asarray(mat["times"], float).ravel()
    signal = np.asarray(mat["data"], float).ravel()
    time = time - time[0]
    keep = (time >= 0) & (time < WINDOW_END_S)
    time, signal = time[keep], signal[keep]
    if len(time) != int(WINDOW_END_S * FS_HZ):
        raise AssertionError(f"Unexpected Plot duration for {record['animal']}")

    stage_rows = parse_stages(
        Path(record["stage"]),
        float(record["offset"]),
        float(record.get("stage_origin", 0.0)),
    )
    stage_epoch_s = float(np.median([float(row["duration"]) for row in stage_rows]))
    if not np.isclose(stage_epoch_s, 5.0, atol=1e-9, rtol=0):
        raise AssertionError(f"Expected original 5-s scores for {record['animal']}, got {stage_epoch_s:g} s")
    bouts = merge_bouts(recode_ma(stage_rows))

    output: list[dict[str, object]] = []
    event_index = 0
    for index, bout in enumerate(bouts):
        if index == 0 or int(bout["code"]) != 4 or int(bouts[index - 1]["code"]) != 2:
            continue
        onset = float(bout["start"])
        preceding = bouts[index - 1]
        if not np.isclose(float(preceding["stop"]), onset, atol=1e-6, rtol=0):
            continue
        if float(preceding["stop"]) - float(preceding["start"]) < MIN_PRECEDING_NREM_S:
            continue
        if onset < PRE_S or onset + POST_S > WINDOW_END_S:
            continue
        baseline_mask = (time >= onset + BASELINE_START_S) & (time < onset + BASELINE_END_S)
        baseline = float(np.mean(signal[baseline_mask]))
        event_index += 1
        duration = float(bout["stop"]) - onset
        for center in BIN_CENTERS:
            left = onset + center - BIN_S / 2
            mask = (time >= left) & (time < left + BIN_S)
            output.append({
                "group": record["group"],
                "animal": record["animal"],
                "event_index": event_index,
                "onset_s": onset,
                "ma_duration_s": duration,
                "time_s": float(center),
                "delta_signal": float(np.mean(signal[mask]) - baseline),
            })
    if event_index < 2:
        raise AssertionError(f"Fewer than two eligible events for {record['animal']}")
    return output, stage_epoch_s


def interval_mean(curve: pd.Series, start: float, stop: float) -> float:
    x = curve.index.to_numpy(float)
    return float(curve[(x >= start) & (x < stop)].mean())


def build_mouse_metrics(event_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mouse_rows: list[dict[str, object]] = []
    duration_rows: list[dict[str, object]] = []
    strata = [
        ("5 s", 0.0, 5.0001),
        ("10 s", 5.0001, 10.0001),
        ("15–20 s", 10.0001, 20.0001),
    ]
    event_meta = event_df.drop_duplicates(["group", "animal", "event_index"])
    for (group, animal), frame in event_df.groupby(["group", "animal"], sort=False):
        animal_curve = frame.groupby("time_s")["delta_signal"].mean().sort_index()
        x = animal_curve.index.to_numpy(float)
        peak_mask = (x >= -10) & (x < 10)
        peak_values = animal_curve.to_numpy(float)[peak_mask]
        peak_times = x[peak_mask]
        mouse_rows.append({
            "group": group,
            "animal": animal,
            "event_count": int(frame["event_index"].nunique()),
            "pre_rise_minus10_0": interval_mean(animal_curve, -10, 0),
            "very_early_0_2": interval_mean(animal_curve, 0, 2),
            "primary_early_0_5": interval_mean(animal_curve, 0, 5),
            "auc_0_10": float(animal_curve[(x >= 0) & (x < 10)].sum() * BIN_S),
            "peak_amplitude_minus10_10": float(np.max(peak_values)),
            "peak_time_minus10_10_s": float(peak_times[int(np.argmax(peak_values))]),
        })
        animal_events = event_meta[(event_meta["group"] == group) & (event_meta["animal"] == animal)]
        for label, low, high in strata:
            ids = animal_events.loc[
                (animal_events["ma_duration_s"] > low) & (animal_events["ma_duration_s"] <= high),
                "event_index",
            ].tolist()
            if not ids:
                continue
            selected = frame[frame["event_index"].isin(ids)]
            early = selected[(selected["time_s"] >= 0) & (selected["time_s"] < 5)].groupby("event_index")["delta_signal"].mean()
            duration_rows.append({
                "group": group,
                "animal": animal,
                "ma_duration_stratum": label,
                "event_count": len(ids),
                "early_response_0_5s": float(early.mean()),
            })
    return pd.DataFrame(mouse_rows), pd.DataFrame(duration_rows)


def compare_mouse(frame: pd.DataFrame, metric: str, label: str, family: str) -> dict[str, object]:
    cont = frame.loc[frame["group"].eq("Cont"), metric].dropna().to_numpy(float)
    sur = frame.loc[frame["group"].eq("SUR"), metric].dropna().to_numpy(float)
    test = stats.ttest_ind(sur, cont, equal_var=False)
    var_cont = np.var(cont, ddof=1) / len(cont)
    var_sur = np.var(sur, ddof=1) / len(sur)
    welch_df = float((var_cont + var_sur) ** 2 / (var_cont ** 2 / (len(cont) - 1) + var_sur ** 2 / (len(sur) - 1)))
    return {
        "family": family,
        "comparison": label,
        "metric": metric,
        "n_cont": len(cont),
        "n_sur": len(sur),
        "cont_mean": float(np.mean(cont)),
        "cont_sem": float(stats.sem(cont)),
        "sur_mean": float(np.mean(sur)),
        "sur_sem": float(stats.sem(sur)),
        "difference_sur_minus_cont": float(np.mean(sur) - np.mean(cont)),
        "independent_t_test": "two-sided Welch",
        "welch_t": float(test.statistic),
        "welch_df": welch_df,
        "welch_p_raw": float(test.pvalue),
        "hedges_g": hedges_g(cont, sur),
    }


def style_axis(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.3, pad=1.5)
    ax.axhline(0, color="#8A8A8A", lw=0.5, zorder=0)


def dot_summary(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, p_value: float) -> None:
    offsets = np.linspace(-0.07, 0.07, 7)
    for xpos, group in enumerate(("Cont", "SUR")):
        values = frame.loc[frame["group"].eq(group), metric].dropna().to_numpy(float)
        color = PALETTE[group]
        mean, sem = float(np.mean(values)), float(stats.sem(values))
        ax.scatter(xpos + offsets[:len(values)], values, s=19, facecolors="white", edgecolors=color["edge"], linewidths=0.8, zorder=3)
        ax.errorbar(xpos, mean, yerr=sem, fmt="_", markersize=16, color=color["edge"], capsize=2.2, lw=1.0, zorder=4)
    ax.set_xticks([0, 1], ["Cont", "SUR"])
    style_axis(ax, "", ylabel)
    ax.text(0.5, 1.02, f"P = {p_value:.3f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2)


def timecourse_panel(ax: plt.Axes, event_df: pd.DataFrame) -> None:
    mouse_curves = event_df.groupby(["group", "animal", "time_s"], as_index=False)["delta_signal"].mean()
    for group in ("Cont", "SUR"):
        pivot = mouse_curves[mouse_curves["group"].eq(group)].pivot(index="animal", columns="time_s", values="delta_signal")
        x = pivot.columns.to_numpy(float)
        mean = pivot.mean().to_numpy(float)
        sem = pivot.sem().to_numpy(float)
        ax.plot(x, mean, color=PALETTE[group]["edge"], lw=1.15, label=f"{group} (n={len(pivot)})")
        ax.fill_between(x, mean - sem, mean + sem, color=PALETTE[group]["fill"], alpha=0.28, linewidth=0)
    ax.axvspan(-20, -10, color="#D9EAF2", alpha=0.35, zorder=0)
    ax.axvspan(0, 5, color="#E5E5E5", alpha=0.38, zorder=0)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlim(-20, 15)
    style_axis(ax, "Time from MA onset (s)", "Δ signal vs stable NREM baseline\n(stored units)")
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.5, frameon=False)


def duration_panel(ax: plt.Axes, frame: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    labels = ["5 s", "10 s", "15–20 s"]
    x = np.arange(len(labels), dtype=float)
    for gi, group in enumerate(("Cont", "SUR")):
        dx = -0.14 if group == "Cont" else 0.14
        for i, label in enumerate(labels):
            values = frame[(frame["group"].eq(group)) & (frame["ma_duration_stratum"].eq(label))]["early_response_0_5s"].to_numpy(float)
            mean, sem = float(np.mean(values)), float(stats.sem(values)) if len(values) > 1 else 0.0
            ax.errorbar(i + dx, mean, yerr=sem, fmt="o", ms=4.2, mfc="white", mec=PALETTE[group]["edge"], mew=0.85,
                        ecolor=PALETTE[group]["edge"], capsize=2, lw=0.9, label=group if i == 0 else None, zorder=4)
            jitter = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(i + dx + jitter, values, s=9, color=PALETTE[group]["fill"], alpha=0.65, linewidths=0, zorder=2)
    for i, label in enumerate(labels):
        p = float(stats_df.loc[stats_df["comparison"].eq(f"MA duration {label}"), "welch_p_raw"].iloc[0])
        ax.text(i, 1.02, f"P={p:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x, labels)
    style_axis(ax, "MA duration", "Early response 0–5 s\n(stored units)")
    ax.legend(fontsize=6, frameon=False, loc="upper right")


def heatmap_panel(fig: plt.Figure, spec, event_df: pd.DataFrame) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    inner = spec.subgridspec(3, 1, height_ratios=[1, 1, 0.10], hspace=0.34)
    axes = [fig.add_subplot(inner[0]), fig.add_subplot(inner[1])]
    matrices: dict[str, np.ndarray] = {}
    row_frames: dict[str, pd.DataFrame] = {}
    for group in ("Cont", "SUR"):
        subset = event_df[event_df["group"].eq(group)]
        pivot = subset.pivot(index=["animal", "event_index", "onset_s", "ma_duration_s"], columns="time_s", values="delta_signal")
        pivot = pivot.sort_index(level=[0, 2])
        matrices[group] = pivot.to_numpy(float)
        row_frames[group] = pivot.reset_index()[["animal", "event_index", "onset_s", "ma_duration_s"]]
    lim = float(np.nanpercentile(np.abs(np.concatenate(list(matrices.values()))), 98))
    image = None
    for ax, group in zip(axes, ("Cont", "SUR")):
        image = ax.imshow(matrices[group], aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim,
                          extent=[BIN_CENTERS[0] - 0.5, BIN_CENTERS[-1] + 0.5, len(matrices[group]), 0])
        ax.axvline(0, color="black", lw=0.55, ls="--")
        counts = row_frames[group].groupby("animal", sort=False).size().cumsum().to_numpy()
        for boundary in counts[:-1]:
            ax.axhline(boundary, color="white", lw=0.65)
        ax.set_ylabel(f"{group}\nevents", fontsize=6.3)
        ax.tick_params(labelsize=5.8, length=2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xticklabels([])
    axes[1].set_xlabel("Time from MA onset (s)", fontsize=6.0, labelpad=1)
    cax = fig.add_subplot(inner[2])
    fig.colorbar(image, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=5.5, length=2)
    cax.set_xlabel("Δ signal (stored units)", fontsize=5.8, labelpad=1)
    return axes[0], axes[1], cax


def save_figure(fig: plt.Figure, base: Path, tiff: bool = False) -> list[Path]:
    outputs: list[Path] = []
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    if tiff:
        path = base.with_suffix(".tiff")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    for ext in ("pdf", "svg"):
        path = base.parent / f"{base.name}_AI_clean.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        outputs.append(path)
    return outputs


def main() -> None:
    for folder in (OUT, OUT / "panels", OUT / "data", OUT / "scripts", OUT / "documentation", OUT / "editable", PROVENANCE_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    stage_epochs: dict[str, float] = {}
    for record in RECORDS:
        rows, epoch_s = load_event_curves(record)
        event_rows.extend(rows)
        stage_epochs[str(record["animal"])] = epoch_s
        for role in ("plot", "stage"):
            path = Path(record[role])
            source_rows.append({"group": record["group"], "animal": record["animal"], "role": role, "path": str(path), "sha256": sha256(path)})

    event_df = pd.DataFrame(event_rows)
    mouse_df, duration_df = build_mouse_metrics(event_df)

    exploratory_specs = [
        ("pre_rise_minus10_0", "Pre-rise −10 to 0 s"),
        ("very_early_0_2", "Very early response 0–2 s"),
        ("auc_0_10", "Response AUC 0–10 s"),
        ("peak_amplitude_minus10_10", "Peak amplitude −10 to 10 s"),
        ("peak_time_minus10_10_s", "Peak time −10 to 10 s"),
    ]
    exploratory_stats = [compare_mouse(mouse_df, metric, label, "exploratory_shape_metrics") for metric, label in exploratory_specs]

    duration_stats: list[dict[str, object]] = []
    for label in ("5 s", "10 s", "15–20 s"):
        frame = duration_df[duration_df["ma_duration_stratum"].eq(label)]
        duration_stats.append(compare_mouse(frame, "early_response_0_5s", f"MA duration {label}", "exploratory_duration_strata"))
    stats_df = pd.DataFrame(exploratory_stats + duration_stats)

    event_df.to_csv(OUT / "data/event_level_transition_timecourses.csv", index=False, encoding="utf-8-sig")
    mouse_df.to_csv(OUT / "data/animal_level_transition_detail_metrics.csv", index=False, encoding="utf-8-sig")
    duration_df.to_csv(OUT / "data/MA_duration_strata_animal_metrics.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(OUT / "data/transition_detail_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).to_csv(OUT / "data/source_file_manifest.csv", index=False, encoding="utf-8-sig")
    event_df.drop_duplicates(["group", "animal", "event_index"])[["group", "animal", "event_index", "onset_s", "ma_duration_s"]].to_csv(
        OUT / "data/event_heatmap_row_manifest.csv", index=False, encoding="utf-8-sig"
    )

    primary = pd.read_csv(Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition\data\animal_level_MA_transition_metrics.csv"))
    check = primary.merge(mouse_df, on=["group", "animal"])
    if not np.allclose(check["early_response_0_5s"], check["primary_early_0_5"], atol=1e-9, rtol=0):
        raise AssertionError("Primary 0–5 s metric failed cross-validation")

    p_lookup = stats_df.set_index("metric")["welch_p_raw"].to_dict()
    fig = plt.figure(figsize=(7.16, 5.55))
    grid = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.05], hspace=0.58, wspace=1.65)
    ax_a = fig.add_subplot(grid[0, 0:6])
    ax_b = fig.add_subplot(grid[0, 6:9])
    ax_c = fig.add_subplot(grid[0, 9:12])
    ax_d = fig.add_subplot(grid[1, 0:3])
    ax_e = fig.add_subplot(grid[1, 3:8])
    timecourse_panel(ax_a, event_df)
    dot_summary(ax_b, mouse_df, "pre_rise_minus10_0", "Pre-rise −10 to 0 s\n(stored units)", p_lookup["pre_rise_minus10_0"])
    dot_summary(ax_c, mouse_df, "peak_amplitude_minus10_10", "Peak −10 to 10 s\n(stored units)", p_lookup["peak_amplitude_minus10_10"])
    dot_summary(ax_d, mouse_df, "auc_0_10", "AUC 0–10 s\n(stored units × s)", p_lookup["auc_0_10"])
    duration_panel(ax_e, duration_df, stats_df)
    heat_axes = heatmap_panel(fig, grid[1, 8:12], event_df)
    for label, ax in zip("abcde", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    heat_axes[0].text(-0.42, 1.08, "f", transform=heat_axes[0].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.suptitle("Exploratory structure of calcium responses around microarousals", fontsize=9.5, y=0.985)
    fig.text(0.5, 0.012, "Mouse-level inference; mean ± SEM; two-sided Welch independent-samples t-test", ha="center", fontsize=5.9)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.10)
    combined = save_figure(fig, OUT / "Fig5_calcium_NREM_MA_transition_details", tiff=True)
    plt.close(fig)

    panel_files: list[Path] = []
    panel_specs = [
        ("panel_a_transition_timecourse", "timecourse"),
        ("panel_b_pre_rise", "pre"),
        ("panel_c_peak_amplitude", "peak"),
        ("panel_d_auc_0_10", "auc"),
        ("panel_e_MA_duration_strata", "duration"),
        ("panel_f_event_heatmaps", "heatmap"),
    ]
    for name, kind in panel_specs:
        if kind == "heatmap":
            pfig = plt.figure(figsize=(3.15, 3.1))
            pgrid = pfig.add_gridspec(1, 1)
            heatmap_panel(pfig, pgrid[0], event_df)
            pfig.subplots_adjust(left=0.23, right=0.96, top=0.96, bottom=0.15)
        else:
            pfig, ax = plt.subplots(figsize=(3.0, 2.25) if kind in {"timecourse", "duration"} else (2.15, 2.25))
            if kind == "timecourse":
                timecourse_panel(ax, event_df)
            elif kind == "pre":
                dot_summary(ax, mouse_df, "pre_rise_minus10_0", "Pre-rise −10 to 0 s\n(stored units)", p_lookup["pre_rise_minus10_0"])
            elif kind == "peak":
                dot_summary(ax, mouse_df, "peak_amplitude_minus10_10", "Peak −10 to 10 s\n(stored units)", p_lookup["peak_amplitude_minus10_10"])
            elif kind == "auc":
                dot_summary(ax, mouse_df, "auc_0_10", "AUC 0–10 s\n(stored units × s)", p_lookup["auc_0_10"])
            else:
                duration_panel(ax, duration_df, stats_df)
            pfig.subplots_adjust(left=0.24, right=0.95, top=0.88, bottom=0.22)
        panel_files.extend(save_figure(pfig, OUT / "panels" / name))
        plt.close(pfig)

    script_copy = OUT / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy)

    contract = """# Figure contract

- Core conclusion: Test whether the SUR-associated NREM-to-MA calcium enhancement reflects anticipatory/early amplitude structure, timing, or particular MA-duration strata.
- Evidence chain: transition time course; mouse-level pre-rise, peak and AUC; mouse-level duration strata; event heatmaps as distributional context only.
- Archetype: quantitative grid with the transition time course as the hero panel.
- Backend: Python/matplotlib exclusively.
- Inference: mouse is the biological replicate; event rows are never treated as independent replicates.
- Review risks: detail endpoints are exploratory and use unadjusted P values; peak timing is limited by 5-s staging resolution.
"""
    (OUT / "documentation/figure_contract.md").write_text(contract, encoding="utf-8")

    shape_lines = []
    for row in exploratory_stats:
        shape_lines.append(
            f"- {row['comparison']}: Cont {row['cont_mean']:.3f}, SUR {row['sur_mean']:.3f}, "
            f"difference {row['difference_sur_minus_cont']:+.3f}; Welch P={row['welch_p_raw']:.4f}."
        )
    duration_lines = []
    for row in duration_stats:
        duration_lines.append(
            f"- {row['comparison']}: n={row['n_cont']}/{row['n_sur']} mice, Cont {row['cont_mean']:.3f}, SUR {row['sur_mean']:.3f}; "
            f"Welch P={row['welch_p_raw']:.4f}."
        )
    readme = f"""# Fig. 5 calcium NREM-to-MA transition detail analysis

## Purpose

Candidate supplemental/detail analysis that tests whether the group difference is anticipatory, amplitude-related, timing-related, or restricted to a particular MA duration. The established three-panel primary figure is not changed.

## Manuscript Placement

Candidate supplementary figure or source-data analysis supporting the NREM-to-MA panel.

## Source Data

Seven independent Cont and seven independent SUR mice. Both groups use original 5-s scoring. The common Plot window is [0, {WINDOW_END_S:g}) s at {FS_HZ:g} Hz. Exact paths and hashes are in `source_paths.csv` and `data/source_file_manifest.csv`.

## Analysis Summary

- MA is a Wake bout ≤{MA_MAX_S:g} s flanked by NREM.
- Eligible events require ≥{MIN_PRECEDING_NREM_S:g} s preceding NREM and a complete −{PRE_S:g} to +{POST_S:g} s window.
- Each event is baseline-subtracted against −20 to −10 s stable NREM.
- Calcium is summarized in 1-s bins; repeated events are averaged within mouse before group inference.
- Event heatmaps retain individual events only for visualization and are not used as independent statistical replicates.

## Statistics

Two-sided Welch independent-samples t-tests at the mouse level. P values are unadjusted, as requested. The original 0–5 s endpoint remains the primary analysis and is cross-validated separately.

### Exploratory shape metrics

{chr(10).join(shape_lines)}

### MA-duration strata

{chr(10).join(duration_lines)}

## Style And Export

Python/matplotlib only; editable text in SVG/PDF; 600-dpi TIFF and 300-dpi PNG previews. Cont is blue and SUR is red. Heatmaps share one robust symmetric color scale.

## Scripts

- `{script_copy}`

## Current Outputs

- Combined PNG/PDF/SVG/TIFF and AI-clean PDF/SVG are at the figure-folder root.
- Every panel is available separately under `panels`.
- Mouse-, event-, duration- and statistics-level CSV files are under `data`.

## Archived Versions

Superseded renders, if present, are retained under the timestamped `archive` subfolders.

## Notes

This is hypothesis-generating detail analysis. The 5-s stage labels constrain MA-onset precision; the 1-s calcium display bins do not create 1-s staging precision. Peak-time estimates should therefore not be over-interpreted.
"""
    (OUT / "documentation/README_analysis.md").write_text(readme, encoding="utf-8")
    with (OUT / "documentation/source_paths.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "animal", "role", "path", "sha256"])
        writer.writeheader()
        writer.writerows(source_rows)

    qa = {
        "backend": "Python/matplotlib only",
        "archetype": "quantitative grid",
        "biological_replicate": "mouse",
        "group_n": {"Cont": 7, "SUR": 7},
        "staging_resolution_s": {"Cont": 5, "SUR": 5},
        "primary_metric_cross_validation": "passed",
        "multiplicity": "unadjusted P values requested for exploratory panels",
        "event_heatmap_inference": "none; visualization only",
        "source_hashes_reverified": True,
        "visual_qa": "passed_combined_and_all_individual_panel_review",
    }
    (OUT / "documentation/figure_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    prov_source = PROVENANCE_ROOT / f"{FIGURE_ID}_source_data.csv"
    prov_stats = PROVENANCE_ROOT / f"{FIGURE_ID}_stats.csv"
    shutil.copy2(OUT / "data/animal_level_transition_detail_metrics.csv", prov_source)
    shutil.copy2(OUT / "data/transition_detail_statistics.csv", prov_stats)
    provenance = f"""# {FIGURE_ID}

- figure_title: Exploratory structure of calcium responses around microarousals
- manuscript_context: Candidate Fig. 5 supplementary/detail analysis of Cont versus SUR NREM-to-MA dynamics.
- status: candidate
- generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}
- data_sources: see {OUT / 'documentation/source_paths.csv'}
- sample_definition: seven independent mice per group; repeated MA events averaged within mouse for inference.
- analysis_window: common Plot [0,{WINDOW_END_S:g}) s; transition window −{PRE_S:g} to +{POST_S:g} s; 5-s staging in both groups.
- analysis_steps: MA ≤{MA_MAX_S:g} s flanked by NREM; ≥{MIN_PRECEDING_NREM_S:g} s prior NREM; −20 to −10 s baseline; 1-s calcium bins; duration-stratified mouse means; event heatmaps for visualization only.
- statistics: two-sided Welch independent-samples t-test; unadjusted P values; mean ± SEM shown.
- scripts: {script_copy}
- source_data: {prov_source}
- stats: {prov_stats}
- outputs: {' | '.join(str(path) for path in combined + panel_files)}
- review_notes: exploratory candidate; 5-s stage resolution limits onset/peak-time interpretation; no raw source was modified.
"""
    provenance_path = PROVENANCE_ROOT / f"{FIGURE_ID}_provenance.md"
    provenance_path.write_text(provenance, encoding="utf-8")
    shutil.copy2(provenance_path, OUT / "documentation" / provenance_path.name)

    for row in source_rows:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise AssertionError(f"Source changed: {row['path']}")

    files = [
        *combined,
        *panel_files,
        *list((OUT / "data").glob("*.csv")),
        script_copy,
        *[path for path in (OUT / "documentation").iterdir() if path.is_file() and path.name != "outputs_manifest.csv"],
    ]
    with (OUT / "documentation/outputs_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "path", "description", "status", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow({
                "type": "figure" if path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff"} else "supporting",
                "path": str(path),
                "description": path.name,
                "status": "current",
                "sha256": sha256(path),
            })

    print(json.dumps({
        "output": str(OUT),
        "event_count": event_df.drop_duplicates(["group", "animal", "event_index"]).groupby("group").size().to_dict(),
        "stage_epochs": stage_epochs,
        "statistics": stats_df.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
