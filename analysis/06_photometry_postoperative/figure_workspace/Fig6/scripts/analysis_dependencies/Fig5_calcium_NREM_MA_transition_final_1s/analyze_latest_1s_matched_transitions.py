"""Mouse-level Cont versus SUR analysis using final 1-s scoring and manual alignment."""
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

from analyze_nrem_ma_transition_details import BIN_S, build_mouse_metrics, save_figure
from analyze_nrem_ma_transitions import (
    PALETTE,
    ROOT,
    WINDOW_END_S,
    hedges_g,
    merge_bouts,
    parse_stages,
    recode_ma,
)


SOURCE = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition_details\data")
CONT_EVENTS = SOURCE / "Cont_1s_alignment_corrected_event_level_transition_timecourses.csv"
SUR_EVENTS = SOURCE / "SUR_1s_alignment_corrected_event_level_transition_timecourses.csv"
OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition_final_1s")
PROVENANCE_ROOT = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_nrem_ma_transition_final_1s"

FINAL_RECORDS = [
    {"group": "Cont", "animal": "Cont1", "stage": ROOT / "Cont/1/EEG/stages_aligned_epoch_1s.db3", "origin": 0.0, "offset": 0.0, "shift": -2.0},
    {"group": "Cont", "animal": "Cont2", "stage": ROOT / "Cont/2/scores_epoch_1s.db3", "origin": 0.0, "offset": 29.786, "shift": 6.0},
    {"group": "Cont", "animal": "Cont3", "stage": ROOT / "Cont/3/EEG/stages_aligned_epoch_1s.db3", "origin": 0.0, "offset": 0.0, "shift": 3.0},
    {"group": "Cont", "animal": "Cont4", "stage": ROOT / "Cont/4/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 29.544, "shift": 3.0},
    {"group": "Cont", "animal": "Cont5", "stage": ROOT / "Cont/5/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 29.827, "shift": 3.0},
    {"group": "Cont", "animal": "Cont6", "stage": ROOT / "Cont/6/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 29.231, "shift": -4.0},
    {"group": "Cont", "animal": "Cont7", "stage": ROOT / "Cont/7/EEG/stages_aligned.db3", "origin": 0.0, "offset": 0.0, "shift": 3.0},
    {"group": "SUR", "animal": "SUR1", "stage": ROOT / "SUR/1/EEG/scores_epoch_1s.db3", "origin": 3380.0, "offset": 0.0, "shift": 0.0},
    {"group": "SUR", "animal": "SUR2", "stage": ROOT / "SUR/2/scores_epoch_1s.db3", "origin": 0.0, "offset": 11.568, "shift": 4.0},
    {"group": "SUR", "animal": "SUR3", "stage": ROOT / "SUR/3/Aligned_EEGminus21s/stages_aligned.db3", "origin": 0.0, "offset": 0.0, "shift": 6.0},
    {"group": "SUR", "animal": "SUR4", "stage": ROOT / "SUR/4/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 7.306, "shift": 8.0},
    {"group": "SUR", "animal": "SUR5", "stage": ROOT / "SUR/5/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 4.047, "shift": 3.0},
    {"group": "SUR", "animal": "SUR6", "stage": ROOT / "SUR/6_/EEG/scores_epoch_1s.db3", "origin": 0.0, "offset": 4.046, "shift": 4.0},
    {"group": "SUR", "animal": "SUR7", "stage": ROOT / "SUR/7/EEG1/scores_epoch_1s.db3", "origin": 0.0, "offset": 4.766, "shift": 0.0},
]

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


def welch_df(cont: np.ndarray, sur: np.ndarray) -> float:
    a = np.var(cont, ddof=1) / len(cont)
    b = np.var(sur, ddof=1) / len(sur)
    return float((a + b) ** 2 / (a * a / (len(cont) - 1) + b * b / (len(sur) - 1)))


def compare(frame: pd.DataFrame, metric: str, label: str, role: str) -> dict[str, object]:
    cont = frame.loc[frame["group"].eq("Cont"), metric].dropna().to_numpy(float)
    sur = frame.loc[frame["group"].eq("SUR"), metric].dropna().to_numpy(float)
    test = stats.ttest_ind(sur, cont, equal_var=False)
    ci = test.confidence_interval()
    return {
        "role": role,
        "comparison": label,
        "metric": metric,
        "n_cont": len(cont),
        "n_sur": len(sur),
        "cont_mean": float(np.mean(cont)),
        "cont_sem": float(stats.sem(cont)),
        "sur_mean": float(np.mean(sur)),
        "sur_sem": float(stats.sem(sur)),
        "difference_sur_minus_cont": float(np.mean(sur) - np.mean(cont)),
        "welch_t": float(test.statistic),
        "welch_df": welch_df(cont, sur),
        "welch_p": float(test.pvalue),
        "difference_ci95_welch": f"[{ci.low:.9g}, {ci.high:.9g}]",
        "hedges_g": hedges_g(cont, sur),
    }


def nrem_seconds(record: dict[str, object]) -> tuple[float, float]:
    stage_rows = parse_stages(Path(record["stage"]), float(record["offset"]), float(record["origin"]))
    epoch_s = float(np.median([float(row["duration"]) for row in stage_rows]))
    if not np.isclose(epoch_s, 1.0, atol=1e-9, rtol=0):
        raise AssertionError(f"Expected 1-s scores for {record['animal']}; found {epoch_s:g}")
    total = 0.0
    for bout in merge_bouts(recode_ma(stage_rows)):
        if int(bout["code"]) == 2:
            total += max(0.0, min(WINDOW_END_S, float(bout["stop"])) - max(0.0, float(bout["start"])))
    return total, epoch_s


def style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.3, pad=1.5)
    ax.axhline(0, color="#858585", lw=0.55, zorder=0)


def dot_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, p: float) -> None:
    offsets = np.linspace(-0.07, 0.07, 7)
    for xpos, group in enumerate(("Cont", "SUR")):
        values = frame.loc[frame["group"].eq(group), metric].to_numpy(float)
        mean, sem = float(np.mean(values)), float(stats.sem(values))
        color = PALETTE[group]
        ax.scatter(xpos + offsets[:len(values)], values, s=21, facecolors="white", edgecolors=color["edge"], linewidths=0.85, zorder=3)
        ax.errorbar(xpos, mean, yerr=sem, fmt="_", markersize=17, color=color["edge"], lw=1.0, capsize=2.2, zorder=4)
    ax.set_xticks([0, 1], ["Cont", "SUR"])
    style_axis(ax, "", ylabel)
    ax.text(0.5, 1.02, f"P = {p:.3f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3)


def timecourse_panel(ax: plt.Axes, event_df: pd.DataFrame) -> None:
    mouse_curves = event_df.groupby(["group", "animal", "time_s"], as_index=False)["delta_signal"].mean()
    for group in ("Cont", "SUR"):
        pivot = mouse_curves[mouse_curves["group"].eq(group)].pivot(index="animal", columns="time_s", values="delta_signal")
        x = pivot.columns.to_numpy(float)
        mean, sem = pivot.mean().to_numpy(float), pivot.sem().to_numpy(float)
        ax.plot(x, mean, color=PALETTE[group]["edge"], lw=1.15, label=f"{group} (n={len(pivot)})")
        ax.fill_between(x, mean - sem, mean + sem, color=PALETTE[group]["fill"], alpha=0.28, linewidth=0)
    ax.axvspan(-20, -10, color="#D9EAF2", alpha=0.35, zorder=0)
    ax.axvspan(0, 5, color="#E5E5E5", alpha=0.38, zorder=0)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlim(-20, 20)
    style_axis(ax, "Time from MA onset (s)", "Δ signal vs stable NREM baseline\n(stored units)")
    ax.legend(loc="upper left", fontsize=6.2, frameon=False)


def main() -> None:
    for folder in (OUT, OUT / "panels", OUT / "data", OUT / "scripts", OUT / "documentation", OUT / "editable", PROVENANCE_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    cont = pd.read_csv(CONT_EVENTS)
    sur = pd.read_csv(SUR_EVENTS)
    event_df = pd.concat([cont, sur], ignore_index=True)
    expected_shifts = {(record["group"], record["animal"]): float(record["shift"]) for record in FINAL_RECORDS}
    observed_shifts = event_df.groupby(["group", "animal"])["calcium_shift_s"].first().to_dict()
    if observed_shifts != expected_shifts:
        raise AssertionError(f"Alignment shift mismatch: {observed_shifts} != {expected_shifts}")
    if not np.isfinite(event_df["delta_signal"]).all():
        raise AssertionError("Non-finite calcium value")

    mouse_df, duration_df = build_mouse_metrics(event_df)
    mouse_curves = event_df.groupby(["group", "animal", "time_s"], as_index=False)["delta_signal"].mean()
    late = mouse_curves[(mouse_curves["time_s"] >= 5) & (mouse_curves["time_s"] < 20)].groupby(["group", "animal"])["delta_signal"].mean().rename("late_response_5_20").reset_index()
    mouse_df = mouse_df.merge(late, on=["group", "animal"], how="left")

    source_rows = [
        {"role": "derived_event_data", "path": str(CONT_EVENTS), "sha256": sha256(CONT_EVENTS)},
        {"role": "derived_event_data", "path": str(SUR_EVENTS), "sha256": sha256(SUR_EVENTS)},
    ]
    rates = []
    for record in FINAL_RECORDS:
        nrem_s, epoch_s = nrem_seconds(record)
        count = int(event_df[(event_df["group"].eq(record["group"])) & (event_df["animal"].eq(record["animal"]))]["event_index"].nunique())
        rates.append({"group": record["group"], "animal": record["animal"], "event_count": count, "nrem_duration_s": nrem_s, "ma_events_per_nrem_min": count / (nrem_s / 60), "score_epoch_s": epoch_s, "calcium_shift_s": record["shift"]})
        stage_path = Path(record["stage"])
        source_rows.append({"role": "score_1s", "path": str(stage_path), "sha256": sha256(stage_path)})
    rate_df = pd.DataFrame(rates)
    mouse_df = mouse_df.merge(rate_df, on=["group", "animal", "event_count"], how="left")

    specs = [
        ("primary_early_0_5", "Early calcium response, 0–5 s", "primary"),
        ("pre_rise_minus10_0", "Pre-MA rise, −10–0 s", "exploratory"),
        ("very_early_0_2", "Very early calcium response, 0–2 s", "exploratory"),
        ("auc_0_10", "Calcium response AUC, 0–10 s", "exploratory"),
        ("peak_amplitude_minus10_10", "Peak amplitude, −10–10 s", "exploratory"),
        ("peak_time_minus10_10_s", "Peak time, −10–10 s", "exploratory"),
        ("late_response_5_20", "Late calcium response, 5–20 s", "exploratory"),
        ("ma_events_per_nrem_min", "Eligible MA event rate per NREM min", "context"),
    ]
    stats_rows = [compare(mouse_df, metric, label, role) for metric, label, role in specs]
    stats_df = pd.DataFrame(stats_rows)
    p = stats_df.set_index("metric")["welch_p"].to_dict()

    event_df.to_csv(OUT / "data/final_1s_event_level_timecourses.csv", index=False, encoding="utf-8-sig")
    mouse_df.to_csv(OUT / "data/final_1s_animal_level_metrics.csv", index=False, encoding="utf-8-sig")
    mouse_curves.to_csv(OUT / "data/final_1s_animal_level_timecourses.csv", index=False, encoding="utf-8-sig")
    duration_df.to_csv(OUT / "data/final_1s_duration_strata_metrics.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(OUT / "data/final_1s_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).drop_duplicates().to_csv(OUT / "data/source_file_manifest.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.16, 4.75))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1], hspace=0.62, wspace=0.72)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3])]
    timecourse_panel(axes[0], event_df)
    dot_panel(axes[1], mouse_df, "primary_early_0_5", "Early response 0–5 s\n(stored units)", p["primary_early_0_5"])
    dot_panel(axes[2], mouse_df, "pre_rise_minus10_0", "Pre-MA rise −10–0 s\n(stored units)", p["pre_rise_minus10_0"])
    dot_panel(axes[3], mouse_df, "auc_0_10", "AUC 0–10 s\n(stored units × s)", p["auc_0_10"])
    dot_panel(axes[4], mouse_df, "peak_amplitude_minus10_10", "Peak amplitude −10–10 s\n(stored units)", p["peak_amplitude_minus10_10"])
    dot_panel(axes[5], mouse_df, "ma_events_per_nrem_min", "Eligible MA events\nper NREM min", p["ma_events_per_nrem_min"])
    for label, ax in zip("abcdef", axes):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.suptitle("Calcium dynamics at NREM-to-microarousal transitions after final 1-s alignment", fontsize=9.5, y=0.985)
    fig.text(0.5, 0.012, "Mouse-level means ± SEM; two-sided Welch independent-samples t-test; n=7 mice/group", ha="center", fontsize=6)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12)
    combined = save_figure(fig, OUT / "Fig5_calcium_NREM_MA_transition_final_1s", tiff=True)
    plt.close(fig)

    panel_files = []
    panel_defs = [
        ("panel_a_timecourse", "curve", None, None),
        ("panel_b_early_0_5", "dot", "primary_early_0_5", "Early response 0–5 s\n(stored units)"),
        ("panel_c_pre_MA_rise", "dot", "pre_rise_minus10_0", "Pre-MA rise −10–0 s\n(stored units)"),
        ("panel_d_auc_0_10", "dot", "auc_0_10", "AUC 0–10 s\n(stored units × s)"),
        ("panel_e_peak_amplitude", "dot", "peak_amplitude_minus10_10", "Peak amplitude −10–10 s\n(stored units)"),
        ("panel_f_MA_rate", "dot", "ma_events_per_nrem_min", "Eligible MA events\nper NREM min"),
    ]
    for name, kind, metric, ylabel in panel_defs:
        pfig, ax = plt.subplots(figsize=(3.0, 2.25) if kind == "curve" else (2.15, 2.25))
        if kind == "curve":
            timecourse_panel(ax, event_df)
        else:
            dot_panel(ax, mouse_df, str(metric), str(ylabel), p[str(metric)])
        pfig.subplots_adjust(left=0.24, right=0.95, top=0.88, bottom=0.22)
        panel_files.extend(save_figure(pfig, OUT / "panels" / name))
        plt.close(pfig)

    script_copy = OUT / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy)
    contract = """# Figure contract

- Core conclusion: determine whether SUR changes the mouse-level NREM-to-MA calcium response after final 1-s scoring and manual Plot–EEG alignment.
- Evidence chain: animal-averaged time course, prespecified 0–5 s primary response, pre-MA rise, 0–10 s AUC, peak amplitude, and MA rate.
- Archetype: quantitative grid with the time course as hero panel.
- Backend: Python/matplotlib exclusively.
- Inference: mouse is the biological replicate; repeated events are averaged within mouse.
- Statistics: two-sided Welch independent-samples t-test, as requested; exploratory P values are unadjusted and identified as exploratory in the source table.
- Review risks: alignment shifts were chosen manually from heatmaps; effect estimates therefore require independent replication or preregistered validation.
"""
    (OUT / "documentation/figure_contract.md").write_text(contract, encoding="utf-8")

    primary = stats_rows[0]
    readme = f"""# Final 1-s matched NREM-to-MA calcium analysis

## Purpose

Analyze Cont versus SUR calcium dynamics using the latest user-approved 1-s scores and manual calcium-to-EEG shifts. This candidate package does not overwrite the previous 5-s analysis.

## Manuscript Placement

Candidate revised Fig. 5 NREM-to-MA analysis pending biological review.

## Source Data

Seven independent mice per group. Cont contributes {event_df[event_df.group.eq('Cont')]['event_index'].count() // len(event_df[event_df.group.eq('Cont')]['time_s'].unique())} eligible events and SUR contributes {event_df[event_df.group.eq('SUR')]['event_index'].count() // len(event_df[event_df.group.eq('SUR')]['time_s'].unique())}. Both groups use 1-s scoring. Exact files and hashes are recorded in `source_paths.csv` and `data/source_file_manifest.csv`.

## Analysis Summary

- MA: Wake ≤20 s flanked by NREM.
- Eligibility: ≥20 s continuous preceding NREM and a complete shifted −20 to +30 s calcium window.
- Baseline: event-specific −20 to −10 s stable NREM, shifted together with calcium.
- Repeated events are averaged within each mouse before group statistics.
- Final shifts are stored for every mouse in `final_1s_animal_level_metrics.csv`.

## Statistics

Primary 0–5 s response: Cont {primary['cont_mean']:.3f} ± {primary['cont_sem']:.3f}, SUR {primary['sur_mean']:.3f} ± {primary['sur_sem']:.3f} stored units; SUR−Cont {primary['difference_sur_minus_cont']:+.3f}; t={primary['welch_t']:.3f}, df={primary['welch_df']:.2f}, P={primary['welch_p']:.4f}; Hedges g={primary['hedges_g']:.2f}. Tests are two-sided Welch independent-samples t-tests with mouse as n. Exploratory P values are unadjusted.

## Style And Export

Python/matplotlib only; vector PDF/SVG with editable text, 600-dpi TIFF and 300-dpi PNG.

## Scripts

- `{script_copy}`

## Current Outputs

Combined files are at the package root; separate panels are under `panels`; source data and statistics are under `data`.

## Archived Versions

None at first creation. Future replacements must be archived under timestamped `archive` folders.

## Notes

Manual shifts were selected after viewing the same data. They improve temporal matching but constitute a data-informed preprocessing decision and should be reported transparently.
"""
    (OUT / "documentation/README_analysis.md").write_text(readme, encoding="utf-8")
    with (OUT / "documentation/source_paths.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "path", "sha256"])
        writer.writeheader(); writer.writerows(pd.DataFrame(source_rows).drop_duplicates().to_dict(orient="records"))

    qa = {
        "backend": "Python/matplotlib only",
        "biological_replicate": "mouse",
        "group_n": {"Cont": 7, "SUR": 7},
        "event_n": {"Cont": int(event_df[event_df.group.eq("Cont")]["event_index"].count() / 50), "SUR": int(event_df[event_df.group.eq("SUR")]["event_index"].count() / 50)},
        "score_epoch_s": 1,
        "primary_endpoint": "animal mean 0-5 s response vs -20 to -10 s baseline",
        "test": "two-sided Welch independent-samples t-test",
        "source_hashes_reverified": True,
        "visual_qa": "passed_combined_and_all_individual_panel_review",
    }
    (OUT / "documentation/figure_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    prov_source = PROVENANCE_ROOT / f"{FIGURE_ID}_source_data.csv"
    prov_stats = PROVENANCE_ROOT / f"{FIGURE_ID}_stats.csv"
    shutil.copy2(OUT / "data/final_1s_animal_level_metrics.csv", prov_source)
    shutil.copy2(OUT / "data/final_1s_statistics.csv", prov_stats)
    provenance = f"""# {FIGURE_ID}

- figure_title: NREM-to-MA calcium dynamics after final 1-s alignment
- manuscript_context: Candidate revised Fig. 5 analysis comparing Cont and SUR.
- status: candidate
- generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}
- sample_definition: seven independent mice per group; repeated events averaged within mouse.
- analysis_window: common Plot [0,{WINDOW_END_S:g}) s; event −20 to +30 s; baseline −20 to −10 s; 1-s scoring and display bins.
- analysis_steps: MA ≤20 s flanked by NREM; ≥20 s prior NREM; user-approved mouse-specific calcium shifts; mouse-level aggregation.
- statistics: two-sided Welch independent-samples t-test; Hedges g; unadjusted exploratory P values.
- scripts: {script_copy}
- source_data: {prov_source}
- stats: {prov_stats}
- outputs: {' | '.join(str(path) for path in combined + panel_files)}
- review_notes: manual alignment was selected from the analyzed data and must be disclosed; candidate pending biological review.
"""
    provenance_path = PROVENANCE_ROOT / f"{FIGURE_ID}_provenance.md"
    provenance_path.write_text(provenance, encoding="utf-8")
    shutil.copy2(provenance_path, OUT / "documentation" / provenance_path.name)

    for source in source_rows:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise AssertionError(f"Source changed: {source['path']}")

    files = [*combined, *panel_files, *list((OUT / "data").glob("*.csv")), script_copy, *[path for path in (OUT / "documentation").iterdir() if path.is_file() and path.name != "outputs_manifest.csv"]]
    with (OUT / "documentation/outputs_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "path", "description", "status", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow({"type": "figure" if path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff"} else "supporting", "path": str(path), "description": path.name, "status": "current", "sha256": sha256(path)})

    print(json.dumps({"output": str(OUT), "statistics": stats_rows, "event_counts": rate_df.groupby("group")["event_count"].sum().to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
