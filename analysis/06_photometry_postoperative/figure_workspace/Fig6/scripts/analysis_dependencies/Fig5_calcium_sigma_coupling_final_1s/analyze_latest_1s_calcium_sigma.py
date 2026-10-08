"""Relate calcium responses to peri-MA EEG sigma power after final 1-s alignment.

The experimental unit is the mouse.  Sigma is defined as 10-16 Hz, matching the
project's EEG microstructure workflow.  Per-event sigma power is expressed in dB
relative to the stable NREM baseline (-20 to -10 s before MA onset).
"""
from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import edfio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from analyze_latest_1s_matched_transitions import CONT_EVENTS, SUR_EVENTS, PALETTE, sha256, welch_df


ROOT = Path(r"F:\1.Sleep\eXdata\VirusGCAMP\PaperUse")
OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_sigma_coupling_final_1s")
PROVENANCE_ROOT = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_sigma_coupling_final_1s"
SIGMA_LOW_HZ = 10.0
SIGMA_HIGH_HZ = 16.0
BASELINE = (-20.0, -10.0)
RESPONSE = (0.0, 5.0)
TIME_BINS = np.arange(-20.0, 20.0, 1.0) + 0.5

# Plot time = EEG time + offset.  Aligned EDFs already start at Plot time zero.
RECORDS = [
    {"group": "Cont", "animal": "Cont1", "edf": ROOT / "Cont/1/EEG/export_200Hz.edf", "offset": 0.0},
    {"group": "Cont", "animal": "Cont2", "edf": ROOT / "Cont/2/EEG/export_200.edf", "offset": 29.786},
    {"group": "Cont", "animal": "Cont3", "edf": ROOT / "Cont/3/EEG/export_200Hz.edf", "offset": 0.0},
    {"group": "Cont", "animal": "Cont4", "edf": ROOT / "Cont/4/EEG/export_200.edf", "offset": 29.544},
    {"group": "Cont", "animal": "Cont5", "edf": ROOT / "Cont/5/EEG/export_200.edf", "offset": 29.827},
    {"group": "Cont", "animal": "Cont6", "edf": ROOT / "Cont/6/EEG/export_200.edf", "offset": 29.231},
    {"group": "Cont", "animal": "Cont7", "edf": ROOT / "Cont/7/EEG/export_200Hz.edf", "offset": 0.0},
    {"group": "SUR", "animal": "SUR1", "edf": ROOT / "SUR/1/Aligned_EEG_56m20s_to84m/EEG_aligned.edf", "offset": 0.0},
    {"group": "SUR", "animal": "SUR2", "edf": ROOT / "SUR/2/export_200Hz.edf", "offset": 11.568},
    {"group": "SUR", "animal": "SUR3", "edf": ROOT / "SUR/3/Aligned_EEGminus21s/EEG_aligned_200Hz.edf", "offset": 0.0},
    {"group": "SUR", "animal": "SUR4", "edf": ROOT / "SUR/4/EEG/export_200.edf", "offset": 7.306},
    {"group": "SUR", "animal": "SUR5", "edf": ROOT / "SUR/5/EEG/export_200.edf", "offset": 4.047},
    {"group": "SUR", "animal": "SUR6", "edf": ROOT / "SUR/6_/EEG/export_200.edf", "offset": 4.046},
    {"group": "SUR", "animal": "SUR7", "edf": ROOT / "SUR/7/EEG1/export_200.edf", "offset": 4.766},
]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.75,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
})


def select_eeg(edf_path: Path) -> tuple[np.ndarray, float, str, float]:
    edf = edfio.read_edf(edf_path)
    labels = [s.label for s in edf.signals]
    # Prefer the same right EEG derivation used during visual alignment.  The
    # two-channel 200-Hz exports call the retained EEG channel Ch1.
    preferred = [i for i, x in enumerate(labels) if "EEG2A-B" in x.upper()]
    if not preferred:
        preferred = [i for i, x in enumerate(labels) if x.upper() == "CH1"]
    if not preferred:
        preferred = [i for i, x in enumerate(labels) if "EEG" in x.upper()]
    if not preferred:
        raise ValueError(f"No EEG channel in {edf_path}: {labels}")
    channel = edf.signals[preferred[0]]
    return np.asarray(channel.data, dtype=float), float(channel.sampling_frequency), channel.label, float(edf.duration)


def interval_mean(frame: pd.DataFrame, column: str, lo: float, hi: float) -> float:
    mask = (frame["time_s"] >= lo) & (frame["time_s"] < hi)
    return float(frame.loc[mask, column].mean())


def fisher_z_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan, math.nan, math.nan
    rho, p = stats.spearmanr(x, y)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return rho, float(np.arctanh(rho)), float(p)


def compare(frame: pd.DataFrame, metric: str, label: str) -> dict[str, object]:
    cont = frame.loc[frame.group.eq("Cont"), metric].dropna().to_numpy(float)
    sur = frame.loc[frame.group.eq("SUR"), metric].dropna().to_numpy(float)
    test = stats.ttest_ind(sur, cont, equal_var=False)
    pooled = math.sqrt(((len(cont)-1)*np.var(cont, ddof=1)+(len(sur)-1)*np.var(sur, ddof=1))/(len(cont)+len(sur)-2))
    g = (1 - 3/(4*(len(cont)+len(sur))-9)) * (np.mean(sur)-np.mean(cont))/pooled if pooled else math.nan
    return {
        "comparison": label, "metric": metric, "n_cont": len(cont), "n_sur": len(sur),
        "cont_mean": float(np.mean(cont)), "cont_sem": float(stats.sem(cont)),
        "sur_mean": float(np.mean(sur)), "sur_sem": float(stats.sem(sur)),
        "difference_sur_minus_cont": float(np.mean(sur)-np.mean(cont)),
        "welch_t": float(test.statistic), "welch_df": welch_df(cont, sur),
        "welch_p": float(test.pvalue), "hedges_g": float(g),
    }


def extract_sigma(events: pd.DataFrame, record: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw, fs, channel, duration = select_eeg(Path(record["edf"]))
    sos = signal.butter(4, [SIGMA_LOW_HZ, SIGMA_HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    sigma = signal.sosfiltfilt(sos, raw)
    sigma_power = np.square(np.abs(signal.hilbert(sigma)))
    subset = events[(events.group == record["group"]) & (events.animal == record["animal"])]
    meta = subset.drop_duplicates("event_index")[["event_index", "onset_s", "ma_duration_s"]]
    rows: list[dict[str, object]] = []
    retained = 0
    for event in meta.itertuples(index=False):
        eeg_onset = float(event.onset_s) - float(record["offset"])
        if eeg_onset + TIME_BINS[0] - 0.5 < 0 or eeg_onset + TIME_BINS[-1] + 0.5 > duration:
            continue
        b0 = int(round((eeg_onset + BASELINE[0]) * fs)); b1 = int(round((eeg_onset + BASELINE[1]) * fs))
        baseline_power = float(np.mean(sigma_power[b0:b1]))
        if not np.isfinite(baseline_power) or baseline_power <= 0:
            continue
        calcium_event = subset[subset.event_index == event.event_index]
        calcium_0_5 = interval_mean(calcium_event, "delta_signal", *RESPONSE)
        retained += 1
        for center in TIME_BINS:
            i0 = int(round((eeg_onset + center - 0.5) * fs)); i1 = int(round((eeg_onset + center + 0.5) * fs))
            power = float(np.mean(sigma_power[i0:i1]))
            rows.append({
                "group": record["group"], "animal": record["animal"], "event_index": int(event.event_index),
                "onset_plot_s": float(event.onset_s), "onset_eeg_s": eeg_onset,
                "ma_duration_s": float(event.ma_duration_s), "time_s": float(center),
                "sigma_power": power, "sigma_change_db": 10.0 * math.log10(power / baseline_power),
                "calcium_response_0_5": calcium_0_5,
            })
    return rows, {"group": record["group"], "animal": record["animal"], "edf_path": str(record["edf"]),
                  "edf_sha256": sha256(Path(record["edf"])), "edf_duration_s": duration,
                  "eeg_channel": channel, "sampling_hz": fs, "plot_minus_eeg_offset_s": float(record["offset"]),
                  "eligible_events_input": int(len(meta)), "eligible_events_retained": retained}


def dot_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, p: float) -> None:
    jitter = np.linspace(-0.075, 0.075, 7)
    for x, group in enumerate(("Cont", "SUR")):
        vals = frame.loc[frame.group.eq(group), metric].dropna().to_numpy(float)
        col = PALETTE[group]
        ax.scatter(x+jitter[:len(vals)], vals, s=25, facecolor="white", edgecolor=col["edge"], lw=0.9, zorder=3)
        ax.errorbar(x, np.mean(vals), yerr=stats.sem(vals), fmt="_", markersize=18, color=col["edge"], lw=1, capsize=2.2)
    ax.set_xticks([0, 1], ["Cont", "SUR"]); ax.set_ylabel(ylabel); ax.axhline(0, color="#888", lw=0.6)
    ax.text(0.5, 1.02, f"P = {p:.3f}", transform=ax.transAxes, ha="center")


def main() -> None:
    for p in (OUT, OUT/"data", OUT/"panels", OUT/"scripts", OUT/"documentation", OUT/"editable", PROVENANCE_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    calcium = pd.concat([pd.read_csv(CONT_EVENTS), pd.read_csv(SUR_EVENTS)], ignore_index=True)
    sigma_rows, manifests = [], []
    for record in RECORDS:
        rows, manifest = extract_sigma(calcium, record); sigma_rows.extend(rows); manifests.append(manifest)
    time_df = pd.DataFrame(sigma_rows)
    event_df = time_df.groupby(["group", "animal", "event_index", "onset_plot_s", "onset_eeg_s", "ma_duration_s", "calcium_response_0_5"], as_index=False).apply(
        lambda x: pd.Series({"sigma_change_db_0_5": interval_mean(x, "sigma_change_db", *RESPONSE),
                             "sigma_suppression_db_0_5": -interval_mean(x, "sigma_change_db", *RESPONSE)}), include_groups=False).reset_index(drop=True)
    mouse_rows = []
    for (group, animal), x in event_df.groupby(["group", "animal"]):
        rho, z, rp = fisher_z_spearman(x["calcium_response_0_5"].to_numpy(float), x["sigma_suppression_db_0_5"].to_numpy(float))
        mouse_rows.append({"group": group, "animal": animal, "event_count": len(x),
                           "calcium_response_0_5": float(x.calcium_response_0_5.mean()),
                           "sigma_change_db_0_5": float(x.sigma_change_db_0_5.mean()),
                           "sigma_suppression_db_0_5": float(x.sigma_suppression_db_0_5.mean()),
                           "event_spearman_rho_calcium_vs_sigma_suppression": rho,
                           "event_spearman_fisher_z": z, "event_spearman_p": rp})
    mouse_df = pd.DataFrame(mouse_rows)
    animal_time = time_df.groupby(["group", "animal", "time_s"], as_index=False)["sigma_change_db"].mean()
    specs = [
        ("sigma_change_db_0_5", "Sigma power change, 0-5 s (dB vs baseline)"),
        ("sigma_suppression_db_0_5", "Sigma suppression, 0-5 s (positive = larger fall)"),
        ("event_spearman_fisher_z", "Within-mouse calcium-sigma coupling (Fisher z of Spearman rho)"),
    ]
    stat_df = pd.DataFrame([compare(mouse_df, m, label) for m, label in specs])
    pmap = stat_df.set_index("metric").welch_p.to_dict()

    time_df.to_csv(OUT/"data/event_level_sigma_timecourses.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUT/"data/event_level_calcium_sigma_metrics.csv", index=False, encoding="utf-8-sig")
    mouse_df.to_csv(OUT/"data/animal_level_calcium_sigma_metrics.csv", index=False, encoding="utf-8-sig")
    animal_time.to_csv(OUT/"data/animal_level_sigma_timecourses.csv", index=False, encoding="utf-8-sig")
    stat_df.to_csv(OUT/"data/calcium_sigma_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifests).to_csv(OUT/"data/source_file_manifest.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65), gridspec_kw={"width_ratios": [1.55, 1, 1], "wspace": 0.55})
    for group in ("Cont", "SUR"):
        pivot = animal_time[animal_time.group == group].pivot(index="animal", columns="time_s", values="sigma_change_db")
        mean, sem = pivot.mean().to_numpy(), pivot.sem().to_numpy(); x = pivot.columns.to_numpy(float); col=PALETTE[group]
        axes[0].plot(x, mean, color=col["edge"], lw=1.2, label=f"{group} (n={len(pivot)})")
        axes[0].fill_between(x, mean-sem, mean+sem, color=col["fill"], alpha=.28, lw=0)
    axes[0].axvspan(-20, -10, color="#D9EAF2", alpha=.35); axes[0].axvspan(0, 5, color="#E5E5E5", alpha=.45)
    axes[0].axvline(0, color="black", ls="--", lw=.7); axes[0].axhline(0, color="#888", lw=.6)
    axes[0].set(xlim=(-20, 20), xlabel="Time from MA onset (s)", ylabel="Sigma power change (dB)\nvs -20 to -10 s baseline")
    axes[0].legend(frameon=False, fontsize=6.3)
    dot_panel(axes[1], mouse_df, "sigma_suppression_db_0_5", "Sigma suppression, 0-5 s (dB)\npositive = larger decrease", pmap["sigma_suppression_db_0_5"])
    dot_panel(axes[2], mouse_df, "event_spearman_fisher_z", "Calcium-sigma coupling per mouse\nFisher z (positive = coupled fall)", pmap["event_spearman_fisher_z"])
    for label, ax in zip("abc", axes): ax.text(-.18, 1.07, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.suptitle("Peri-microarousal calcium response and EEG sigma suppression", fontsize=9.5, y=.99)
    fig.text(.5, .015, "10-16 Hz Hilbert power; mouse-level means +/- SEM; two-sided Welch independent-samples t-test", ha="center", fontsize=5.8)
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.23)
    base = OUT/"Fig5_calcium_sigma_coupling_final_1s"
    for ext, kw in [("png", {"dpi":300}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi":600})]: fig.savefig(base.with_suffix('.'+ext), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

    conclusion = stat_df.set_index("metric")
    text = f"""# Calcium-sigma coupling analysis\n\n- Sigma definition: {SIGMA_LOW_HZ:g}-{SIGMA_HIGH_HZ:g} Hz, fourth-order zero-phase Butterworth band-pass followed by Hilbert power.\n- Event baseline: {BASELINE[0]:g} to {BASELINE[1]:g} s before EEG-defined MA onset.\n- Early window: {RESPONSE[0]:g} to {RESPONSE[1]:g} s. Power change is 10 log10(response/baseline).\n- Experimental unit: mouse (n=7/group); group tests are two-sided Welch independent-samples t-tests.\n- Sigma suppression SUR-Cont: {conclusion.loc['sigma_suppression_db_0_5','difference_sur_minus_cont']:.3f} dB, P={conclusion.loc['sigma_suppression_db_0_5','welch_p']:.4g}.\n- Calcium-sigma coupling Fisher-z SUR-Cont: {conclusion.loc['event_spearman_fisher_z','difference_sur_minus_cont']:.3f}, P={conclusion.loc['event_spearman_fisher_z','welch_p']:.4g}.\n\nPositive suppression means a larger post-MA decrease. Association does not establish that calcium caused the EEG change.\n"""
    (OUT/"documentation/README.md").write_text(text, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT/"scripts"/Path(__file__).name)
    checks = []
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): checks.append({"path": str(p), "sha256": sha256(p)})
    pd.DataFrame(checks).to_csv(OUT/"data/output_checksums.csv", index=False, encoding="utf-8-sig")
    prov = {"figure_id": FIGURE_ID, "output_folder": str(OUT), "script": str(Path(__file__)), "sigma_band_hz": [10,16], "baseline_s": list(BASELINE), "response_s": list(RESPONSE)}
    import json
    (PROVENANCE_ROOT/f"{FIGURE_ID}.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(stat_df.to_string(index=False)); print('\nAnimal metrics:\n', mouse_df.to_string(index=False)); print('\nOutput:', OUT)


if __name__ == "__main__":
    main()
