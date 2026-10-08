from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from analyze_chr2_spindle_so_5s_10s import (
    OUT as SPINDLE_OUT,
    FIG_DIR as SPINDLE_FIG_DIR,
    PROV_DIR,
    discover_records,
    parse_scores_tsv,
    load_5s_stim_table,
    make_10s_stim_table,
    read_edf,
    STATE_WAKE,
    STATE_NREM,
    STATE_REM,
    STATE_MA,
    EPOCH_SEC,
)


ROOT = Path(r"F:\Sleep")
OUT = ROOT / "outputs" / "virus_chr2_time" / "laser_state_occurrence"
FIG_DIR = ROOT / "Figure" / "VirusCHR2_laser_state_occurrence_20260711"

STATE_ORDER = [
    ("Wake", STATE_WAKE),
    ("NREM", STATE_NREM),
    ("REM", STATE_REM),
    ("MA", STATE_MA),
]


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def build_stim_table(records: list[dict]) -> pd.DataFrame:
    for rec in records:
        rec["edf_duration_s"] = read_edf(rec["edf"]).duration_sec
    return pd.concat(
        [
            load_5s_stim_table(),
            make_10s_stim_table([r for r in records if r["condition"] == "10s"]),
        ],
        ignore_index=True,
    )


def epoch_laser_mask(epoch_starts: np.ndarray, stims: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if stims.empty:
        return np.zeros_like(epoch_starts, dtype=bool), np.zeros_like(epoch_starts, dtype=bool)
    first = float(stims["stim_s"].min())
    last = float((stims["stim_s"] + stims["stim_duration_s"]).max())
    in_session = (epoch_starts >= first) & (epoch_starts < last)
    laser = np.zeros_like(epoch_starts, dtype=bool)
    epoch_ends = epoch_starts + EPOCH_SEC
    for _, row in stims.iterrows():
        a = float(row["stim_s"])
        b = float(row["stim_s"] + row["stim_duration_s"])
        laser |= (epoch_starts < b) & (epoch_ends > a)
    laser &= in_session
    no_laser = in_session & ~laser
    return laser, no_laser


def recode_ma_20s(states: np.ndarray, max_ma_epochs: int = 4) -> np.ndarray:
    """Recode short Wake bouts <=20 s flanked by NREM as MA, using 5-s epochs."""
    out = states.copy()
    wake = states == STATE_WAKE
    if not wake.any():
        return out
    changes = np.diff(np.r_[0, wake.astype(np.int8), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        length = end - start
        left_nrem = start > 0 and out[start - 1] == STATE_NREM
        right_nrem = end < len(out) and out[end] == STATE_NREM
        if length <= max_ma_epochs and left_nrem and right_nrem:
            out[start:end] = STATE_MA
    return out


def compute_occurrence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = discover_records()
    stim_table = build_stim_table(records)
    rows = []
    qc_rows = []
    for rec in records:
        scores = parse_scores_tsv(rec["scores"])
        stims = stim_table[(stim_table["mouse"] == rec["mouse"]) & (stim_table["condition"] == rec["condition"])]
        epoch_starts = scores["time_s"].to_numpy(dtype=float)
        states_raw = scores["state"].to_numpy(dtype=int)
        states = recode_ma_20s(states_raw)
        laser_mask, no_laser_mask = epoch_laser_mask(epoch_starts, stims)
        qc_rows.append(
            {
                "mouse": rec["mouse"],
                "condition": rec["condition"],
                "scores": str(rec["scores"]),
                "stim_trials": len(stims),
                "laser_epochs": int(laser_mask.sum()),
                "no_laser_epochs": int(no_laser_mask.sum()),
                "raw_wake_epochs": int((states_raw == STATE_WAKE).sum()),
                "ma20_epochs": int((states == STATE_MA).sum()),
            }
        )
        for laser_label, mask in [("No laser", no_laser_mask), ("Laser", laser_mask)]:
            denom = int(mask.sum())
            for state_name, state_code in STATE_ORDER:
                rows.append(
                    {
                        "mouse": rec["mouse"],
                        "condition": rec["condition"],
                        "laser_status": laser_label,
                        "state": state_name,
                        "epochs": int(((states == state_code) & mask).sum()),
                        "total_epochs": denom,
                        "percent": 100 * float(((states == state_code) & mask).sum()) / denom if denom else np.nan,
                    }
                )
    session_df = pd.DataFrame(rows)
    qc_df = pd.DataFrame(qc_rows)

    # For the one-graph summary, avoid double-counting the same mouse across 5s and 10s sessions:
    # average each mouse/laser/state across available stimulation durations.
    mouse_df = (
        session_df.groupby(["mouse", "laser_status", "state"], as_index=False)["percent"]
        .mean()
        .assign(condition="combined")
    )
    return session_df, mouse_df, qc_df


def make_stats(session_df: pd.DataFrame, mouse_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("combined", mouse_df), ("5s", session_df[session_df["condition"] == "5s"]), ("10s", session_df[session_df["condition"] == "10s"])]:
        for state_name, _ in STATE_ORDER:
            sub = df[df["state"] == state_name]
            piv = sub.pivot(index="mouse", columns="laser_status", values="percent").dropna()
            if not {"No laser", "Laser"}.issubset(piv.columns) or len(piv) < 2:
                continue
            diff = piv["Laser"] - piv["No laser"]
            try:
                wilcoxon_p = stats.wilcoxon(piv["No laser"], piv["Laser"]).pvalue if len(piv) >= 3 and np.any(diff != 0) else np.nan
            except ValueError:
                wilcoxon_p = np.nan
            rows.append(
                {
                    "analysis": label,
                    "state": state_name,
                    "n_mice": len(piv),
                    "no_laser_mean": piv["No laser"].mean(),
                    "no_laser_sem": sem(piv["No laser"]),
                    "laser_mean": piv["Laser"].mean(),
                    "laser_sem": sem(piv["Laser"]),
                    "delta_laser_minus_no_laser": diff.mean(),
                    "paired_t_p": stats.ttest_rel(piv["No laser"], piv["Laser"], nan_policy="omit").pvalue,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_state_occurrence(plot_df: pd.DataFrame, stats_df: pd.DataFrame, analysis: str, filename_stub: str, title: str | None = None) -> list[Path]:
    colors = {"No laser": "#E6E6E6", "Laser": "#C7C9FF"}
    edge = {"No laser": "#555555", "Laser": "#6E73D9"}
    states = [s for s, _ in STATE_ORDER]
    x = np.arange(len(states))
    width = 0.32

    fig, ax = plt.subplots(figsize=(2.55, 2.25))
    summary = plot_df.groupby(["laser_status", "state"])["percent"].agg(["mean", sem]).reset_index()
    offsets = {"No laser": -width / 2, "Laser": width / 2}
    for laser_status in ["No laser", "Laser"]:
        sub = summary[summary["laser_status"] == laser_status].set_index("state").reindex(states)
        ax.bar(
            x + offsets[laser_status],
            sub["mean"],
            yerr=sub["sem"],
            width=width,
            color=colors[laser_status],
            edgecolor=edge[laser_status],
            linewidth=0.8,
            capsize=2,
            label=laser_status,
            zorder=2,
        )
        for i, state in enumerate(states):
            vals = plot_df[(plot_df["laser_status"] == laser_status) & (plot_df["state"] == state)]["percent"].to_numpy()
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) else []
            ax.scatter(
                np.full(len(vals), x[i] + offsets[laser_status]) + jitter,
                vals,
                s=12,
                facecolors="white",
                edgecolors=edge[laser_status],
                linewidths=0.7,
                zorder=3,
            )

    for i, state in enumerate(states):
        stat = stats_df[(stats_df["analysis"] == analysis) & (stats_df["state"] == state)]
        if stat.empty:
            continue
        stars = p_to_stars(float(stat.iloc[0]["paired_t_p"]))
        if not stars:
            continue
        y = plot_df[plot_df["state"] == state]["percent"].max() + 5
        ax.plot([x[i] - width / 2, x[i] + width / 2], [y, y], color="black", lw=0.8)
        ax.text(x[i], y + 1.5, stars, ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(80, np.nanmax(plot_df["percent"]) + 15))
    ax.set_ylabel("%", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(pad=0.8)

    paths = []
    for ext in ["png", "pdf", "svg"]:
        p = FIG_DIR / f"{filename_stub}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_combined(mouse_df: pd.DataFrame, stats_df: pd.DataFrame) -> list[Path]:
    return plot_state_occurrence(mouse_df, stats_df, "combined", "virus_chr2_laser_state_occurrence_combined")


def write_metadata(paths: list[Path], session_df: pd.DataFrame, mouse_df: pd.DataFrame, stats_df: pd.DataFrame, qc_df: pd.DataFrame) -> None:
    source_paths = [
        {"role": "stimulation schedule", "path": str(SPINDLE_OUT / "stimulation_schedule_5s_10s.csv"), "description": "Stimulus timing reused from spindle/SO workflow"},
        {"role": "analysis script", "path": str(ROOT / "analysis_code" / "virus_chr2_spindle_so" / "plot_chr2_laser_state_occurrence.py"), "description": "Laser vs no-laser state occurrence analysis"},
    ]
    for _, row in qc_df.iterrows():
        source_paths.append({"role": f"{row['condition']} mouse {row['mouse']} scores", "path": row["scores"], "description": "5-s sleep scoring TSV"})
    pd.DataFrame(source_paths).to_csv(FIG_DIR / "source_paths.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"path": str(p), "format": p.suffix.lstrip("."), "note": "final figure output"} for p in paths]).to_csv(FIG_DIR / "outputs_manifest.csv", index=False, encoding="utf-8-sig")
    readme = f"""# VirusCHR2 Laser State Occurrence

Purpose: quantify Wake, NREM, REM, and MA occurrence during laser stimulation epochs versus inter-stimulus no-laser epochs.

Definition:
- Laser: 5-s scoring epochs overlapping a blue-light stimulation window.
- No laser: epochs within the same stimulation session, between first and last stimulation, excluding laser-overlapping epochs.
- Combined summary averages 5 s and 10 s sessions within each mouse first, then treats mouse as biological n.
- Separate 5s-only and 10s-only figures use only that stimulation duration.

Source data:
- Session-level source table: `{OUT / 'laser_state_occurrence_by_session.csv'}`
- Mouse-level combined source table: `{OUT / 'laser_state_occurrence_by_mouse_combined.csv'}`
- Statistics: `{OUT / 'laser_state_occurrence_stats.csv'}`

Statistics: paired Laser vs No laser comparisons at mouse level; paired t-test and Wilcoxon p-values exported.

Outputs:
{chr(10).join('- ' + str(p) for p in paths)}
"""
    (FIG_DIR / "README_analysis.md").write_text(readme, encoding="utf-8")
    shutil.copy2(ROOT / "analysis_code" / "virus_chr2_spindle_so" / "plot_chr2_laser_state_occurrence.py", FIG_DIR / "plot_chr2_laser_state_occurrence.py")
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    (PROV_DIR / "virus_chr2_laser_state_occurrence_provenance.md").write_text(readme.replace("# VirusCHR2 Laser State Occurrence", "# Provenance: VirusCHR2 Laser State Occurrence"), encoding="utf-8")
    mouse_df.to_csv(PROV_DIR / "virus_chr2_laser_state_occurrence_source_data.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(PROV_DIR / "virus_chr2_laser_state_occurrence_stats.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    session_df, mouse_df, qc_df = compute_occurrence()
    stats_df = make_stats(session_df, mouse_df)
    session_df.to_csv(OUT / "laser_state_occurrence_by_session.csv", index=False, encoding="utf-8-sig")
    mouse_df.to_csv(OUT / "laser_state_occurrence_by_mouse_combined.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(OUT / "laser_state_occurrence_qc.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(OUT / "laser_state_occurrence_stats.csv", index=False, encoding="utf-8-sig")
    paths = plot_combined(mouse_df, stats_df)
    for cond, title in [("5s", "5 s"), ("10s", "10 s")]:
        cond_df = session_df[session_df["condition"] == cond].copy()
        paths.extend(
            plot_state_occurrence(
                cond_df,
                stats_df,
                cond,
                f"virus_chr2_laser_state_occurrence_{cond}_only",
                title=title,
            )
        )
    for p in paths:
        shutil.copy2(p, OUT / p.name)
    write_metadata(paths, session_df, mouse_df, stats_df, qc_df)
    print("outputs:")
    for p in paths:
        print(p)
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
