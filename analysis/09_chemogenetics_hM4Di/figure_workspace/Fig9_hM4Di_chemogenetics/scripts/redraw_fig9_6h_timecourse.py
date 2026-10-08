from __future__ import annotations

import csv
import hashlib
import itertools
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(r"F:\1.Sleep\PHD稿件")
WORK = ROOT / "Figure_Workspace" / "Fig9_hM4Di_chemogenetics"
FORMAL = ROOT / "Figures" / "Fig9_hM4Di_chemogenetics"
SOURCE = FORMAL / "data" / "ma_bout_review_current" / "sleep_architecture_ma_current_source_data.csv"
DATA_NAME = "sleep_state_6h_timecourse_current"
PROJECT_NAME = "Fig9_sleep_state_6h_timecourse_SD.opju"
GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
ANIMALS = [f"NO{i}" for i in range(1, 8)]
STATE_CODES = {"NREM": 2, "REM": 3, "WAKE": 1}
COHORTS = {
    "mCherry": {"prefix": "baseline", "colors": ("#BFBFBF", "#595959")},
    "hM3Dq": {"prefix": "POCD", "colors": ("#F3B697", "#EA6B32")},
}
YMAX = {"NREM": 80.0, "REM": 20.0, "WAKE": 100.0}
YSTEP = {"NREM": 20.0, "REM": 5.0, "WAKE": 20.0}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_scores(path: Path) -> np.ndarray:
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select start_time_seconds, score from sleep_scores_table "
            "order by start_time_seconds, start_time_sub_seconds"
        ).fetchall()
    values = np.asarray(rows)
    times = values[:, 0].astype(np.int64)
    scores = values[:, 1].astype(np.int16)
    if len(scores) != 17280 or not np.all(np.diff(times) == 5):
        raise RuntimeError(f"Expected 17,280 consecutive 5-s epochs: {path}")
    if not set(np.unique(scores)).issubset({0, 1, 2, 3}):
        raise RuntimeError(f"Unexpected score values: {path}: {np.unique(scores)}")
    return scores


def analyze(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in source.itertuples(index=False):
        path = Path(record.scores_db_path)
        current_hash = sha256(path)
        if current_hash != record.scores_db_sha256:
            raise RuntimeError(f"Staging changed after current inventory was frozen: {path}")
        scores = read_scores(path)
        for bin_index, chunk in enumerate(np.split(scores, 4), start=1):
            valid = int(np.sum(chunk != 0))
            if valid == 0:
                raise RuntimeError(f"No valid epochs in bin {bin_index}: {path}")
            for state, code in STATE_CODES.items():
                rows.append({
                    "group": record.group,
                    "cohort": "mCherry" if record.group.startswith("baseline+") else "hM3Dq",
                    "condition": record.group.split("+")[1],
                    "animal": record.animal,
                    "state": state,
                    "time_bin": bin_index,
                    "bin_start_h": (bin_index - 1) * 6,
                    "bin_end_h": bin_index * 6,
                    "valid_epochs": valid,
                    "artifact_epochs": int(np.sum(chunk == 0)),
                    "state_epochs": int(np.sum(chunk == code)),
                    "percent_of_valid": float(np.sum(chunk == code) / valid * 100.0),
                    "scores_db_path": str(path),
                    "scores_db_sha256": current_hash,
                })
    frame = pd.DataFrame(rows)
    group_animals = source.groupby("group")["animal"].apply(lambda x: sorted(x.tolist())).to_dict()
    if any(group_animals.get(group) != ANIMALS for group in GROUPS):
        raise RuntimeError(f"Unexpected animal pairing: {group_animals}")
    return frame


def holm(values: list[float]) -> list[float]:
    arr = np.asarray(values, float)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted_ranked = np.maximum.accumulate(ranked * (len(arr) - np.arange(len(arr))))
    adjusted = np.empty_like(arr)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()


def exact_sign_flip(values: np.ndarray, statistic) -> tuple[float, float, int]:
    values = np.asarray(values, float)
    observed = float(statistic(values))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(float(statistic(values * np.asarray(signs))))
    null = np.asarray(null)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - np.finfo(float).eps * 10))
    return observed, p_value, len(null)


def calculate_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_CODES:
        for cohort, cohort_info in COHORTS.items():
            sub = frame[(frame.state == state) & (frame.cohort == cohort)]
            wide = sub.pivot(index="animal", columns=["condition", "time_bin"], values="percent_of_valid").sort_index()
            sal = wide["SAL"].to_numpy(float)
            cno = wide["CNO"].to_numpy(float)
            differences = cno - sal
            effect, p_main, permutations = exact_sign_flip(
                differences.mean(axis=1), lambda x: np.mean(x)
            )
            rows.append({
                "state": state, "cohort": cohort, "analysis_role": "condition_main_effect",
                "comparison": "CNO vs SAL averaged across four 6-h bins",
                "test": f"exact paired sign-flip permutation ({permutations} allocations)",
                "n_animals": len(differences), "time_bin": "all",
                "effect_CNO_minus_SAL_percentage_points": effect, "statistic": effect,
                "p_raw": p_main, "p_holm_within_four_bins": np.nan,
            })
            interaction_stat = lambda matrix: np.sum((np.mean(matrix, axis=0) - np.mean(matrix)) ** 2)
            observed_interaction = float(interaction_stat(differences))
            null_interactions = []
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
                null_interactions.append(interaction_stat(differences * np.asarray(signs)[:, None]))
            p_interaction = float(np.mean(np.asarray(null_interactions) >= observed_interaction - np.finfo(float).eps * 10))
            rows.append({
                "state": state, "cohort": cohort, "analysis_role": "condition_by_time_interaction",
                "comparison": "CNO-SAL difference varies across four 6-h bins",
                "test": f"exact paired sign-flip permutation ({len(null_interactions)} allocations)",
                "n_animals": len(differences), "time_bin": "all",
                "effect_CNO_minus_SAL_percentage_points": np.nan, "statistic": observed_interaction,
                "p_raw": p_interaction, "p_holm_within_four_bins": np.nan,
            })
            bin_indices = []
            p_values = []
            for bin_index in range(1, 5):
                test = wilcoxon(sal[:, bin_index - 1], cno[:, bin_index - 1], method="exact")
                bin_indices.append(len(rows))
                p_values.append(float(test.pvalue))
                rows.append({
                    "state": state, "cohort": cohort, "analysis_role": "time_bin_pairwise",
                    "comparison": f"CNO vs SAL in time bin {bin_index}",
                    "test": "exact two-sided paired Wilcoxon", "n_animals": 7,
                    "time_bin": bin_index,
                    "effect_CNO_minus_SAL_percentage_points": float(np.mean(cno[:, bin_index - 1] - sal[:, bin_index - 1])),
                    "statistic": float(test.statistic), "p_raw": float(test.pvalue),
                    "p_holm_within_four_bins": np.nan,
                })
            for row_index, adjusted in zip(bin_indices, holm(p_values)):
                rows[row_index]["p_holm_within_four_bins"] = adjusted
    return pd.DataFrame(rows)


def export_panel(frame: pd.DataFrame, state: str, cohort: str, destination: Path) -> None:
    info = COHORTS[cohort]
    fig, ax = plt.subplots(figsize=(2.35, 2.05))
    for condition, color, label in zip(("SAL", "CNO"), info["colors"], ("SAL", "CNO")):
        sub = frame[(frame.state == state) & (frame.cohort == cohort) & (frame.condition == condition)]
        pivot = sub.pivot(index="animal", columns="time_bin", values="percent_of_valid").sort_index()
        mean = pivot.mean(axis=0).to_numpy(float)
        sd = pivot.std(axis=0, ddof=1).to_numpy(float)
        # Each value is a mean over a 6-h interval and is therefore plotted at
        # the interval midpoint rather than at the interval boundary.
        x = np.asarray([3.0, 9.0, 15.0, 21.0])
        ax.errorbar(
            x, mean, yerr=sd, color=color, marker="o", markersize=3.2,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8,
            linewidth=1.15, elinewidth=0.85, capsize=2.3, capthick=0.85,
            label=label, zorder=3,
        )
    ax.set_xlim(0, 24)
    ax.set_ylim(0, YMAX[state])
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["ZT0", "ZT6", "ZT12", "ZT18", "ZT24"])
    ax.set_yticks(np.arange(0, YMAX[state] + 0.01, YSTEP[state]))
    ax.set_xlabel("Zeitgeber time", labelpad=7)
    ax.set_ylabel(f"{state} (%)")
    ax.set_title(state, fontsize=7.5, pad=20)
    ax.text(0.98, 0.94, cohort, transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    # Standard 12:12 light-dark schedule: ZT0-12 light, ZT12-24 dark.
    ax.add_patch(Rectangle((0.0, 1.015), 0.5, 0.025, transform=ax.transAxes,
                           facecolor="white", edgecolor="black", linewidth=0.65,
                           clip_on=False))
    ax.add_patch(Rectangle((0.5, 1.015), 0.5, 0.025, transform=ax.transAxes,
                           facecolor="black", edgecolor="black", linewidth=0.65,
                           clip_on=False))
    ax.text(0.25, 1.05, "Light", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.3)
    ax.text(0.75, 1.05, "Dark", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.3)
    legend_location = "upper left" if state == "REM" else "lower left"
    ax.legend(loc=legend_location, frameon=False, fontsize=6, handlelength=1.7, borderpad=0.1)
    ax.grid(False)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.22, top=0.83)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=600, facecolor="white")
    fig.savefig(destination.parent / f"{destination.name}_AI_clean.pdf", facecolor="white")
    fig.savefig(destination.parent / f"{destination.name}_AI_clean.svg", facecolor="white")
    plt.close(fig)


def export_combined(panel_dir: Path, destination: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(5.0, 6.3))
    for row, state in enumerate(("NREM", "REM", "WAKE")):
        for col, cohort in enumerate(("mCherry", "hM3Dq")):
            image = plt.imread(panel_dir / f"Fig9_{state}_6h_{cohort}.png")
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.01, hspace=0.01)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def build_origin_project(frame: pd.DataFrame, summary: pd.DataFrame, project: Path) -> None:
    import originpro as op

    op.attach()
    try:
        op.new(asksave=False)
        for state in STATE_CODES:
            for cohort in COHORTS:
                sheet = op.new_sheet("w", lname=f"{state} {cohort} 6h")
                sheet.name = (state[:2] + cohort[:5]).replace("3", "")[:12]
                sub = frame[(frame.state == state) & (frame.cohort == cohort)]
                sheet.from_df(sub[["animal", "condition", "time_bin", "percent_of_valid"]].reset_index(drop=True))
                summary_sheet = op.new_sheet("w", lname=f"{state} {cohort} summary")
                summary_sheet.name = ("S" + state[:2] + cohort[:5]).replace("3", "")[:12]
                table = summary[(summary.state == state) & (summary.cohort == cohort)].copy()
                summary_sheet.from_df(table.reset_index(drop=True))
        op.save(str(project))
    finally:
        op.detach()


def archive_current(root: Path, stamp: str) -> None:
    archive = root / "archive" / stamp / "sleep_state_6h_timecourse"
    stems = [f"Fig9_{state}_6h_{cohort}" for state in STATE_CODES for cohort in COHORTS]
    stems.append("Fig9_sleep_state_6h_timecourse_6panels")
    for stem in stems:
        for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
            for parent in (root / "panels", root):
                source = parent / f"{stem}{extension}"
                if source.exists():
                    relative = source.relative_to(root)
                    target = archive / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    for relative in (Path("data") / DATA_NAME, Path("editable") / PROJECT_NAME):
        source = root / relative
        if source.exists():
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)


def update_manifest() -> None:
    path = FORMAL / "documentation" / "outputs_manifest.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if "_6h_" not in row.get("path", "") and "6h_timecourse" not in row.get("path", "")]
    for state in STATE_CODES:
        for cohort in COHORTS:
            stem = f"Fig9_{state}_6h_{cohort}"
            for kind, extension in (("preview_png", ".png"), ("panel_pdf", ".pdf"), ("panel_svg", ".svg")):
                rows.append({"type": kind, "path": str(FORMAL / "panels" / f"{stem}{extension}"), "description": f"{state} 6-h time course in {cohort}; mean +/- SD", "status": "current"})
    rows.extend([
        {"type": "origin_project", "path": str(FORMAL / "editable" / PROJECT_NAME), "description": "Editable Origin data project for six 6-h sleep-state time courses", "status": "current"},
        {"type": "source_data", "path": str(FORMAL / "data" / DATA_NAME / "animal_level_sleep_state_6h.csv"), "description": "Animal-level state occupancy in four 6-h bins", "status": "current"},
        {"type": "stats", "path": str(FORMAL / "data" / DATA_NAME / "sleep_state_6h_statistics.csv"), "description": "Paired condition, time-interaction and time-bin statistics", "status": "current"},
    ])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("type", "path", "description", "status"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = pd.read_csv(SOURCE)
    frame = analyze(source)
    summary = (
        frame.groupby(["state", "cohort", "condition", "time_bin"], as_index=False)["percent_of_valid"]
        .agg(n="count", mean="mean", sd=lambda x: x.std(ddof=1), sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    )
    frame["zt_center_h"] = frame["time_bin"].map({1: 3, 2: 9, 3: 15, 4: 21})
    frame["zt_interval"] = frame["time_bin"].map({1: "ZT0-6", 2: "ZT6-12", 3: "ZT12-18", 4: "ZT18-24"})
    summary["zt_center_h"] = summary["time_bin"].map({1: 3, 2: 9, 3: 15, 4: 21})
    summary["zt_interval"] = summary["time_bin"].map({1: "ZT0-6", 2: "ZT6-12", 3: "ZT12-18", 4: "ZT18-24"})
    stats = calculate_stats(frame)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for root in (WORK, FORMAL):
        archive_current(root, stamp)

    stage = WORK / "panels" / "sleep_state_6h_latest"
    stage.mkdir(parents=True, exist_ok=True)
    for state in STATE_CODES:
        for cohort in COHORTS:
            export_panel(frame, state, cohort, stage / f"Fig9_{state}_6h_{cohort}")
    export_combined(stage, WORK / "Fig9_sleep_state_6h_timecourse_6panels")

    data_dir = WORK / "data" / DATA_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "animal_level_sleep_state_6h.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(data_dir / "sleep_state_6h_group_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(data_dir / "sleep_state_6h_statistics.csv", index=False, encoding="utf-8-sig")
    project = WORK / "editable" / PROJECT_NAME
    project.parent.mkdir(parents=True, exist_ok=True)
    build_origin_project(frame, summary, project)

    global_provenance = Path(r"F:\Sleep\outputs\figure_provenance")
    global_provenance.mkdir(parents=True, exist_ok=True)
    for root in (WORK, FORMAL):
        for state in STATE_CODES:
            for cohort in COHORTS:
                stem = f"Fig9_{state}_6h_{cohort}"
                for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
                    target = root / "panels" / f"{stem}{extension}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stage / f"{stem}{extension}", target)
        for extension in (".png", ".pdf"):
            source_combined = WORK / f"Fig9_sleep_state_6h_timecourse_6panels{extension}"
            target_combined = root / source_combined.name
            if source_combined != target_combined:
                shutil.copy2(source_combined, target_combined)
        target_data = root / "data" / DATA_NAME
        if target_data != data_dir:
            shutil.copytree(data_dir, target_data, dirs_exist_ok=True)
        target_project = root / "editable" / PROJECT_NAME
        target_project.parent.mkdir(parents=True, exist_ok=True)
        if target_project != project:
            shutil.copy2(project, target_project)
        target_script = root / "scripts" / Path(__file__).name
        target_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), target_script)

        note = f"""# Fig9 sleep-state 6-h time courses

- generated_at: {stamp}
- status: formal Fig9 supplementary panels
- source: latest manually reviewed scores.db3 files frozen in data/{DATA_NAME}/animal_level_sleep_state_6h.csv
- sample: mCherry/baseline and hM3Dq/POCD cohorts; SAL and CNO are paired within NO1-NO7
- analysis window: full 24 h aligned to ZT0, four consecutive 6-h bins (4320 five-second epochs per bin)
- display timing: values for ZT0-6, ZT6-12, ZT12-18 and ZT18-24 are plotted at interval midpoints ZT3, ZT9, ZT15 and ZT21; axis boundaries are ZT0, 6, 12, 18 and 24
- light-dark schedule: ZT0-12 Light and ZT12-24 Dark, shown by the bar above each panel
- denominator: non-artifact epochs within each bin; score 0 is excluded from the denominator
- state codes: WAKE=1, NREM=2, REM=3
- display: arithmetic mean +/- sample SD; hollow symbols; no animal IDs; no uncorrected significance marks
- inference: exact paired sign-flip permutation for condition main effect and condition-by-time interaction; exact paired Wilcoxon within each bin with Holm correction over four bins
- editable: editable/{PROJECT_NAME}
- script: scripts/{Path(__file__).name}
"""
        docs = root / "documentation"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "fig9_sleep_state_6h_timecourse_provenance.md").write_text(note, encoding="utf-8")
        (global_provenance / "fig9_sleep_state_6h_timecourse_provenance.md").write_text(note, encoding="utf-8")
        frame.to_csv(global_provenance / "fig9_sleep_state_6h_timecourse_source_data.csv", index=False, encoding="utf-8-sig")
        stats.to_csv(global_provenance / "fig9_sleep_state_6h_timecourse_stats.csv", index=False, encoding="utf-8-sig")

    update_manifest()
    print(stats.to_string(index=False))
    print(FORMAL / "panels")


if __name__ == "__main__":
    main()
