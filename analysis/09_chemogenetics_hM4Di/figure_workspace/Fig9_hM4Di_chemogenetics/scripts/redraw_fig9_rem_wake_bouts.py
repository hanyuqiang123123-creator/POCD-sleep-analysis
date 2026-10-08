from __future__ import annotations

import csv
import hashlib
import itertools
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(r"F:\1.Sleep\PHD稿件")
WORK = ROOT / "Figure_Workspace" / "Fig9_hM4Di_chemogenetics"
FORMAL = ROOT / "Figures" / "Fig9_hM4Di_chemogenetics"
SOURCE = FORMAL / "data" / "ma_bout_review_current" / "sleep_architecture_ma_current_source_data.csv"
DATA_DIR_NAME = "rem_wake_bouts_current"
PROJECT_NAME = "Fig9_REM_WAKE_bout_metrics_SD.opju"
GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
ANIMALS = [f"NO{i}" for i in range(1, 8)]

SPECS = [
    {"metric": "rem_bout_duration_s", "stem": "Fig9_REM_bout_duration", "title": "REM", "ylabel": "Duration (s)", "ymax": 110.0, "ystep": 20.0},
    {"metric": "rem_bout_frequency_h", "stem": "Fig9_REM_bout_frequency", "title": "REM", "ylabel": "Frequency / h", "ymax": 4.0, "ystep": 0.5},
    {"metric": "wake_bout_duration_s", "stem": "Fig9_WAKE_bout_duration", "title": "WAKE", "ylabel": "Duration (s)", "ymax": 320.0, "ystep": 50.0},
    {"metric": "wake_bout_frequency_h", "stem": "Fig9_WAKE_bout_frequency", "title": "WAKE", "ylabel": "Frequency / h", "ymax": 35.0, "ystep": 5.0},
]

ORIGIN_DIR = WORK / "scripts" / "origin_scripts"
sys.path.insert(0, str(ORIGIN_DIR))
sys.path.insert(0, str(WORK / "scripts"))
import rebuild_fig9_current_sleep_architecture_ma_origin as arch  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        raise RuntimeError(f"Unexpected sleep scores in {path}: {np.unique(scores)}")
    return scores


def state_bouts(scores: np.ndarray, state: int) -> np.ndarray:
    edges = np.r_[0, np.flatnonzero(np.diff(scores) != 0) + 1, len(scores)]
    durations = [(end - start) * 5.0 for start, end in zip(edges[:-1], edges[1:]) if scores[start] == state]
    if not durations:
        raise RuntimeError(f"No bouts for state {state}")
    return np.asarray(durations, float)


def analyze(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in source.itertuples(index=False):
        path = Path(record.scores_db_path)
        if sha256(path) != record.scores_db_sha256:
            raise RuntimeError(f"Current scores.db3 differs from approved source inventory: {path}")
        scores = read_scores(path)
        valid_hours = float(np.sum(scores != 0) * 5.0 / 3600.0)
        rem = state_bouts(scores, 3)
        wake = state_bouts(scores, 1)
        rows.append({
            "group": record.group,
            "animal": record.animal,
            "scores_db_path": str(path),
            "scores_db_sha256": record.scores_db_sha256,
            "valid_hours": valid_hours,
            "rem_bout_count": len(rem),
            "rem_bout_duration_s": float(rem.mean()),
            "rem_bout_frequency_h": float(len(rem) / valid_hours),
            "wake_bout_count": len(wake),
            "wake_bout_duration_s": float(wake.mean()),
            "wake_bout_frequency_h": float(len(wake) / valid_hours),
        })
    frame = pd.DataFrame(rows)
    counts = frame.groupby("group").size().reindex(GROUPS).astype(int).to_dict()
    if counts != {group: 7 for group in GROUPS}:
        raise RuntimeError(f"Unexpected cohort: {counts}")
    for group in GROUPS:
        if frame.loc[frame.group == group, "animal"].sort_values().tolist() != ANIMALS:
            raise RuntimeError(f"Animal mismatch in {group}")
    return frame


def holm(values: list[float]) -> list[float]:
    values_array = np.asarray(values, float)
    order = np.argsort(values_array)
    ranked = values_array[order]
    adjusted_ranked = np.maximum.accumulate(ranked * (len(ranked) - np.arange(len(ranked))))
    adjusted = np.empty_like(values_array)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()


def star(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def exact_interaction(frame: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    wide = (
        frame.assign(cohort=frame.group.str.split("+").str[0], treatment=frame.group.str.split("+").str[1])
        .pivot(index=["cohort", "animal"], columns="treatment", values=metric)
        .reset_index()
    )
    wide["change"] = wide["CNO"] - wide["SAL"]
    baseline = wide.loc[wide.cohort == "baseline", "change"].to_numpy(float)
    pocd = wide.loc[wide.cohort == "POCD", "change"].to_numpy(float)
    observed = float(pocd.mean() - baseline.mean())
    pooled = np.r_[baseline, pocd]
    permutations = []
    for selected in itertools.combinations(range(len(pooled)), len(pocd)):
        mask = np.zeros(len(pooled), bool)
        mask[list(selected)] = True
        permutations.append(float(pooled[mask].mean() - pooled[~mask].mean()))
    p_value = float(np.mean(np.abs(permutations) >= abs(observed) - np.finfo(float).eps * 10))
    return observed, p_value, len(permutations)


def calculate_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair_indices = {"baseline": [], "POCD": []}
    for spec in SPECS:
        metric = spec["metric"]
        for cohort in ("baseline", "POCD"):
            wide = frame[frame.group.str.startswith(cohort + "+")].pivot(index="animal", columns="group", values=metric).sort_index()
            sal = wide[f"{cohort}+SAL"].to_numpy(float)
            cno = wide[f"{cohort}+CNO"].to_numpy(float)
            test = wilcoxon(sal, cno, alternative="two-sided", method="exact")
            index = len(rows)
            rows.append({
                "metric": metric,
                "analysis_role": "within_cohort_pairwise",
                "comparison": f"{cohort} SAL vs CNO" if cohort == "baseline" else "POCD SAL vs CNO",
                "group_a": f"{cohort}+SAL", "group_b": f"{cohort}+CNO",
                "design": "paired", "test": "exact two-sided paired Wilcoxon",
                "n_a": 7, "n_b": 7, "statistic": float(test.statistic),
                "effect": float(np.mean(cno - sal)), "p_raw": float(test.pvalue),
            })
            pair_indices[cohort].append(index)

    for cohort, indices in pair_indices.items():
        adjusted = holm([rows[index]["p_raw"] for index in indices])
        for index, p_value in zip(indices, adjusted):
            rows[index]["p_holm_across_four_bout_endpoints"] = p_value
            rows[index]["stars"] = star(p_value)

    model_indices = []
    for spec in SPECS:
        metric = spec["metric"]
        baseline = frame.loc[frame.group == "baseline+SAL", metric].to_numpy(float)
        pocd = frame.loc[frame.group == "POCD+SAL", metric].to_numpy(float)
        test = mannwhitneyu(baseline, pocd, alternative="two-sided", method="exact")
        model_indices.append(len(rows))
        rows.append({
            "metric": metric, "analysis_role": "model_effect_at_sal",
            "comparison": "baseline SAL vs POCD SAL", "group_a": "baseline+SAL", "group_b": "POCD+SAL",
            "design": "independent cohorts", "test": "exact two-sided Mann-Whitney U",
            "n_a": 7, "n_b": 7, "statistic": float(test.statistic),
            "effect": float(pocd.mean() - baseline.mean()), "p_raw": float(test.pvalue),
        })
    adjusted = holm([rows[index]["p_raw"] for index in model_indices])
    for index, p_value in zip(model_indices, adjusted):
        rows[index]["p_holm_across_four_bout_endpoints"] = p_value
        rows[index]["stars"] = star(p_value)

    for spec in SPECS:
        effect, p_value, permutations = exact_interaction(frame, spec["metric"])
        rows.append({
            "metric": spec["metric"], "analysis_role": "overall_interaction",
            "comparison": "cohort by treatment interaction",
            "group_a": "baseline CNO-SAL change", "group_b": "POCD CNO-SAL change",
            "design": "2x2 mixed design", "test": f"exact two-sided difference-in-differences permutation ({permutations} allocations)",
            "n_a": 7, "n_b": 7, "statistic": effect, "effect": effect,
            "p_raw": p_value, "p_holm_across_four_bout_endpoints": np.nan, "stars": star(p_value),
        })
    return pd.DataFrame(rows)


def archive_current(root: Path, stamp: str) -> None:
    archive = root / "archive" / stamp / "rem_wake_bout_metrics"
    for spec in SPECS:
        for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
            source = root / "panels" / f"{spec['stem']}{extension}"
            if source.exists():
                destination = archive / "panels" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    for relative in (Path("data") / DATA_DIR_NAME, Path("editable") / PROJECT_NAME):
        source = root / relative
        if source.exists():
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)


def update_manifest() -> None:
    path = FORMAL / "documentation" / "outputs_manifest.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    stems = {spec["stem"] for spec in SPECS}
    rows = [row for row in rows if not any(stem in row.get("path", "") for stem in stems)]
    for spec in SPECS:
        stem = spec["stem"]
        description = f"Current {spec['title']} {spec['ylabel']} panel; mean +/- SD"
        for kind, extension in (("preview_png", ".png"), ("panel_pdf", ".pdf"), ("panel_svg", ".svg")):
            rows.append({"type": kind, "path": str(FORMAL / "panels" / f"{stem}{extension}"), "description": description, "status": "current"})
        rows.append({"type": "source_data", "path": str(FORMAL / "data" / DATA_DIR_NAME / "animal_level_rem_wake_bouts.csv"), "description": "Latest animal-level REM/WAKE bout metrics", "status": "current"})
        rows.append({"type": "stats", "path": str(FORMAL / "data" / DATA_DIR_NAME / "rem_wake_bout_statistics.csv"), "description": "Paired tests, model-effect tests and interaction tests", "status": "current"})
    rows.append({"type": "origin_project", "path": str(FORMAL / "editable" / PROJECT_NAME), "description": "Editable Origin project for four REM/WAKE bout panels", "status": "current"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("type", "path", "description", "status"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = pd.read_csv(SOURCE)
    frame = analyze(source)
    statistics = calculate_statistics(frame)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for root in (WORK, FORMAL):
        archive_current(root, stamp)

    data_dir = WORK / "data" / DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "animal_level_rem_wake_bouts.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(data_dir / "rem_wake_bout_statistics.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for spec in SPECS:
        for group in GROUPS:
            values = frame.loc[frame.group == group, spec["metric"]].to_numpy(float)
            summary_rows.append({"metric": spec["metric"], "group": group, "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1), "sem": values.std(ddof=1) / np.sqrt(len(values))})
    pd.DataFrame(summary_rows).to_csv(data_dir / "rem_wake_bout_group_summary.csv", index=False, encoding="utf-8-sig")

    stage_dir = WORK / "panels" / "rem_wake_bouts_latest"
    stage_dir.mkdir(parents=True, exist_ok=True)
    arch.mean_sem = lambda values: (float(np.mean(values)), float(np.std(values, ddof=1)))
    graphs = []
    op.attach()
    try:
        for spec in SPECS:
            metric_stats = statistics[statistics.metric == spec["metric"]].copy()
            graph = arch.build_graph(frame, pd.DataFrame(), metric_stats, spec)
            baseline = metric_stats[metric_stats.comparison == "baseline SAL vs CNO"]
            if not baseline.empty and isinstance(baseline.iloc[0].stars, str) and baseline.iloc[0].stars:
                top = max(frame.loc[frame.group == "baseline+SAL", spec["metric"]].max(), frame.loc[frame.group == "baseline+CNO", spec["metric"]].max())
                arch.add_bracket(graph[0], arch.XPOS[0], arch.XPOS[1], top + spec["ymax"] * 0.055, spec["ymax"] * 0.02, baseline.iloc[0].stars)
            model = metric_stats[metric_stats.comparison == "baseline SAL vs POCD SAL"]
            if not model.empty:
                model_stars = str(model.iloc[0].stars).strip()
                if model_stars and model_stars.lower() != "nan":
                    top = max(
                        frame.loc[frame.group == "baseline+SAL", spec["metric"]].max(),
                        frame.loc[frame.group == "POCD+SAL", spec["metric"]].max(),
                    )
                    arch.add_bracket(
                        graph[0], arch.XPOS[0], arch.XPOS[2],
                        top + spec["ymax"] * 0.15, spec["ymax"] * 0.02, model_stars,
                    )
            arch.export_graph(graph, stage_dir, spec["stem"])
            graphs.append(graph)
        project = WORK / "editable" / PROJECT_NAME
        graphs[-1].activate()
        op.save(str(project))
    finally:
        op.detach()

    global_provenance = Path(r"F:\Sleep\outputs\figure_provenance")
    global_provenance.mkdir(parents=True, exist_ok=True)
    for root in (WORK, FORMAL):
        for spec in SPECS:
            for extension in (".png", ".pdf", ".svg", "_AI_clean.pdf", "_AI_clean.svg"):
                destination = root / "panels" / f"{spec['stem']}{extension}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stage_dir / f"{spec['stem']}{extension}", destination)
        target_data = root / "data" / DATA_DIR_NAME
        if target_data != data_dir:
            shutil.copytree(data_dir, target_data, dirs_exist_ok=True)
        target_project = root / "editable" / PROJECT_NAME
        target_project.parent.mkdir(parents=True, exist_ok=True)
        if target_project != project:
            shutil.copy2(project, target_project)
        scripts = root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), scripts / "redraw_fig9_rem_wake_bouts.py")

        for spec in SPECS:
            metric_stats = statistics[statistics.metric == spec["metric"]]
            note = f"""# {spec['stem']}

- generated_at: {stamp}
- status: formal Fig9 single panel
- source: current manually reviewed scores.db3 inventory recorded in data/{DATA_DIR_NAME}/animal_level_rem_wake_bouts.csv
- sample_definition: baseline and POCD are independent cohorts; SAL and CNO are repeated within animal; NO1-NO7 per cell; one mouse is one biological replicate
- bout_definition: maximal consecutive epochs of the same state; each epoch is 5 s; score-0 epochs break bouts
- duration: arithmetic mean bout duration in seconds within animal
- frequency: bout count divided by non-artifact recording hours
- display: arithmetic mean +/- sample SD; hollow animal points; same-animal SAL-CNO trajectories; no animal IDs; no n.s. labels
- inference: exact paired Wilcoxon within cohort, exact Mann-Whitney U for baseline+SAL versus POCD+SAL, and exact difference-in-differences permutation; Holm adjustment across the four bout endpoints for each pairwise/model-effect family
- editable: editable/{PROJECT_NAME}
- output: panels/{spec['stem']}.*
- script: scripts/redraw_fig9_rem_wake_bouts.py

{metric_stats.to_csv(index=False)}
"""
            docs = root / "documentation"
            docs.mkdir(parents=True, exist_ok=True)
            (docs / f"{spec['stem']}_provenance.md").write_text(note, encoding="utf-8")
            (global_provenance / f"{spec['stem']}_provenance.md").write_text(note, encoding="utf-8")
            metric_stats.to_csv(global_provenance / f"{spec['stem']}_stats.csv", index=False, encoding="utf-8-sig")
            frame[["group", "animal", spec["metric"]]].to_csv(global_provenance / f"{spec['stem']}_source_data.csv", index=False, encoding="utf-8-sig")

    update_manifest()
    print(statistics.to_string(index=False))
    print(FORMAL / "panels")


if __name__ == "__main__":
    main()
