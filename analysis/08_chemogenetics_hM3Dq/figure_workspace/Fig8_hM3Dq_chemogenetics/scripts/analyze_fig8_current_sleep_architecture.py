from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(r"F:\1.Sleep\eXdata\hm3Dq\CNO")
PACKAGE = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig8_hM3Dq_chemogenetics")
OUT = PACKAGE / "data" / "Sleep_architecture_current"
GROUPS = ["mCherry+SAL", "mCherry+CNO", "hmdq3+SAL", "hmdq3+CNO"]
XPOS = [0.72, 1.28, 2.72, 3.28]
EXPECTED = {group: 6 for group in GROUPS}
SPECS = [
    ("A_NREM", "nrem_percent", "NREM (%)", 80.0, 20.0),
    ("B_REM", "rem_percent", "REM (%)", 8.0, 2.0),
    ("C_Wake", "wake_percent", "Wake (%)", 100.0, 20.0),
]


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
    if len(values) < 2 or not np.allclose(np.diff(values[:, 0]), 5.0):
        raise RuntimeError(f"Non-consecutive 5-s scores: {path}")
    scores = values[:, 1].astype(np.int16)
    if not set(np.unique(scores)).issubset({0, 1, 2, 3}):
        raise RuntimeError(f"Unexpected score code in {path}: {np.unique(scores)}")
    return scores


def suffix_number(text: str) -> int:
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(text)
    return int(match.group(1))


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def exact_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = wilcoxon(b, a, alternative="two-sided", method="exact")
    return float(result.statistic), float(result.pvalue)


def point_x(values: np.ndarray, center: float) -> np.ndarray:
    x = np.full(len(values), center, dtype=float)
    rounded = np.round(values, 4)
    for value in np.unique(rounded):
        indices = np.flatnonzero(rounded == value)
        if len(indices) > 1:
            x[indices] += (np.arange(len(indices)) - (len(indices) - 1) / 2) * 0.05
    return x


def make_origin_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = []
    for group in GROUPS:
        subset = frame.loc[frame.group == group].sort_values("pair_id")
        grouped.append(subset)
    output: dict[str, pd.Series] = {}
    for i, subset in enumerate(grouped):
        output[f"Raw_{i}"] = pd.Series(subset[metric].to_numpy(float))
    output["Bar_X"] = pd.Series(XPOS)
    output["Mean"] = pd.Series([subset[metric].mean() for subset in grouped])
    output["SEM"] = pd.Series([subset[metric].std(ddof=1) / np.sqrt(len(subset)) for subset in grouped])
    for i, (center, subset) in enumerate(zip(XPOS, grouped)):
        values = subset[metric].to_numpy(float)
        output[f"PointX_{i}"] = pd.Series(point_x(values, center))
        output[f"PointY_{i}"] = pd.Series(values)

    pairs = []
    for virus, left_group, right_group, left_x, right_x in (
        ("mCherry", "mCherry+SAL", "mCherry+CNO", XPOS[0], XPOS[1]),
        ("hM3Dq", "hmdq3+SAL", "hmdq3+CNO", XPOS[2], XPOS[3]),
    ):
        left = frame.loc[frame.group == left_group].set_index("pair_id")
        right = frame.loc[frame.group == right_group].set_index("pair_id")
        for pair_id in sorted(set(left.index) & set(right.index)):
            pairs.append(([left_x, right_x], [left.loc[pair_id, metric], right.loc[pair_id, metric]]))
    if len(pairs) != 12:
        raise RuntimeError(f"Expected 12 pair lines, found {len(pairs)}")
    for j, (xs, ys) in enumerate(pairs):
        output[f"PairX_{j}"] = pd.Series(xs)
        output[f"PairY_{j}"] = pd.Series(ys)
    for i, subset in enumerate(grouped):
        output[f"ID_{i}"] = pd.Series(subset.plot_id.tolist())
    return pd.DataFrame(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in GROUPS:
        folders = sorted((p for p in (ROOT / group).iterdir() if p.is_dir()), key=lambda p: suffix_number(p.name))
        for folder in folders:
            db = folder / "scores.db3"
            if not db.is_file():
                continue
            scores = read_scores(db)
            artifact = int(np.sum(scores == 0))
            wake_scored = int(np.sum(scores == 1))
            nrem = int(np.sum(scores == 2))
            rem = int(np.sum(scores == 3))
            total = len(scores)
            wake_with_artifact = wake_scored + artifact
            pair_id = suffix_number(folder.name)
            display_id = folder.name.replace("_", "-")
            rows.append(
                {
                    "group": group,
                    "virus": "mCherry" if group.startswith("mCherry") else "hM3Dq",
                    "drug": "SAL" if group.endswith("SAL") else "CNO",
                    "animal": folder.name,
                    "plot_id": display_id,
                    "pair_id": pair_id,
                    "scores_db_path": str(db),
                    "scores_db_sha256": sha256(db),
                    "epoch_seconds": 5.0,
                    "total_epochs": total,
                    "artifact_epochs_as_wake": artifact,
                    "scored_wake_epochs": wake_scored,
                    "wake_epochs_including_artifact": wake_with_artifact,
                    "nrem_epochs": nrem,
                    "rem_epochs": rem,
                    "nrem_percent": 100.0 * nrem / total,
                    "rem_percent": 100.0 * rem / total,
                    "wake_percent": 100.0 * wake_with_artifact / total,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.groupby("group").size().to_dict() != EXPECTED:
        raise RuntimeError(frame.groupby("group").size().to_dict())
    if not np.allclose(frame[["nrem_percent", "rem_percent", "wake_percent"]].sum(axis=1), 100.0):
        raise RuntimeError("State percentages do not sum to 100")

    summary_rows = []
    stat_rows = []
    for stem, metric, _, _, _ in SPECS:
        for group in GROUPS:
            values = frame.loc[frame.group == group, metric].to_numpy(float)
            summary_rows.append(
                {
                    "panel": stem,
                    "metric": metric,
                    "group": group,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "sem": values.std(ddof=1) / np.sqrt(len(values)),
                    "median": np.median(values),
                }
            )
        for virus, sal_group, cno_group in (
            ("mCherry", "mCherry+SAL", "mCherry+CNO"),
            ("hM3Dq", "hmdq3+SAL", "hmdq3+CNO"),
        ):
            sal = frame.loc[frame.group == sal_group].sort_values("pair_id")
            cno = frame.loc[frame.group == cno_group].sort_values("pair_id")
            if sal.pair_id.tolist() != cno.pair_id.tolist():
                raise RuntimeError(f"Pair mismatch: {virus}")
            statistic, p_raw = exact_paired(sal[metric].to_numpy(float), cno[metric].to_numpy(float))
            stat_rows.append(
                {
                    "panel": stem,
                    "metric": metric,
                    "virus": virus,
                    "n_pairs": len(sal),
                    "sal_mean": sal[metric].mean(),
                    "cno_mean": cno[metric].mean(),
                    "difference_cno_minus_sal": cno[metric].mean() - sal[metric].mean(),
                    "wilcoxon_w": statistic,
                    "p_raw": p_raw,
                    "pairing_note": "confirmed same IDs" if virus == "mCherry" else "numeric-suffix filename match; verify identity",
                }
            )
    stats = pd.DataFrame(stat_rows)
    stats["p_holm_across_3_states"] = np.nan
    for virus in ("mCherry", "hM3Dq"):
        indices = stats.index[stats.virus == virus]
        stats.loc[indices, "p_holm_across_3_states"] = holm(stats.loc[indices, "p_raw"].tolist())
    stats["stars_raw"] = np.where(stats.p_raw < 0.001, "***", np.where(stats.p_raw < 0.01, "**", np.where(stats.p_raw < 0.05, "*", "")))
    stats["stars_holm"] = np.where(stats.p_holm_across_3_states < 0.001, "***", np.where(stats.p_holm_across_3_states < 0.01, "**", np.where(stats.p_holm_across_3_states < 0.05, "*", "")))

    frame.to_csv(OUT / "sleep_architecture_current_source_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(OUT / "sleep_architecture_current_group_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(OUT / "sleep_architecture_current_statistics.csv", index=False, encoding="utf-8-sig")

    config = []
    origin_dir = OUT / "origin_ready"
    origin_dir.mkdir(exist_ok=True)
    for stem, metric, ylabel, ymax, step in SPECS:
        table = make_origin_table(frame, metric)
        table.to_csv(origin_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        hmdq = stats.loc[(stats.panel == stem) & (stats.virus == "hM3Dq")].iloc[0]
        brackets = []
        if hmdq.stars_raw:
            brackets.append(
                {
                    "a": 2,
                    "b": 3,
                    "p": float(hmdq.p_raw),
                    "p_holm_across_3_states": float(hmdq.p_holm_across_3_states),
                    "comparison": "hM3Dq SAL vs CNO numeric-suffix filename-matched paired sensitivity",
                    "stars": hmdq.stars_raw,
                }
            )
        config.append(
            {
                "stem": stem,
                "key": metric,
                "ylabel": ylabel,
                "ymax": ymax,
                "step": step,
                "brackets": brackets,
                "csv": str(origin_dir / f"{stem}.csv"),
            }
        )
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(frame.groupby("group").size().to_string())
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
