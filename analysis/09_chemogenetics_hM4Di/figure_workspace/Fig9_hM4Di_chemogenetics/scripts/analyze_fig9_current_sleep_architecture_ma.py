from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"F:\1.Sleep\eXdata\hm4Di")
OUT = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig9_hM4Di_chemogenetics\data")
GROUPS = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
EXPECTED = {group: 7 for group in GROUPS}
METRICS = ["nrem_percent_of_valid", "rem_percent_of_valid", "wake_percent_of_valid", "ma_per_nrem_h"]
COMPARISONS = [
    ("baseline SAL vs CNO", "baseline+SAL", "baseline+CNO"),
    ("POCD SAL vs CNO", "POCD+SAL", "POCD+CNO"),
]


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_scores(path: Path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "select start_time_seconds, score from sleep_scores_table "
            "order by start_time_seconds, start_time_sub_seconds"
        ).fetchall()
    arr = np.asarray(rows)
    if len(arr) < 2 or not np.allclose(np.diff(arr[:, 0]), 5.0):
        raise RuntimeError(f"Non-consecutive 5-s scores: {path}")
    return arr[:, 1].astype(np.int16)


def count_mas(scores: np.ndarray) -> int:
    edges = np.r_[0, np.flatnonzero(np.diff(scores) != 0) + 1, len(scores)]
    count = 0
    for a, b in zip(edges[:-1], edges[1:]):
        state = int(scores[a])
        if state == 1 and 0 < a and b < len(scores) and (b - a) <= 3:
            if scores[a - 1] == 2 and scores[b] == 2:
                count += 1
    return count


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in GROUPS:
        for folder in sorted((p for p in (ROOT / group).iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
            match = re.fullmatch(r"N?O?(\d+)", folder.name, flags=re.IGNORECASE)
            if not match or int(match.group(1)) not in range(1, 8):
                continue
            animal = f"NO{int(match.group(1))}"
            db = folder / "scores.db3"
            if not db.is_file():
                continue
            scores = read_scores(db)
            wake, nrem, rem = [int(np.sum(scores == code)) for code in (1, 2, 3)]
            valid = wake + nrem + rem
            ma_count = count_mas(scores)
            nrem_h = nrem * 5.0 / 3600.0
            rows.append({
                "group": group, "animal": animal, "source_folder_name": folder.name, "scores_db_path": str(db),
                "scores_db_sha256": sha256(db), "epoch_seconds": 5.0,
                "total_epochs": len(scores), "artifact_epochs": int(np.sum(scores == 0)),
                "valid_epochs": valid, "wake_epochs": wake, "nrem_epochs": nrem, "rem_epochs": rem,
                "wake_percent_of_valid": 100 * wake / valid,
                "nrem_percent_of_valid": 100 * nrem / valid,
                "rem_percent_of_valid": 100 * rem / valid,
                "ma_count": ma_count, "ma_per_nrem_h": ma_count / nrem_h,
            })
    frame = pd.DataFrame(rows)
    counts = frame.groupby("group").size().to_dict()
    if counts != EXPECTED:
        raise RuntimeError(f"Unexpected current cohort: {counts}")
    if not np.allclose(frame[["wake_percent_of_valid", "nrem_percent_of_valid", "rem_percent_of_valid"]].sum(axis=1), 100):
        raise RuntimeError("State percentages do not sum to 100")

    summary_rows, stat_rows = [], []
    for metric in METRICS:
        for group in GROUPS:
            values = frame.loc[frame.group == group, metric].to_numpy(float)
            summary_rows.append({"metric": metric, "group": group, "n": len(values),
                                 "mean": values.mean(), "sd": values.std(ddof=1),
                                 "sem": values.std(ddof=1) / np.sqrt(len(values)),
                                 "median": np.median(values)})
        for comparison, ga, gb in COMPARISONS:
            a = frame.loc[frame.group == ga].set_index("animal")[metric].sort_index()
            b = frame.loc[frame.group == gb].set_index("animal")[metric].sort_index()
            if a.index.tolist() != b.index.tolist() or a.index.tolist() != [f"NO{i}" for i in range(1, 8)]:
                raise RuntimeError(f"Pairing mismatch for {ga}/{gb}: {a.index.tolist()} vs {b.index.tolist()}")
            test = wilcoxon(a.to_numpy(float), b.to_numpy(float), alternative="two-sided", method="exact")
            stat_rows.append({"metric": metric, "comparison": comparison, "group_a": ga, "group_b": gb,
                              "n_pairs": len(a), "paired_ids": "|".join(a.index),
                              "test": "exact two-sided paired Wilcoxon", "W": float(test.statistic),
                              "p_raw": float(test.pvalue)})
    stats = pd.DataFrame(stat_rows)
    stats["p_holm_across_4_endpoints"] = np.nan
    for comparison, _, _ in COMPARISONS:
        idx = stats.index[stats.comparison == comparison]
        stats.loc[idx, "p_holm_across_4_endpoints"] = holm(stats.loc[idx, "p_raw"].to_numpy())
    # Author decision: MAs is a prespecified standalone primary endpoint and is
    # therefore displayed using its exact paired Wilcoxon p value. The older
    # four-endpoint Holm value is retained as a transparent sensitivity column.
    is_ma = stats["metric"].eq("ma_per_nrem_h")
    stats["endpoint_role"] = np.where(
        is_ma, "prespecified standalone primary endpoint", "secondary sleep-stage endpoint"
    )
    stats["p_for_display"] = np.where(is_ma, stats["p_raw"], stats["p_holm_across_4_endpoints"])
    stats["multiplicity_note"] = np.where(
        is_ma,
        "No cross-endpoint adjustment: MAs was prespecified as a standalone primary endpoint.",
        "Conservative Holm sensitivity value across the four historical panel endpoints.",
    )
    p = stats["p_for_display"]
    stats["stars"] = np.where(p < .001, "***", np.where(p < .01, "**", np.where(p < .05, "*", "")))

    frame.to_csv(OUT / "sleep_architecture_ma_current_source_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(OUT / "sleep_architecture_ma_current_group_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(OUT / "sleep_architecture_ma_current_statistics.csv", index=False, encoding="utf-8-sig")
    print(frame.groupby("group").size().to_string())
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
