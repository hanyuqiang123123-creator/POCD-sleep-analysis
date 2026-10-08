"""Recalculate Fig8 score-derived metrics with MA <=15 s / WAKE bout >15 s.

The approved 17,220-epoch window, input inventory, spectra and 6-h tables are
unchanged. One current DB has a single subsequently edited 5-s epoch; use its
verified prior backup to freeze the formal Fig8 score vector for this rerun.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"F:\1.Sleep\PHD稿件")
WORK = ROOT / "Figure_Workspace" / "Fig8_hM3Dq_chemogenetics"
OUT = Path(r"C:\Users\USER\Documents\ChatGPT\PHD文章\Fig8_sustained_WAKE_15s_candidate")
sys.path.insert(0, str(WORK / "scripts"))
import redraw_fig8_requested_16panel_set as old  # noqa: E402

FROZEN_NO2 = Path(
    r"F:\1.Sleep\eXdata\hm3Dq\CNO\hmdq3+SAL\NO2\scores_backups"
    r"\scores_before_SleepWorkbench_20260927_001719_608342.db3"
)


def inventory_from_prior() -> pd.DataFrame:
    inventory = pd.read_csv(old.SOURCE, sep="\t")
    inventory = inventory[inventory.group.isin(old.GROUPS)].copy()
    for idx, row in inventory.iterrows():
        db, edf = Path(str(row.scores_db3)), Path(str(row.edf_path))
        if (not db.exists() or not edf.exists()) and row.group == "hmdq3+CNO":
            current = Path(r"F:\1.Sleep\eXdata\hm3Dq\CNO\hmdq3+CNO") / str(row.subject_for_sensitivity)
            current_db, current_edf = current / "scores.db3", current / "export_SO200Hz.edf"
            if current_db.exists() and current_edf.exists():
                inventory.at[idx, "animal"] = str(row.subject_for_sensitivity)
                inventory.at[idx, "scores_db3"] = str(current_db)
                inventory.at[idx, "edf_path"] = str(current_edf)
    if inventory.groupby("group").size().to_dict() != {group: 6 for group in old.GROUPS}:
        raise RuntimeError("Fig8 group membership changed")
    return inventory


def main() -> None:
    inventory = inventory_from_prior()
    old_audit = pd.read_csv(WORK / "data" / "Fig8_requested_score_input_audit.csv")
    no2 = (inventory.group == "hmdq3+SAL") & (inventory.subject_for_sensitivity == "NO2")
    if no2.sum() != 1 or not FROZEN_NO2.exists():
        raise RuntimeError("Frozen NO2 score source missing or ambiguous")
    current_no2 = Path(inventory.loc[no2, "scores_db3"].iloc[0])
    _, current_scores = old.read_scores(current_no2)
    _, frozen_scores = old.read_scores(FROZEN_NO2)
    differences = np.flatnonzero(current_scores[:old.ANALYSIS_EPOCHS] != frozen_scores[:old.ANALYSIS_EPOCHS])
    if differences.tolist() != [1] or frozen_scores[1] != 0 or current_scores[1] != 1:
        raise RuntimeError("NO2 edit no longer matches audited single-epoch 0-to-1 change")
    inventory.loc[no2, "scores_db3"] = str(FROZEN_NO2)
    for record in inventory.itertuples(index=False):
        path = Path(record.scores_db3)
        expected = old_audit[(old_audit.group == record.group) &
                             (old_audit.subject == record.subject_for_sensitivity)]
        is_frozen_no2 = record.group == "hmdq3+SAL" and record.subject_for_sensitivity == "NO2"
        if len(expected) != 1 or (not is_frozen_no2 and old.sha256(path) != expected.iloc[0].sha256):
            raise RuntimeError(f"Score source changed since formal Fig8 redraw: {path}")
    old.MA_MAX_SECONDS = 15
    old.SUSTAINED_WAKE_MIN_SECONDS_EXCLUSIVE = 15
    full, time, audit = old.prepare_score_metrics(inventory)
    full["ma_count_5_to_15s_nrem_flanked"] = full.pop("ma_count_5_to_20s_nrem_flanked")
    full["sustained_wake_definition_seconds"] = ">15"
    full["ma_definition_seconds"] = "5-15, NREM-flanked"
    prior = pd.read_csv(WORK / "data" / "Fig8_requested_full_record_metrics.csv")
    key = ["group", "subject"]
    before = prior.set_index(key).sort_index()
    after = full.set_index(key).sort_index()
    if before.index.tolist() != after.index.tolist():
        raise RuntimeError("Fig8 animal identities or pairings changed")
    changed = {"wake_mean_bout_seconds", "wake_bout_frequency_per_valid_h",
               "ma_count_5_to_20s_nrem_flanked", "ma_frequency_per_nrem_h",
               "ma_mean_duration_seconds", "brief_wake_not_nrem_flanked_count",
               "sustained_wake_definition_seconds", "ma_definition_seconds", "scores_db3"}
    common = [column for column in before.columns if column in after.columns and column not in changed]
    for column in common:
        a, b = before[column], after[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-11, atol=1e-11):
                raise RuntimeError(f"Unrelated Fig8 metric changed: {column}")
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            raise RuntimeError(f"Unrelated Fig8 field changed: {column}")
    old_time = pd.read_csv(WORK / "data" / "Fig8_requested_6h_timecourse_metrics.csv")
    time_keys = ["group", "subject", "state", "bin_index"]
    time = time.sort_values(time_keys).reset_index(drop=True)
    old_time = old_time.sort_values(time_keys).reset_index(drop=True)
    if not time.equals(old_time):
        if time.shape != old_time.shape or time.columns.tolist() != old_time.columns.tolist():
            raise RuntimeError(f"Fig8 time-course schema changed: {time.shape} vs {old_time.shape}")
        for column in time.columns:
            a, b = time[column], old_time[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True,
                                   rtol=1e-11, atol=1e-11)
            else:
                same = a.fillna("").astype(str).equals(b.fillna("").astype(str))
            if not same:
                raise RuntimeError(f"Fig8 time-course field changed: {column}")
    bands = pd.read_csv(WORK / "data" / "Fig8_requested_band_power_metrics.csv")
    stats = old.build_stats(full, time, bands)
    stats.loc[stats.state == "WAKE", "metric_definition"] = (
        "sustained WAKE runs >15 s; 5-15 s NREM-flanked events excluded as MA"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT / "Fig8_requested_full_record_metrics.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(OUT / "Fig8_requested_16panel_statistics.csv", index=False, encoding="utf-8-sig")
    full[["group", "virus", "drug", "subject", "scores_db3", "valid_hours",
          "ma_count_5_to_15s_nrem_flanked", "ma_frequency_per_nrem_h",
          "ma_mean_duration_seconds", "brief_wake_not_nrem_flanked_count",
          "sustained_wake_definition_seconds", "ma_definition_seconds"]].to_csv(
              OUT / "Fig8_requested_microarousal_classification_metrics.csv", index=False,
              encoding="utf-8-sig")
    audit.to_csv(OUT / "Fig8_requested_score_input_audit.csv", index=False, encoding="utf-8-sig")
    print("ANIMALS", len(full), "HASHES_MATCH", len(audit), flush=True)
    print(stats[(stats.family == "bout") & (stats.state == "WAKE")][
        ["panel", "virus", "mean_sal", "mean_cno", "p_raw", "p_adjusted", "display_annotation"]
    ].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
