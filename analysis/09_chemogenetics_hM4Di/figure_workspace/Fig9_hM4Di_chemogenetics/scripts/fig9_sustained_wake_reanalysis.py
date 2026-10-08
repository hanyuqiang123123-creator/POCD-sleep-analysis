"""Stage Fig9 sustained-WAKE animal data and revised four-endpoint inference.

Uses the existing, hash-checked Fig9 24-hour scoring inventory. This script does
not modify any raw scores or formal figure files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"F:\1.Sleep\PHD稿件")
WORK = ROOT / "Figure_Workspace" / "Fig9_hM4Di_chemogenetics"
sys.path.insert(0, str(WORK / "scripts"))
import redraw_fig9_rem_wake_bouts as old  # noqa: E402


OUT = Path(r"C:\Users\USER\Documents\ChatGPT\PHD文章\Fig9_sustained_WAKE_15s_candidate")
SUSTAINED_WAKE_MIN_SECONDS_EXCLUSIVE = 15.0


def main() -> None:
    source = pd.read_csv(old.SOURCE)
    frame = old.analyze(source)
    audit = []
    for index, record in frame.iterrows():
        scores = old.read_scores(Path(record.scores_db_path))
        all_wake = old.state_bouts(scores, 1)
        sustained = all_wake[all_wake > SUSTAINED_WAKE_MIN_SECONDS_EXCLUSIVE]
        if len(sustained) == 0:
            raise RuntimeError(f"No sustained WAKE bout for {record.group}/{record.animal}")
        frame.loc[index, "wake_bout_count_all"] = len(all_wake)
        frame.loc[index, "wake_bout_count_sustained"] = len(sustained)
        frame.loc[index, "wake_bout_count_excluded_5_15s"] = len(all_wake) - len(sustained)
        frame.loc[index, "wake_bout_duration_s"] = float(sustained.mean())
        frame.loc[index, "wake_bout_frequency_h"] = float(len(sustained) / record.valid_hours)
        audit.append({
            "group": record.group,
            "animal": record.animal,
            "all_wake_bouts": len(all_wake),
            "sustained_wake_bouts_gt15s": len(sustained),
            "excluded_5_15s_wake_bouts": len(all_wake) - len(sustained),
            "sustained_wake_duration_mean_s": float(sustained.mean()),
            "sustained_wake_frequency_per_valid_h": float(len(sustained) / record.valid_hours),
            "valid_scored_hours": float(record.valid_hours),
        })
    stats = old.calculate_statistics(frame)
    summary = (frame.melt(id_vars=["group", "animal"],
                          value_vars=[spec["metric"] for spec in old.SPECS],
                          var_name="metric", value_name="value")
               .groupby(["metric", "group"], sort=False).value
               .agg(n="size", mean="mean", sd=lambda x: x.std(ddof=1),
                    sem=lambda x: x.sem(ddof=1)).reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "animal_level_rem_sustained_wake_bouts.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(OUT / "rem_sustained_wake_bout_statistics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "rem_sustained_wake_group_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(OUT / "sustained_wake_inclusion_audit.csv", index=False, encoding="utf-8-sig")
    print(summary[summary.metric.str.startswith("wake_")].to_string(index=False))
    print(stats[stats.metric.str.startswith("wake_")].to_string(index=False))
    print(f"STAGED {OUT}")


if __name__ == "__main__":
    main()
