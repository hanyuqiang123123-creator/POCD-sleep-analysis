from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5")
SOURCE = ROOT / "Anal" / "Gcamp_FQ" / "state_level_reanalysis" / "data" / "Gcamp_state_record_means_all8.csv"
SUR_SOURCE = ROOT / "data" / "sur_expansion_qc" / "SUR_confirmed_state_means_timestamp_aligned.csv"
DATA = ROOT / "data"
STATES = ["Wake", "NREM", "REM", "MA"]
EXCLUSIONS = {
    "20250603Trail5": (
        "Fiber placement did not lie directly above the viral-expression region "
        "(user-confirmed technical QC exclusion, 2026-09-05)"
    )
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (len(p) - rank)))
        out[idx] = running
    return out.tolist()


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    delta = a - b
    delta = delta[delta != 0]
    ranks = stats.rankdata(np.abs(delta))
    pos = ranks[delta > 0].sum()
    neg = ranks[delta < 0].sum()
    return float((pos - neg) / (pos + neg))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    base_all = pd.read_csv(SOURCE)
    if list(base_all.columns) != ["record_id", *STATES] or len(base_all) != 8:
        raise ValueError("Expected the audited 8-record Wake/NREM/REM/MA table")
    if base_all[STATES].isna().any().any():
        raise ValueError("Source table contains missing state values")

    sur_long = pd.read_csv(SUR_SOURCE)
    if sur_long["record_id"].nunique() != 6 or not sur_long["status"].str.contains("user-confirmed").all():
        raise ValueError("Expected six user-confirmed independent SUR records and offsets")
    sur_wide = (
        sur_long.pivot(index="record_id", columns="state", values="mean_intensity")
        .reindex(columns=STATES)
        .reset_index()
    )
    raw_all = pd.concat([base_all, sur_wide], ignore_index=True)
    raw_all.insert(1, "source_set", ["legacy_audited"] * len(base_all) + ["confirmed_SUR_expansion"] * len(sur_wide))

    unknown = set(EXCLUSIONS) - set(raw_all["record_id"])
    if unknown:
        raise ValueError(f"Exclusion IDs absent from audited source: {sorted(unknown)}")
    four_state_ineligible = set(raw_all.loc[raw_all[STATES].isna().any(axis=1), "record_id"])
    if four_state_ineligible != {"SUR_gcamp20250513_20250513_2"}:
        raise ValueError(f"Unexpected incomplete four-state records: {sorted(four_state_ineligible)}")
    exclusion_log = raw_all.copy()
    exclusion_log.insert(
        2,
        "analysis_status",
        exclusion_log["record_id"].map(
            lambda value: (
                "excluded_technical_QC"
                if value in EXCLUSIONS
                else "retained_for_NREM_MA_but_not_four_state"
                if value in four_state_ineligible
                else "included_complete_four_state"
            )
        ),
    )
    exclusion_log.insert(
        3,
        "reason",
        exclusion_log["record_id"].map(
            lambda value: EXCLUSIONS.get(
                value,
                "No sustained Wake epochs after recoding Wake bouts shorter than 15 s as MA"
                if value in four_state_ineligible
                else "Passed available QC and has all four states",
            )
        ),
    )
    exclusion_log.to_csv(
        DATA / "Fig5E_GCaMP_state_activity_exclusion_log.csv", index=False, encoding="utf-8-sig"
    )
    raw = raw_all.loc[
        ~raw_all["record_id"].isin(set(EXCLUSIONS) | four_state_ineligible), ["record_id", *STATES]
    ].reset_index(drop=True)
    if len(raw) != 12 or raw[STATES].isna().any().any():
        raise ValueError("Expected twelve complete four-state records")

    nrem_ma = raw_all.loc[~raw_all["record_id"].isin(EXCLUSIONS), ["record_id", "NREM", "MA"]].dropna()
    if len(nrem_ma) != 13:
        raise ValueError("Expected thirteen records with paired NREM and MA values")
    nrem_ma.to_csv(DATA / "Fig5E_GCaMP_state_activity_NREM_MA_all_available_source_data.csv", index=False, encoding="utf-8-sig")
    primary = stats.wilcoxon(nrem_ma["NREM"], nrem_ma["MA"], method="exact")
    pd.DataFrame(
        [{
            "analysis": "primary_all_available",
            "comparison": "NREM vs MA",
            "test": "exact paired Wilcoxon",
            "statistic": float(primary.statistic),
            "p_raw": float(primary.pvalue),
            "n_records": len(nrem_ma),
            "multiplicity_rule": "single primary contrast",
        }]
    ).to_csv(DATA / "Fig5E_GCaMP_state_activity_NREM_MA_all_available_stats.csv", index=False, encoding="utf-8-sig")

    source_out = DATA / "Fig5E_GCaMP_state_activity_source_data.csv"
    raw.to_csv(source_out, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        {
            "state": STATES,
            "state_index": np.arange(1, 5),
            "n_records": [len(raw)] * 4,
            "mean": [raw[s].mean() for s in STATES],
            "sample_sd": [raw[s].std(ddof=1) for s in STATES],
            "sem": [raw[s].sem(ddof=1) for s in STATES],
        }
    )
    summary.to_csv(DATA / "Fig5E_GCaMP_state_activity_summary.csv", index=False, encoding="utf-8-sig")

    friedman = stats.friedmanchisquare(*[raw[s].to_numpy() for s in STATES])
    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for i, left in enumerate(STATES):
        for right in STATES[i + 1 :]:
            a, b = raw[left].to_numpy(), raw[right].to_numpy()
            result = stats.wilcoxon(a, b, method="exact")
            raw_p.append(float(result.pvalue))
            comparisons.append(
                {
                    "analysis": "pairwise",
                    "comparison": f"{left} vs {right}",
                    "comparison_role": "complete_case_posthoc",
                    "test": "exact paired Wilcoxon",
                    "statistic": float(result.statistic),
                    "p_raw": float(result.pvalue),
                    "effect": "paired rank-biserial",
                    "effect_value": rank_biserial(a, b),
                    "n_records": len(raw),
                }
            )
    adjusted = holm(raw_p)
    for row, p_adj in zip(comparisons, adjusted, strict=True):
        row["p_holm_6"] = p_adj
        row["multiplicity_rule"] = "Holm adjustment across all 6 complete-case pairs"
        row["p_for_inference"] = p_adj
        row["display"] = stars(p_adj)

    stats_rows = [
        {
            "analysis": "omnibus",
            "comparison": "Wake/NREM/REM/MA",
            "comparison_role": "omnibus",
            "test": "Friedman repeated-measures test",
            "statistic": float(friedman.statistic),
            "p_raw": float(friedman.pvalue),
            "effect": "Kendall W",
            "effect_value": float(friedman.statistic / (len(raw) * (len(STATES) - 1))),
            "n_records": len(raw),
            "p_holm_6": "",
            "multiplicity_rule": "omnibus test",
            "p_for_inference": float(friedman.pvalue),
            "display": "",
        },
        *comparisons,
    ]
    pd.DataFrame(stats_rows).to_csv(DATA / "Fig5E_GCaMP_state_activity_stats.csv", index=False, encoding="utf-8-sig")

    origin = pd.DataFrame(index=np.arange(len(raw)))
    origin["StateIndex"] = pd.Series([1, 2, 3, 4])
    origin["Mean"] = pd.Series(summary["mean"].to_numpy())
    origin["SEM"] = pd.Series(summary["sem"].to_numpy())
    for index, state in enumerate(STATES, start=1):
        origin[f"{state}_X"] = float(index)
        origin[f"{state}_Raw"] = raw[state].to_numpy()
    origin.insert(0, "RecordID", raw["record_id"])
    origin.to_csv(DATA / "Fig5E_GCaMP_state_activity_Origin_import.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"friedman={friedman.statistic:.6f} p={friedman.pvalue:.9f}")
    print(pd.DataFrame(comparisons)[["comparison", "p_raw", "p_holm_6", "display"]].to_string(index=False))


if __name__ == "__main__":
    main()
