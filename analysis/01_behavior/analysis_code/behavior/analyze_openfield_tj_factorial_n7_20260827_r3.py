#!/usr/bin/env python3
"""Factorial analysis of the current n=7/group open-field workbook."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from patsy import build_design_matrices
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence


SOURCE = Path(r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\openfilenphr\tj.xls")
OUTPUT_DIR = SOURCE.parent / "analysis_openfield_tj_factorial_n7_20260827_r3"
GROUP_ORDER = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
PREFIX_TO_TREATMENT = {
    "c": "baseline+SAL",
    "s": "baseline+CNO",
    "eyp": "POCD+SAL",
    "nphr": "POCD+CNO",
}
METRICS = [
    "TrackedTime_s",
    "TotalDistance_m",
    "CenterTime_s",
    "BorderTime_s",
    "Distance_m_per_min",
    "CenterPctTracked",
    "BorderPctTracked",
]
REQUESTED_METRICS = ["TotalDistance_m", "CenterTime_s", "BorderTime_s"]
COMPARISON_PAIRS = [
    ("baseline+SAL", "POCD+SAL"),
    ("baseline+CNO", "POCD+CNO"),
    ("baseline+SAL", "baseline+CNO"),
    ("POCD+SAL", "POCD+CNO"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(SOURCE, sheet_name="Sheet1", header=None)
    rows = raw.iloc[2:].loc[raw.iloc[2:, 0].notna()].copy()
    current_summary_schema = (
        raw.shape[1] >= 18
        and str(raw.iloc[1, 1]).strip() == "动物名称"
        and str(raw.iloc[1, 2]).strip() == "总路程(mm)"
    )

    if current_summary_schema:
        data = pd.DataFrame(
            {
                "Record": pd.to_numeric(rows.iloc[:, 0], errors="raise").astype(int),
                "Treatment": rows.iloc[:, 1].astype(str).str.strip(),
                "TotalDistance_mm": pd.to_numeric(rows.iloc[:, 2], errors="raise"),
                "TrackedTime_s": pd.to_numeric(rows.iloc[:, 3], errors="raise"),
                "BorderDistance_mm": pd.to_numeric(rows.iloc[:, 6], errors="raise"),
                "BorderTime_s": pd.to_numeric(rows.iloc[:, 7], errors="raise"),
                "CenterDistance_mm": pd.to_numeric(rows.iloc[:, 12], errors="raise"),
                "CenterTime_s": pd.to_numeric(rows.iloc[:, 13], errors="raise"),
            }
        ).reset_index(drop=True)
        data["Animal"] = "source-record-" + data["Record"].astype(str)
        data["ExperimentDateTime"] = "not present in summary sheet"
        data["OriginalPrefix"] = "group label supplied in workbook"
        source_schema = "group-labelled summary sheet"
    else:
        data = pd.DataFrame(
            {
                "Record": pd.to_numeric(rows.iloc[:, 0], errors="raise").astype(int),
                "Animal": rows.iloc[:, 2].astype(str).str.strip(),
                "ExperimentDateTime": rows.iloc[:, 4].astype(str).str.strip(),
                "TotalDistance_mm": pd.to_numeric(rows.iloc[:, 5], errors="raise"),
                "TrackedTime_s": pd.to_numeric(rows.iloc[:, 6], errors="raise"),
                "BorderDistance_mm": pd.to_numeric(rows.iloc[:, 9], errors="raise"),
                "BorderTime_s": pd.to_numeric(rows.iloc[:, 10], errors="raise"),
                "CenterDistance_mm": pd.to_numeric(rows.iloc[:, 15], errors="raise"),
                "CenterTime_s": pd.to_numeric(rows.iloc[:, 16], errors="raise"),
            }
        ).reset_index(drop=True)
        data["OriginalPrefix"] = data["Animal"].str.extract(r"^([^\-]+)")[0].str.lower()
        data["Treatment"] = data["OriginalPrefix"].map(PREFIX_TO_TREATMENT)
        if data["Treatment"].isna().any():
            unknown = sorted(data.loc[data["Treatment"].isna(), "OriginalPrefix"].unique())
            raise SystemExit(f"Unexpected prefixes: {unknown}")
        source_schema = "legacy animal-level export"

    unexpected_groups = sorted(set(data["Treatment"]) - set(GROUP_ORDER))
    if unexpected_groups:
        raise SystemExit(f"Unexpected treatment labels: {unexpected_groups}")
    counts = data.groupby("Treatment").size().reindex(GROUP_ORDER)
    if len(data) != 28 or not (counts == 7).all():
        raise SystemExit(f"Expected 28 records and n=7/group, found {counts.to_dict()}")

    factors = data["Treatment"].str.split("+", regex=False, expand=True)
    data["Surgery"] = factors[0]
    data["Drug"] = factors[1]
    data["TotalDistance_m"] = data["TotalDistance_mm"] / 1000
    data["Distance_m_per_min"] = data["TotalDistance_m"] / data["TrackedTime_s"] * 60
    data["CenterPctTracked"] = data["CenterTime_s"] / data["TrackedTime_s"] * 100
    data["BorderPctTracked"] = data["BorderTime_s"] / data["TrackedTime_s"] * 100
    data["ZoneCoveragePct"] = (
        (data["CenterTime_s"] + data["BorderTime_s"])
        / data["TrackedTime_s"]
        * 100
    )
    data.attrs["source_schema"] = source_schema
    return data


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    pooled = math.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2)
    )
    if pooled == 0:
        return float("nan")
    d_value = (np.mean(x) - np.mean(y)) / pooled
    return float(d_value * (1 - 3 / (4 * (len(x) + len(y)) - 9)))


def design_row(model, treatment: str) -> np.ndarray:
    surgery, drug = treatment.split("+")
    frame = pd.DataFrame({"Surgery": [surgery], "Drug": [drug]})
    matrix = build_design_matrices([model.model.data.design_info], frame)[0]
    return np.asarray(matrix)[0]


def model_contrast(model, data: pd.DataFrame, metric: str, group_a: str, group_b: str) -> dict:
    result = model.t_test(design_row(model, group_a) - design_row(model, group_b))
    interval = np.asarray(result.conf_int(alpha=0.05)).reshape(-1, 2)[0]
    x = data.loc[data["Treatment"] == group_a, metric].to_numpy(float)
    y = data.loc[data["Treatment"] == group_b, metric].to_numpy(float)
    return {
        "Metric": metric,
        "Comparison": f"{group_a} vs {group_b}",
        "GroupA": group_a,
        "GroupB": group_b,
        "N_A": len(x),
        "N_B": len(y),
        "Mean_A": float(np.mean(x)),
        "Mean_B": float(np.mean(y)),
        "MeanDifference_A_minus_B": float(np.asarray(result.effect).squeeze()),
        "CI95_low": float(interval[0]),
        "CI95_high": float(interval[1]),
        "t": float(np.asarray(result.tvalue).squeeze()),
        "df": float(model.df_resid),
        "p_raw": float(np.asarray(result.pvalue).squeeze()),
        "Hedges_g": hedges_g(x, y),
    }


def analyze(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summaries = []
    anova_rows = []
    contrast_rows = []
    qc_metrics = {}

    for metric in METRICS:
        summary = (
            data.groupby("Treatment")[metric]
            .agg(N="count", Mean="mean", SD="std", SEM="sem", Median="median", Min="min", Max="max")
            .reindex(GROUP_ORDER)
        )
        for group, row in summary.iterrows():
            summaries.append({"Metric": metric, "Treatment": group, **row.to_dict()})

        model = ols(f"{metric} ~ C(Surgery, Sum) * C(Drug, Sum)", data=data).fit()
        anova = sm.stats.anova_lm(model, typ=3)
        residual_ss = float(anova.loc["Residual", "sum_sq"])
        for effect in [
            "C(Surgery, Sum)",
            "C(Drug, Sum)",
            "C(Surgery, Sum):C(Drug, Sum)",
        ]:
            effect_ss = float(anova.loc[effect, "sum_sq"])
            anova_rows.append(
                {
                    "Metric": metric,
                    "Effect": effect,
                    "sum_sq": effect_ss,
                    "df": float(anova.loc[effect, "df"]),
                    "F": float(anova.loc[effect, "F"]),
                    "p": float(anova.loc[effect, "PR(>F)"]),
                    "partial_eta_squared": effect_ss / (effect_ss + residual_ss),
                }
            )

        comparisons = [
            model_contrast(model, data, metric, group_a, group_b)
            for group_a, group_b in COMPARISON_PAIRS
        ]
        reject, adjusted_p, _, _ = multipletests(
            [item["p_raw"] for item in comparisons], method="holm"
        )
        for item, p_holm, significant in zip(comparisons, adjusted_p, reject):
            item["p_holm"] = float(p_holm)
            item["significant_holm_0_05"] = bool(significant)
            contrast_rows.append(item)

        arrays = [
            data.loc[data["Treatment"] == group, metric].to_numpy(float)
            for group in GROUP_ORDER
        ]
        brown_forsythe = stats.levene(*arrays, center="median")
        shapiro = stats.shapiro(model.resid)
        influence = OLSInfluence(model)
        qc_metrics[metric] = {
            "shapiro_residual_W": float(shapiro.statistic),
            "shapiro_residual_p": float(shapiro.pvalue),
            "brown_forsythe_statistic": float(brown_forsythe.statistic),
            "brown_forsythe_p": float(brown_forsythe.pvalue),
            "maximum_absolute_externally_studentized_residual": float(
                np.max(np.abs(influence.resid_studentized_external))
            ),
            "maximum_cooks_distance": float(np.max(influence.cooks_distance[0])),
        }

    correlations = {}
    for metric in REQUESTED_METRICS:
        result = stats.pearsonr(data["TrackedTime_s"], data[metric])
        correlations[metric] = {
            "r": float(result.statistic),
            "p": float(result.pvalue),
        }
    quality_control = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "source_schema": data.attrs.get("source_schema", "unknown"),
        "records": int(len(data)),
        "group_sizes": {
            group: int((data["Treatment"] == group).sum()) for group in GROUP_ORDER
        },
        "source_record_ids_by_group": {
            group: data.loc[data["Treatment"] == group, "Record"].astype(int).tolist()
            for group in GROUP_ORDER
        },
        "tracked_time_min_s": float(data["TrackedTime_s"].min()),
        "tracked_time_max_s": float(data["TrackedTime_s"].max()),
        "zone_coverage_mean_percent": float(data["ZoneCoveragePct"].mean()),
        "tracked_time_correlations": correlations,
        "metric_diagnostics": qc_metrics,
    }
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(anova_rows),
        pd.DataFrame(contrast_rows),
        quality_control,
    )


def write_outputs(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    anova: pd.DataFrame,
    contrasts: pd.DataFrame,
    qc: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "openfield_factorial_n7_source_data.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "openfield_factorial_n7_group_summary.csv", index=False, encoding="utf-8-sig")
    anova.to_csv(OUTPUT_DIR / "openfield_factorial_n7_type3_anova.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUTPUT_DIR / "openfield_factorial_n7_model_contrasts.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "openfield_factorial_n7_quality_control.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_index = summary.set_index(["Metric", "Treatment"])
    anova_index = anova.set_index(["Metric", "Effect"])
    lines = [
        "# Open-field factorial analysis, n=7/group",
        "",
        f"- Source: `{SOURCE}`",
        f"- SHA-256: `{qc['source_sha256']}`",
        f"- Source schema: {qc['source_schema']}.",
        "- Groups are read directly from the current workbook; legacy c/s/eyp/nphr prefix mapping remains supported.",
        "- Model: independent-samples 2 x 2 Type III ANOVA (Surgery x Drug).",
        "- Simple effects: model-based two-sided contrasts with Holm correction across four comparisons per metric.",
        "",
        "## Requested outcomes",
        "",
    ]
    for metric in REQUESTED_METRICS:
        lines.append(f"### {metric}")
        for group in GROUP_ORDER:
            row = summary_index.loc[(metric, group)]
            lines.append(
                f"- {group}: n={int(row['N'])}, mean={row['Mean']:.3f}, SEM={row['SEM']:.3f}."
            )
        for effect in [
            "C(Surgery, Sum)",
            "C(Drug, Sum)",
            "C(Surgery, Sum):C(Drug, Sum)",
        ]:
            row = anova_index.loc[(metric, effect)]
            lines.append(f"- {effect}: F={row['F']:.4f}, p={row['p']:.6f}.")
        significant = contrasts.loc[
            (contrasts["Metric"] == metric) & contrasts["significant_holm_0_05"]
        ]
        if significant.empty:
            lines.append("- Holm-adjusted simple effects: none significant.")
        else:
            for row in significant.itertuples(index=False):
                lines.append(f"- {row.Comparison}: Holm p={row.p_holm:.6f}.")
        lines.append("")

    lines.extend(
        [
            "## Tracking-duration note",
            "",
            f"Effective tracked time ranges from {qc['tracked_time_min_s']:.2f} to {qc['tracked_time_max_s']:.2f} s. Absolute outcomes remain duration-sensitive; normalized outcomes are included in the complete CSV tables.",
        ]
    )
    (OUTPUT_DIR / "README_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    summary, anova, contrasts, qc = analyze(data)
    write_outputs(data, summary, anova, contrasts, qc)
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "source_sha256": qc["source_sha256"],
                "group_sizes": qc["group_sizes"],
                "requested_anova": json.loads(
                    anova.loc[anova["Metric"].isin(REQUESTED_METRICS)].to_json(orient="records")
                ),
                "significant_requested_contrasts": json.loads(
                    contrasts.loc[
                        contrasts["Metric"].isin(REQUESTED_METRICS)
                        & contrasts["significant_holm_0_05"]
                    ].to_json(orient="records")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
