from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway


FIGURE_ID = "fc_contextual_20260317_reanalysis_20260829"
EXP_PATH = Path(
    r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\20260317-TEST\CONTEXTUAL\CONTEXTUAL.exp"
)
ANALYZER_DIR = Path(r"F:\hyq\软件\PACKWIN_Analyzer")
ANALYZER_SOURCE = ANALYZER_DIR / "packwin_analyzer.py"
OUTDIR = Path(r"F:\Sleep\Figure\FC_contextual_4groups_20260317_reanalysis_20260829")
PROVENANCE_DIR = Path(r"F:\Sleep\outputs\figure_provenance")
SCRIPT_PATH = Path(__file__).resolve()
LABEL_AUDIT = EXP_PATH.parent / "packwin_label_rename_audit_20260828.json"

GROUP_ORDER = ["baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO"]
GROUP_FACTORS = {
    "baseline+SAL": ("baseline", "SAL"),
    "baseline+CNO": ("baseline", "CNO"),
    "POCD+SAL": ("POCD", "SAL"),
    "POCD+CNO": ("POCD", "CNO"),
}
PALETTE = {
    "baseline+SAL": {"fill": "#4DBBD5", "edge": "#368DA8"},
    "baseline+CNO": {"fill": "#00A087", "edge": "#007B69"},
    "POCD+SAL": {"fill": "#F39B7F", "edge": "#C96F59"},
    "POCD+CNO": {"fill": "#8491B4", "edge": "#626E91"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def animal_code(subject: str) -> str:
    match = re.search(r"(\d+)\s*$", subject)
    if not match:
        raise ValueError(f"Cannot derive animal code from {subject!r}")
    return f"S{int(match.group(1))}"


def p_text(value: float) -> str:
    return "P < 0.0001" if value < 0.0001 else f"P = {value:.4f}"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.compression": 0,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def load_data() -> pd.DataFrame:
    if not EXP_PATH.is_file() or not ANALYZER_SOURCE.is_file():
        raise FileNotFoundError("PACKWIN experiment or analyzer source is missing")
    sys.path.insert(0, str(ANALYZER_DIR))
    from packwin_analyzer import Experiment

    rows: list[dict[str, object]] = []
    for result in Experiment(EXP_PATH).analyze():
        if result.group not in GROUP_FACTORS:
            raise ValueError(f"Unexpected group label: {result.group}")
        condition, treatment = GROUP_FACTORS[result.group]
        rows.append(
            {
                "raw_id": int(result.raw_id),
                "group": result.group,
                "condition": condition,
                "treatment": treatment,
                "subject": result.subject,
                "animal_code": animal_code(result.subject),
                "timestamp": result.timestamp,
                "duration_s": float(result.duration_s),
                "freezing_s": float(result.freezing_s),
                "activity_s": float(result.activity_s),
                "freezing_pct": float(result.freezing_pct),
                "freezing_episodes": int(result.episodes),
                "mean_activity_pct": float(result.mean_value),
                "max_activity_pct": float(result.max_value),
                "sample_rate_hz": int(result.sample_rate),
                "activity_threshold_pct": float(result.lower),
                "minimum_freezing_ms": int(result.freezing_ms),
                "protocol": result.protocol,
            }
        )
    frame = pd.DataFrame(rows)
    frame["group"] = pd.Categorical(frame["group"], GROUP_ORDER, ordered=True)
    frame = frame.sort_values(["group", "raw_id"]).reset_index(drop=True)
    if set(frame["group"].astype(str)) != set(GROUP_ORDER):
        raise ValueError("Not all four expected groups are present")
    if frame["raw_id"].duplicated().any() or frame["animal_code"].duplicated().any():
        raise ValueError("Duplicate session or animal code")
    if len(frame) != 26:
        raise ValueError(f"Expected 26 sessions, found {len(frame)}")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        values = frame.loc[frame["group"] == group, "freezing_pct"].to_numpy(float)
        n = len(values)
        sd = float(np.std(values, ddof=1))
        sem = sd / math.sqrt(n)
        critical = float(stats.t.ppf(0.975, n - 1))
        rows.append(
            {
                "group": group,
                "n": n,
                "mean_freezing_pct": float(np.mean(values)),
                "sd_freezing_pct": sd,
                "sem_freezing_pct": sem,
                "ci95_lower": float(np.mean(values) - critical * sem),
                "ci95_upper": float(np.mean(values) + critical * sem),
                "median_freezing_pct": float(np.median(values)),
                "min_freezing_pct": float(np.min(values)),
                "max_freezing_pct": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def analyze_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays = [
        frame.loc[frame["group"] == group, "freezing_pct"].to_numpy(float)
        for group in GROUP_ORDER
    ]
    model = ols("freezing_pct ~ C(group)", data=frame).fit()
    f_value, p_value = stats.f_oneway(*arrays)
    shapiro_w, shapiro_p = stats.shapiro(model.resid)
    levene_w, levene_p = stats.levene(*arrays, center="median")
    welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)
    kruskal_h, kruskal_p = stats.kruskal(*arrays)
    grand_mean = float(frame["freezing_pct"].mean())
    ss_between = sum(len(a) * (float(np.mean(a)) - grand_mean) ** 2 for a in arrays)
    ss_within = sum(float(np.sum((a - np.mean(a)) ** 2)) for a in arrays)
    ss_total = ss_between + ss_within
    df1 = 3
    df2 = len(frame) - 4
    ms_within = ss_within / df2
    overall = pd.DataFrame(
        [
            {
                "test": "One-way ANOVA",
                "statistic": f_value,
                "df1": df1,
                "df2": df2,
                "p_value": p_value,
                "effect_size_name": "eta_squared",
                "effect_size": ss_between / ss_total,
                "secondary_effect_size_name": "omega_squared",
                "secondary_effect_size": (ss_between - df1 * ms_within) / (ss_total + ms_within),
            },
            {
                "test": "Welch ANOVA",
                "statistic": float(welch.statistic),
                "df1": float(welch.df_num),
                "df2": float(welch.df_denom),
                "p_value": float(welch.pvalue),
            },
            {
                "test": "Kruskal-Wallis",
                "statistic": kruskal_h,
                "df1": df1,
                "df2": np.nan,
                "p_value": kruskal_p,
            },
            {
                "test": "Shapiro-Wilk residuals",
                "statistic": shapiro_w,
                "df1": np.nan,
                "df2": np.nan,
                "p_value": shapiro_p,
            },
            {
                "test": "Brown-Forsythe/Levene",
                "statistic": levene_w,
                "df1": df1,
                "df2": df2,
                "p_value": levene_p,
            },
        ]
    )

    tukey_result = pairwise_tukeyhsd(
        frame["freezing_pct"].to_numpy(float), frame["group"].astype(str).to_numpy()
    )
    tukey = pd.DataFrame(
        tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0],
    ).rename(
        columns={
            "group1": "group_1",
            "group2": "group_2",
            "meandiff": "mean_difference_group2_minus_group1",
            "p-adj": "p_adjusted",
            "lower": "ci95_lower",
            "upper": "ci95_upper",
        }
    )

    factorial_model = ols(
        "freezing_pct ~ C(condition, Sum) * C(treatment, Sum)", data=frame
    ).fit()
    table = sm.stats.anova_lm(factorial_model, typ=3)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    term_labels = {
        "C(condition, Sum)": "Condition (POCD vs baseline)",
        "C(treatment, Sum)": "Treatment (CNO vs SAL)",
        "C(condition, Sum):C(treatment, Sum)": "Condition x treatment interaction",
        "Residual": "Residual",
    }
    factorial_rows: list[dict[str, object]] = []
    for term, row in table.iterrows():
        if term == "Intercept":
            continue
        ss = float(row["sum_sq"])
        factorial_rows.append(
            {
                "term": term_labels.get(term, term),
                "sum_sq": ss,
                "df": float(row["df"]),
                "F": float(row["F"]) if pd.notna(row["F"]) else np.nan,
                "p_value": float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else np.nan,
                "partial_eta_squared": ss / (ss + residual_ss) if term != "Residual" else np.nan,
            }
        )
    return overall, tukey, pd.DataFrame(factorial_rows)


def spread_labels(values: np.ndarray, minimum_gap: float = 3.5) -> np.ndarray:
    order = np.argsort(values)
    output = values.astype(float).copy()
    for index in range(1, len(order)):
        previous = order[index - 1]
        current = order[index]
        output[current] = max(output[current], output[previous] + minimum_gap)
    overflow = max(0.0, float(output.max()) - 103.0)
    output -= overflow
    underflow = max(0.0, 3.0 - float(output.min()))
    output += underflow
    return output


def draw_figure(frame: pd.DataFrame, summary: pd.DataFrame, anova_p: float) -> list[Path]:
    configure_style()
    fig, ax = plt.subplots(figsize=(4.45, 2.75))
    x_positions = np.arange(4, dtype=float)
    indexed = summary.set_index("group").loc[GROUP_ORDER]
    means = indexed["mean_freezing_pct"].to_numpy(float)
    sems = indexed["sem_freezing_pct"].to_numpy(float)

    for x, group, mean in zip(x_positions, GROUP_ORDER, means, strict=True):
        style = PALETTE[group]
        ax.bar(x, mean, width=0.54, color=style["fill"], edgecolor=style["edge"], linewidth=0.6, zorder=1)
    ax.errorbar(x_positions, means, yerr=sems, fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)

    for x, group in zip(x_positions, GROUP_ORDER, strict=True):
        subset = frame.loc[frame["group"] == group].sort_values("raw_id")
        values = subset["freezing_pct"].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        point_x = x + jitter
        ax.scatter(point_x, values, s=25, facecolors="white", edgecolors=PALETTE[group]["edge"], linewidths=0.95, zorder=5)
        sides = np.array(["left" if i % 2 else "right" for i in range(len(values))])
        label_y = np.zeros(len(values), dtype=float)
        for side in ("left", "right"):
            indices = np.flatnonzero(sides == side)
            label_y[indices] = spread_labels(values[indices])
        for i, row in enumerate(subset.itertuples(index=False)):
            left = sides[i] == "left"
            label_x = x - 0.33 if left else x + 0.33
            ax.annotate(
                row.animal_code,
                xy=(point_x[i], values[i]),
                xytext=(label_x, label_y[i]),
                ha="right" if left else "left",
                va="center",
                fontsize=5,
                color=PALETTE[group]["edge"],
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": PALETTE[group]["edge"], "linewidth": 0.45, "alpha": 0.7, "shrinkA": 1.5, "shrinkB": 2},
                zorder=6,
            )

    counts = indexed["n"].astype(int).tolist()
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(GROUP_ORDER, counts, strict=True)])
    ax.set_ylabel("Freezing (%)")
    ax.set_title("Contextual freezing", fontsize=8, fontweight="bold", pad=6)
    ax.text(0.5, 0.975, f"One-way ANOVA: {p_text(anova_p)}", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    ax.set_xlim(-0.72, 3.72)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(False)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.87)

    base = OUTDIR / "FC_contextual_freezing_4groups_reanalysis_labeled"
    outputs = [base.with_suffix(ext) for ext in (".pdf", ".svg", ".png", ".tiff")]
    fig.savefig(outputs[0], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(outputs[1], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(outputs[3], dpi=600, facecolor="white", edgecolor="none", bbox_inches="tight")
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
        if hasattr(artist, "set_clip_path"):
            artist.set_clip_path(None)
    clean_pdf = OUTDIR / "FC_contextual_freezing_4groups_reanalysis_labeled_ai_clean.pdf"
    clean_svg = OUTDIR / "FC_contextual_freezing_4groups_reanalysis_labeled_ai_clean.svg"
    fig.savefig(clean_pdf, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(clean_svg, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return outputs + [clean_pdf, clean_svg]


def write_records(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    tukey: pd.DataFrame,
    factorial: pd.DataFrame,
    figures: list[Path],
) -> None:
    source_paths = [EXP_PATH, EXP_PATH.with_suffix(".ssn"), EXP_PATH.with_suffix(".ini"), EXP_PATH.with_suffix(".raw"), ANALYZER_SOURCE, SCRIPT_PATH]
    if LABEL_AUDIT.exists():
        source_paths.append(LABEL_AUDIT)
    pd.DataFrame(
        [
            {
                "role": "script" if path.suffix == ".py" else ("audit" if path.suffix == ".json" else "raw"),
                "path": str(path),
                "description": "Analysis source" if path.suffix == ".py" else ("Prior label-correction audit" if path.suffix == ".json" else "PACKWIN source file"),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ]
    ).to_csv(OUTDIR / "source_paths.csv", index=False)

    anova = overall.loc[overall["test"] == "One-way ANOVA"].iloc[0]
    factors = factorial.loc[factorial["term"] != "Residual"]
    group_lines = "\n".join(
        f"- {r.group}: n={int(r.n)}, {r.mean_freezing_pct:.2f} ± {r.sem_freezing_pct:.2f}% (mean ± SEM)"
        for r in summary.itertuples(index=False)
    )
    factor_lines = "\n".join(
        f"- {r.term}: F={r.F:.4f}, {p_text(r.p_value)}, partial η²={r.partial_eta_squared:.4f}"
        for r in factors.itertuples(index=False)
    )
    readme = f"""# FC contextual freezing reanalysis (2026-03-17)

## Figure contract

- Core conclusion: test whether contextual freezing differs among the four baseline/POCD × SAL/CNO groups.
- Evidence: animal-level freezing percentages, group mean ± SEM, and exact one-way ANOVA.
- Archetype: single-panel quantitative comparison.
- Backend: Python/Matplotlib only.
- Review risks: small group sizes (n=6–7), unequal cell sizes, and interpretation of a non-significant result.

## Sample and endpoint

- Biological replicate: one animal / one PACKWIN session.
- Total n={len(frame)}; no exclusions.
- Endpoint: freezing percentage during the full {frame['duration_s'].iloc[0]:.0f}-s contextual session.
- Threshold: {frame['activity_threshold_pct'].iloc[0]:g}%; minimum freezing duration: {int(frame['minimum_freezing_ms'].iloc[0])} ms; sample rate: {int(frame['sample_rate_hz'].iloc[0])} Hz.

## Group results

{group_lines}

## Statistics

- One-way ANOVA: F({int(anova.df1)}, {int(anova.df2)})={anova.statistic:.4f}, {p_text(anova.p_value)}, η²={anova.effect_size:.4f}, ω²={anova.secondary_effect_size:.4f}.
- No Tukey-adjusted comparison reached P < 0.05.
- Factorial sensitivity analysis:
{factor_lines}
- Assumption checks, Welch ANOVA, Kruskal–Wallis, and all Tukey results are preserved in the CSV tables.

Interpretation: no statistically significant four-group difference was detected at α=0.05. This does not establish equivalence.

## Reproduction

- Script: `{SCRIPT_PATH}`
- Command: `python -X utf8 "{SCRIPT_PATH}"`
- Raw source and hashes: `source_paths.csv`
- Animal-level source: `animal_level_freezing.csv`
- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}
"""
    (OUTDIR / "README_analysis.md").write_text(readme, encoding="utf-8")

    provenance_source = PROVENANCE_DIR / f"{FIGURE_ID}_source_data.csv"
    provenance_stats = PROVENANCE_DIR / f"{FIGURE_ID}_stats.csv"
    shutil.copy2(OUTDIR / "animal_level_freezing.csv", provenance_source)
    pd.concat(
        [overall.assign(section="overall"), factorial.assign(section="factorial"), tukey.assign(section="tukey")],
        ignore_index=True,
        sort=False,
    ).to_csv(provenance_stats, index=False, float_format="%.10g")
    provenance = f"""# {FIGURE_ID}

- Figure title: Contextual freezing, four-group reanalysis
- Status: draft
- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}
- Manuscript context: behavioral validation of contextual fear memory across baseline/POCD and SAL/CNO conditions.

## Data sources

- `{EXP_PATH}` and matching `.ssn/.ini/.raw` files.

## Sample definition

- One animal per PACKWIN session; n=26; no exclusions; groups ordered as {', '.join(GROUP_ORDER)}.
- Animal names are read directly from the currently corrected experiment definition.

## Analysis window and steps

- Full 0–180 s contextual session.
- PACKWIN activity traces were parsed with the local analyzer using threshold 13% and minimum freezing duration 1000 ms.
- Freezing percentage was summarized as mean ± SEM; all individual animals were displayed.

## Statistics

- One-way ANOVA F(3,22)={anova.statistic:.4f}, p={anova.p_value:.10g}.
- Type-III condition × treatment factorial ANOVA and sensitivity analyses are in `{provenance_stats}`.

## Scripts and outputs

- Script: `{SCRIPT_PATH}`
- Source data: `{provenance_source}`
- Stats: `{provenance_stats}`
- Figure folder: `{OUTDIR}`

## Review notes

- Non-significant results should not be described as proof of equivalence.
- The prior EXP group/animal-label correction audit is included in `source_paths.csv`.
"""
    (PROVENANCE_DIR / f"{FIGURE_ID}_provenance.md").write_text(provenance, encoding="utf-8")

    deliverables = figures + [
        OUTDIR / "animal_level_freezing.csv",
        OUTDIR / "group_summary.csv",
        OUTDIR / "statistics_overall.csv",
        OUTDIR / "tukey_pairwise.csv",
        OUTDIR / "two_way_anova.csv",
        OUTDIR / "source_paths.csv",
        OUTDIR / "README_analysis.md",
        SCRIPT_PATH,
        provenance_source,
        provenance_stats,
        PROVENANCE_DIR / f"{FIGURE_ID}_provenance.md",
    ]
    pd.DataFrame(
        [
            {"path": str(path), "format_notes": path.suffix.lstrip(".").upper(), "sha256": sha256_file(path)}
            for path in deliverables
        ]
    ).to_csv(OUTDIR / "outputs_manifest.csv", index=False)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    summary = summarize(frame)
    overall, tukey, factorial = analyze_statistics(frame)
    frame.to_csv(OUTDIR / "animal_level_freezing.csv", index=False, float_format="%.8f")
    summary.to_csv(OUTDIR / "group_summary.csv", index=False, float_format="%.8f")
    overall.to_csv(OUTDIR / "statistics_overall.csv", index=False, float_format="%.10g")
    tukey.to_csv(OUTDIR / "tukey_pairwise.csv", index=False, float_format="%.10g")
    factorial.to_csv(OUTDIR / "two_way_anova.csv", index=False, float_format="%.10g")
    anova_p = float(overall.loc[overall["test"] == "One-way ANOVA", "p_value"].iloc[0])
    figures = draw_figure(frame, summary, anova_p)
    write_records(frame, summary, overall, tukey, factorial, figures)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + overall.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\n" + factorial.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\n" + tukey.to_string(index=False))
    print(json.dumps({"outdir": str(OUTDIR), "figure": str(figures[2])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
