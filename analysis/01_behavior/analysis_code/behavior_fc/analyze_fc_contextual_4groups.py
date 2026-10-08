from __future__ import annotations

import hashlib
import math
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
import scipy.stats as scipy_stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway


FIGURE_ID = "fc_contextual_20260317_four_groups"
EXP_PATH = Path(
    r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\20260317-TEST\CONTEXTUAL\CONTEXTUAL.exp"
)
ANALYZER_DIR = Path(r"F:\hyq\软件\PACKWIN_Analyzer")
ANALYZER_SOURCE = ANALYZER_DIR / "packwin_analyzer.py"
OUTDIR = Path(r"F:\Sleep\Figure\FC_contextual_4groups_20260317_20260828")
PROVENANCE_DIR = Path(r"F:\Sleep\outputs\figure_provenance")
DATE_AUDIT = Path(
    r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\_date_backup_20240317-TEST_20240317_to_20260317_20260828-202949\date_correction_audit.json"
)

GROUP_ORDER = ["CON", "SEV", "ASD", "ASD+SEV"]
GROUP_ALIASES = {
    "CON": "CON",
    "SEV": "SEV",
    "ASD": "ASD",
    "ASD+SEV": "ASD+SEV",
    "baseline+SAL": "CON",
    "POCD+SAL": "SEV",
    "baseline+CNO": "ASD",
    "POCD+CNO": "ASD+SEV",
}
FACTOR_MAP = {
    "CON": ("Absent", "Absent"),
    "SEV": ("Absent", "Present"),
    "ASD": ("Present", "Absent"),
    "ASD+SEV": ("Present", "Present"),
}
PALETTE = {
    "CON": {"fill": "#4DBBD5", "edge": "#368DA8"},
    "SEV": {"fill": "#F39B7F", "edge": "#C96F59"},
    "ASD": {"fill": "#00A087", "edge": "#007B69"},
    "ASD+SEV": {"fill": "#8491B4", "edge": "#626E91"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def p_text(p_value: float) -> str:
    if p_value < 0.0001:
        return "P < 0.0001"
    return f"P = {p_value:.4f}"


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return math.nan
    pooled_variance = (
        ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_variance <= 0:
        return 0.0
    d_value = (np.mean(second) - np.mean(first)) / math.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(correction * d_value)


def load_animal_data() -> pd.DataFrame:
    if not EXP_PATH.exists():
        raise FileNotFoundError(EXP_PATH)
    if not ANALYZER_SOURCE.exists():
        raise FileNotFoundError(ANALYZER_SOURCE)
    sys.path.insert(0, str(ANALYZER_DIR))
    from packwin_analyzer import Experiment

    results = Experiment(EXP_PATH).analyze()
    rows: list[dict[str, object]] = []
    for result in results:
        source_group = GROUP_ALIASES.get(result.group)
        if source_group not in FACTOR_MAP:
            raise ValueError(f"Unexpected group: {result.group}")
        asd_factor, sev_factor = FACTOR_MAP[source_group]
        rows.append(
            {
                "raw_id": result.raw_id,
                "group": source_group,
                "group_file_label": result.group,
                "subject": result.subject,
                "timestamp": result.timestamp,
                "ASD_factor": asd_factor,
                "SEV_factor": sev_factor,
                "duration_s": result.duration_s,
                "freezing_s": result.freezing_s,
                "activity_s": result.activity_s,
                "freezing_pct": result.freezing_pct,
                "freezing_episodes": result.episodes,
                "mean_activity_pct": result.mean_value,
                "max_activity_pct": result.max_value,
                "sample_rate_hz": result.sample_rate,
                "activity_threshold_pct": result.lower,
                "minimum_freezing_ms": result.freezing_ms,
                "protocol": result.protocol,
            }
        )
    frame = pd.DataFrame(rows)
    frame["group"] = pd.Categorical(frame["group"], GROUP_ORDER, ordered=True)
    frame = frame.sort_values(["group", "raw_id"]).reset_index(drop=True)
    observed = set(frame["group"].astype(str))
    if observed != set(GROUP_ORDER):
        raise ValueError(f"Expected {GROUP_ORDER}, found {sorted(observed)}")
    if frame["raw_id"].duplicated().any():
        raise ValueError("Duplicate raw session identifiers")
    return frame


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        values = frame.loc[frame["group"] == group, "freezing_pct"].to_numpy(float)
        n_value = len(values)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        sem = sd / math.sqrt(n_value)
        t_critical = float(scipy_stats.t.ppf(0.975, n_value - 1))
        rows.append(
            {
                "group": group,
                "n": n_value,
                "mean_freezing_pct": mean,
                "sd_freezing_pct": sd,
                "sem_freezing_pct": sem,
                "ci95_lower": mean - t_critical * sem,
                "ci95_upper": mean + t_critical * sem,
                "median_freezing_pct": float(np.median(values)),
                "q1_freezing_pct": float(np.percentile(values, 25)),
                "q3_freezing_pct": float(np.percentile(values, 75)),
                "min_freezing_pct": float(np.min(values)),
                "max_freezing_pct": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def run_statistics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    arrays = [
        frame.loc[frame["group"] == group, "freezing_pct"].to_numpy(float)
        for group in GROUP_ORDER
    ]
    model_one_way = ols("freezing_pct ~ C(group)", data=frame).fit()
    shapiro_w, shapiro_p = scipy_stats.shapiro(model_one_way.resid)
    levene_w, levene_p = scipy_stats.levene(*arrays, center="median")
    f_value, anova_p = scipy_stats.f_oneway(*arrays)
    kruskal_h, kruskal_p = scipy_stats.kruskal(*arrays)
    welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)

    grand_mean = float(frame["freezing_pct"].mean())
    ss_between = sum(
        len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in arrays
    )
    ss_within = sum(float(np.sum((values - np.mean(values)) ** 2)) for values in arrays)
    ss_total = ss_between + ss_within
    df_between = len(arrays) - 1
    df_within = len(frame) - len(arrays)
    ms_within = ss_within / df_within
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    overall = pd.DataFrame(
        [
            {
                "test": "One-way ANOVA",
                "statistic": f_value,
                "df1": df_between,
                "df2": df_within,
                "p_value": anova_p,
                "effect_size_name": "eta_squared",
                "effect_size": eta_squared,
                "secondary_effect_size_name": "omega_squared",
                "secondary_effect_size": omega_squared,
            },
            {
                "test": "Welch ANOVA",
                "statistic": float(welch.statistic),
                "df1": float(welch.df_num),
                "df2": float(welch.df_denom),
                "p_value": float(welch.pvalue),
                "effect_size_name": "",
                "effect_size": math.nan,
                "secondary_effect_size_name": "",
                "secondary_effect_size": math.nan,
            },
            {
                "test": "Kruskal-Wallis sensitivity analysis",
                "statistic": kruskal_h,
                "df1": df_between,
                "df2": math.nan,
                "p_value": kruskal_p,
                "effect_size_name": "",
                "effect_size": math.nan,
                "secondary_effect_size_name": "",
                "secondary_effect_size": math.nan,
            },
            {
                "test": "Shapiro-Wilk test of one-way ANOVA residuals",
                "statistic": shapiro_w,
                "df1": math.nan,
                "df2": math.nan,
                "p_value": shapiro_p,
                "effect_size_name": "",
                "effect_size": math.nan,
                "secondary_effect_size_name": "",
                "secondary_effect_size": math.nan,
            },
            {
                "test": "Brown-Forsythe/Levene test (median centered)",
                "statistic": levene_w,
                "df1": df_between,
                "df2": df_within,
                "p_value": levene_p,
                "effect_size_name": "",
                "effect_size": math.nan,
                "secondary_effect_size_name": "",
                "secondary_effect_size": math.nan,
            },
        ]
    )

    tukey_result = pairwise_tukeyhsd(
        endog=frame["freezing_pct"].to_numpy(float),
        groups=frame["group"].astype(str).to_numpy(),
        alpha=0.05,
    )
    tukey = pd.DataFrame(
        tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0],
    )
    tukey = tukey.rename(
        columns={
            "group1": "group_1",
            "group2": "group_2",
            "meandiff": "mean_difference_group2_minus_group1",
            "p-adj": "p_adjusted",
            "lower": "ci95_lower",
            "upper": "ci95_upper",
        }
    )
    tukey["hedges_g_group2_minus_group1"] = [
        hedges_g(
            frame.loc[frame["group"] == row.group_1, "freezing_pct"].to_numpy(float),
            frame.loc[frame["group"] == row.group_2, "freezing_pct"].to_numpy(float),
        )
        for row in tukey.itertuples(index=False)
    ]
    tukey["significance"] = [stars(float(p)) for p in tukey["p_adjusted"]]

    factorial_model = ols(
        "freezing_pct ~ C(ASD_factor, Sum) * C(SEV_factor, Sum)", data=frame
    ).fit()
    factorial_table = sm.stats.anova_lm(factorial_model, typ=3)
    residual_ss = float(factorial_table.loc["Residual", "sum_sq"])
    term_names = {
        "C(ASD_factor, Sum)": "ASD main effect",
        "C(SEV_factor, Sum)": "SEV main effect",
        "C(ASD_factor, Sum):C(SEV_factor, Sum)": "ASD x SEV interaction",
        "Residual": "Residual",
    }
    factorial_rows: list[dict[str, object]] = []
    for term, row in factorial_table.iterrows():
        if term == "Intercept":
            continue
        ss_value = float(row["sum_sq"])
        factorial_rows.append(
            {
                "term": term_names.get(term, term),
                "sum_sq": ss_value,
                "df": float(row["df"]),
                "F": float(row["F"]) if pd.notna(row["F"]) else math.nan,
                "p_value": float(row["PR(>F)"])
                if pd.notna(row["PR(>F)"])
                else math.nan,
                "partial_eta_squared": (
                    ss_value / (ss_value + residual_ss) if term != "Residual" else math.nan
                ),
            }
        )
    factorial = pd.DataFrame(factorial_rows)
    return overall, tukey, factorial, tukey_result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.compression": 0,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.frameon": False,
        }
    )


def draw_figure(
    frame: pd.DataFrame, summary: pd.DataFrame, one_way_p: float
) -> list[Path]:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.05, 2.35))
    x_positions = np.arange(len(GROUP_ORDER), dtype=float)
    means = summary.set_index("group").loc[GROUP_ORDER, "mean_freezing_pct"].to_numpy(float)
    sems = summary.set_index("group").loc[GROUP_ORDER, "sem_freezing_pct"].to_numpy(float)

    for x_value, group, mean in zip(x_positions, GROUP_ORDER, means, strict=True):
        style = PALETTE[group]
        ax.bar(
            x_value,
            mean,
            width=0.56,
            color=style["fill"],
            edgecolor=style["edge"],
            linewidth=0.55,
            zorder=1,
        )
    ax.errorbar(
        x_positions,
        means,
        yerr=sems,
        fmt="none",
        ecolor="black",
        elinewidth=0.85,
        capsize=2.3,
        capthick=0.85,
        zorder=4,
    )

    for x_value, group in zip(x_positions, GROUP_ORDER, strict=True):
        group_data = frame.loc[frame["group"] == group].sort_values("raw_id")
        values = group_data["freezing_pct"].to_numpy(float)
        jitter = np.linspace(-0.115, 0.115, len(values))
        ax.scatter(
            x_value + jitter,
            values,
            s=22,
            facecolors="white",
            edgecolors=PALETTE[group]["edge"],
            linewidths=0.9,
            alpha=0.9,
            zorder=5,
        )

    group_ns = summary.set_index("group").loc[GROUP_ORDER, "n"].astype(int).tolist()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{group}\n(n={n_value})" for group, n_value in zip(GROUP_ORDER, group_ns, strict=True)]
    )
    ax.set_ylabel("Freezing (%)", fontsize=6.5)
    ax.set_title("Contextual freezing", fontsize=7.2, fontweight="bold", pad=5)
    ax.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=6)
    ax.text(
        0.5,
        0.975,
        f"One-way ANOVA: {p_text(one_way_p)}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6,
    )
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(False)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.88)

    base = OUTDIR / "FC_contextual_freezing_4groups"
    outputs = [
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
        base.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(outputs[1], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(
        outputs[2],
        dpi=600,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
    )

    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
        if hasattr(artist, "set_clip_path"):
            artist.set_clip_path(None)
    clean_pdf = OUTDIR / "FC_contextual_freezing_4groups_ai_clean.pdf"
    clean_svg = OUTDIR / "FC_contextual_freezing_4groups_ai_clean.svg"
    fig.savefig(clean_pdf, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(clean_svg, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return outputs + [clean_pdf, clean_svg]


def write_metadata(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    tukey: pd.DataFrame,
    factorial: pd.DataFrame,
    figure_outputs: list[Path],
) -> None:
    source_files = [
        EXP_PATH,
        EXP_PATH.with_suffix(".ini"),
        EXP_PATH.with_suffix(".ssn"),
        EXP_PATH.with_suffix(".raw"),
        ANALYZER_SOURCE,
        Path(__file__).resolve(),
    ]
    source_rows = []
    descriptions = {
        ".exp": "PACKWIN experiment definition and group/subject mapping",
        ".ini": "PACKWIN protocol and freezing-analysis parameters",
        ".ssn": "PACKWIN per-session metadata",
        ".raw": "PACKWIN activity signal and event records",
        ".py": "Analysis or plotting source code",
    }
    for path in source_files:
        source_rows.append(
            {
                "role": "script" if path.suffix == ".py" else "raw",
                "path": str(path),
                "description": descriptions[path.suffix],
                "sha256": sha256_file(path),
            }
        )
    if DATE_AUDIT.exists():
        source_rows.append(
            {
                "role": "audit",
                "path": str(DATE_AUDIT),
                "description": "Date-correction audit; documents unchanged RAW sample blocks",
                "sha256": sha256_file(DATE_AUDIT),
            }
        )
    pd.DataFrame(source_rows).to_csv(OUTDIR / "source_paths.csv", index=False)

    overall_p = float(
        overall.loc[overall["test"] == "One-way ANOVA", "p_value"].iloc[0]
    )
    tukey_significant = tukey.loc[tukey["reject"].astype(bool)]
    factorial_effects = factorial.loc[factorial["term"] != "Residual"]
    group_lines = [
        f"- {row.group}: n={int(row.n)}, {row.mean_freezing_pct:.2f} ± {row.sem_freezing_pct:.2f}% (mean ± SEM)"
        for row in summary.itertuples(index=False)
    ]
    factor_lines = [
        f"- {row.term}: F={row.F:.4f}, {p_text(row.p_value)}, partial η²={row.partial_eta_squared:.4f}"
        for row in factorial_effects.itertuples(index=False)
    ]
    if tukey_significant.empty:
        tukey_conclusion = "No Tukey-adjusted pairwise comparison reached P < 0.05."
    else:
        tukey_conclusion = "; ".join(
            f"{row.group_1} vs {row.group_2}: adjusted {p_text(float(row.p_adjusted))}"
            for row in tukey_significant.itertuples(index=False)
        )
    threshold_values = [float(value) for value in sorted(frame["activity_threshold_pct"].unique())]
    freezing_ms_values = [int(value) for value in sorted(frame["minimum_freezing_ms"].unique())]
    sample_rates = [int(value) for value in sorted(frame["sample_rate_hz"].unique())]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    readme = f"""# FC contextual freezing: four-group analysis

## Purpose

Compare animal-level contextual freezing among CON, SEV, ASD, and ASD+SEV groups in the 2026-03-17 fear-conditioning contextual test.

## Sample definition

- Biological replicate: one animal / one PACKWIN session.
- Total n={len(frame)}; no exclusions were applied.
- Session duration: {frame['duration_s'].min():.0f} s for every animal.
- Recorded activity threshold(s): {threshold_values}%.
- Recorded minimum freezing duration(s): {freezing_ms_values} ms.
- Sample rate(s): {sample_rates} Hz.

## Group results

{chr(10).join(group_lines)}

## Statistical results

- Primary four-group test: one-way ANOVA, F({int(overall.iloc[0].df1)}, {int(overall.iloc[0].df2)})={overall.iloc[0].statistic:.4f}, {p_text(overall_p)}, η²={overall.iloc[0].effect_size:.4f}, ω²={overall.iloc[0].secondary_effect_size:.4f}.
- {tukey_conclusion}
- Welch ANOVA and Kruskal-Wallis sensitivity results are stored in `statistics_overall.csv`.
- Type-III 2×2 factorial ANOVA with sum contrasts (used because cell sizes are 7/6/7/6):
{chr(10).join(factor_lines)}

Interpretation: the dataset does not provide statistically significant evidence of a contextual-freezing difference among the four groups at α=0.05. This is not proof that the groups are identical; individual values and effect sizes are retained in the source/statistics tables.

## Analysis steps

1. Parsed the PACKWIN `.exp/.ssn/.ini/.raw` quartet with the local PACKWIN analyzer.
2. Recomputed freezing per animal with each recorded session's threshold and minimum-freezing duration.
3. Used freezing percentage over the full 180-s contextual session as the primary endpoint.
4. Computed group mean, SD, SEM, 95% CI, median, IQR, and range.
5. Ran one-way ANOVA, Tukey HSD, assumption checks, Welch ANOVA, Kruskal-Wallis sensitivity analysis, and a Type-III 2×2 ASD×SEV factorial ANOVA.
6. Plotted mean ± SEM with every animal shown as a hollow point.

## Data and scripts

- Raw experiment: `{EXP_PATH}`
- Derived animal-level table: `{OUTDIR / 'animal_level_freezing.csv'}`
- Analysis script: `{Path(__file__).resolve()}`
- Figure provenance: `{PROVENANCE_DIR / (FIGURE_ID + '_provenance.md')}`
- Source file hashes: `{OUTDIR / 'source_paths.csv'}`

## Outputs

- Vector: `FC_contextual_freezing_4groups.pdf`, `.svg`
- Illustrator-clean vector: `FC_contextual_freezing_4groups_ai_clean.pdf`, `.svg`
- Preview: `FC_contextual_freezing_4groups.png`
- Tables: animal-level source, group summary, overall statistics, Tukey comparisons, and factorial ANOVA.

## Notes

- The internal recording dates were previously corrected from 2024 to 2026. The correction audit is listed in `source_paths.csv`; RAW sample blocks were unchanged.
- Status: draft analysis; generated {generated_at}.
"""
    (OUTDIR / "README_analysis.md").write_text(readme, encoding="utf-8")

    table_outputs = [
        OUTDIR / "animal_level_freezing.csv",
        OUTDIR / "group_summary.csv",
        OUTDIR / "statistics_overall.csv",
        OUTDIR / "tukey_pairwise.csv",
        OUTDIR / "two_way_anova.csv",
        OUTDIR / "source_paths.csv",
        OUTDIR / "README_analysis.md",
    ]
    provenance_outputs = [
        PROVENANCE_DIR / f"{FIGURE_ID}_provenance.md",
        PROVENANCE_DIR / f"{FIGURE_ID}_source_data.csv",
        PROVENANCE_DIR / f"{FIGURE_ID}_stats.csv",
    ]
    table_outputs.extend(path for path in provenance_outputs if path.exists())
    format_notes = {
        ".pdf": "Editable vector PDF",
        ".svg": "Editable SVG with live text",
        ".png": "600-dpi raster preview",
        ".csv": "UTF-8 comma-separated data table",
        ".md": "Human-readable analysis record",
    }
    manifest_rows = []
    for path in figure_outputs + table_outputs:
        manifest_rows.append(
            {
                "path": str(path),
                "format_notes": format_notes[path.suffix],
                "sha256": sha256_file(path),
            }
        )
    manifest_rows.append(
        {
            "path": str(Path(__file__).resolve()),
            "format_notes": "Reproducible analysis and plotting script",
            "sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    pd.DataFrame(manifest_rows).to_csv(OUTDIR / "outputs_manifest.csv", index=False)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_animal_data()
    summary = group_summary(frame)
    overall, tukey, factorial, _ = run_statistics(frame)

    animal_path = OUTDIR / "animal_level_freezing.csv"
    group_path = OUTDIR / "group_summary.csv"
    overall_path = OUTDIR / "statistics_overall.csv"
    tukey_path = OUTDIR / "tukey_pairwise.csv"
    factorial_path = OUTDIR / "two_way_anova.csv"
    frame.to_csv(animal_path, index=False, float_format="%.8f")
    summary.to_csv(group_path, index=False, float_format="%.8f")
    overall.to_csv(overall_path, index=False, float_format="%.10g")
    tukey.to_csv(tukey_path, index=False, float_format="%.10g")
    factorial.to_csv(factorial_path, index=False, float_format="%.10g")

    provenance_source = PROVENANCE_DIR / f"{FIGURE_ID}_source_data.csv"
    provenance_stats = PROVENANCE_DIR / f"{FIGURE_ID}_stats.csv"
    shutil.copy2(animal_path, provenance_source)
    combined_stats = pd.concat(
        [
            overall.assign(section="overall"),
            factorial.assign(section="factorial"),
            tukey.assign(section="tukey_pairwise"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_stats.to_csv(provenance_stats, index=False, float_format="%.10g")

    one_way_p = float(
        overall.loc[overall["test"] == "One-way ANOVA", "p_value"].iloc[0]
    )
    figure_outputs = draw_figure(frame, summary, one_way_p)
    write_metadata(frame, summary, overall, tukey, factorial, figure_outputs)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(factorial.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(tukey.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
