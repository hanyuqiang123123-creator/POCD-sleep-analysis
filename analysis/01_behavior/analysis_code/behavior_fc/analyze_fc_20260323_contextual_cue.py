from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_fc_contextual_4groups import (  # noqa: E402
    FACTOR_MAP,
    GROUP_ORDER,
    PALETTE,
    configure_style,
    group_summary,
    p_text,
    run_statistics,
    sha256_file,
)


FIGURE_ID = "fc_contextual_cue_20260323_four_groups"
DATA_ROOT = Path(r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\20260323-TEST")
CONTEXTUAL_EXP = DATA_ROOT / "CONTEXTUAL.exp"
CUE_EXP = DATA_ROOT / "新建文件夹" / "CUE.exp"
ANALYZER_DIR = Path(r"F:\hyq\软件\PACKWIN_Analyzer")
ANALYZER_SOURCE = ANALYZER_DIR / "packwin_analyzer.py"
HELPER_SOURCE = SCRIPT_DIR / "analyze_fc_contextual_4groups.py"
OUTDIR = Path(r"F:\Sleep\Figure\FC_contextual_cue_4groups_20260323_20260828")
PROVENANCE_DIR = Path(r"F:\Sleep\outputs\figure_provenance")
DATE_AUDIT = Path(
    r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\_date_backup_20260323-TEST_20240323_to_20260323_20260828-203524\date_correction_audit.json"
)

ENDPOINTS = [
    "Contextual total (0-180 s)",
    "CUE total (0-270 s)",
    "CUE State 1 (0-180 s)",
    "CUE State 2 (180-210 s)",
    "CUE State 3 (210-270 s)",
]
MAIN_ENDPOINTS = [
    "Contextual total (0-180 s)",
    "CUE State 2 (180-210 s)",
]
CUE_STATE_ENDPOINTS = [
    "CUE State 1 (0-180 s)",
    "CUE State 2 (180-210 s)",
    "CUE State 3 (210-270 s)",
]


def add_row(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    endpoint: str,
    raw_id: int,
    group: str,
    subject: str,
    timestamp: str,
    start_s: float,
    end_s: float,
    freezing_s: float,
    freezing_pct: float,
    episodes: int,
    sample_rate: int,
    threshold: float,
    minimum_freezing_ms: int,
    protocol: str,
) -> None:
    normalized_group = group.upper()
    if normalized_group not in FACTOR_MAP:
        raise ValueError(f"Unexpected group label: {group}")
    asd_factor, sev_factor = FACTOR_MAP[normalized_group]
    rows.append(
        {
            "dataset": dataset,
            "endpoint": endpoint,
            "raw_id": raw_id,
            "group": normalized_group,
            "subject": subject,
            "subject_key": f"{dataset}:{normalized_group}:{subject}",
            "timestamp": timestamp,
            "ASD_factor": asd_factor,
            "SEV_factor": sev_factor,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "freezing_s": freezing_s,
            "freezing_pct": freezing_pct,
            "freezing_episodes": episodes,
            "sample_rate_hz": sample_rate,
            "activity_threshold_pct": threshold,
            "minimum_freezing_ms": minimum_freezing_ms,
            "protocol": protocol,
        }
    )


def load_long_data() -> pd.DataFrame:
    for path in (CONTEXTUAL_EXP, CUE_EXP, ANALYZER_SOURCE, HELPER_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(ANALYZER_DIR))
    from packwin_analyzer import Experiment

    rows: list[dict[str, object]] = []
    contextual = Experiment(CONTEXTUAL_EXP).analyze()
    cue = Experiment(CUE_EXP).analyze()

    for result in contextual:
        add_row(
            rows,
            dataset="CONTEXTUAL",
            endpoint=ENDPOINTS[0],
            raw_id=result.raw_id,
            group=result.group,
            subject=result.subject,
            timestamp=result.timestamp,
            start_s=0.0,
            end_s=result.duration_s,
            freezing_s=result.freezing_s,
            freezing_pct=result.freezing_pct,
            episodes=result.episodes,
            sample_rate=result.sample_rate,
            threshold=result.lower,
            minimum_freezing_ms=result.freezing_ms,
            protocol=result.protocol,
        )

    cue_endpoint_by_state = {
        "State 1": ENDPOINTS[2],
        "State 2": ENDPOINTS[3],
        "State 3": ENDPOINTS[4],
    }
    for result in cue:
        add_row(
            rows,
            dataset="CUE",
            endpoint=ENDPOINTS[1],
            raw_id=result.raw_id,
            group=result.group,
            subject=result.subject,
            timestamp=result.timestamp,
            start_s=0.0,
            end_s=result.duration_s,
            freezing_s=result.freezing_s,
            freezing_pct=result.freezing_pct,
            episodes=result.episodes,
            sample_rate=result.sample_rate,
            threshold=result.lower,
            minimum_freezing_ms=result.freezing_ms,
            protocol=result.protocol,
        )
        if [state.name for state in result.states] != ["State 1", "State 2", "State 3"]:
            raise ValueError(f"Unexpected CUE state layout for {result.subject}")
        for state in result.states:
            add_row(
                rows,
                dataset="CUE",
                endpoint=cue_endpoint_by_state[state.name],
                raw_id=result.raw_id,
                group=result.group,
                subject=result.subject,
                timestamp=result.timestamp,
                start_s=state.start_s,
                end_s=state.end_s,
                freezing_s=state.freezing_s,
                freezing_pct=state.freezing_pct,
                episodes=state.episodes,
                sample_rate=result.sample_rate,
                threshold=result.lower,
                minimum_freezing_ms=result.freezing_ms,
                protocol=result.protocol,
            )

    frame = pd.DataFrame(rows)
    frame["group"] = pd.Categorical(frame["group"], GROUP_ORDER, ordered=True)
    frame["endpoint"] = pd.Categorical(frame["endpoint"], ENDPOINTS, ordered=True)
    frame = frame.sort_values(["endpoint", "group", "raw_id"]).reset_index(drop=True)

    expected_counts = {"CON": 7, "SEV": 8, "ASD": 7, "ASD+SEV": 8}
    for endpoint in ENDPOINTS:
        subset = frame.loc[frame["endpoint"] == endpoint]
        observed = {
            str(group): int(count)
            for group, count in subset.groupby("group", observed=True, sort=False).size().items()
        }
        if observed != expected_counts:
            raise ValueError(f"{endpoint}: expected {expected_counts}, found {observed}")
        if subset["subject_key"].duplicated().any():
            raise ValueError(f"Duplicate animal rows within {endpoint}")
    return frame


def analyze_endpoints(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    overall_tables: list[pd.DataFrame] = []
    tukey_tables: list[pd.DataFrame] = []
    factorial_tables: list[pd.DataFrame] = []
    nonparametric_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        subset = frame.loc[frame["endpoint"] == endpoint].copy()
        summary = group_summary(subset)
        overall, tukey, factorial, _ = run_statistics(subset)
        summary.insert(0, "endpoint", endpoint)
        overall = overall.rename(columns={"test": "statistical_test"})
        overall.insert(0, "endpoint", endpoint)
        tukey.insert(0, "endpoint", endpoint)
        factorial.insert(0, "endpoint", endpoint)
        summaries.append(summary)
        overall_tables.append(overall)
        tukey_tables.append(tukey)
        factorial_tables.append(factorial)
        endpoint_pair_rows: list[dict[str, object]] = []
        for group_1, group_2 in combinations(GROUP_ORDER, 2):
            values_1 = subset.loc[subset["group"] == group_1, "freezing_pct"].to_numpy(float)
            values_2 = subset.loc[subset["group"] == group_2, "freezing_pct"].to_numpy(float)
            u_statistic, raw_p = scipy_stats.mannwhitneyu(
                values_1, values_2, alternative="two-sided", method="auto"
            )
            rank_biserial = 2.0 * float(u_statistic) / (len(values_1) * len(values_2)) - 1.0
            endpoint_pair_rows.append(
                {
                    "endpoint": endpoint,
                    "group_1": group_1,
                    "group_2": group_2,
                    "n_group_1": len(values_1),
                    "n_group_2": len(values_2),
                    "mann_whitney_U_group_1": float(u_statistic),
                    "p_value": float(raw_p),
                    "rank_biserial_group_1_minus_group_2": rank_biserial,
                }
            )
        adjusted = multipletests(
            [row["p_value"] for row in endpoint_pair_rows], method="holm"
        )[1]
        for row, adjusted_p in zip(endpoint_pair_rows, adjusted, strict=True):
            row["p_holm_within_endpoint"] = float(adjusted_p)
            row["significant_0_05"] = bool(adjusted_p < 0.05)
        nonparametric_rows.extend(endpoint_pair_rows)

    summary_all = pd.concat(summaries, ignore_index=True)
    overall_all = pd.concat(overall_tables, ignore_index=True)
    tukey_all = pd.concat(tukey_tables, ignore_index=True)
    factorial_all = pd.concat(factorial_tables, ignore_index=True)

    overall_all["p_holm_across_5_endpoints"] = np.nan
    one_way_mask = overall_all["statistical_test"] == "One-way ANOVA"
    one_way_p = overall_all.loc[one_way_mask, "p_value"].to_numpy(float)
    overall_all.loc[one_way_mask, "p_holm_across_5_endpoints"] = multipletests(
        one_way_p, method="holm"
    )[1]

    factorial_all["p_holm_across_5_endpoints_within_term"] = np.nan
    for term in ["ASD main effect", "SEV main effect", "ASD x SEV interaction"]:
        term_mask = factorial_all["term"] == term
        term_p = factorial_all.loc[term_mask, "p_value"].to_numpy(float)
        factorial_all.loc[
            term_mask, "p_holm_across_5_endpoints_within_term"
        ] = multipletests(term_p, method="holm")[1]
    return (
        summary_all,
        overall_all,
        tukey_all,
        factorial_all,
        pd.DataFrame(nonparametric_rows),
    )


def endpoint_p(overall: pd.DataFrame, endpoint: str, statistical_test: str) -> float:
    row = overall.loc[
        (overall["endpoint"] == endpoint)
        & (overall["statistical_test"] == statistical_test)
    ]
    return float(row["p_value"].iloc[0])


def draw_bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    endpoint: str,
    title: str,
    panel_label: str,
    p_value: float,
    test_label: str,
) -> None:
    data = frame.loc[frame["endpoint"] == endpoint]
    stats = summary.loc[summary["endpoint"] == endpoint].set_index("group")
    x_positions = np.arange(len(GROUP_ORDER), dtype=float)
    means = stats.loc[GROUP_ORDER, "mean_freezing_pct"].to_numpy(float)
    sems = stats.loc[GROUP_ORDER, "sem_freezing_pct"].to_numpy(float)
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
        values = (
            data.loc[data["group"] == group]
            .sort_values("raw_id")["freezing_pct"]
            .to_numpy(float)
        )
        jitter = np.linspace(-0.115, 0.115, len(values))
        ax.scatter(
            x_value + jitter,
            values,
            s=20,
            facecolors="white",
            edgecolors=PALETTE[group]["edge"],
            linewidths=0.85,
            alpha=0.9,
            zorder=5,
        )
    counts = stats.loc[GROUP_ORDER, "n"].astype(int).tolist()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{group}\n(n={n_value})" for group, n_value in zip(GROUP_ORDER, counts, strict=True)]
    )
    ax.set_title(title, fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5,
        0.97,
        f"{test_label}: {p_text(p_value)}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
    )
    ax.text(
        -0.13,
        1.07,
        panel_label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    ax.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(False)


def save_figure_set(fig: plt.Figure, base_name: str) -> list[Path]:
    base = OUTDIR / base_name
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".svg"), base.with_suffix(".png")]
    fig.savefig(outputs[0], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(outputs[1], facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(
        outputs[2], dpi=600, facecolor="white", edgecolor="none", bbox_inches="tight"
    )
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
        if hasattr(artist, "set_clip_path"):
            artist.set_clip_path(None)
    clean_pdf = OUTDIR / f"{base_name}_ai_clean.pdf"
    clean_svg = OUTDIR / f"{base_name}_ai_clean.svg"
    fig.savefig(clean_pdf, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(clean_svg, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return outputs + [clean_pdf, clean_svg]


def draw_figures(
    frame: pd.DataFrame, summary: pd.DataFrame, overall: pd.DataFrame
) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.35), sharey=True)
    draw_bar_panel(
        axes[0],
        frame,
        summary,
        MAIN_ENDPOINTS[0],
        "Contextual freezing (0–180 s)",
        "A",
        endpoint_p(overall, MAIN_ENDPOINTS[0], "Kruskal-Wallis sensitivity analysis"),
        "Kruskal–Wallis",
    )
    draw_bar_panel(
        axes[1],
        frame,
        summary,
        MAIN_ENDPOINTS[1],
        "CUE State 2 freezing (180–210 s)",
        "B",
        endpoint_p(overall, MAIN_ENDPOINTS[1], "Kruskal-Wallis sensitivity analysis"),
        "Kruskal–Wallis",
    )
    axes[0].set_ylabel("Freezing (%)", fontsize=6.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.88, wspace=0.22)
    main_outputs = save_figure_set(fig, "FC_contextual_cue_primary_4groups")

    profile_fig, ax = plt.subplots(figsize=(3.25, 2.35))
    x_positions = np.arange(3, dtype=float)
    for group in GROUP_ORDER:
        group_summary_rows = (
            summary.loc[
                summary["endpoint"].isin(CUE_STATE_ENDPOINTS)
                & (summary["group"] == group)
            ]
            .set_index("endpoint")
            .loc[CUE_STATE_ENDPOINTS]
        )
        means = group_summary_rows["mean_freezing_pct"].to_numpy(float)
        sems = group_summary_rows["sem_freezing_pct"].to_numpy(float)
        ax.errorbar(
            x_positions,
            means,
            yerr=sems,
            color=PALETTE[group]["edge"],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=PALETTE[group]["edge"],
            markeredgewidth=0.85,
            markersize=4.0,
            linewidth=1.0,
            elinewidth=0.8,
            capsize=2.0,
            label=group,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["State 1\n0–180 s", "State 2\n180–210 s", "State 3\n210–270 s"])
    ax.set_ylabel("Freezing (%)", fontsize=6.5)
    ax.set_title("CUE phase profile", fontsize=7.2, fontweight="bold", pad=5)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=6)
    ax.legend(fontsize=5.5, ncol=2, loc="lower right", handlelength=1.5)
    ax.grid(False)
    profile_fig.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.88)
    profile_outputs = save_figure_set(profile_fig, "FC_cue_phase_profile_4groups")
    return main_outputs + profile_outputs


def write_records(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    tukey: pd.DataFrame,
    factorial: pd.DataFrame,
    nonparametric_pairwise: pd.DataFrame,
    figure_outputs: list[Path],
) -> None:
    source_files = []
    for exp_path in (CONTEXTUAL_EXP, CUE_EXP):
        source_files.extend(exp_path.with_suffix(suffix) for suffix in (".exp", ".ini", ".ssn", ".raw"))
    source_files.extend([ANALYZER_SOURCE, HELPER_SOURCE, Path(__file__).resolve()])
    descriptions = {
        ".exp": "PACKWIN experiment definition and group/subject mapping",
        ".ini": "PACKWIN protocol and freezing-analysis parameters",
        ".ssn": "PACKWIN per-session metadata",
        ".raw": "PACKWIN activity signal and event records",
        ".py": "Analysis or plotting source code",
    }
    source_rows = [
        {
            "role": "script" if path.suffix == ".py" else "raw",
            "path": str(path),
            "description": descriptions[path.suffix],
            "sha256": sha256_file(path),
        }
        for path in source_files
    ]
    if DATE_AUDIT.exists():
        source_rows.append(
            {
                "role": "audit",
                "path": str(DATE_AUDIT),
                "description": "Date-correction audit documenting unchanged RAW sample blocks",
                "sha256": sha256_file(DATE_AUDIT),
            }
        )
    pd.DataFrame(source_rows).to_csv(OUTDIR / "source_paths.csv", index=False)

    summary_index = summary.set_index(["endpoint", "group"])
    group_sections = []
    for endpoint in MAIN_ENDPOINTS:
        group_sections.append(f"### {endpoint}")
        for group in GROUP_ORDER:
            row = summary_index.loc[(endpoint, group)]
            group_sections.append(
                f"- {group}: n={int(row['n'])}, {row['mean_freezing_pct']:.2f} ± {row['sem_freezing_pct']:.2f}% (mean ± SEM)"
            )

    endpoint_stat_lines = []
    for endpoint in ENDPOINTS:
        anova_row = overall.loc[
            (overall["endpoint"] == endpoint)
            & (overall["statistical_test"] == "One-way ANOVA")
        ].iloc[0]
        kruskal_row = overall.loc[
            (overall["endpoint"] == endpoint)
            & (overall["statistical_test"] == "Kruskal-Wallis sensitivity analysis")
        ].iloc[0]
        endpoint_stat_lines.append(
            f"- {endpoint}: Kruskal–Wallis H={kruskal_row['statistic']:.4f}, {p_text(kruskal_row['p_value'])}; one-way ANOVA F({int(anova_row['df1'])}, {int(anova_row['df2'])})={anova_row['statistic']:.4f}, {p_text(anova_row['p_value'])}; ANOVA Holm-adjusted P={anova_row['p_holm_across_5_endpoints']:.4f}."
        )

    tukey_primary = tukey.loc[tukey["endpoint"].isin(MAIN_ENDPOINTS)]
    significant_primary = tukey_primary.loc[tukey_primary["reject"].astype(bool)]
    tukey_note = (
        "No Tukey-adjusted pairwise comparison was significant for either primary panel."
        if significant_primary.empty
        else "; ".join(
            f"{row.endpoint}: {row.group_1} vs {row.group_2}, adjusted P={row.p_adjusted:.4f}"
            for row in significant_primary.itertuples(index=False)
        )
    )
    nonparametric_primary = nonparametric_pairwise.loc[
        nonparametric_pairwise["endpoint"].isin(MAIN_ENDPOINTS)
    ]
    nonparametric_note = (
        "No Holm-adjusted pairwise Mann–Whitney comparison was significant for either primary panel."
        if not nonparametric_primary["significant_0_05"].any()
        else "One or more Holm-adjusted pairwise Mann–Whitney comparisons were significant; see `pairwise_mannwhitney_holm.csv`."
    )
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    readme = f"""# FC 2026-03-23 contextual and cued freezing

## Purpose

Compare animal-level freezing among CON, SEV, ASD, and ASD+SEV groups in the contextual and cued fear-conditioning datasets.

## Sample definition

- Biological replicate: one animal / one PACKWIN session.
- Each dataset: CON n=7, SEV n=8, ASD n=7, ASD+SEV n=8; total n=30; no exclusions.
- CONTEXTUAL duration: 180 s. CUE duration: 270 s.
- CUE states: State 1 0–180 s, State 2 180–210 s, State 3 210–270 s.
- Recorded activity threshold: 12%; minimum freezing duration: 1000 ms.
- CONTEXTUAL and CUE subject labels/group ordering differ, so cross-file animal pairing was not assumed.

## Primary panel group results

{chr(10).join(group_sections)}

## Omnibus statistical results

{chr(10).join(endpoint_stat_lines)}

- {nonparametric_note}
- Parametric sensitivity analysis: {tukey_note}
- Full assumption checks, Welch ANOVA, Kruskal-Wallis sensitivity tests, Tukey comparisons, and 2×2 factorial ANOVA results are stored in the CSV tables.

## Interpretation

The report separates the two experiments and uses animal-level values. CUE State 2 is displayed as the putative cue interval because it is the only 30-s phase in the CUED protocol, but the source labels are generic (`State 1–3`); confirm this phase identity before manuscript submission. The main panels use Kruskal–Wallis inference because their ANOVA residuals were non-normal and values showed ceiling effects. The CUE phase-profile plot is descriptive, while group comparisons are tested separately within each endpoint.

## Analysis steps

1. Parsed both PACKWIN `.exp/.ssn/.ini/.raw` quartets with the local PACKWIN analyzer.
2. Recomputed freezing with the recorded 12% activity threshold and 1000-ms minimum freezing duration.
3. Calculated CONTEXTUAL total, CUE total, and State 1–3 animal-level freezing percentages.
4. For each endpoint, ran Kruskal–Wallis with Holm-adjusted pairwise Mann–Whitney tests, plus one-way ANOVA, Tukey HSD, residual Shapiro-Wilk, median-centered Levene, Welch ANOVA, and Type-III 2×2 ASD×SEV ANOVA as parametric/exploratory analyses.
5. Holm-adjusted the five one-way omnibus P values; factorial P values were Holm-adjusted within each term across endpoints.
6. Plotted mean ± SEM and all animals for the two primary panels; added a descriptive CUE phase profile.

## Data, script, and provenance

- Animal-level data: `{OUTDIR / 'animal_level_all_endpoints.csv'}`
- Group summary: `{OUTDIR / 'endpoint_group_summary.csv'}`
- Analysis script: `{Path(__file__).resolve()}`
- Figure provenance: `{PROVENANCE_DIR / (FIGURE_ID + '_provenance.md')}`
- Source hashes: `{OUTDIR / 'source_paths.csv'}`

## Outputs

- Primary two-panel figure: `FC_contextual_cue_primary_4groups.*`
- CUE phase profile: `FC_cue_phase_profile_4groups.*`
- PDF, SVG, 600-dpi PNG, and Illustrator-clean PDF/SVG are provided.
- Statistical tables include overall tests, Tukey comparisons, and factorial ANOVA.

## Notes

- The internal recording dates were corrected from 2024 to 2026; the audit listed in `source_paths.csv` documents unchanged RAW sample blocks.
- Factorial ASD×SEV structure is inferred from group names and should be confirmed.
- Status: draft analysis; generated {generated_at}.
"""
    (OUTDIR / "README_analysis.md").write_text(readme, encoding="utf-8")

    table_outputs = [
        OUTDIR / "animal_level_all_endpoints.csv",
        OUTDIR / "endpoint_group_summary.csv",
        OUTDIR / "statistics_overall.csv",
        OUTDIR / "tukey_pairwise.csv",
        OUTDIR / "pairwise_mannwhitney_holm.csv",
        OUTDIR / "factorial_anova.csv",
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
        ".py": "Reproducible analysis and plotting script",
    }
    output_paths = figure_outputs + table_outputs + [Path(__file__).resolve(), HELPER_SOURCE]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path),
                "format_notes": format_notes[path.suffix],
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ]
    )
    manifest.to_csv(OUTDIR / "outputs_manifest.csv", index=False)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_long_data()
    summary, overall, tukey, factorial, nonparametric_pairwise = analyze_endpoints(frame)

    animal_path = OUTDIR / "animal_level_all_endpoints.csv"
    summary_path = OUTDIR / "endpoint_group_summary.csv"
    overall_path = OUTDIR / "statistics_overall.csv"
    tukey_path = OUTDIR / "tukey_pairwise.csv"
    nonparametric_path = OUTDIR / "pairwise_mannwhitney_holm.csv"
    factorial_path = OUTDIR / "factorial_anova.csv"
    frame.to_csv(animal_path, index=False, float_format="%.8f")
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    overall.to_csv(overall_path, index=False, float_format="%.10g")
    tukey.to_csv(tukey_path, index=False, float_format="%.10g")
    nonparametric_pairwise.to_csv(
        nonparametric_path, index=False, float_format="%.10g"
    )
    factorial.to_csv(factorial_path, index=False, float_format="%.10g")

    shutil.copy2(animal_path, PROVENANCE_DIR / f"{FIGURE_ID}_source_data.csv")
    combined_stats = pd.concat(
        [
            overall.assign(section="overall"),
            factorial.assign(section="factorial"),
            tukey.assign(section="tukey_pairwise"),
            nonparametric_pairwise.assign(section="mannwhitney_pairwise"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_stats.to_csv(
        PROVENANCE_DIR / f"{FIGURE_ID}_stats.csv", index=False, float_format="%.10g"
    )

    figure_outputs = draw_figures(frame, summary, overall)
    write_records(
        frame,
        summary,
        overall,
        tukey,
        factorial,
        nonparametric_pairwise,
        figure_outputs,
    )

    print("PRIMARY GROUP SUMMARY")
    print(
        summary.loc[summary["endpoint"].isin(MAIN_ENDPOINTS)].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nONE-WAY ANOVA BY ENDPOINT")
    print(
        overall.loc[overall["statistical_test"] == "One-way ANOVA"].to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    print("\nFACTORIAL ANOVA FOR PRIMARY ENDPOINTS")
    print(
        factorial.loc[
            factorial["endpoint"].isin(MAIN_ENDPOINTS)
            & (factorial["term"] != "Residual")
        ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
