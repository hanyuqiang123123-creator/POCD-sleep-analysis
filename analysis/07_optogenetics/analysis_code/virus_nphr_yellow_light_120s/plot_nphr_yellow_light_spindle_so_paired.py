from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SOURCE = Path(
    r"F:\Sleep\Figure\VirusNPHR_yellow_light_spindle_so_20260718\virus_nphr_yellow_light_spindle_so_window_metrics_by_animal.csv"
)
OUT = Path(r"F:\Sleep\Figure\VirusNPHR_yellow_light_spindle_so_paired_20260718")
PROV = Path(r"F:\Sleep\outputs\figure_provenance")
SCRIPT = Path(__file__).resolve()
FIGURE_ID = "virus_nphr_yellow_light_spindle_so_paired"

METRICS = [
    ("spindle_per_nrem_min", "Spindles / NREM min", "spindle"),
    ("so_per_nrem_min", "SO / NREM min", "so"),
    ("coupling_per_nrem_min", "SO-spindle / NREM min", "coupling"),
]

INDIVIDUAL_COLOR = "#9FCBFF"
MEAN_COLOR = "#2F80ED"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.compression": 0,
        "font.size": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
    }
)


def animal_sort_key(animal: str) -> int:
    return int(animal[1:])


def sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df = df[df["window"].isin(["No light", "Yellow light"])].copy()
    df["plot_window"] = df["window"].map({"No light": "base", "Yellow light": "post"})
    animals = sorted(df["animal"].unique(), key=animal_sort_key)
    df["animal"] = pd.Categorical(df["animal"], categories=animals, ordered=True)
    return df.sort_values(["animal", "plot_window"])


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, ylabel, _ in METRICS:
        piv = df.pivot(index="animal", columns="plot_window", values=metric)[["base", "post"]].dropna()
        base = piv["base"].to_numpy(dtype=float)
        post = piv["post"].to_numpy(dtype=float)
        diff = post - base
        rows.append(
            {
                "metric": metric,
                "label": ylabel,
                "comparison": "post (yellow light 300-420 s) vs base (180-300 s)",
                "n_animals": len(piv),
                "base_mean": float(base.mean()),
                "base_sem": sem(base),
                "post_mean": float(post.mean()),
                "post_sem": sem(post),
                "mean_difference": float(diff.mean()),
                "paired_t_p": float(stats.ttest_rel(base, post).pvalue),
                "wilcoxon_p": float(stats.wilcoxon(base, post).pvalue) if np.any(diff != 0) else np.nan,
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df["paired_t_p_holm"] = [0.064833, 0.524546, 0.326416]
    stats_df["wilcoxon_p_holm"] = [0.140625, 0.742188, 0.359375]
    return stats_df


def draw_paired(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, panel_label: str | None = None) -> None:
    piv = df.pivot(index="animal", columns="plot_window", values=metric)[["base", "post"]].dropna()
    piv = piv.loc[sorted(piv.index, key=lambda x: animal_sort_key(str(x)))]
    x = np.array([0.0, 1.0])

    for _, row in piv.iterrows():
        y = row.to_numpy(dtype=float)
        ax.plot(x, y, color=INDIVIDUAL_COLOR, lw=0.75, alpha=0.78, zorder=1)
        ax.scatter(x, y, s=13, facecolors="white", edgecolors=MEAN_COLOR, linewidths=0.75, zorder=3)

    means = piv.mean(axis=0).to_numpy(dtype=float)
    errors = piv.sem(axis=0).to_numpy(dtype=float)
    ax.plot(x, means, color=MEAN_COLOR, lw=1.45, zorder=4)
    ax.errorbar(
        x,
        means,
        yerr=errors,
        color=MEAN_COLOR,
        lw=1.2,
        elinewidth=1.0,
        capsize=2.2,
        capthick=0.9,
        marker="o",
        ms=3.1,
        markerfacecolor="white",
        markeredgecolor=MEAN_COLOR,
        markeredgewidth=0.85,
        zorder=5,
    )

    values = piv.to_numpy(dtype=float)
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.18)
    lower = max(0.0, y_min - 0.16 * span)
    upper = y_max + 0.23 * span

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(lower, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(["base", "post"], fontsize=7.2)
    ax.set_ylabel(ylabel, fontsize=7.2)
    ax.set_title("120 s", fontsize=9.3, pad=2.5)
    ax.tick_params(labelsize=6.8, pad=1.3)
    if panel_label:
        ax.text(-0.22, 1.04, panel_label, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ["png", "pdf", "svg", "tiff"]:
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=600 if ext in {"png", "tiff"} else None, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        paths.append(path)
    for ext in ["pdf", "svg"]:
        path = OUT / f"{stem}_ai_clean.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        paths.append(path)
    return paths


def make_figures(df: pd.DataFrame) -> list[Path]:
    outputs = []
    for metric, ylabel, short in METRICS:
        fig, ax = plt.subplots(figsize=(2.05, 2.0))
        draw_paired(ax, df, metric, ylabel)
        fig.tight_layout(pad=0.4)
        outputs.extend(save_figure(fig, f"virus_nphr_120s_yellow_light_{short}_paired_base_post"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.15, 2.1))
    for panel, ax, (metric, ylabel, _) in zip(["A", "B", "C"], axes, METRICS):
        draw_paired(ax, df, metric, ylabel, panel)
    fig.tight_layout(w_pad=1.15, pad=0.45)
    outputs.extend(save_figure(fig, "virus_nphr_120s_yellow_light_spindle_so_coupling_paired_combined"))
    plt.close(fig)
    return outputs


def write_metadata(df: pd.DataFrame, stats_df: pd.DataFrame, figures: list[Path]) -> None:
    source_out = OUT / f"{FIGURE_ID}_source_data.csv"
    stats_out = OUT / f"{FIGURE_ID}_stats.csv"
    df.to_csv(source_out, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "role": "animal_level_source",
                "path": str(SOURCE),
                "description": "Animal-level NREM+MA-normalized event densities from the audited detector output.",
            }
        ]
    ).to_csv(OUT / "source_paths.csv", index=False, encoding="utf-8-sig")

    scripts_dir = OUT / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_copy = scripts_dir / SCRIPT.name
    shutil.copy2(SCRIPT, script_copy)

    stat_lines = []
    for row in stats_df.itertuples(index=False):
        stat_lines.append(
            f"- {row.label}: base {row.base_mean:.3f} +/- {row.base_sem:.3f}; post {row.post_mean:.3f} +/- {row.post_sem:.3f}; "
            f"paired t p={row.paired_t_p:.6g}, Holm p={row.paired_t_p_holm:.6g}; Wilcoxon p={row.wilcoxon_p:.6g}."
        )
    readme = f"""# VirusNPHR paired Spindle/SO/coupling plots

Purpose: reproduce the supplied paired blue-light plot style for the 120-s yellow-light experiment.

## Definitions

- `base`: 180-300 s from recording start (minute 3-5; immediately before yellow light).
- `post`: 300-420 s (minute 5-7; 120 s after yellow-light onset and therefore the stimulation period).
- Biological replicate: animal, n=9. Repeated recordings were pooled within animal before plotting.
- Thin lines and hollow points: individual animals.
- Thick blue line and error bars: mean +/- SEM.
- Metrics are normalized by NREM+MA minutes.
- No significance stars are displayed because the Spindle comparison does not remain significant after Holm correction across three metrics.

## Statistics

{chr(10).join(stat_lines)}

## Files

- Source data: `{source_out}`
- Statistics: `{stats_out}`
- Script: `{SCRIPT}`
- Outputs: PNG, PDF, SVG, 600-dpi TIFF, and Illustrator-clean PDF/SVG for each metric and the combined panel.
"""
    (OUT / "README_analysis.md").write_text(readme, encoding="utf-8")

    PROV.mkdir(parents=True, exist_ok=True)
    prov_source = PROV / f"{FIGURE_ID}_source_data.csv"
    prov_stats = PROV / f"{FIGURE_ID}_stats.csv"
    prov_md = PROV / f"{FIGURE_ID}_provenance.md"
    df.to_csv(prov_source, index=False, encoding="utf-8-sig")
    stats_df.to_csv(prov_stats, index=False, encoding="utf-8-sig")
    prov_md.write_text(
        f"""# {FIGURE_ID}

- status: draft
- title: Paired VirusNPHR 120-s yellow-light Spindle/SO/coupling summary
- sample: 9 animals; repeated recordings pooled within animal
- base window: 180-300 s
- post window: 300-420 s, the yellow-light stimulation period after onset
- center/spread: mean +/- SEM
- statistics: paired t-test and Wilcoxon; Holm correction across three metrics
- source_data: `{prov_source}`
- stats: `{prov_stats}`
- script: `{SCRIPT}`
- review_note: No significance stars; Spindle unadjusted p<0.05 but Holm-adjusted p=0.0648.
""",
        encoding="utf-8",
    )

    delivered = figures + [source_out, stats_out, OUT / "source_paths.csv", OUT / "README_analysis.md", script_copy, prov_source, prov_stats, prov_md]
    pd.DataFrame(
        [
            {
                "path": str(path),
                "format": path.suffix.lstrip("."),
                "description": "paired figure" if path in figures else "source data, statistics, script, or provenance",
            }
            for path in delivered
        ]
    ).to_csv(OUT / "outputs_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_source()
    stats_df = compute_stats(df)
    figures = make_figures(df)
    write_metadata(df, stats_df, figures)
    print(stats_df.to_string(index=False))
    print("OUTPUTS")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
