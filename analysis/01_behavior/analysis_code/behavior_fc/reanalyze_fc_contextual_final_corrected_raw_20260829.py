from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(SCRIPT_DIR))
import reanalyze_fc_contextual_six_group_swaps_20260829 as pipeline  # noqa: E402


analysis = pipeline.analysis
FIGURE_ID = "fc_contextual_20260317_final_corrected_raw_20260829"
OUTDIR = Path(r"F:\Sleep\Figure\FC_contextual_20260317_final_corrected_raw_20260829")
PROVENANCE_DIR = Path(r"F:\Sleep\outputs\figure_provenance")


def draw_final_two_way_figure(
    frame: pd.DataFrame, summary: pd.DataFrame, one_way_p: float
) -> list[Path]:
    _, _, factorial = analysis.analyze_statistics(frame)
    factorial_index = factorial.set_index("term")
    p_condition = float(
        factorial_index.loc["Condition (POCD vs baseline)", "p_value"]
    )
    p_treatment = float(
        factorial_index.loc["Treatment (CNO vs SAL)", "p_value"]
    )
    p_interaction = float(
        factorial_index.loc["Condition x treatment interaction", "p_value"]
    )

    analysis.configure_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.95))
    x_positions = np.arange(4, dtype=float)
    indexed = summary.set_index("group").loc[analysis.GROUP_ORDER]
    means = indexed["mean_freezing_pct"].to_numpy(float)
    sems = indexed["sem_freezing_pct"].to_numpy(float)

    for x, group, mean in zip(
        x_positions, analysis.GROUP_ORDER, means, strict=True
    ):
        style = analysis.PALETTE[group]
        ax.bar(
            x,
            mean,
            width=0.54,
            color=style["fill"],
            edgecolor=style["edge"],
            linewidth=0.6,
            zorder=1,
        )
    ax.errorbar(
        x_positions,
        means,
        yerr=sems,
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=4,
    )

    for x, group in zip(x_positions, analysis.GROUP_ORDER, strict=True):
        subset = frame.loc[frame["group"] == group].sort_values("raw_id")
        values = subset["freezing_pct"].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        point_x = x + jitter
        ax.scatter(
            point_x,
            values,
            s=25,
            facecolors="white",
            edgecolors=analysis.PALETTE[group]["edge"],
            linewidths=0.95,
            zorder=5,
        )
        sides = np.array(["left" if i % 2 else "right" for i in range(len(values))])
        label_y = np.zeros(len(values), dtype=float)
        for side in ("left", "right"):
            indices = np.flatnonzero(sides == side)
            label_y[indices] = analysis.spread_labels(values[indices])
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
                color=analysis.PALETTE[group]["edge"],
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": analysis.PALETTE[group]["edge"],
                    "linewidth": 0.45,
                    "alpha": 0.7,
                    "shrinkA": 1.5,
                    "shrinkB": 2,
                },
                zorder=6,
            )

    counts = indexed["n"].astype(int).tolist()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            f"{group}\n(n={n})"
            for group, n in zip(analysis.GROUP_ORDER, counts, strict=True)
        ]
    )
    ax.set_ylabel("Freezing (%)")
    ax.set_title("Contextual freezing", fontsize=8, fontweight="bold", pad=6)
    ax.text(
        0.5,
        0.985,
        "Two-way ANOVA",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.945,
        (
            f"Condition {analysis.p_text(p_condition)}  |  "
            f"Treatment {analysis.p_text(p_treatment)}  |  "
            f"Interaction {analysis.p_text(p_interaction)}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )
    ax.set_xlim(-0.72, 3.72)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(False)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.23, top=0.87)

    base = OUTDIR / "FC_contextual_freezing_final_corrected_two_way"
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
    clean_pdf = OUTDIR / "FC_contextual_freezing_final_corrected_two_way_ai_clean.pdf"
    clean_svg = OUTDIR / "FC_contextual_freezing_final_corrected_two_way_ai_clean.svg"
    fig.savefig(clean_pdf, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(clean_svg, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return outputs + [clean_pdf, clean_svg]


def update_final_records() -> None:
    readme_path = OUTDIR / "README_analysis.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme += """

## Final raw-data reanalysis

- This revision was parsed afresh from the current corrected PACKWIN `.exp/.ssn/.ini/.raw` quartet.
- No previously exported animal-level or summary CSV was used as an analysis input.
- The figure reports the Type-III 2×2 condition × treatment ANOVA because this matches the factorial experimental design.
- One-way ANOVA and Tukey HSD remain available as complementary four-cell analyses.
"""
    readme_path.write_text(readme, encoding="utf-8")

    provenance_path = PROVENANCE_DIR / f"{FIGURE_ID}_provenance.md"
    provenance = provenance_path.read_text(encoding="utf-8")
    provenance += """

## Final analysis note

- Freshly recomputed from the corrected PACKWIN quartet, not from a prior CSV.
- Primary figure annotation: Type-III two-way ANOVA with sum contrasts.
"""
    provenance_path.write_text(provenance, encoding="utf-8")

    manifest_path = OUTDIR / "outputs_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    for path, note in [
        (readme_path, "Final raw-data reanalysis record"),
        (provenance_path, "Final raw-data figure provenance"),
    ]:
        manifest = pipeline.upsert_manifest(manifest, path, note)
    manifest.to_csv(manifest_path, index=False)


def main() -> int:
    pipeline.FIGURE_ID = FIGURE_ID
    pipeline.OUTDIR = OUTDIR
    pipeline.PROVENANCE_DIR = PROVENANCE_DIR
    pipeline.SCRIPT_PATH = SCRIPT_PATH
    analysis.draw_figure = draw_final_two_way_figure
    status = pipeline.main()
    update_final_records()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
