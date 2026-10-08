from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

W = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig3_sleep_microarchitecture_CON_SUR")
OUT = W / "review" / "MA_SO_coupling_correlations_P3"
DATA = W / "data" / "MA_SO_coupling_correlations_P3"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

COLORS = {
    "CON": {"edge": "#646464", "fill": "#d5d5d5"},
    "SUR": {"edge": "#a94118", "fill": "#ee8a55"},
}

ma = pd.read_csv(W / "data/H_MA_vs_Spindle_D3_latest_source_data.csv")
ma = ma[["group", "animal", "MA_20s", "spindle_number_per_NREM_min"]]
so = pd.read_csv(W / "data/SO_refresh/animal_density.csv")
so = so.loc[so["day"].eq("P3"), ["group", "animal", "density"]].rename(columns={"density": "SO_density"})
coupling = pd.read_csv(W / "data/Coupling_Fig8_method/animal_metrics.csv")
coupling = coupling.loc[coupling["day"].eq("P3"), ["group", "animal", "coupled_density"]]
source = ma.merge(so, on=["group", "animal"], validate="one_to_one")
source = source.merge(coupling, on=["group", "animal"], validate="one_to_one")
assert len(source) == 12 and source.groupby("group").size().eq(6).all()
source.to_csv(DATA / "source_data.csv", index=False, encoding="utf-8-sig")

rows = []
for y in ["spindle_number_per_NREM_min", "SO_density", "coupled_density"]:
    r, p = stats.pearsonr(source["MA_20s"], source[y])
    model = smf.ols(f"{y} ~ MA_20s + C(group)", data=source).fit()
    rows.append({
        "outcome": y, "analysis": "pooled Pearson", "n": 12,
        "estimate": r, "p": p, "note": "unadjusted for group",
    })
    rows.append({
        "outcome": y, "analysis": "OLS MA slope adjusted for group", "n": 12,
        "estimate": model.params["MA_20s"], "p": model.pvalues["MA_20s"],
        "note": "model: outcome ~ MA_20s + group",
    })
    for group, sub in source.groupby("group"):
        rg, pg = stats.pearsonr(sub["MA_20s"], sub[y])
        rows.append({
            "outcome": y, "analysis": f"within {group} Pearson", "n": len(sub),
            "estimate": rg, "p": pg, "note": "exploratory; n=6",
        })
stats_df = pd.DataFrame(rows)
stats_df.to_csv(DATA / "statistics.csv", index=False, encoding="utf-8-sig")

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
specs = [
    ("SO_density", "MA–SO density correlation", "SO number / NREM min", (6.8, 9.8)),
    ("coupled_density", "MA–SO-coupled spindle correlation", "SO-coupled spindles / NREM min", (0.08, 0.35)),
]
for ax, (y, title, ylabel, ylim) in zip(axes, specs):
    for group in ["CON", "SUR"]:
        sub = source[source["group"].eq(group)]
        ax.scatter(sub["MA_20s"], sub[y], s=32, facecolors="white",
                   edgecolors=COLORS[group]["edge"], linewidths=1.2, label=group, zorder=3)
    slope, intercept, r, p, _ = stats.linregress(source["MA_20s"], source[y])
    xx = np.linspace(source["MA_20s"].min(), source["MA_20s"].max(), 100)
    ax.plot(xx, intercept + slope * xx, color="#aaaaaa", lw=1.2, zorder=2)
    ax.set(xlim=(100, 600), ylim=ylim, xlabel="MA count", ylabel=ylabel, title=title)
    ax.text(0.06, 0.08, f"Pooled r = {r:.3f}\nP = {p:.3f}",
            transform=ax.transAxes, ha="left", va="bottom")
    ax.legend(frameon=False, loc="upper right")
fig.suptitle("Exploratory pooled correlations at P3", fontsize=10, fontweight="bold")
for suffix in ["png", "pdf", "svg"]:
    kwargs = {"dpi": 600} if suffix == "png" else {}
    fig.savefig(OUT / f"MA_SO_coupling_correlations_P3_candidate.{suffix}", bbox_inches="tight", **kwargs)
plt.close(fig)

summary = {
    row["outcome"]: {
        "pooled_r": row["estimate"], "pooled_p": row["p"],
        "group_adjusted_MA_p": stats_df.loc[
            (stats_df.outcome.eq(row["outcome"])) &
            stats_df.analysis.eq("OLS MA slope adjusted for group"), "p"
        ].iloc[0],
    }
    for row in rows if row["analysis"] == "pooled Pearson"
}
(DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(stats_df.to_string(index=False))
