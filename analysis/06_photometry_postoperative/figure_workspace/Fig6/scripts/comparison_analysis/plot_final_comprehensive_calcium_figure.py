"""Comprehensive final 1-s NREM-to-MA calcium figure: curve, heatmaps and 7 endpoints."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SOURCE = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition_final_1s")
SIGMA_SOURCE = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_sigma_coupling_final_1s")
OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_comprehensive_final_1s")
PROV = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_nrem_ma_comprehensive_final_1s"
BLUE = {"edge": "#3697B8", "fill": "#7FC3DF"}
RED = {"edge": "#C94E52", "fill": "#EE6A70"}
COLORS = {"Cont": BLUE, "SUR": RED}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "pdf.compression": 0,
    "font.size": 6.5, "axes.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timecourse(ax: plt.Axes, event: pd.DataFrame) -> None:
    mouse = event.groupby(["group", "animal", "time_s"], as_index=False).delta_signal.mean()
    for group in ("Cont", "SUR"):
        p = mouse[mouse.group.eq(group)].pivot(index="animal", columns="time_s", values="delta_signal")
        x = p.columns.to_numpy(float); mean = p.mean().to_numpy(); sem = p.sem().to_numpy(); c = COLORS[group]
        ax.plot(x, mean, color=c["edge"], lw=1.25, label=f"{group} (n={len(p)})")
        ax.fill_between(x, mean-sem, mean+sem, color=c["fill"], alpha=.30, lw=0)
    ax.axvspan(-20, -10, color="#DCECF3", alpha=.55, zorder=0)
    ax.axvspan(0, 5, color="#E6E6E6", alpha=.50, zorder=0)
    ax.axvline(0, color="black", ls="--", lw=.75); ax.axhline(0, color="#888", lw=.55)
    ax.set(xlim=(-20, 30), xlabel="Time from MA onset (s)", ylabel="Delta signal vs -20 to -10 s baseline\n(stored units)")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.2)


def heatmap(ax: plt.Axes, event: pd.DataFrame, group: str, im_holder: list) -> None:
    f = event[event.group.eq(group)].copy()
    rows=[]; centers=[]; labels=[]; boundaries=[]; start=0
    for animal, x in f.groupby("animal", sort=False):
        pivot=x.pivot(index="event_index", columns="time_s", values="delta_signal").sort_index()
        rows.append(pivot.to_numpy()); n=len(pivot); centers.append(start+(n-1)/2); labels.append(animal); start += n; boundaries.append(start-.5)
    matrix=np.vstack(rows); times=np.sort(f.time_s.unique())
    im=ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-20, vmax=20,
                 extent=[times.min()-.5,times.max()+.5,len(matrix)-.5,-.5])
    im_holder.append(im)
    ax.axvline(0, color="black", ls="--", lw=.7)
    for y in boundaries[:-1]: ax.axhline(y, color="white", lw=.65)
    ax.set_yticks(centers, labels); ax.set_xlim(-20,30); ax.set_xlabel("")
    ax.set_ylabel("Mouse / events"); ax.set_title(f"{group}: {len(matrix)} events", fontsize=7, pad=2)


def dotplot(ax: plt.Axes, mouse: pd.DataFrame, metric: str, ylabel: str, p: float) -> None:
    jitter=np.linspace(-.075,.075,7)
    values=[]
    for x,group in enumerate(("Cont","SUR")):
        v=mouse.loc[mouse.group.eq(group),metric].to_numpy(float); values.extend(v); c=COLORS[group]
        ax.scatter(x+jitter, v, s=19, facecolor="white", edgecolor=c["edge"], lw=.8, zorder=3)
        ax.errorbar(x, v.mean(), yerr=stats.sem(v), fmt="_", markersize=15, color=c["edge"], lw=.9, capsize=2)
    ax.set_xticks([0,1],["Cont","SUR"]); ax.set_ylabel(ylabel, labelpad=2); ax.axhline(0,color="#999",lw=.5)
    ax.tick_params(labelsize=5.8, pad=1.2)
    ax.text(.5,1.015,f"P = {p:.3f}",transform=ax.transAxes,ha="center",va="bottom",fontsize=6.1)
    span=max(values)-min(values)
    if span>0: ax.margins(y=.16)


def sigma_timecourse(ax: plt.Axes, time: pd.DataFrame) -> None:
    for group in ("Cont", "SUR"):
        p=time[time.group.eq(group)].pivot(index="animal",columns="time_s",values="sigma_change_db")
        x=p.columns.to_numpy(float); mean=p.mean().to_numpy(); sem=p.sem().to_numpy(); c=COLORS[group]
        ax.plot(x,mean,color=c["edge"],lw=1.15,label=f"{group} (n={len(p)})")
        ax.fill_between(x,mean-sem,mean+sem,color=c["fill"],alpha=.28,lw=0)
    ax.axvspan(-20,-10,color="#DCECF3",alpha=.55); ax.axvspan(0,5,color="#E6E6E6",alpha=.50)
    ax.axvline(0,color="black",ls="--",lw=.7); ax.axhline(0,color="#888",lw=.55)
    ax.set(xlim=(-20,20),xlabel="Time from MA onset (s)",ylabel="Sigma power change (dB)\nvs -20 to -10 s baseline")
    ax.legend(frameon=False,fontsize=5.8)


def main() -> None:
    for p in (OUT, OUT/"panels", OUT/"data", OUT/"scripts", OUT/"documentation", OUT/"editable", PROV): p.mkdir(parents=True, exist_ok=True)
    event=pd.read_csv(SOURCE/"data/final_1s_event_level_timecourses.csv")
    mouse=pd.read_csv(SOURCE/"data/final_1s_animal_level_metrics.csv")
    tests=pd.read_csv(SOURCE/"data/final_1s_statistics.csv")
    pmap=tests.set_index("metric").welch_p.to_dict()
    sigma_mouse=pd.read_csv(SIGMA_SOURCE/"data/animal_level_calcium_sigma_metrics_v2.csv")
    sigma_time=pd.read_csv(SIGMA_SOURCE/"data/animal_level_sigma_timecourses.csv")
    sigma_tests=pd.read_csv(SIGMA_SOURCE/"data/calcium_sigma_statistics_v2.csv")
    sigma_p=sigma_tests.set_index("metric").welch_p.to_dict()

    fig=plt.figure(figsize=(7.16,11.25))
    outer=fig.add_gridspec(5,1,height_ratios=[1.15,1.30,1.0,1.0,2.05],hspace=.69)
    ax_a=fig.add_subplot(outer[0,0]); timecourse(ax_a,event)
    hm=outer[1,0].subgridspec(2,2,height_ratios=[1,.075],hspace=.34,wspace=.34)
    ax_b=fig.add_subplot(hm[0,0]); ax_c=fig.add_subplot(hm[0,1]); ims=[]
    heatmap(ax_b,event,"Cont",ims); heatmap(ax_c,event,"SUR",ims)
    cax=fig.add_subplot(hm[1,:])
    cb=fig.colorbar(ims[0],cax=cax,orientation="horizontal")
    cb.set_label("Delta signal vs -20 to -10 s baseline (stored units)",labelpad=2)
    row3=outer[2,0].subgridspec(1,4,wspace=.70); axes3=[fig.add_subplot(row3[0,i]) for i in range(4)]
    row4=outer[3,0].subgridspec(1,3,wspace=.62); axes4=[fig.add_subplot(row4[0,i]) for i in range(3)]
    specs=[
        ("primary_early_0_5","Response 0-5 s\n(stored units)"),
        ("pre_rise_minus10_0","Pre-MA -10 to 0 s\n(stored units)"),
        ("very_early_0_2","Response 0-2 s\n(stored units)"),
        ("auc_0_10","AUC 0-10 s\n(stored units x s)"),
        ("peak_amplitude_minus10_10","Peak amplitude\n-10 to 10 s"),
        ("peak_time_minus10_10_s","Peak time\n(s from MA onset)"),
        ("late_response_5_20","Late response 5-20 s\n(stored units)"),
    ]
    stat_axes=axes3+axes4
    for ax,(metric,label) in zip(stat_axes,specs): dotplot(ax,mouse,metric,label,float(pmap[metric]))
    sigma_grid=outer[4,0].subgridspec(2,2,wspace=.46,hspace=.62)
    sigma_axes=[fig.add_subplot(sigma_grid[i,j]) for i in range(2) for j in range(2)]
    sigma_timecourse(sigma_axes[0],sigma_time)
    dotplot(sigma_axes[1],sigma_mouse,"sigma_suppression_db_0_5","Sigma suppression, 0-5 s (dB)\npositive = larger decrease",float(sigma_p["sigma_suppression_db_0_5"]))
    dotplot(sigma_axes[2],sigma_mouse,"event_spearman_fisher_z","Concurrent calcium-Sigma coupling\nFisher z, 0-5 s",float(sigma_p["event_spearman_fisher_z"]))
    dotplot(sigma_axes[3],sigma_mouse,"delayed_fisher_z","Early calcium vs delayed Sigma fall\n0-5 s calcium; 5-10 s Sigma",float(sigma_p["delayed_fisher_z"]))
    all_axes=[ax_a,ax_b,ax_c]+stat_axes+sigma_axes
    for letter,ax in zip("abcdefghijklmn",all_axes): ax.text(-.105,1.06,letter,transform=ax.transAxes,fontsize=8.5,fontweight="bold",va="top")
    fig.suptitle("Calcium dynamics at NREM-to-microarousal transitions",fontsize=10,y=.992)
    fig.text(.5,.009,"Final 1-s scoring and corrected alignment; mouse-level means +/- SEM; two-sided Welch independent-samples t-test (n=7/group)",ha="center",fontsize=5.8)
    fig.subplots_adjust(left=.09,right=.985,top=.975,bottom=.04)
    base=OUT/"Fig5_calcium_NREM_MA_comprehensive_final_1s"
    outputs=[]
    for ext,kw in (("png",{"dpi":300}),("pdf",{}),("svg",{}),("tiff",{"dpi":600})):
        p=base.with_suffix('.'+ext); fig.savefig(p,bbox_inches="tight",facecolor="white",**kw); outputs.append(p)
    plt.close(fig)

    # Package the exact derived inputs used by the figure.
    for name in ("final_1s_event_level_timecourses.csv","final_1s_animal_level_metrics.csv","final_1s_statistics.csv","source_file_manifest.csv"):
        shutil.copy2(SOURCE/"data"/name,OUT/"data"/name)
    for src_name,dst_name in (("animal_level_calcium_sigma_metrics_v2.csv","sigma_animal_level_metrics.csv"),
                              ("animal_level_sigma_timecourses.csv","sigma_animal_level_timecourses.csv"),
                              ("calcium_sigma_statistics_v2.csv","sigma_statistics.csv"),
                              ("window_sensitivity_analysis.csv","sigma_window_sensitivity.csv")):
        shutil.copy2(SIGMA_SOURCE/"data"/src_name,OUT/"data"/dst_name)
    shutil.copy2(Path(__file__),OUT/"scripts"/Path(__file__).name)
    readme=(
        "# Comprehensive NREM-to-MA calcium figure\n\n"
        "Panels a-c show the animal-level mean time course and event-level heatmaps. "
        "Panels d-j show seven mouse-level calcium endpoints as individual mice plus mean +/- SEM. "
        "Panels k-n add peri-MA Sigma dynamics, mean 0-5 s Sigma suppression, concurrent calcium-Sigma coupling, and exploratory delayed coupling between 0-5 s calcium and 5-10 s Sigma suppression. "
        "Heatmaps share a fixed -20 to +20 stored-unit color scale. The shaded baseline is -20 to -10 s and the primary response window is 0 to 5 s. "
        "All displayed group P values are two-sided Welch independent-samples t-tests with the mouse as the experimental unit (n=7/group).\n"
    )
    (OUT/"documentation/README.md").write_text(readme,encoding="utf-8")
    source_manifest=[]
    for p in (SOURCE/"data/final_1s_event_level_timecourses.csv",SOURCE/"data/final_1s_animal_level_metrics.csv",SOURCE/"data/final_1s_statistics.csv",
              SIGMA_SOURCE/"data/animal_level_calcium_sigma_metrics_v2.csv",SIGMA_SOURCE/"data/animal_level_sigma_timecourses.csv",SIGMA_SOURCE/"data/calcium_sigma_statistics_v2.csv"):
        source_manifest.append({"path":str(p),"sha256":sha256(p)})
    pd.DataFrame(source_manifest).to_csv(OUT/"data/comprehensive_figure_source_manifest.csv",index=False,encoding="utf-8-sig")
    checks=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): checks.append({"path":str(p),"sha256":sha256(p)})
    pd.DataFrame(checks).to_csv(OUT/"data/output_checksums.csv",index=False,encoding="utf-8-sig")
    prov={"figure_id":FIGURE_ID,"output_folder":str(OUT),"script":str(Path(__file__)),"source_folder":str(SOURCE),"sigma_source_folder":str(SIGMA_SOURCE),"outputs":[str(x) for x in outputs],"experimental_unit":"mouse","n_per_group":7,"test":"two-sided Welch independent-samples t-test"}
    (PROV/f"{FIGURE_ID}.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")
    print(OUT); print(tests[tests.metric.isin([x[0] for x in specs])][["metric","cont_mean","cont_sem","sur_mean","sur_sem","welch_t","welch_df","welch_p","hedges_g"]].to_string(index=False))

if __name__=="__main__": main()
