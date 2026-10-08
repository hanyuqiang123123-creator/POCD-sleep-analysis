"""Add the exploratory delayed calcium-sigma coupling panel to the final analysis."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_sigma_coupling_final_1s")
DATA = OUT / "data"
PALETTE = {"Cont": {"edge": "#3697B8", "fill": "#7FC3DF"}, "SUR": {"edge": "#C94E52", "fill": "#EE6A70"}}

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                     "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,
                     "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.75})

def test(frame, metric):
    a=frame.loc[frame.group.eq("Cont"),metric].dropna().to_numpy(float)
    b=frame.loc[frame.group.eq("SUR"),metric].dropna().to_numpy(float)
    q=stats.ttest_ind(b,a,equal_var=False)
    return a,b,float(q.statistic),float(q.pvalue)

def dots(ax, frame, metric, ylabel, p):
    jitter=np.linspace(-.075,.075,7)
    for x,g in enumerate(("Cont","SUR")):
        v=frame.loc[frame.group.eq(g),metric].dropna().to_numpy(float); c=PALETTE[g]
        ax.scatter(x+jitter[:len(v)],v,s=24,facecolor="white",edgecolor=c["edge"],lw=.9,zorder=3)
        ax.errorbar(x,v.mean(),yerr=stats.sem(v),fmt="_",markersize=18,color=c["edge"],lw=1,capsize=2.2)
    ax.set_xticks([0,1],["Cont","SUR"]); ax.set_ylabel(ylabel); ax.axhline(0,color="#888",lw=.6)
    ax.text(.5,1.02,f"P = {p:.3f}",transform=ax.transAxes,ha="center")

def main():
    mouse=pd.read_csv(DATA/"animal_level_calcium_sigma_metrics.csv")
    delayed=pd.read_csv(DATA/"delayed_5_10s_coupling_by_mouse.csv").rename(columns={"rho":"delayed_rho","fisher_z":"delayed_fisher_z"})
    mouse=mouse.merge(delayed[["group","animal","delayed_rho","delayed_fisher_z"]],on=["group","animal"])
    time=pd.read_csv(DATA/"animal_level_sigma_timecourses.csv")
    sens=pd.read_csv(DATA/"window_sensitivity_analysis.csv")
    raw=sens.p_coupling.to_numpy(float); order=np.argsort(raw)
    adj_sorted=np.maximum.accumulate((len(raw)-np.arange(len(raw)))*raw[order]); adj=np.empty_like(raw); adj[order]=np.minimum(adj_sorted,1)
    sens["holm_adjusted_p_coupling"]=adj
    sens.to_csv(DATA/"window_sensitivity_analysis.csv",index=False,encoding="utf-8-sig")

    stats_rows=[]
    for metric,label in [("sigma_suppression_db_0_5","Sigma suppression, 0-5 s"),
                         ("event_spearman_fisher_z","Concurrent calcium-sigma coupling, 0-5 s"),
                         ("delayed_fisher_z","Early calcium 0-5 s versus delayed sigma suppression 5-10 s")]:
        a,b,t,p=test(mouse,metric)
        pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        g=(1-3/(4*(len(a)+len(b))-9))*(b.mean()-a.mean())/pooled
        stats_rows.append({"comparison":label,"metric":metric,"n_cont":len(a),"n_sur":len(b),"cont_mean":a.mean(),"cont_sem":stats.sem(a),"sur_mean":b.mean(),"sur_sem":stats.sem(b),"difference_sur_minus_cont":b.mean()-a.mean(),"welch_t":t,"welch_p":p,"hedges_g":g})
    pd.DataFrame(stats_rows).to_csv(DATA/"calcium_sigma_statistics_v2.csv",index=False,encoding="utf-8-sig")
    mouse.to_csv(DATA/"animal_level_calcium_sigma_metrics_v2.csv",index=False,encoding="utf-8-sig")

    fig,axs=plt.subplots(2,2,figsize=(7.16,4.45),gridspec_kw={"wspace":.46,"hspace":.62}); ax=axs.ravel()
    for g in ("Cont","SUR"):
        p=time[time.group.eq(g)].pivot(index="animal",columns="time_s",values="sigma_change_db")
        x=p.columns.to_numpy(float); m=p.mean().to_numpy(); se=p.sem().to_numpy(); c=PALETTE[g]
        ax[0].plot(x,m,color=c["edge"],lw=1.2,label=f"{g} (n={len(p)})"); ax[0].fill_between(x,m-se,m+se,color=c["fill"],alpha=.28,lw=0)
    ax[0].axvspan(-20,-10,color="#D9EAF2",alpha=.35); ax[0].axvspan(0,5,color="#E5E5E5",alpha=.45)
    ax[0].axvline(0,color="black",ls="--",lw=.7); ax[0].axhline(0,color="#888",lw=.6)
    ax[0].set(xlim=(-20,20),xlabel="Time from MA onset (s)",ylabel="Sigma power change (dB)\nvs -20 to -10 s baseline"); ax[0].legend(frameon=False,fontsize=6.3)
    p0=test(mouse,"sigma_suppression_db_0_5")[3]; p1=test(mouse,"event_spearman_fisher_z")[3]; p2=test(mouse,"delayed_fisher_z")[3]
    dots(ax[1],mouse,"sigma_suppression_db_0_5","Sigma suppression, 0-5 s (dB)\npositive = larger decrease",p0)
    dots(ax[2],mouse,"event_spearman_fisher_z","Concurrent calcium-sigma coupling\nFisher z, 0-5 s",p1)
    dots(ax[3],mouse,"delayed_fisher_z","Early calcium vs delayed sigma fall\n0-5 s calcium; 5-10 s sigma",p2)
    for l,a in zip("abcd",ax): a.text(-.16,1.07,l,transform=a.transAxes,fontsize=9,fontweight="bold",va="top")
    fig.suptitle("Peri-microarousal calcium response and EEG sigma suppression",fontsize=9.5,y=.99)
    fig.text(.5,.015,"10-16 Hz Hilbert power; mouse-level means +/- SEM; two-sided Welch independent-samples t-test",ha="center",fontsize=5.8)
    fig.subplots_adjust(left=.09,right=.985,top=.89,bottom=.12)
    base=OUT/"Fig5_calcium_sigma_coupling_final_1s_v2"
    for ext,kw in (("png",{"dpi":300}),("pdf",{}),("svg",{}),("tiff",{"dpi":600})):
        fig.savefig(base.with_suffix('.'+ext),bbox_inches="tight",facecolor="white",**kw)
    plt.close(fig)
    shutil.copy2(Path(__file__),OUT/"scripts"/Path(__file__).name)
    print(pd.DataFrame(stats_rows).to_string(index=False)); print('\n',sens.to_string(index=False))

if __name__=="__main__": main()
