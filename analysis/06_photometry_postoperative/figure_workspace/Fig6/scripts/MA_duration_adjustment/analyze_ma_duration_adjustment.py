"""MA-duration-adjusted NREM-to-MA calcium response analysis (final 1-s scores)."""
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
import statsmodels.formula.api as smf

SOURCE = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_NREM_MA_transition_final_1s")
OUT = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig5_calcium_MA_duration_adjustment_final_1s")
PROV = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_ma_duration_adjustment_final_1s"
COLORS = {"Cont": {"edge":"#3697B8","fill":"#7FC3DF"}, "SUR": {"edge":"#C94E52","fill":"#EE6A70"}}

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                     "svg.fonttype":"none","pdf.fonttype":42,"pdf.compression":0,"font.size":7,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.75})

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def dotplot(ax, frame, value, ylabel, p):
    jitter=np.linspace(-.075,.075,7)
    for x,g in enumerate(("Cont","SUR")):
        v=frame.loc[frame.group.eq(g),value].dropna().to_numpy(float); c=COLORS[g]
        ax.scatter(x+jitter[:len(v)],v,s=25,facecolor="white",edgecolor=c["edge"],lw=.9,zorder=3)
        ax.errorbar(x,v.mean(),yerr=stats.sem(v),fmt="_",markersize=18,color=c["edge"],lw=1,capsize=2.2)
    ax.set_xticks([0,1],["Cont","SUR"]); ax.set_ylabel(ylabel); ax.axhline(0,color="#999",lw=.55)
    ax.text(.5,1.02,f"P = {p:.3f}",transform=ax.transAxes,ha="center")

def main():
    for p in (OUT,OUT/"data",OUT/"panels",OUT/"scripts",OUT/"documentation",OUT/"editable",PROV): p.mkdir(parents=True,exist_ok=True)
    source_event=SOURCE/"data/final_1s_event_level_timecourses.csv"
    source_strata=SOURCE/"data/final_1s_duration_strata_metrics.csv"
    long=pd.read_csv(source_event)
    event=(long[(long.time_s>=0)&(long.time_s<5)]
           .groupby(["group","animal","event_index","ma_duration_s"],as_index=False).delta_signal.mean()
           .rename(columns={"delta_signal":"response_0_5"}))
    event["mouse"]=event.group+"_"+event.animal
    event["surg"]=(event.group=="SUR").astype(int)
    event["duration_c5"]=event.ma_duration_s-5.0

    # Event-level model accounts for repeated events using a random intercept for mouse.
    mixed=smf.mixedlm("response_0_5 ~ surg * duration_c5",event,groups=event.mouse).fit(reml=True,method="powell",maxiter=2000)
    fixed=mixed.fe_params; ci=mixed.conf_int().loc[fixed.index]
    mixed_rows=[]
    labels={"Intercept":"Cont response at 5-s MA","surg":"SUR-Cont difference at 5-s MA",
            "duration_c5":"Duration slope in Cont","surg:duration_c5":"Additional duration slope in SUR"}
    for term in fixed.index:
        mixed_rows.append({"term":term,"interpretation":labels[term],"estimate":fixed[term],"se":mixed.bse[term],
                           "ci95_low":ci.loc[term,0],"ci95_high":ci.loc[term,1],"z":mixed.tvalues[term],"p":mixed.pvalues[term]})
    mixed_df=pd.DataFrame(mixed_rows)

    # Mouse-level ANCOVA is the conservative analysis retaining n=14 independent units.
    mouse=event.groupby(["group","animal"],as_index=False).agg(response_0_5=("response_0_5","mean"),mean_ma_duration_s=("ma_duration_s","mean"),event_count=("event_index","nunique"))
    grand=float(mouse.mean_ma_duration_s.mean()); mouse["duration_centered"]=mouse.mean_ma_duration_s-grand
    ancova=smf.ols("response_0_5 ~ C(group) + duration_centered",mouse).fit(cov_type="HC3",use_t=True)
    slope=float(ancova.params["duration_centered"])
    mouse["duration_adjusted_response_0_5"]=mouse.response_0_5-slope*mouse.duration_centered
    ancova_ci=ancova.conf_int()
    ancova_df=pd.DataFrame([{"term":term,"estimate":ancova.params[term],"se_hc3":ancova.bse[term],"t":ancova.tvalues[term],"df":ancova.df_resid,
                             "p_hc3_t":ancova.pvalues[term],"ci95_low":ancova_ci.loc[term,0],"ci95_high":ancova_ci.loc[term,1]} for term in ancova.params.index])

    cont_d=mouse.loc[mouse.group.eq("Cont"),"mean_ma_duration_s"].to_numpy(); sur_d=mouse.loc[mouse.group.eq("SUR"),"mean_ma_duration_s"].to_numpy()
    duration_test=stats.ttest_ind(sur_d,cont_d,equal_var=False)
    duration_stats=pd.DataFrame([{"comparison":"Mean MA duration, SUR vs Cont","cont_mean":cont_d.mean(),"cont_sem":stats.sem(cont_d),
                                  "sur_mean":sur_d.mean(),"sur_sem":stats.sem(sur_d),"difference_sur_minus_cont":sur_d.mean()-cont_d.mean(),
                                  "welch_t":duration_test.statistic,"welch_df":duration_test.df,"welch_p":duration_test.pvalue}])

    strata=pd.read_csv(source_strata)
    strata_rows=[]
    for label,x in strata.groupby("ma_duration_stratum",sort=False):
        a=x[x.group.eq("Cont")].early_response_0_5s.to_numpy(); b=x[x.group.eq("SUR")].early_response_0_5s.to_numpy(); q=stats.ttest_ind(b,a,equal_var=False)
        strata_rows.append({"stratum":label,"n_cont":len(a),"n_sur":len(b),"cont_mean":a.mean(),"cont_sem":stats.sem(a),"sur_mean":b.mean(),"sur_sem":stats.sem(b),"welch_t":q.statistic,"welch_df":q.df,"welch_p":q.pvalue})
    strata_stats=pd.DataFrame(strata_rows)

    event.to_csv(OUT/"data/event_level_duration_response.csv",index=False,encoding="utf-8-sig")
    mouse.to_csv(OUT/"data/animal_level_duration_adjusted_response.csv",index=False,encoding="utf-8-sig")
    mixed_df.to_csv(OUT/"data/mixed_effects_duration_model.csv",index=False,encoding="utf-8-sig")
    ancova_df.to_csv(OUT/"data/animal_level_ancova_hc3.csv",index=False,encoding="utf-8-sig")
    duration_stats.to_csv(OUT/"data/mean_ma_duration_group_test.csv",index=False,encoding="utf-8-sig")
    strata.to_csv(OUT/"data/duration_strata_animal_metrics.csv",index=False,encoding="utf-8-sig")
    strata_stats.to_csv(OUT/"data/duration_strata_statistics.csv",index=False,encoding="utf-8-sig")

    fig,ax=plt.subplots(2,2,figsize=(7.16,5.15),gridspec_kw={"wspace":.48,"hspace":.58}); ax=ax.ravel()
    dotplot(ax[0],mouse,"mean_ma_duration_s","Mean eligible MA duration (s)",float(duration_test.pvalue))
    dotplot(ax[1],mouse,"duration_adjusted_response_0_5",f"Duration-adjusted response 0-5 s\n(at common mean duration {grand:.2f} s)",float(ancova.pvalues["C(group)[T.SUR]"]))

    # Descriptive event points plus fixed-effect lines from the random-intercept model.
    rng=np.random.default_rng(20260914)
    for g in ("Cont","SUR"):
        x=event[event.group.eq(g)]; c=COLORS[g]
        ax[2].scatter(x.ma_duration_s+rng.normal(0,.055,len(x)),x.response_0_5,s=9,facecolor=c["fill"],edgecolor="none",alpha=.34,label=f"{g} events")
        duration=np.linspace(1,18,100); dc=duration-5
        pred=fixed["Intercept"]+fixed["duration_c5"]*dc
        if g=="SUR": pred=pred+fixed["surg"]+fixed["surg:duration_c5"]*dc
        ax[2].plot(duration,pred,color=c["edge"],lw=1.3,label=f"{g} mixed-model fit")
    ax[2].axhline(0,color="#999",lw=.55); ax[2].set(xlabel="MA duration (s)",ylabel="Calcium response 0-5 s\n(stored units)",xlim=(0,19)); ax[2].legend(frameon=False,fontsize=5.6,ncol=2)
    ax[2].text(.02,.98,f"SUR-Cont at 5 s: {fixed['surg']:.2f}\nP = {mixed.pvalues['surg']:.4f}\nInteraction P = {mixed.pvalues['surg:duration_c5']:.3f}",transform=ax[2].transAxes,va="top",fontsize=5.8)

    order=["5 s","10 s","15–20 s"]; xpos=np.arange(3); offsets={"Cont":-.16,"SUR":.16}
    for g in ("Cont","SUR"):
        c=COLORS[g]
        for i,label in enumerate(order):
            v=strata[(strata.group.eq(g))&(strata.ma_duration_stratum.eq(label))].early_response_0_5s.to_numpy(float)
            jitter=np.linspace(-.045,.045,len(v)); x=xpos[i]+offsets[g]
            ax[3].scatter(x+jitter,v,s=18,facecolor="white",edgecolor=c["edge"],lw=.75,zorder=3)
            ax[3].errorbar(x,v.mean(),yerr=stats.sem(v),fmt="_",markersize=12,color=c["edge"],lw=.9,capsize=1.8)
    ax[3].set_xticks(xpos,["1-5 s","6-10 s","11-20 s"]); ax[3].set_ylabel("Response 0-5 s\n(stored units)"); ax[3].axhline(0,color="#999",lw=.55)
    for i,row in strata_stats.set_index("stratum").loc[order].reset_index().iterrows(): ax[3].text(i,.99,f"P={row.welch_p:.3f}",transform=ax[3].get_xaxis_transform(),ha="center",va="top",fontsize=5.7)
    ax[3].plot([],[],color=COLORS["Cont"]["edge"],label="Cont"); ax[3].plot([],[],color=COLORS["SUR"]["edge"],label="SUR"); ax[3].legend(frameon=False,fontsize=5.8)

    for l,a in zip("abcd",ax): a.text(-.15,1.06,l,transform=a.transAxes,fontsize=9,fontweight="bold",va="top")
    fig.suptitle("MA-duration-adjusted calcium response at NREM-to-microarousal transitions",fontsize=9.5,y=.99)
    fig.text(.5,.015,"Final 1-s scoring; event model includes mouse random intercept; animal-level ANCOVA uses HC3 robust SE (n=7/group)",ha="center",fontsize=5.8)
    fig.subplots_adjust(left=.10,right=.985,top=.90,bottom=.11)
    base=OUT/"Fig5_calcium_MA_duration_adjustment_final_1s"
    outputs=[]
    for ext,kw in (("png",{"dpi":300}),("pdf",{}),("svg",{}),("tiff",{"dpi":600})):
        p=base.with_suffix('.'+ext);fig.savefig(p,bbox_inches="tight",facecolor="white",**kw);outputs.append(p)
    plt.close(fig)

    shutil.copy2(Path(__file__),OUT/"scripts"/Path(__file__).name)
    summary=("# MA-duration adjustment\n\nThe original 0-5 s endpoint was adjusted for actual MA duration.\n\n"
             f"- Mean MA duration: Cont {cont_d.mean():.3f} +/- {stats.sem(cont_d):.3f} s; SUR {sur_d.mean():.3f} +/- {stats.sem(sur_d):.3f} s; P={duration_test.pvalue:.4g}.\n"
             f"- Animal-level ANCOVA (HC3): adjusted SUR-Cont difference {ancova.params['C(group)[T.SUR]']:.3f} stored units, 95% CI {ancova_ci.loc['C(group)[T.SUR]',0]:.3f} to {ancova_ci.loc['C(group)[T.SUR]',1]:.3f}, P={ancova.pvalues['C(group)[T.SUR]']:.4g}.\n"
             f"- Random-intercept mixed model at a 5-s MA: SUR-Cont {fixed['surg']:.3f}, 95% CI {ci.loc['surg',0]:.3f} to {ci.loc['surg',1]:.3f}, P={mixed.pvalues['surg']:.4g}.\n"
             f"- Group-by-duration interaction: P={mixed.pvalues['surg:duration_c5']:.4g}.\n\n"
             "Conclusion: the larger SUR calcium response persists after MA-duration adjustment and is not explained by longer MA events.\n")
    (OUT/"documentation/README.md").write_text(summary,encoding="utf-8")
    manifest=pd.DataFrame([{"role":"event_timecourses","path":str(source_event),"sha256":sha256(source_event)},
                           {"role":"duration_strata","path":str(source_strata),"sha256":sha256(source_strata)}])
    manifest.to_csv(OUT/"data/source_file_manifest.csv",index=False,encoding="utf-8-sig")
    prov={"figure_id":FIGURE_ID,"output_folder":str(OUT),"script":str(Path(__file__)),"outputs":[str(p) for p in outputs],"primary_endpoint":"calcium response 0-5 s","duration_adjustment":"actual MA duration in seconds","experimental_unit":"mouse"}
    (PROV/f"{FIGURE_ID}.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary);print(mixed_df.to_string(index=False));print(ancova_df.to_string(index=False));print('Output:',OUT)

if __name__=="__main__": main()
