from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel
import shutil
S=Path('F:/Sleep/outputs/virus_chr2_time')
O=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig7/data/ChR2_10s/exclude_mouse12')
O.mkdir(parents=True,exist_ok=True)
rows=[]
def test(df,metric,after,label):
 p=df.pivot(index='mouse',columns='window',values=metric)[['baseline',after]].dropna()
 r=ttest_rel(p[after],p.baseline)
 rows.append(dict(metric=label,n=len(p),baseline=p.baseline.mean(),baseline_sem=p.baseline.sem(),post=p[after].mean(),post_sem=p[after].sem(),t=r.statistic,p=r.pvalue))
w=pd.read_csv(S/'spindle_so_5s_10s/window_metrics_by_mouse_5s_10s.csv')
w=w[(w.condition=='10s')&(w.mouse!=12)]
w.to_csv(O/'eligible_window_mouse_metrics.csv',index=False)
for metric in ['spindle_per_nrem_min','so_per_nrem_min','coupling_per_nrem_min','pct_spindles_coupled','pct_so_coupled']:
 test(w,metric,'post_0_100',metric)
t=pd.read_csv(S/'spindle_so_5s_10s/window_metrics_by_trial_5s_10s.csv')
t=t[(t.condition=='10s')&(t.mouse!=12)]
t.to_csv(O/'all_trial_metrics.csv',index=False)
t['seconds']=t.window.map({'baseline':90,'post_0_100':100,'early_0_30':30,'late_30_100':70})
for kind in ['spindle','so','coupling']:t[kind+'_per_recording_min']=60*t[kind+'_count']/t.seconds
t['nrem_ma_percent']=100*t.nrem_sec/t.seconds
a=t.groupby(['mouse','window'],as_index=False)[['spindle_per_recording_min','so_per_recording_min','coupling_per_recording_min','nrem_ma_percent']].mean()
a.to_csv(O/'all_trial_mouse_metrics.csv',index=False)
for metric in a.columns[2:]:test(a,metric,'post_0_100',metric)
st=pd.read_csv(S/'laser_brainstate_0p1_10s/laser_brainstate_0p1_window_summary_by_mouse_with_ma.csv')
st=st[st.mouse!=12];st.to_csv(O/'state_window_mouse_metrics.csv',index=False)
for state in st.state.unique():test(st[st.state==state],'percent','laser',state+'_percent')
for name,folder in [('peri_stimulus_by_mouse_5s_10s.csv','spindle_so_5s_10s'),('laser_brainstate_0p1_timecourse_by_mouse_with_ma.csv','laser_brainstate_0p1_10s')]:
 d=pd.read_csv(S/folder/name);d=d[d.mouse!=12]
 if 'condition' in d:d=d[d.condition=='10s']
 d.to_csv(O/name,index=False)
out=pd.DataFrame(rows);out.to_csv(O/'paired_statistics.csv',index=False)
(O/'README.md').write_text('''# 10-s ChR2 reanalysis excluding mouse 12
User-requested exclusion, 2026-09-15. User reports a problem with this mouse; specific QC reason/evidence pending. This is post-hoc exclusion after viewing the outcome and must be disclosed; not a validated technical-failure exclusion. Full-sample original results and acquisition files retained unchanged.

Only 10-s condition excludes mouse12, remaining mice 07-11. 5-s and NpHR unchanged; confirm scope if defect also affects other recordings. All outcomes recomputed, not only spindle. Paired two-sided t-tests at mouse level, unadjusted for multiplicity; mean +/- SEM. Original eligible-window density uses NREM+MA >=20 s, baseline -100:-10 s and post 0:100 s. All-trial event rates normalize unequal durations to recording minutes. State comparison retains original laser/baseline windows. Original figures/OPJU not updated by this statistics-only run.
''',encoding='utf-8')
shutil.copy2(__file__,O/Path(__file__).name)
print(out.to_string(index=False))
