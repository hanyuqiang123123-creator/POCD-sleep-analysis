from pathlib import Path
from datetime import datetime
import shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Contract: descriptive 2h trajectory relative to each animal's POD1 NREM mean.
# Single quantitative panel; partial cohort; no baseline-change or significance claim.
B=Path('F:/1.Sleep/PHD稿件')
P=B/'Figure_Workspace/Fig2_sleep_architecture_CON_SUR/POD1_SWA'
O=B/'Figures/Fig2_sleep_architecture_CON_SUR/POD1_SWA'
ID='POD1_SWA_percent_daily_mean_2h'
for d in [O,O/'panels',P/'panels',P/'scripts',P/'documentation']:d.mkdir(parents=True,exist_ok=True)
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
for folder in [P,O]:
 for f in list(folder.glob(ID+'*'))+list((folder/'panels').glob(ID+'*')):
  if f.is_file():
   target=B/'Figure_Archive/Fig2_sleep_architecture_CON_SUR/POD1_SWA'/stamp/folder.parent.name/f.relative_to(folder)
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
src=P/'data/animal_swa.csv';d=pd.read_csv(src);d=d[d.period.str.startswith('ZT')].copy()
d['ZT_mid']=d.period.str.slice(2,4).astype(int)+1
d['swa_percent_daily_mean']=d.normalized_percent
assert d.groupby('group').animal.nunique().to_dict()=={'CON':6,'SUR':5}
g=d.groupby(['group','ZT_mid']).swa_percent_daily_mean.agg(['mean','std','count']).reset_index()
g['sem']=g['std']/np.sqrt(g['count'])
d.to_csv(P/'data'/f'{ID}_animal.csv',index=False);g.to_csv(P/'data'/f'{ID}_summary.csv',index=False)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':11,'axes.linewidth':1.1,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
fig,ax=plt.subplots(figsize=(7.5,3.65));fig.subplots_adjust(left=.14,right=.97,bottom=.24,top=.80)
for group,color,n in [('CON','#7066B3',6),('SUR','#DD784C',5)]:
 s=g[g.group==group].sort_values('ZT_mid')
 ax.errorbar(s.ZT_mid,s['mean'],yerr=s['sem'],color=color,marker='o',markersize=3.8,linewidth=1.25,elinewidth=1,capsize=2.5,label=f'{group} (n = {n})')
# Normalized values are centered on 100; no additional reference line.
ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24])
low=float((g['mean']-g['sem']).min());high=float((g['mean']+g['sem']).max())
lo=np.floor(min(low,100)/10)*10-5;hi=np.ceil(max(high,100)/10)*10+5
ax.set_ylim(lo,hi);ax.set_yticks(np.arange(np.ceil(lo/10)*10,hi,10))
ax.set_xlabel('Time since lights on (h)',fontweight='bold',labelpad=8)
ax.set_ylabel('Normalized SWA',fontweight='bold',labelpad=8)
ax.grid(color='#EFEFEF',lw=.8);ax.set_axisbelow(True)
for side in ['right','top']:ax.spines[side].set_visible(False)
ax.tick_params(length=5,width=1)
for start,color in [(0,'white'),(12,'#20205C')]:
 ax.add_patch(Rectangle((start,1.015),12,.045,transform=ax.get_xaxis_transform(),facecolor=color,edgecolor='black',lw=1,clip_on=False))
ax.legend(loc='lower right',bbox_to_anchor=(1,1.09),ncol=2,handlelength=2.1,columnspacing=2)
fig.text(.14,.93,'Postoperative day 1',fontweight='bold',fontsize=11)
fig.text(.14,.08,'2-h bins; mean ± SEM. 100% = each animal’s daily NREM mean.',fontsize=8)
fig.text(.14,.035,'Exploratory: incomplete cohort; artifact verification pending.',fontsize=8,color='#555555')
for ext in ['png','pdf','svg']:
 f=P/f'{ID}.{ext}';fig.savefig(f,dpi=300,facecolor='white')
 for dest in [O,P/'panels',O/'panels']:shutil.copy2(f,dest/f.name)
for ext in ['pdf','svg']:
 for dest in [P,O,P/'panels',O/'panels']:shutil.copy2(dest/f'{ID}.{ext}',dest/f'{ID}_AI_clean.{ext}')
plt.close(fig)
shutil.copy2(__file__,P/'scripts'/Path(__file__).name)
note=f'''# POD1 SWA time-course figure
Status: exploratory partial cohort, CON 6 / SUR 5; SUR NO.03 awaits EEG-score identity resolution.
Generated: {stamp}. Figure 2 candidate single quantitative panel, Python matplotlib, 7.5 x 3.65 in.
Source: {src}; SHA256 {hashlib.sha256(src.read_bytes()).hexdigest()}.
Estimator/preprocessing/source files: README_analysis.md and source_paths.csv in this documentation folder.
2h bins [0,2),...,[22,24), centers 1,3,...,23. ZT0=08:00, ZT12=20:00.
Animal relative SWA = 100*(bin mean SWA / same animal full-recording valid NREM mean SWA).
Not relative to preoperative baseline. Equal animal weighting; error bars are sample SEM across animals, not windows.
Top bar white ZT0–12 light, dark ZT12–24 dark. No grey intervention period is supported by these data.
No significant 2h differences under existing 12-comparison Holm correction; no stars or significance brackets shown.
Verified artifact masks unavailable: descriptive exploration only. No added smoothing or invented values.
Outputs: {O}; individual panel in panels; CSVs in data; reproducible script in scripts.
'''
(P/'documentation'/f'{ID}_provenance.md').write_text(note,encoding='utf-8')
prov=Path('F:/Sleep/outputs/figure_provenance')
(prov/f'{ID}_provenance.md').write_text(note,encoding='utf-8');d.to_csv(prov/f'{ID}_source_data.csv',index=False)
pd.DataFrame([{'path':str(f)} for folder in [P,O] for f in folder.rglob('*') if f.is_file()]).to_csv(P/'documentation/outputs_manifest.csv',index=False)
print(O/f'{ID}.png')
