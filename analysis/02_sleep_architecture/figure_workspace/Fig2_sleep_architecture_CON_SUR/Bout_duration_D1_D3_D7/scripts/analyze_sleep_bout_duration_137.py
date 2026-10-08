from pathlib import Path
from datetime import datetime
import hashlib, shutil, json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

ROOT=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig2_sleep_architecture_CON_SUR')
PKG=ROOT/'Bout_duration_D1_D3_D7'
for sub in ['data','scripts','documentation']: (PKG/sub).mkdir(parents=True,exist_ok=True)
if (PKG/'data/bout_events.csv').exists():
    shutil.copytree(PKG,ROOT.parent.parent/'Figure_Archive/Fig2_sleep_architecture_CON_SUR/Bout_duration_D1_D3_D7'/datetime.now().strftime('%Y%m%d_%H%M%S'))
states={1:'Wake',2:'NREM',3:'REM'}
events=[]; means=[]; sources=[]; qc=[]
for day in [1,3,7]:
    files=sorted((ROOT/f'POD{day}_hourly/data').glob('*_scores.tsv'))
    assert len(files)==12
    for f in files:
        raw=f.read_bytes(); lines=raw.decode('utf-8-sig').splitlines()
        k=next(i for i,l in enumerate(lines) if l.startswith('Date\t'))
        rows=[l.split('\t') for l in lines[k+1:] if l.strip()]
        scores=np.array([int(float(l[4])) for l in rows]); times=np.array([float(l[3]) for l in rows])
        assert np.array_equal(times,np.arange(17280)*5)
        animal=f.stem.rsplit(f'_P{day}_scores',1)[0]; group=animal.split('_')[0]
        sources.append(dict(day=day,animal=animal,path=str(f),sha256=hashlib.sha256(raw).hexdigest()))
        starts=np.r_[0,np.flatnonzero(np.diff(scores)!=0)+1]; ends=np.r_[starts[1:],len(scores)]
        animal_events=[]
        for start,end in zip(starts,ends):
            state=states.get(scores[start])
            if state is None: continue
            reason=[]
            if start==0: reason.append('left_recording_boundary')
            elif scores[start-1] not in states: reason.append('left_unscored')
            if end==len(scores): reason.append('right_recording_boundary')
            elif scores[end] not in states: reason.append('right_unscored')
            item=dict(day=day,group=group,animal=animal,state=state,start_s=int(start*5),end_s=int(end*5),
                      start_epoch_1based=int(start+1),end_epoch_1based=int(end),duration_s=int((end-start)*5),
                      hour=int(start*5//3600),phase='Light' if start*5<43200 else 'Dark',included=not reason,exclusion=';'.join(reason))
            events.append(item); animal_events.append(item)
        b=pd.DataFrame(animal_events); valid=b[b.included]
        qc.append(dict(day=day,animal=animal,group=group,valid_bouts=len(valid),excluded_bouts=len(b)-len(valid),unscored_epochs=int((scores==255).sum())))
        windows=[('Daily','24h',valid)]+[('Phase',p,valid[valid.phase==p]) for p in ['Light','Dark']]+[('Hourly',str(h),valid[valid.hour==h]) for h in range(24)]
        for scope,window,part in windows:
            for state in ['NREM','REM','Wake']:
                x=part.loc[part.state==state,'duration_s']
                means.append(dict(day=day,group=group,animal=animal,state=state,scope=scope,window=window,n_bouts=len(x),mean_duration_s=x.mean(),median_duration_s=x.median()))
animal=pd.DataFrame(means); results=[]
for (day,state,scope,window),g in animal.groupby(['day','state','scope','window']):
    a=g.loc[g.group=='CON','mean_duration_s'].dropna().to_numpy(); b=g.loc[g.group=='SUR','mean_duration_s'].dropna().to_numpy()
    test=ttest_ind(a,b,equal_var=False) if min(len(a),len(b))>=3 else None
    results.append(dict(day=day,state=state,scope=scope,window=window,n_CON=len(a),n_SUR=len(b),
                        mean_CON_s=np.mean(a) if len(a) else np.nan,SEM_CON_s=np.std(a,ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan,
                        mean_SUR_s=np.mean(b) if len(b) else np.nan,SEM_SUR_s=np.std(b,ddof=1)/np.sqrt(len(b)) if len(b)>1 else np.nan,
                        t=test.statistic if test else np.nan,df=test.df if test else np.nan,p_raw=test.pvalue if test else np.nan))
tests=pd.DataFrame(results)
tests['family']=tests.apply(lambda r:f'Hourly_D{r.day}_{r.state}_24' if r.scope=='Hourly' else f'{r.scope}_{r.state}_'+('6' if r.scope=='Phase' else '3'),axis=1)
for family,ix in tests.groupby('family').groups.items():
    p=tests.loc[ix,'p_raw']; adj=multipletests(p.fillna(1),method='holm')[1]
    tests.loc[ix,'p_holm']=np.where(p.notna(),adj,np.nan)
tests['significant']=tests.p_holm<.05
pd.DataFrame(events).to_csv(PKG/'data/bout_events.csv',index=False)
animal.to_csv(PKG/'data/animal_bout_duration.csv',index=False)
tests.to_csv(PKG/'data/bout_duration_Welch_Holm.csv',index=False)
pd.DataFrame(qc).to_csv(PKG/'data/quality_control.csv',index=False)
pd.DataFrame(sources).to_csv(PKG/'documentation/source_paths.csv',index=False)
assert animal.groupby(['day','group']).animal.nunique().eq(6).all()
assert all(x['end_s']>x['start_s'] for x in events)
assert not animal.duplicated(['day','animal','scope','window','state']).any()
readme='''# Bout duration: postoperative days 1, 3 and 7

## Question and status
Exploratory comparison of the mean duration of complete NREM, REM and Wake bouts between CON and SUR.
Six mice per group; animal is the biological replicate. No plots generated in this analysis.

## Source and extraction
Uses the verified scoring TSV copies in POD1_hourly, POD3_hourly and POD7_hourly.
5-s epochs, 08:00–08:00, 17,280 epochs per animal/day; codes 1 Wake, 2 NREM, 3 REM, 255 unscored.
A bout is a maximal uninterrupted run of one state. No short-bout merging or microarousal reassignment.
Runs touching either recording edge or unscored epochs are excluded as potentially censored.
Complete bouts are assigned by onset to hour and phase, preserving full duration across boundaries.
Light=onset 0–12 h, Dark=12–24 h. Hourly bins are [h,h+1).
This estimates mean duration of bouts beginning in a window, not time spent in that state in the window.
Each animal contributes its arithmetic mean of eligible bout durations in seconds.
A window without a bout is missing, not zero. Both groups must have >=3 animal means for a test;
varying sample sizes are explicitly recorded and sparse hourly results require caution.
Events near day boundaries are excluded, so means describe observed complete bouts rather than a
censoring-adjusted survival estimate. This can especially affect long Wake bouts.
All events, exclusions, animal means, denominators, and source hashes are retained.

## Statistics
Two-sided independent-samples Welch t-tests on animal means; sample SEM (ddof=1).
No pooling of bouts across mice as independent replicates. Normality remains a small-sample assumption.
Holm families fixed before calculating tests: hourly=24 hours per state/day;
phase=6 comparisons (2 phases x 3 days) per state; daily=3 days per state.
These are separate exploratory analysis families; there is no global correction covering every
state and every aggregation. Undefined tests remain in their family as p=1 for correction purposes.
Significance means adjusted p<0.05 under the declared family, not confirmatory study-wide significance.
These pointwise contrasts do not constitute a repeated-measures group-by-time test.
Do not select a different family or correction based on which becomes significant.

## Relation to old Figure 2 duration tables
The older phase tables use rounded state percentages divided by phase event counts.
This analysis recomputes exact complete-bout lengths from current scoring and assigns by onset;
boundary treatment, censoring exclusions and source revisions can produce different estimates.
No previous artwork or statistics were overwritten.

## Reproduction
python -X utf8 scripts/analyze_sleep_bout_duration_137.py
'''
readme+='\nGenerated: '+datetime.now().isoformat(timespec='seconds')+'\nSciPy: '+scipy.__version__+'\n'
(PKG/'documentation/README_analysis.md').write_text(readme,encoding='utf-8')
dest=PKG/'scripts'/Path(__file__).name
if Path(__file__).resolve()!=dest: shutil.copy2(__file__,dest)
prov=Path('F:/Sleep/outputs/figure_provenance')
for src,name in [(PKG/'documentation/README_analysis.md','bout_duration_D1_D3_D7_provenance.md'),(PKG/'data/animal_bout_duration.csv','bout_duration_D1_D3_D7_source_data.csv'),(PKG/'data/bout_duration_Welch_Holm.csv','bout_duration_D1_D3_D7_stats.csv')]:shutil.copy2(src,prov/name)
pd.DataFrame([dict(path=str(f),type=f.suffix,status='current') for f in PKG.rglob('*') if f.is_file()]).to_csv(PKG/'documentation/outputs_manifest.csv',index=False)
print('DAILY AND PHASE RESULTS')
print(tests[tests.scope!='Hourly'][['day','state','scope','window','n_CON','n_SUR','mean_CON_s','SEM_CON_s','mean_SUR_s','SEM_SUR_s','p_raw','p_holm']].to_string(index=False))
print('HOURLY SIGNIFICANT')
print(tests[(tests.scope=='Hourly')&tests.significant].to_string(index=False))
print('Hourly missing animal means',int(animal.loc[animal.scope=='Hourly','mean_duration_s'].isna().sum()))
