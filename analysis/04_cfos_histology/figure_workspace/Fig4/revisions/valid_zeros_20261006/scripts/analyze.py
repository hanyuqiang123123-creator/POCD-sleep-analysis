from pathlib import Path
import sys,json,hashlib
P=Path(__file__).parent;sys.path.insert(0,str(P/'python_deps'))
import pandas as pd,numpy as np
from scipy import stats
R=Path('F:/1.Sleep');W=R/'PHD稿件/Figure_Workspace/Fig4'
O=P/'output';O.mkdir(exist_ok=True)
for sub in ['data','panels','editable','documentation']: (O/sub).mkdir(exist_ok=True)
src=R/'outputs/fig4_bba_projectpython_rerun/projectpython_braian_midbrain_animal_density.csv'
oldpath=W/'panels/E_Brainstem_volcano/data/volcano_results_original.csv'
d=pd.read_csv(src);old=pd.read_csv(oldpath)
assert len(d)==440 and not d.duplicated(['animal','region']).any()
assert not (d.density.dropna()<0).any()
d['valid_zero_inclusive']=np.isfinite(d.density)&d.density.ge(0)
d.to_csv(O/'data/animal_density_zero_inclusive.csv',index=False,encoding='utf-8-sig')
out=[]
for region,g in d.groupby('region',sort=False):
 c=g.loc[g.group.eq('Control')&g.valid_zero_inclusive,'density'].to_numpy()
 s=g.loc[g.group.eq('Surgery')&g.valid_zero_inclusive,'density'].to_numpy()
 rec=dict(region=region,control_n=len(c),surgery_n=len(s),control_mean=np.mean(c) if len(c) else np.nan,surgery_mean=np.mean(s) if len(s) else np.nan)
 eligible=len(c)>=2 and len(s)>=2;rec['eligible']=eligible
 if eligible:
  t,p=stats.ttest_ind(s,c,equal_var=True)
  fc=np.log2(np.mean(s)/np.mean(c)) if np.mean(c)>0 and np.mean(s)>0 else (np.inf if np.mean(s)>0 else (-np.inf if np.mean(c)>0 else np.nan))
  rec.update(log2_fc=fc,p_value=p,neg_log10_p=-np.log10(p) if p>0 else np.inf,t_statistic=t,significance='Up (Surgery)' if p<.05 and fc>0 else ('Down (Surgery)' if p<.05 and fc<0 else 'Not Significant'))
 out.append(rec)
alltests=pd.DataFrame(out); alltests.to_csv(O/'data/all_region_eligibility.csv',index=False,encoding='utf-8-sig')
new=alltests[alltests.eligible].set_index('region').loc[old.region].reset_index()
assert len(new)==48 and alltests.eligible.sum()==48
assert np.isfinite(new[['log2_fc','p_value']]).all().all()
new.to_csv(O/'data/volcano_results_zero_inclusive.csv',index=False,encoding='utf-8-sig')
compare=new.merge(old,on='region',suffixes=('_new','_old'))
compare['changed']=~np.isclose(compare.p_value_new,compare.p_value_old,rtol=1e-10,atol=1e-12)
compare.to_csv(O/'data/old_new_comparison.csv',index=False,encoding='utf-8-sig')
# Reproduce historical strictly positive rule as an independent arithmetic check.
for _,r in old.iterrows():
 g=d[d.region.eq(r.region)&d.density.gt(0)]
 c=g[g.group.eq('Control')].density;s=g[g.group.eq('Surgery')].density
 assert np.isclose(stats.ttest_ind(s,c,equal_var=True).pvalue,r.p_value,rtol=1e-10,atol=1e-12)
# Heatmap always used the unfiltered density field, including real zeros.
prev=pd.read_csv(P.parent/'source_audit_20261005/regional_density_49x8.csv',index_col=0)
matrix=d.pivot(index='region',columns='animal',values='density').reindex(index=prev.index,columns=prev.columns)
assert np.allclose(matrix,prev,equal_nan=True)
z=matrix.sub(matrix.mean(axis=1),axis=0).div(matrix.std(axis=1,ddof=1),axis=0)
matrix.to_csv(O/'data/heatmap_density_verified.csv',encoding='utf-8-sig');z.to_csv(O/'data/heatmap_zscore_verified.csv',encoding='utf-8-sig')
# Check numeric zeros directly against measured QuPath area/count rows.
raw_records=[];hashes={str(src):hashlib.sha256(src.read_bytes()).hexdigest(),str(oldpath):hashlib.sha256(oldpath.read_bytes()).hexdigest()}
for _,r in d[d.density.eq(0)].iterrows():
 hits=[]
 for f in sorted((R/'bba_braian/Qupath_projects'/r.animal/'results').glob('*_regions.txt')):
  raw=pd.read_csv(f,sep='\t');q=raw[raw['Name'].eq(r.region)]
  if len(q):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
   for _,row in q.iterrows():
    raw_records.append(dict(animal=r.animal,region=r.region,file=str(f),classification=row['Classification'],area_um2=row['Area um^2'],FITC_count=row['Num FITC']))
    if pd.notna(row['Area um^2']) and row['Area um^2']>0 and row['Num FITC']==0:hits.append(1)
 assert hits,(r.animal,r.region,'No measured zero count found')
pd.DataFrame(raw_records).to_csv(O/'data/raw_zero_count_evidence.csv',index=False,encoding='utf-8-sig')
(O/'documentation/input_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
# Retain original plotted row ordering, styles and threshold columns.
plot=pd.read_csv(W/'panels/E_Brainstem_volcano/data/volcano_origin_plot_data.csv')
for i,row in plot.iterrows():
 r=new[new.region.eq(row.region)].iloc[0]
 for col in ['log2_fc','p_value','neg_log10_p','significance']:plot.at[i,col]=r[col]
 for x,y in [('bg_x','bg_y'),('sig_x','sig_y'),('ew_x','ew_y')]:
  if pd.notna(row[x]):plot.at[i,x]=r.log2_fc;plot.at[i,y]=r.neg_log10_p
plot.to_csv(O/'data/volcano_origin_plot_data.csv',index=False,encoding='utf-8-sig')
report=dict(zero_definition='0 means a measured absence of positive cells; blank means missing data. Author correction 2026-10-06.',valid_zero_animal_regions=d[d.density.eq(0)][['animal','region']].to_dict('records'),heatmap_unchanged=True,heatmap_missing=int(matrix.isna().sum().sum()),heatmap_zero_cells=int(matrix.eq(0).sum().sum()),testable_regions=48,changed_regions=compare[compare.changed][['region','control_n','surgery_n','log2_fc_old','log2_fc_new','p_value_old','p_value_new']].to_dict('records'),nominal_significant=new[new.p_value<.05][['region','p_value']].to_dict('records'),excluded_regions=alltests[~alltests.eligible][['region','control_n','surgery_n']].to_dict('records'),historical_48_P_values_reproduced=True,all_input_hashes_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()),pseudocount_added=False)
(O/'documentation/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
