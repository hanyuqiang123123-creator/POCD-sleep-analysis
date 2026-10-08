from pathlib import Path
import pandas as pd,numpy as np,shutil,datetime,json
from scipy.stats import t,f
from statsmodels.stats.multitest import multipletests
r=Path(r'F:\1.Sleep\PHD稿件\Figure_Workspace\Fig10_Golgi_spine_density\behavior');a=r/'archive'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');a.mkdir(parents=True)
for d in ['data','documentation','editable','exports','previews']:shutil.copytree(r/d,a/d)
p=Path(r'F:\1.Sleep\eXdata\BehaviorTest\hm4Di\NOR\Nornphr3.0\nortest\tongji.xls');shutil.copy2(p,r/'data/NOR_corrected_tongji.xls')
x=pd.read_excel(p).iloc[1:];src=pd.read_csv(r/'data/hm4Di_behavior_source_data.csv');changes=[]
for _,row in x.iterrows():
 record=int(row.iloc[0]);old=float(row.iloc[14]);new=float(row.iloc[23]);value=(new-old)/(new+old)
 mask=(src.Metric=='DiscriminationIndex')&(src.Subject==f'source-record-{record}');assert mask.sum()==1
 before=float(src.loc[mask,'Value'].iloc[0]);assert src.loc[mask,'Group'].iloc[0]==row.iloc[1]
 if abs(before-value)>1e-12:changes.append(dict(record=record,before=before,after=value,old_time=old,new_time=new))
 src.loc[mask,'Value']=value;src.loc[mask,'SourcePath']=str(p)
src.to_csv(r/'data/hm4Di_behavior_source_data.csv',index=False)
nor=src[src.Metric=='DiscriminationIndex'];nor.to_csv(r/'data/E_nor_discrimination_index_animal_level_source.csv',index=False)
groups=['baseline+SAL','baseline+CNO','POCD+SAL','POCD+CNO'];vals=[nor[nor.Group==g].Value.to_numpy() for g in groups];means=np.array([v.mean() for v in vals]);ns=np.array([len(v) for v in vals]);df=sum(ns)-4;mse=sum(((v-v.mean())**2).sum() for v in vals)/df
stats=pd.read_csv(r/'data/hm4Di_behavior_stats.csv');pairs=[(0,2),(1,3),(0,1),(2,3)];ts=[];ps=[]
for i,j in pairs:
 z=(means[i]-means[j])/np.sqrt(mse*(1/ns[i]+1/ns[j]));ts.append(z);ps.append(2*t.sf(abs(z),df))
adj=multipletests(ps,method='holm')[1]
for k,(i,j) in enumerate(pairs):
 mask=(stats.Metric=='DiscriminationIndex')&(stats.EffectOrComparison==groups[i]+' vs '+groups[j]);stats.loc[mask,'Statistic']=ts[k];stats.loc[mask,'p_raw']=ps[k];stats.loc[mask,'p_adjusted']=adj[k];stats.loc[mask,'Stars']='***' if adj[k]<.001 else '**' if adj[k]<.01 else '*' if adj[k]<.05 else ''
for effect,c in [('Surgery',[-.5,-.5,.5,.5]),('Drug',[-.5,.5,-.5,.5]),('Surgery x Drug',[1,-1,-1,1])]:
 c=np.array(c);F=(c@means)**2/(mse*np.sum(c*c/ns));mask=(stats.Metric=='DiscriminationIndex')&(stats.EffectOrComparison==effect);stats.loc[mask,'Statistic']=F;stats.loc[mask,'p_raw']=f.sf(F,1,df)
stats.to_csv(r/'data/hm4Di_behavior_stats.csv',index=False)
for name in ['origin_panel_summary.csv','hm4Di_behavior_group_summary.csv']:
 q=pd.read_csv(r/'data'/name)
 for g,v in zip(groups,vals):
  mask=(q.Metric=='DiscriminationIndex')&(q.Group==g)
  for key,value in dict(N=len(v),Mean=v.mean(),SEM=v.std(ddof=1)/np.sqrt(len(v)),SD=v.std(ddof=1),Min=v.min(),Max=v.max()).items():
   if key in q:q.loc[mask,key]=value
 q.to_csv(r/'data'/name,index=False)
result=dict(changes=changes,mean=means.tolist(),sem=[v.std(ddof=1)/np.sqrt(len(v)) for v in vals],p_holm_POCD=float(adj[3]),stars='***' if adj[3]<.001 else '**' if adj[3]<.01 else '*')
(r/'documentation/NOR_correction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
