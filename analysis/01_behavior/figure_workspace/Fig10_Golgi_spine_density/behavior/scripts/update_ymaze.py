from pathlib import Path
import csv,datetime,shutil,json,math
import numpy as np,pandas as pd
from scipy.stats import t,f
from statsmodels.stats.multitest import multipletests
r=Path(r'F:\1.Sleep\PHD稿件\Figure_Workspace\Fig10_Golgi_spine_density\behavior');p=Path(r'F:\1.Sleep\eXdata\BehaviorTest\hm4Di\Y-maze\Ymaze_with_alternation.slk');stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');a=r/'archive'/stamp;a.mkdir(parents=True)
for d in ['data','editable','documentation','exports','previews']:shutil.copytree(r/d,a/d)
shutil.copy2(p,a/p.name)
raw=p.read_bytes();lines=raw.decode('latin1').splitlines();rows={};x=y=1
for line in lines:
 if not line.startswith(('C;','F;')):continue
 for z in line.split(';')[1:]:
  if z.startswith('X'):x=int(z[1:])
  elif z.startswith('Y'):y=int(z[1:])
  elif z.startswith('K') and line.startswith('C;'):rows.setdefault(y,{})[x]=z[1:].strip('"')
z=rows[29];assert z[1]=='POCD+SAL';n=sum(float(z[i]) for i in [12,13,14]);alt=sum(float(z[i]) for i in range(15,21));value=alt/(n-2)*100;assert 0<=alt<=n-2
# Fill the blank rate cell only; preserve all other bytes/lines.
x=y=1;idx=None
for i,line in enumerate(lines):
 if not line.startswith(('F;','C;')):continue
 for token in line.split(';')[1:]:
  if token.startswith('X'):x=int(token[1:])
  elif token.startswith('Y'):y=int(token[1:])
 if y==29 and x==21 and line.startswith('F;'):idx=i;break
assert idx is not None and not lines[idx+1].startswith('C;K')
lines.insert(idx+1,'C;K'+format(value,'.15g'));p.write_bytes(('\r\n'.join(lines)+'\r\n').encode('latin1'));shutil.copy2(p,r/'data/Ymaze_corrected_with_alternation.slk')
s=pd.read_csv(r/'data/hm4Di_behavior_source_data.csv');mask=(s.Metric=='SpontaneousAlternation_pct')&(s.Subject=='source-record-28');before=float(s.loc[mask,'Value'].iloc[0]);s.loc[mask,'Value']=value;s.to_csv(r/'data/hm4Di_behavior_source_data.csv',index=False)
d=s[s.Metric=='SpontaneousAlternation_pct'];d.to_csv(r/'data/D_ymaze_spontaneous_alternation_animal_level_source.csv',index=False);groups=['baseline+SAL','baseline+CNO','POCD+SAL','POCD+CNO'];vals=[d[d.Group==g].Value.to_numpy() for g in groups];means=np.array([v.mean() for v in vals]);ns=np.array([len(v) for v in vals]);df=sum(ns)-4;mse=sum(((v-v.mean())**2).sum() for v in vals)/df
st=pd.read_csv(r/'data/hm4Di_behavior_stats.csv');pairs=[(0,2),(1,3),(0,1),(2,3)];ts=[(means[i]-means[j])/np.sqrt(mse*(1/ns[i]+1/ns[j])) for i,j in pairs];ps=[2*t.sf(abs(v),df) for v in ts];adj=multipletests(ps,method='holm')[1]
star=lambda p:'****' if p<.0001 else '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
for k,(i,j) in enumerate(pairs):
 mask=(st.Metric=='SpontaneousAlternation_pct')&(st.EffectOrComparison==groups[i]+' vs '+groups[j]);st.loc[mask,'Statistic']=ts[k];st.loc[mask,'p_raw']=ps[k]
 for ix in st.index[mask]:
  primary=st.loc[ix,'TestFamily']=='Prespecified primary contrast';st.loc[ix,'p_adjusted']=np.nan if primary else adj[k];st.loc[ix,'Stars']=star(ps[k] if primary else adj[k])
for effect,c in [('Surgery',[-.5,-.5,.5,.5]),('Drug',[-.5,.5,-.5,.5]),('Surgery x Drug',[1,-1,-1,1])]:
 c=np.array(c);F=(c@means)**2/(mse*np.sum(c*c/ns));mask=(st.Metric=='SpontaneousAlternation_pct')&(st.EffectOrComparison==effect);st.loc[mask,'Statistic']=F;st.loc[mask,'p_raw']=f.sf(F,1,df)
st.to_csv(r/'data/hm4Di_behavior_stats.csv',index=False)
for name in ['origin_panel_summary.csv','hm4Di_behavior_group_summary.csv']:
 q=pd.read_csv(r/'data'/name)
 for g,v in zip(groups,vals):
  mask=(q.Metric=='SpontaneousAlternation_pct')&(q.Group==g)
  for key,num in dict(N=len(v),Mean=v.mean(),SEM=v.std(ddof=1)/np.sqrt(len(v)),SD=v.std(ddof=1),Min=v.min(),Max=v.max()).items():
   if key in q:q.loc[mask,key]=num
 q.to_csv(r/'data'/name,index=False)
result=dict(record=28,entries=n,alternations=alt,before=before,after=value,mean_POCD_SAL=means[2],sem_POCD_SAL=vals[2].std(ddof=1)/np.sqrt(len(vals[2])),p_primary=ps[3],p_holm_POCD=adj[3],p_holm_SAL=adj[0]);(r/'documentation/Ymaze_correction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
