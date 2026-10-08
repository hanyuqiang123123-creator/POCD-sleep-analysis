from pathlib import Path
from itertools import combinations
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
W=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig3_sleep_microarchitecture_CON_SUR');D=W/'data/Spindle_refresh'
p=D/'animal_density.csv';df=pd.read_csv(p)
assert len(df)==96 and df.source_status.eq('latest_export').all()
assert df.artifact_qc.str.startswith('verified').all()
assert not df.duplicated(['group','animal','day']).any()
rows=[]
for day in ['Pre']+['P'+str(i) for i in range(1,8)]:
 a=df[(df.group=='CON')&(df.day==day)].sort_values('animal').density.to_numpy()
 b=df[(df.group=='SUR')&(df.day==day)].sort_values('animal').density.to_numpy()
 assert len(a)==len(b)==6 and np.isfinite(np.r_[a,b]).all()
 pool=np.r_[a,b];obs=abs(b.mean()-a.mean());extreme=0
 for ids in combinations(range(12),6):
  mask=np.zeros(12,bool);mask[list(ids)]=True
  extreme+=abs(pool[mask].mean()-pool[~mask].mean())>=obs-1e-12
 raw=extreme/924
 check=permutation_test((a,b),lambda x,y: np.mean(x)-np.mean(y),permutation_type='independent',vectorized=False,n_resamples=np.inf,alternative='two-sided').pvalue
 assert abs(raw-check)<1e-10
 rows.append(dict(day=day,n_CON=6,n_SUR=6,mean_CON=a.mean(),SEM_CON=a.std(ddof=1)/np.sqrt(6),mean_SUR=b.mean(),SEM_SUR=b.std(ddof=1)/np.sqrt(6),difference_SUR_minus_CON=b.mean()-a.mean(),p_raw=raw,assignments=924,test='Exact two-sided independent permutation of mean difference',correction='Holm across Pre and P1-P7 (8 comparisons)'))
st=pd.DataFrame(rows);st['p_Holm']=multipletests(st.p_raw,method='holm')[1]
st['symbol']=st.p_Holm.map(lambda p:'***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '')
st.to_csv(D/'statistics.csv',index=False,encoding='utf-8-sig')
(D/'statistics_method.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unit='animal, 6 per group',contrast='CON versus SUR at each timepoint',family_size=8,method='Exact two-sided permutation; all 924 group allocations',multiplicity='Holm',repeated_measures='Each timepoint tested separately; no pooling of 96 observations as independent. Holm valid under dependence across days.',assumption='Independent animals and exchangeability under the null; manual/AI decisions treated as fixed.',scope='Timepoint group contrasts only; not a group-by-time interaction test.',crosscheck='Exact p-values independently checked against scipy permutation_test'),indent=2),encoding='utf-8')
print(st[['day','mean_CON','mean_SUR','difference_SUR_minus_CON','p_raw','p_Holm','symbol']].to_string(index=False))
