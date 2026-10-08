from pathlib import Path
import csv,json,hashlib,itertools,platform
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

OUT=Path(__file__).resolve().parent
W=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig10_Golgi_spine_density')
RAW=Path('F:/1.Sleep/eXdata/golgi')
GROUPS=['Baseline+SAL','Baseline+CNO','POCD+SAL','POCD+CNO']
PAIRS=[(0,1),(2,3),(0,2),(1,3)]
inputs={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    inputs[str(p)]=sha(p)
    return pd.read_csv(p)
def save(df,name):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def summary(df,value):
    return df.groupby('group')[value].agg(n='count',mean='mean',SD='std',SEM='sem').reindex(GROUPS).reset_index()

def exact_permutation(a,b):
    z=np.r_[a,b];observed=abs(np.mean(b)-np.mean(a));ds=[]
    for ids in itertools.combinations(range(len(z)),len(a)):
        mask=np.zeros(len(z),dtype=bool);mask[list(ids)]=True
        ds.append(abs(z[~mask].mean()-z[mask].mean()))
    return float(np.mean(np.asarray(ds)>=observed-1e-12))

def compare(df,value,endpoint,mode):
    rows=[]
    for i,j in PAIRS:
        a=df.loc[df.group==GROUPS[i],value].to_numpy(float)
        b=df.loc[df.group==GROUPS[j],value].to_numpy(float)
        va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b)
        se=np.sqrt(va+vb);dfree=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
        t=stats.ttest_ind(b,a,equal_var=False);diff=b.mean()-a.mean();margin=stats.t.ppf(.975,dfree)*se
        rows.append(dict(endpoint=endpoint,aggregation=mode,comparison=f'{GROUPS[j]} minus {GROUPS[i]}',n_first=len(a),n_second=len(b),difference=diff,SE=se,df=dfree,t=t.statistic,p_raw=t.pvalue,ci95_low=diff-margin,ci95_high=diff+margin,exact_permutation_p=exact_permutation(a,b)))
    q=pd.DataFrame(rows);q['p_Holm_4']=multipletests(q.p_raw,method='holm')[1];q['exact_permutation_p_Holm_4']=multipletests(q.exact_permutation_p,method='holm')[1]
    return q

def factorial(df,value,endpoint,mode):
    d=df.copy();d['condition']=d.group.str.split('+').str[0];d['drug']=d.group.str.split('+').str[1]
    fit=ols(f'{value} ~ C(condition, Sum)*C(drug, Sum)',d).fit()
    tab=anova_lm(fit,typ=3).reset_index().rename(columns={'index':'term'})
    tab['endpoint']=endpoint;tab['aggregation']=mode
    return tab

# Record figure hashes before analysis; no drawing or export code is called.
artfiles=list(Path('C:/Users/USER/Desktop/图片/Last/Main').glob('Fig8*'))
art_before={str(p):sha(p) for p in artfiles if p.is_file()}
old=read(W/'current_review/data/animal_data.csv')
audit=read(W/'current_review/data/source_audit.csv')
old_tests=read(W/'current_review/data/pairwise_comparisons.csv')
current={}
for group in GROUPS:
    for p in (RAW/'260108'/group).glob('NO*/*/*manual_review.json'):
        h=sha(p);current.setdefault(h,[]).append((group,p))
maps=[]
for row in audit.to_dict('records'):
    found=current.get(row['sha256'],[]);assert len(found)==1,(row,found)
    group,p=found[0];assert group==row['group']
    inputs[str(p)]=row['sha256']
    maps.append(dict(group=group,animal_id=row['animal_id'],mouse=p.parents[1].name,current_sample=p.parent.name,current_review_file=str(p),historical_review_file=row['review_json'],sha256=row['sha256']))
maps=pd.DataFrame(maps);save(maps,'spine_hash_verified_mapping.csv')
mapped=old.merge(maps,on=['group','animal_id'],validate='one_to_one')
assert len(mapped)==37
save(mapped,'spine_original_values_with_mouse.csv')
mouse=mapped.groupby(['group','mouse'],as_index=False).agg(n_sample_records=('animal_id','size'),density_equal_sample_mean=('density_spines_per_10um','mean'),total_spines=('spines','sum'),total_length_um=('length_xy_um','sum'),n_segments=('segments','sum'))
mouse['density_length_weighted']=10*mouse.total_spines/mouse.total_length_um
assert mouse.groupby('group').size().to_dict()=={g:3 for g in GROUPS}
save(mouse,'spine_mouse_values.csv')
tabs=[];anovas=[];summaries=[]
# Primary: equal weighting of included sample/neuron summaries within mouse,
# then equal weighting of mice. Secondary: pooled counts / pooled lengths.
for value,mode in [('density_equal_sample_mean','primary_equal_sample_within_mouse'),('density_length_weighted','sensitivity_total_count_over_total_length')]:
    q=compare(mouse,value,'Spine_density',mode);q=q.merge(old_tests[['comparison','p_holm']].rename(columns={'p_holm':'old_record_level_p_Holm'}),on='comparison',validate='one_to_one');tabs.append(q)
    anovas.append(factorial(mouse,value,'Spine_density',mode))
    q=summary(mouse,value);q['aggregation']=mode;summaries.append(q)
save(pd.concat(tabs),'spine_comparisons.csv');save(pd.concat(anovas),'spine_factorial_ANOVA.csv');save(pd.concat(summaries),'spine_group_summaries.csv')

# Morphology matching: use numerical content over ALL groups, not folder ID.
oldcurve=read(W/'morphology/data/sholl_source_long.csv')
oldlength=read(W/'morphology/data/total_dendritic_length_source.csv')
cells=[]
for group in GROUPS:
    for folder in (RAW/group).glob('NO*/*'):
        if not folder.is_dir():continue
        row=dict(current_group=group,mouse=folder.parent.name,sample=folder.name,path=str(folder))
        p=folder/'Values.csv';q=folder/'SNT Measurements.csv'
        if p.exists():
            d=read(p);v={int(float(a)):float(b) for a,b in d.iloc[:,:2].to_numpy()};v.setdefault(0,1.)
            row['curve']=np.array([v.get(i,0.) for i in range(0,401,20)])
        if q.exists():
            d=read(q);c=next(c for c in d if c.startswith('Branch length') and c.endswith('[Sum]'))
            row['length']=float(d[c].iloc[0])
        cells.append(row)
matches=[];records=[];curverows=[]
for endpoint,table in [('length',oldlength),('sholl',oldcurve)]:
    for (g,s),part in table.groupby(['group','sample']):
        y=part.total_dendritic_length_um.iloc[0] if endpoint=='length' else part.sort_values('radius_um').intersections.to_numpy()
        found=[r for r in cells if (('length' in r and np.isclose(r['length'],y,rtol=1e-10,atol=1e-8)) if endpoint=='length' else ('curve' in r and np.allclose(r['curve'],y,rtol=1e-10,atol=1e-8)))]
        assert len(found)==1,(endpoint,g,s,len(found));r=found[0]
        matches.append(dict(endpoint=endpoint,old_group=g,old_sample=s,current_group=r['current_group'],current_mouse=r['mouse'],current_sample=r['sample'],current_path=r['path'],group_changed=(g!=r['current_group'])))
        val=float(y) if endpoint=='length' else float(np.trapezoid(y,dx=20))
        for scenario,group in [('old_group_labels',g),('current_folder_labels',r['current_group'])]:
            records.append(dict(scenario=scenario,endpoint=endpoint,group=group,mouse=r['mouse'],sample=r['sample'],value=val))
            if endpoint=='sholl':
                for radius,v in zip(range(0,401,20),y):curverows.append(dict(scenario=scenario,group=group,mouse=r['mouse'],sample=r['sample'],radius_um=radius,intersections=v))
save(pd.DataFrame(matches),'morphology_content_verified_mapping.csv')
rec=pd.DataFrame(records);save(rec,'morphology_preserved_cell_values.csv')
mo=rec.groupby(['scenario','endpoint','group','mouse'],as_index=False).agg(value=('value','mean'),n_cells=('value','size'))
save(mo,'morphology_mouse_values.csv')
mc=[];ma=[];ms=[]
for (scenario,endpoint),part in mo.groupby(['scenario','endpoint']):
    title='Dendritic_length_um' if endpoint=='length' else 'Sholl_AUC_0_400_um_exploratory'
    mc.append(compare(part,'value',title,scenario));ma.append(factorial(part,'value',title,scenario))
    q=summary(part,'value');q['endpoint']=title;q['scenario']=scenario;ms.append(q)
save(pd.concat(mc),'morphology_comparisons_TWO_LABEL_SCENARIOS.csv');save(pd.concat(ma),'morphology_factorial_ANOVA_TWO_LABEL_SCENARIOS.csv');save(pd.concat(ms),'morphology_group_summaries_TWO_LABEL_SCENARIOS.csv')
curves=pd.DataFrame(curverows).groupby(['scenario','group','mouse','radius_um'],as_index=False).intersections.mean()
save(curves,'sholl_mouse_mean_curves.csv')
save(curves.groupby(['scenario','group','radius_um']).intersections.agg(mean='mean',SD='std',SEM='sem',n_mice='count').reset_index(),'sholl_mouse_weighted_summary.csv')
assert art_before=={p:sha(p) for p in art_before}
assert all(sha(p)==h for p,h in inputs.items())
(OUT/'verification.json').write_text(json.dumps(dict(source_files_read=len(inputs),all_input_hashes_unchanged=True,spine_records_matched=37,spine_mice=12,morphology_each_endpoint_records_matched=32,morphology_group_label_conflict='All 17 POCD cell profiles and lengths match opposite old treatment labels; baseline unchanged',figure_hashes_unchanged=art_before,software=dict(Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),method_notes=['Four fixed Welch contrasts with Holm per endpoint; same contrasts as old spine analysis.','3 independent mice per group, group-specific IDs; same NO label across groups is not treated as pairing.','Exact unpaired permutation (20 allocations at 3 vs 3) is a sensitivity analysis, no normality assumption.','Sholl AUC is an exploratory new scalar summary, not a test of the entire radius-dependent profile.','Primary aggregation chosen before calculation: mean included sample densities within mouse; length-weighted density also fully reported.','No raw measurement, sample eligibility, image, manuscript or original statistics overwritten.']),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'input_hashes.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.concat(tabs)[['aggregation','comparison','p_raw','p_Holm_4','old_record_level_p_Holm','exact_permutation_p']].to_string(index=False))
print(pd.concat(anovas).to_string(index=False))
print(pd.concat(mc)[['endpoint','aggregation','comparison','p_Holm_4']].to_string(index=False))
