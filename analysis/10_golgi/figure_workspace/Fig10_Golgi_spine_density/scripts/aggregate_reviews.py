from pathlib import Path
import json,csv,hashlib,math,shutil
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'F:\1.Sleep\eXdata\golgi\Baseline+SAL\260108')
QC=Path(r'C:\Users\USER\Documents\ChatGPT\PHD文章\Golgi_spine_QC_20260927')
GROUPS=['Baseline+SAL','Baseline+CNO','POCD+SAL','POCD+CNO']
def write(name,rows):
 with (ROOT/'data'/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
segments=[];animals=[];sources=[];versions=[]
for group in GROUPS:
 for folder in sorted((SOURCE/group).iterdir(),key=lambda p:int(p.name) if p.name.isdigit() else 999):
  if not folder.is_dir():continue
  files=list(folder.glob('folder*_manual_review*.json'))
  if not files:raise ValueError(f'No review: {folder}')
  chosen=max(files,key=lambda p:p.stat().st_mtime_ns)
  for f in files:
   j=json.loads(f.read_text(encoding='utf-8-sig'))
   versions.append(dict(group=group,animal=folder.name,file=str(f),selected=f==chosen,mtime_ns=f.stat().st_mtime_ns,included=sum(p['status']=='include' for s in j['segments'] for p in s['points']),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  j=json.loads(chosen.read_text(encoding='utf-8-sig'));q=QC/f'folder{folder.name}_density'
  lengthfile=q/'segment_measurements.csv'
  if not lengthfile.exists():
   lengthfile=q/'reviewed_segment_results.csv'
  with lengthfile.open(encoding='utf-8-sig') as f:lengthrows=list(csv.DictReader(f))
  lengths={r['segment']:float(r.get('projected_length_um') or r.get('length_um')) for r in lengthrows}
  subtotal=[]
  for s in j['segments']:
   points=s['points'];assert len({p['id'] for p in points})==len(points)
   counts=Counter(p['status'] for p in points);assert set(counts)<= {'include','exclude','uncertain'}
   L=lengths[s['segment']];assert L>0
   row=dict(group=group,animal_id=folder.name,segment=s['segment'],included_green=counts['include'],orange_reference=counts['uncertain'],excluded=counts['exclude'],length_xy_um=L,density_spines_per_10um=10*counts['include']/L,review_json=str(chosen),count_basis='all saved green points; folder1 mixed original+manual' if folder.name=='1' else 'complete user green recount')
   subtotal.append(row);segments.append(row)
  N=sum(r['included_green'] for r in subtotal);L=sum(r['length_xy_um'] for r in subtotal)
  animals.append(dict(group=group,animal_id=folder.name,segments=len(subtotal),spines=N,length_xy_um=L,density_spines_per_10um=10*N/L))
  dest=ROOT/'data'/'review_sources'/group/folder.name;dest.mkdir(parents=True,exist_ok=True)
  shutil.copy2(chosen,dest/chosen.name);shutil.copy2(lengthfile,dest/lengthfile.name)
  sources.append(dict(group=group,animal_id=folder.name,source_json=str(chosen),sha256=hashlib.sha256(chosen.read_bytes()).hexdigest(),length_source=str(lengthfile)))
stats=[]
for g in GROUPS:
 v=[a['density_spines_per_10um'] for a in animals if a['group']==g];n=len(v);mean=sum(v)/n;sem=math.sqrt(sum((x-mean)**2 for x in v)/(n-1)/n) if n>1 else ''
 stats.append(dict(group=g,n_animals=n,mean=mean,sem=sem,unit='spines/10 um',inference='descriptive only; baseline groups n=1'))
for name,rows in [('segment_data.csv',segments),('animal_data.csv',animals),('group_summary.csv',stats),('source_audit.csv',sources),('review_versions.csv',versions)]:write(name,rows)
print(json.dumps(stats,indent=2));print('Animals',len(animals),'segments',len(segments))
