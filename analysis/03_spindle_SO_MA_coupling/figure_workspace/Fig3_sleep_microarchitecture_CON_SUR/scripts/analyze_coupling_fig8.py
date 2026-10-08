from pathlib import Path
import sys,json,ast,heapq,hashlib,gc
from dataclasses import asdict
from datetime import datetime
import numpy as np
import pandas as pd
sys.path.insert(0,'D:/SleepWorkbench_source_20260913/SleepWorkbench')
from sleepworkbench.workbench_core import read_edf_signals,ScoreDatabase
from sleepworkbench.analysis_fingerprint import score_fingerprint
from sleepworkbench.artifact_store import load_verified_artifact_masks
from sleepworkbench.spindle_detection import SpindleDetectionConfig,detect_multichannel_spindles
from sleepworkbench.spindle_review_store import SpindleReviewStore,build_spindle_review_scope

WORK=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig3_sleep_microarchitecture_CON_SUR')
OUT=WORK/'data/Coupling_Fig8_method';OUT.mkdir(exist_ok=True)
(OUT/'events').mkdir(exist_ok=True)
ref=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig8_hM3Dq_chemogenetics/scripts/redraw_fig8_new_so_spindle_coupling.py')
tree=ast.parse(ref.read_text(encoding='utf-8-sig'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='couple_events'],type_ignores=[]),str(ref),'exec'))
source=pd.read_csv(WORK/'data/SO_refresh/animal_density.csv')
config=SpindleDetectionConfig.recommended_mouse()
rows=[]; failures=[]
for i,r in enumerate(source.itertuples(index=False),1):
 try:
  edf=Path(r.source_edf);db=ScoreDatabase(Path(r.source_db));h,signals=read_edf_signals(edf);fs=h['sample_rates'][0]
  meta=json.loads(Path(r.metadata_path).read_text(encoding='utf-8-sig'))
  assert score_fingerprint(db.scores,db.epoch_seconds)==meta['score_sha256']
  start=datetime.strptime(h['start_date']+' '+h['start_time'],'%d.%m.%y %H.%M.%S').timestamp()
  assert start==db.rows[0].start_seconds and len(signals[0])/fs==len(db.scores)*db.epoch_seconds
  masks,audits=load_verified_artifact_masks(edf,epoch_count=len(db.scores),epoch_seconds=db.epoch_seconds,sample_rate=fs,emg_channel_index=1,eeg_channel_indices=[0],channel_labels=h['labels'],verify_checksum=True)
  result=detect_multichannel_spindles([(0,h['labels'][0],signals[0],h['physical_dimensions'][0])],fs,db.scores,db.epoch_seconds,config=config,artifact_masks=masks)
  store=SpindleReviewStore();scope=build_spindle_review_scope(source_edf=edf,source_db=Path(r.source_db),session_number=db.session_number,sample_rate=fs,config=config)
  restored=store.apply_reviews(scope,result.events);audit=restored.audit_metadata(store.path)
  nrem=result.channel_results[0].nrem_seconds/60
  assert abs(nrem-r.nrem_minutes)<1e-8, f'NREM denominator mismatch {nrem} vs {r.nrem_minutes}'
  sp=result.events_frame();sp=sp[sp.accepted.astype(str).str.lower().eq('true')].copy()
  sop=Path(r.summary_path).with_name('export_200Hz_so_events.tsv')
  so=pd.read_csv(sop,sep='\t');so=so[(so.channel_index_0based==0)&so.accepted.astype(str).str.lower().eq('true')].copy()
  assert len(so)==r.SO_accepted
  coupled=couple_events(sp,so)
  assert len({x['spindle_event_id'] for x in coupled})==len(coupled)<=len(sp)
  stem=f'{r.group}_{r.animal}_{r.day}'
  sp.to_csv(OUT/'events'/f'{stem}_spindles.csv',index=False)
  pd.DataFrame(coupled).to_csv(OUT/'events'/f'{stem}_coupled.csv',index=False)
  row=dict(group=r.group,animal=r.animal,day=r.day,nrem_minutes=nrem,spindle_count=len(sp),so_count=len(so),coupled_count=len(coupled),uncoupled_count=len(sp)-len(coupled),coupled_density=len(coupled)/nrem,uncoupled_density=(len(sp)-len(coupled))/nrem,coupled_percent=100*len(coupled)/len(sp) if len(sp) else np.nan,amplitude_percentile=r.amplitude_percentile,artifact_validity=r.artifact_validity,source_edf=str(edf),source_db=r.source_db,so_event_path=str(sop),so_event_sha256=hashlib.sha256(sop.read_bytes()).hexdigest(),spindle_reviews_restored=audit.get('restored_count',0),spindle_reviews_excluded=audit.get('excluded_count',0))
  rows.append(row)
  print(f'{i}/96 {stem}: spindle={len(sp)} coupled={len(coupled)} density={len(coupled)/nrem:.4f}',flush=True)
  del signals,result;gc.collect()
 except Exception as e:
  failures.append(dict(group=r.group,animal=r.animal,day=r.day,error=str(e)));print('FAILED',failures[-1],flush=True)
 pd.DataFrame(rows).to_csv(OUT/'animal_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(failures).to_csv(OUT/'failures.csv',index=False)
(OUT/'method.json').write_text(json.dumps(dict(fig8_script=str(ref),fig8_script_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),spindle_config=asdict(config),definition='Accepted spindle peak within accepted SO interval; nearest down peak resolves overlaps; count each spindle once',phase_analysis=False,records=len(rows),failures=failures),indent=2),encoding='utf-8')
assert len(rows)==96 and not failures
