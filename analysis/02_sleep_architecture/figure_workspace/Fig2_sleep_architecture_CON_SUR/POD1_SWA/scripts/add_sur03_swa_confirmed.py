from pathlib import Path
import sys, sqlite3, hashlib, json, shutil
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.stats import ttest_ind, t
from statsmodels.stats.multitest import multipletests

BACKEND=Path('F:/Sleep/EEGsoftware/EEGSoftware/AutoFQ/SleepWorkbench')
sys.path.insert(0,str(BACKEND))
from sleepworkbench.slow_wave_activity import analyze_slow_wave_activity, SlowWaveActivityConfig
from sleepworkbench.edf_preprocessing import read_edf_metadata
BASE=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig2_sleep_architecture_CON_SUR')
OUT=BASE/'POD1_SWA'
for name in ['data','scripts','documentation']:(OUT/name).mkdir(parents=True,exist_ok=True)
src=pd.read_csv(BASE/'POD1_hourly/documentation/source_paths.csv')
records=pd.read_csv(OUT/'data/animal_swa.csv').query("animal != 'SUR_NO.03'").to_dict('records'); audit=pd.read_csv(OUT/'documentation/source_paths.csv').query("animal != 'SUR_NO.03'").to_dict('records'); blocked=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for r in src.itertuples():
 if r.animal!='SUR_NO.03':continue
 db=Path(r.path);edf=db.parent/'export.edf';m=read_edf_metadata(edf)
 with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
  q=np.array(c.execute('select start_time_seconds, end_time_seconds, score from sleep_scores_table where session_number=0 order by start_time_seconds').fetchall())
 assert len(q)==17280 and np.all(np.diff(q[:,0])==5)
 start=datetime.strptime(m['start_date']+' '+m['start_time'],'%d.%m.%y %H.%M.%S').replace(tzinfo=timezone(timedelta(hours=8))).timestamp()
 # User explicitly confirmed POD1 EEG and scoring are the same recording,
 # with a date-only error; use relative epoch offsets, preserving source metadata.
 assert np.all(q[:,1]-q[:,0]==5)
 confirmed_offset=float(start-q[0,0])
 (OUT/'documentation/SUR_NO03_alignment_confirmation.json').write_text(json.dumps(dict(animal=r.animal,confirmation='User confirmed matching recording; date error only',edf_start=start,score_start=int(q[0,0]),relative_alignment_offset_seconds=confirmed_offset,originals_modified=False),indent=2),encoding='utf-8')
 ch=1;fs=m['sample_rates'][ch];assert fs==600 and m['duration_seconds']==86400
 ns=m['samples_per_record'];mm=np.memmap(edf,dtype='<i2',mode='r',offset=m['header_bytes'],shape=(m['record_count'],sum(ns)))
 a=np.asarray(mm[:,sum(ns[:ch]):sum(ns[:ch+1])]).reshape(-1).astype(float)
 a=(a-m['digital_min'][ch])*(m['physical_max'][ch]-m['physical_min'][ch])/(m['digital_max'][ch]-m['digital_min'][ch])+m['physical_min'][ch]
 a=sosfiltfilt(butter(4,.1,btype='highpass',fs=fs,output='sos'),a)
 a=resample_poly(a,1,3);del mm
 result=analyze_slow_wave_activity([(ch,m['labels'][ch],a,m['physical_dimensions'][ch])],200,q[:,2],5,config=SlowWaveActivityConfig())
 z=result.channel_results[0];del a
 np.savez_compressed(OUT/'data'/f'{r.animal}_windows.npz',center_s=z.window_center_seconds,swa_uv2=z.smoothed_swa_uv2,normalized_percent=z.normalized_swa_percent,raw_swa_uv2=z.raw_swa_uv2)
 group=r.animal.split('_')[0]
 for label,lo,hi in [('24h',0,86400),('Light',0,43200),('Dark',43200,86400)]+[(f'ZT{i:02d}-{i+2:02d}',i*3600,(i+2)*3600) for i in range(0,24,2)]:
  keep=(z.window_center_seconds>=lo)&(z.window_center_seconds<hi)&np.isfinite(z.smoothed_swa_uv2)
  records.append(dict(animal=r.animal,group=group,period=label,valid_windows=int(keep.sum()),swa_uv2=float(np.mean(z.smoothed_swa_uv2[keep])) if keep.any() else np.nan,normalized_percent=float(np.mean(z.normalized_swa_percent[keep])) if keep.any() else np.nan))
 audit.append(dict(animal=r.animal,edf=str(edf),edf_sha256=sha(edf),db=str(db),db_sha256=sha(db),channel=m['labels'][ch],unit=m['physical_dimensions'][ch],nrem_epochs=int((q[:,2]==2).sum()),artifact_mask='missing; exploratory',preprocessing='raw EDF; fourth-order 0.1Hz zero-phase highpass; polyphase 600->200Hz'))
 pd.DataFrame(records).to_csv(OUT/'data/animal_swa.csv',index=False)
 pd.DataFrame(audit).to_csv(OUT/'documentation/source_paths.csv',index=False)
 print('DONE',r.animal,records[-15]['swa_uv2'],flush=True)
pd.DataFrame(blocked).to_csv(OUT/'documentation/unresolved_alignment.csv',index=False)

shutil.copy2(__file__,OUT/"scripts"/Path(__file__).name)
