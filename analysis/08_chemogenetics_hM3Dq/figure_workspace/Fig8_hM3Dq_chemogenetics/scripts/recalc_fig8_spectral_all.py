"""Recompute all sleep states from hash-matched archived inputs, without writes to inputs."""
from pathlib import Path
import argparse, json, sqlite3, hashlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DEFAULT=Path('F:/1.Sleep/PHD稿件/Figure_Workspace/Fig8_hM3Dq_chemogenetics')
RAW=Path('F:/Sleep/Figure/EEG_hm3dq_CNO_spindle_SO_article_package_20260729/01_raw_inputs')
OLD=Path('F:/Sleep/Figure/Fig_hm3Dq_EEG_Spectral_Profile_Origin_20260903/source_data')
BANDS={'Delta':(.5,4),'Theta':(4,8),'Sigma':(10,15)}
STATES={'NREM':2,'REM':3,'Wake':1}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_eeg(p):
 with p.open('rb') as f:
  h=f.read(256);n=int(h[252:256]);sh=f.read(256*n);pos=0;fields=[]
  for width in [16,80,8,8,8,8,8,80,8,32]:
   fields.append([sh[pos+i*width:pos+(i+1)*width].decode('latin1').strip() for i in range(n)]);pos+=n*width
  spr=list(map(int,fields[8]));records=int(h[236:244]);dur=float(h[244:252]);f.seek(int(h[184:192]))
  z=np.fromfile(f,dtype='<i2').reshape(records,sum(spr))[:,:spr[0]].reshape(-1).astype(float)
  lo,hi,dlo,dhi=[float(fields[i][0]) for i in [3,4,5,6]]
  assert fields[2][0] in ['uV','µV']
  return ((z-dlo)*(hi-lo)/(dhi-dlo)+lo).astype(np.float32),spr[0]/dur,fields[0][0]
def interp_psd(psd,f,x):
 ix=np.clip(np.searchsorted(f,x),1,len(f)-1);a=(x-f[ix-1])/(f[ix]-f[ix-1])
 return psd[:,ix-1]*(1-a)+psd[:,ix]*a
def integrate(psd,f,lo,hi,interp=True):
 if not interp:
  m=(f>=lo)&(f<=hi);return np.trapezoid(psd[:,m],f[m],axis=1)
 x=np.r_[lo,f[(f>lo)&(f<hi)],hi]
 return np.trapezoid(interp_psd(psd,f,x),x,axis=1)
def holm(p):
 p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.maximum.accumulate(np.minimum(1,p[order]*np.arange(len(p),0,-1)));return out
def main(W):
 D=W/'data/Spectral_DTS';D.mkdir(parents=True,exist_ok=True);(D/'epochs').mkdir(exist_ok=True)
 inv=pd.read_csv(OLD/'input_inventory.csv');oldbands=pd.read_csv(OLD/'animal_level_band_power_delta_merged.csv')
 oldspec=pd.read_csv(OLD/'animal_level_spectrum_summary.csv')
 rows=[];checks=[];inputs=[];spectra=[]
 for r in inv.itertuples():
  folder=RAW/r.group/r.animal;p=folder/'export_SO200Hz.edf';db=folder/'scores.db3'
  assert sha(p)==r.edf_sha256 and sha(db)==r.db3_sha256
  with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:arr=c.execute('select start_time_seconds,score from sleep_scores_table order by start_time_seconds,start_time_sub_seconds').fetchall()
  times=np.array([a[0] for a in arr]);scores=np.array([a[1] for a in arr]);assert np.all(np.diff(times)==5)
  mask=scores==0;side=folder/'export_SO200Hz.artifacts.json'
  if r.artifact_qc=='verified':
   obj=json.loads(side.read_text());rec=obj['records']['eeg0_emg1'];assert obj['source_fingerprint']['sha256']==r.edf_sha256
   for a,b in rec['final_artifact_ranges']:mask[a:b+1]=True
  assert mask.sum()==r.effective_artifact_epochs
  eeg,fs,label=read_eeg(p);assert fs==200 and len(eeg)==len(scores)*1000;epochs=eeg.reshape(-1,1000)
  tap=np.sqrt(2/1001)*np.sin(np.pi*np.arange(1,6)[:,None]*np.arange(1,1001)[None,:]/1001)
  f=np.fft.rfftfreq(1000,1/fs);grid=np.r_[.5,f[(f>.5)&(f<=25)]]
  info=dict(group=r.group,animal=r.animal,edf=str(p),db3=str(db),edf_sha256=sha(p),db3_sha256=sha(db),artifact_path=str(side) if side.exists() else '',artifact_sha256=sha(side) if side.exists() else '',artifact_qc=r.artifact_qc,eeg_label=label,epoch_seconds=5,sample_rate_hz=fs,start_epoch_index0=0,end_epoch_index0=len(scores)-1,score_start_unix=times[0],score_end_unix=times[-1]+5,artifact_epochs=int(mask.sum()))
  for stage,code in STATES.items():
   valid=(scores==code)&~mask;assert np.isfinite(epochs[valid]).all();idx=np.flatnonzero(valid);vals=epochs[valid].astype(float)
   allrel=[];allabs=[];old=[];psds=[];oldcurves=[]
   for j in range(0,len(vals),256):
    chunk=vals[j:j+256];psd=sum(abs(np.fft.rfft(chunk*t,axis=1))**2/(fs*np.sum(tap[0]**2)) for t in tap)/5;psd[:,1:-1]*=2
    den=integrate(psd,f,.5,25);assert (den>0).all();ab=np.column_stack([integrate(psd,f,*bounds) for bounds in BANDS.values()])
    allabs.append(ab);allrel.append(ab/den[:,None]*100);psds.append(interp_psd(psd,f,grid)/den[:,None]*100)
    oldden=integrate(psd,f,1,25,False)
    old.append(np.column_stack([integrate(psd,f,a,b,False)/oldden*100 for a,b in [(1,4.4),(6,9),(9,13),(13,25)]]))
    oldcurves.append(psd[:,(f>=1)&(f<=25)]/oldden[:,None]*100)
   rel=np.vstack(allrel);ab=np.vstack(allabs);legacy=np.vstack(old).mean(axis=0);sp=np.vstack(psds).mean(axis=0)
   assert np.isclose(np.trapezoid(sp,grid),100,atol=1e-9)
   for k,b in enumerate(['Delta','Theta','Alpha','Beta']):
    o=oldbands[(oldbands.group==r.group)&(oldbands.animal==r.animal)&(oldbands.stage==stage)&(oldbands.band==b)].iloc[0]
    error=legacy[k]-o.relative_mean_percent;assert abs(error)<.0001,(r.animal,stage,b,error)
    checks.append(dict(group=r.group,animal=r.animal,stage=stage,metric=b,old=o.relative_mean_percent,reproduced=legacy[k],error=error))
   oldcurve=oldspec[(oldspec.group==r.group)&(oldspec.animal==r.animal)&(oldspec.stage==stage)].sort_values('frequency_hz')
   err=np.max(np.abs(np.vstack(oldcurves).mean(axis=0)-oldcurve.mean_relative_spectrum.to_numpy()));assert err<.0001,(stage,err)
   checks.append(dict(group=r.group,animal=r.animal,stage=stage,metric='PSD_max_abs_error',error=err))
   for k,(b,(lo,hi)) in enumerate(BANDS.items()):rows.append(dict(group=r.group,animal=r.animal,stage=stage,band=b,low_hz=lo,high_hz=hi,valid_epochs=len(vals),relative_mean_percent=rel[:,k].mean(),absolute_mean=ab[:,k].mean()))
   spectra.append(pd.DataFrame(dict(group=r.group,animal=r.animal,stage=stage,frequency_hz=grid,relative_psd_percent_per_hz=sp)))
   pd.DataFrame({'epoch_index0':idx,'start_s':idx*5,**{b:rel[:,k] for k,b in enumerate(BANDS)}}).to_csv(D/'epochs'/f'{r.group}_{r.animal}_{stage}.csv',index=False)
   info[stage+'_valid_epochs']=len(vals)
  inputs.append(info);print(r.group,r.animal,'three states validated/recalculated',flush=True)
 df=pd.DataFrame(rows);df.to_csv(D/'animal_level_band_power.csv',index=False);pd.concat(spectra).to_csv(D/'animal_level_spectra.csv',index=False)
 pd.DataFrame(checks).to_csv(D/'legacy_reproduction_QA.csv',index=False);pd.DataFrame(inputs).to_csv(D/'input_inventory.csv',index=False)
 stats=[]
 for stage in STATES:
  for b in BANDS:
   a=df[(df.band==b)&(df.stage==stage)&(df.group=='cont1')].relative_mean_percent.to_numpy();c=df[(df.band==b)&(df.stage==stage)&(df.group=='hmdq3')].relative_mean_percent.to_numpy()
   assert len(a)==8 and len(c)==6 and len(np.unique(np.r_[a,c]))==14
   u,p=mannwhitneyu(a,c,alternative='two-sided',method='exact')
   stats.append(dict(stage=stage,band=b,n_CON=8,n_hM3Dq=6,CON_mean=a.mean(),CON_SEM=a.std(ddof=1)/np.sqrt(8),hM3Dq_mean=c.mean(),hM3Dq_SEM=c.std(ddof=1)/np.sqrt(6),U=u,p_raw=p))
 st=pd.DataFrame(stats);st['p_Holm_3']=st.groupby('stage').p_raw.transform(lambda p:holm(p));st['p_Holm_9_sensitivity']=holm(st.p_raw)
 st.to_csv(D/'statistics.csv',index=False)
 # The expanded analysis must reproduce the already accepted NREM values.
 prev=pd.read_csv(W/'data/NREM_Delta_Theta_Sigma/animal_level_band_power.csv')
 merged=df[df.stage=='NREM'].merge(prev,on=['group','animal','stage','band'],suffixes=('_new','_prior'))
 assert len(merged)==42 and np.allclose(merged.relative_mean_percent_new,merged.relative_mean_percent_prior,atol=1e-10)
 (D/'parameters.json').write_text(json.dumps(dict(bands=BANDS,reference=[.5,25],states=STATES,epoch_seconds=5,sample_rate=200,tapers='five orthogonal sine; same legacy estimator',boundary_method='linear interpolation; exact-edge trapezoid integration',aggregation='mean per-epoch relative power/PSD within each state and mouse, then equal animal weighting',statistics='two-sided exact Mann-Whitney U; Holm three bands per state; all-nine sensitivity also reported'),indent=2),encoding='utf-8')
 print(st.to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=DEFAULT);main(p.parse_args().workspace)
