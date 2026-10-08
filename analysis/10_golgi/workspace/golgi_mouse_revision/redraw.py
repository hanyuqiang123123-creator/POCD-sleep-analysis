from pathlib import Path
import json, shutil, hashlib, csv
import numpy as np
import pandas as pd
import pymupdf as fitz
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'figure'
OUT.mkdir(exist_ok=True)
(OUT/'panels').mkdir(exist_ok=True)
ROOT=Path(r'F:\1.Sleep\PHD稿件')
WORK=ROOT/'Figure_Workspace/Fig10_Golgi_spine_density/relayout_NOR_Ymaze'
DATA=BASE.parent/'source_audit_20261005/golgi_mouse_level_check'
SPECS={s['id']:s for s in json.loads((BASE.parent/'origin_fig1_fig8/specs.json').read_text(encoding='utf8'))}
PW,PH=183/25.4*72/4,124
AW,AH=SPECS['F8D']['aw'],SPECS['F8D']['ah']; L,T=35,34
GROUPS=['Baseline+SAL','Baseline+CNO','POCD+SAL','POCD+CNO']
plt.rcParams.update({'font.family':'Arial','font.size':6.5,'axes.linewidth':.7,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
 'pdf.fonttype':42,'legend.frameon':False,'axes.unicode_minus':True})
contract={'conclusion':'EW inhibition increases hippocampal pyramidal-neuron spine density in the primary mouse-level analysis; arbor summaries do not establish equivalence.',
 'evidence':'A/D representative images; B length; C Sholl; E spine density; F-K existing cognitive evidence.',
 'archetype':'image plate + quant','backend':'Python','dimensions_mm':[183,PH*3/72*25.4],
 'unit':'3 independent mice per group for B,C,E; within-mouse sample averages; group mean and SEM across mice. B/E display all nested cell/sample points as requested.',
 'changes':['B: all traced-cell points with mouse-level mean/SEM and confirmed current labels','C: mouse-averaged curves and SEM; confirmed current labels','E: all sample points with mouse-level mean/SEM, both corrected comparisons marked *'],
 'image_integrity':'All representative image and behavior panels retained byte-for-byte at individual-panel level; no pixel adjustments, relabeling or scale-bar invention.',
 'limitations':'Small n; sensitivity analyses in manuscript and source tables; microscopy scale calibration and sampling details remain author queries.'}
(BASE/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf8')
for p in (WORK/'panels').glob('Fig8_*'):
 if p.is_file(): shutil.copy2(p,OUT/'panels'/p.name)

def make(letter,title):
 f=plt.figure(figsize=(PW/72,PH/72),facecolor='white')
 f.text(4/PW,1-10/PH,letter,fontsize=10,fontweight='bold',va='center')
 f.text((L+AW/2)/PW,1-17/PH,title,ha='center',va='center',fontsize=7.5)
 ax=f.add_axes([L/PW,1-(T+AH)/PH,AW/PW,AH/PH])
 ax.tick_params(length=2,width=.7,pad=2,labelsize=6.5)
 return f,ax

def save(f,letter):
 stem=OUT/'panels'/f'Fig8_{letter}'
 for ext in ['pdf','svg','png']: f.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
 f.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 for ext in ['pdf','svg']:shutil.copy2(stem.with_suffix('.'+ext),stem.with_name(stem.name+'_AI_clean.'+ext))
 plt.close(f)

spine=pd.read_csv(DATA/'spine_mouse_values.csv')
length=pd.read_csv(DATA/'morphology_mouse_values_CONFIRMED_CURRENT_GROUPS.csv').query("endpoint == 'length'")
spine_points=pd.read_csv(DATA/'spine_original_values_with_mouse.csv')
length_points=pd.read_csv(DATA/'morphology_preserved_cell_values.csv').query("scenario == 'current_folder_labels' and endpoint == 'length'")
stats=pd.read_csv(DATA/'spine_comparisons.csv').query("aggregation == 'primary_equal_sample_within_mouse'")
qa={'bars':{},'retained_panel_hashes':{},'spine_adjusted_p':stats[['comparison','p_Holm_4']].to_dict('records')}
for letter,sid,df,col in [('B','F8D',length,'value'),('E','F8C',spine,'density_equal_sample_mean')]:
 s=SPECS[sid];f,ax=make(letter,s['title']); rows=[]
 for i,g in enumerate(GROUPS):
  d=df[df.group==g].sort_values('mouse'); vals=d[col].to_numpy();assert len(vals)==3
  m=float(vals.mean());sem=float(vals.std(ddof=1)/np.sqrt(3));x=s['x'][i]
  edge=np.array(s['edges'][i])/255;fill=np.array(s['fills'][i])/255
  ax.bar(x,m,width=s['bar_width'],color=fill,edgecolor=edge,lw=.7,zorder=2)
  points=(length_points if letter=='B' else spine_points)
  points=points[points.group==g].sort_values(['mouse','sample' if letter=='B' else 'current_sample'])
  py=points['value' if letter=='B' else 'density_spines_per_10um'].to_numpy()
  # Fixed horizontal jitter separates nested observations; it encodes no additional variable.
  offsets=np.random.default_rng(8100+i).permutation(np.linspace(-.075,.075,len(py)))
  ax.scatter(x+offsets,py,s=7.5,facecolor='white',edgecolor=edge,lw=.65,zorder=4)
  ax.errorbar(x,m,yerr=sem,color='black',lw=.7,capsize=2,capthick=.7,zorder=5)
  rows.append({'group':g,'mouse_ids':d.mouse.tolist(),'values':vals.tolist(),'mean':m,'SEM':sem,'n_mice':3,'displayed_point_count':len(py),'displayed_values':py.tolist(),'point_mouse_ids':points.mouse.tolist()})
 ax.set(xlim=s['xlim'],ylim=s['ylim'],yticks=s['yticks'],xticks=s['x'],xticklabels=['SAL','CNO','SAL','CNO'])
 ax.set_ylabel(s['ylabel'],fontsize=7,labelpad=4);ax.yaxis.set_label_coords((10-L)/AW,.5)
 ax.yaxis.label.set_verticalalignment('center')
 for x,lab in [(0,'Baseline'),(1,'POCD')]:ax.text(x,-.28,lab,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6.5)
 if letter=='E':
  for b,comparison in zip(s['brackets'],['POCD+SAL minus Baseline+SAL','POCD+CNO minus POCD+SAL']):
   p=float(stats.set_index('comparison').loc[comparison,'p_Holm_4']);stars='****' if p<.0001 else '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
   assert stars=='*'
   x1,x2=s['x'][b['i']],s['x'][b['j']]; y=b['y']; h=.5
   ax.plot([x1,x1,x2,x2],[y-h,y,y,y-h],color='black',lw=.7,clip_on=False)
   ax.annotate(stars,((x1+x2)/2,y),xytext=(0,2),textcoords='offset points',ha='center',va='bottom',fontsize=7.5)
 qa['bars'][letter]=rows; save(f,letter)

df=pd.read_csv(DATA/'sholl_mouse_weighted_summary.csv').query("scenario == 'current_folder_labels'")
f,ax=make('C','Sholl analysis');s=SPECS['F8B']
for i,g in enumerate(GROUPS):
 d=df[df.group==g].sort_values('radius_um');assert (d.n_mice==3).all()
 x,y,e=d.radius_um.to_numpy(),d['mean'].to_numpy(),d.SEM.to_numpy();c=np.array(s['colors'][i])/255
 ax.fill_between(x,y-e,y+e,color=c,alpha=.18,lw=0)
 ax.plot(x,y,color=c,lw=.7,label=g.replace('+',' + '))
ax.set(xlim=(0,400),ylim=(0,9),xticks=[0,100,200,300,400],yticks=[0,2,4,6,8])
ax.set_xlabel('Distance from soma (µm)',fontsize=7,labelpad=3)
ax.set_ylabel('Intersections',fontsize=7,labelpad=4)
f.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.005),ncol=2,fontsize=5.2,handlelength=1.5,handletextpad=.4,columnspacing=.8,labelspacing=.3)
save(f,'C')
positions={'A':(0,0),'B':(2,0),'C':(3,0),'D':(0,1),'E':(1,1),'F':(2,1),'G':(3,1),'H':(0,2),'I':(1,2),'J':(2,2),'K':(3,2)}
combined=fitz.open();page=combined.new_page(width=4*PW,height=3*PH)
for letter,(c,r) in positions.items():
 path=OUT/'panels'/f'Fig8_{letter}.pdf';sub=fitz.open(path)
 page.show_pdf_page(fitz.Rect(c*PW,r*PH,(c+(2 if letter=='A' else 1))*PW,(r+1)*PH),sub,0)
 if letter not in 'BCE':
  assert path.read_bytes()==(WORK/'panels'/path.name).read_bytes()
  qa['retained_panel_hashes'][letter]=hashlib.sha256(path.read_bytes()).hexdigest()
combined.save(OUT/'Fig8_relayout.pdf',garbage=4,deflate=True)
page.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(OUT/'Fig8_relayout.png')
(OUT/'Fig8_relayout.svg').write_text(page.get_svg_image(text_as_path=False),encoding='utf8')
pix=page.get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False)
Image.frombytes('RGB',(pix.width,pix.height),pix.samples).save(OUT/'Fig8_relayout.tiff',compression='tiff_lzw',dpi=(600,600))
for ext in ['pdf','svg']:shutil.copy2(OUT/f'Fig8_relayout.{ext}',OUT/f'Fig8_relayout_AI_clean.{ext}')
qa['updated_panels']=['B','C','E'];qa['contract']=contract
(BASE/'figure_QA.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf8')
print(OUT/'Fig8_relayout.png')
