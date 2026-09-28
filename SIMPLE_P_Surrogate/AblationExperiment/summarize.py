from pathlib import Path
import json, numpy as np, csv
ROOT=Path(__file__).resolve().parent
VARIANTS=['Core','Band','Local','HFB','FNO12_matched','FNO44_matched','HFS_Core']
LABELS={'Core':'Core','Band':'Core + spectral branch','Local':'Core + local branch','HFB':'Full HFB','FNO12_matched':'FNO, 12 modes','FNO44_matched':'FNO, 44 modes','HFS_Core':'Core + frequency scaling'}
SEEDS=list(range(20260924,20260934))
def gather(require_complete=True):
 allrows=[]; groups={};pairs={}
 for mode,folder in [('hybrid','KdV_Hybrid'),('pde','KdV_Physics')]:
  groups[mode]={}
  for v in VARIANTS:
   rows=[]
   for seed in SEEDS:
    p=ROOT/folder/v/f'seed_{seed}'/'metrics.json'
    if not p.exists():
     if require_complete:raise RuntimeError(f'Missing {p}')
     continue
    m=json.loads(p.read_text());rows.append(m)
    allrows.append({'mode':mode,'variant':v,'seed':seed,'params':m['params'],'selected_epoch':m['selected_epoch'],'training_s':m['training_s'],'inference_s':m['inference_s_single_field'],**m['selected']})
   if not rows:continue
   g={'n':len(rows),'params':rows[0]['params'],'label':LABELS[v],'seeds':[r['seed'] for r in rows]}
   for k in ['relative_l2','high_band_error_over_total_l2','high_band_relative_l2','pde_mse','pde_mse_absolute','reference_high_energy_fraction','training_s','inference_s_single_field']:
    a=np.array([r[k] if k in r else r['selected'][k] for r in rows])
    g[k]={'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None,'values':a.tolist()}
   groups[mode][v]=g
  pairs[mode]={}
  for ref,alt in [('Core','Band'),('Core','Local'),('Core','HFB'),('Local','HFB'),('Band','HFB'),('FNO12_matched','HFB'),('FNO44_matched','HFB'),('HFS_Core','HFB')]:
   if ref not in groups[mode] or alt not in groups[mode]:continue
   a,b=groups[mode][ref],groups[mode][alt]
   if a['seeds']!=b['seeds']:continue
   pair={}
   for k in ['relative_l2','high_band_error_over_total_l2']:
    x=np.array(a[k]['values']);y=np.array(b[k]['values']);d=100*(x-y)
    ix=np.random.default_rng(91024).integers(0,len(d),(20000,len(d)))
    lo,hi=np.quantile(d[ix].mean(1),[.025,.975])
    pair[k]={'difference_percentage_points':float(d.mean()),'bootstrap95':[float(lo),float(hi)],'wins':int((d>0).sum()),'n':len(d),'relative_reduction_percent':float(100*(x.mean()-y.mean())/x.mean())}
   pairs[mode][ref+'__'+alt]=pair
 result={'groups':groups,'paired_comparisons':pairs,'runs':len(allrows),'fixed_split':True,'bootstrap':'20000 paired resamples of training seeds; percentile CI conditional on fixed test functions'}
 (ROOT/'results_summary.json').write_text(json.dumps(result,indent=2))
 if allrows:
  with (ROOT/'all_runs.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 return result
if __name__=='__main__':
 r=gather('--partial' not in __import__('sys').argv)
 print('Completed',r['runs'])
 for mode,g in r['groups'].items():
  for v,a in g.items():print(mode,v,a['n'],round(a['relative_l2']['mean']*100,3),round(a['high_band_error_over_total_l2']['mean']*100,3))
