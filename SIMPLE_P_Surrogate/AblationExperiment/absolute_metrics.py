from pathlib import Path
import json, csv, hashlib
import numpy as np
from summarize import VARIANTS, SEEDS, LABELS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'absolute_error_revision'
OUT.mkdir(exist_ok=True)
groups={}; rows=[]; hashes={}; reference=None
for mode,folder in [('pde','KdV_Physics'),('hybrid','KdV_Hybrid')]:
    groups[mode]={}
    for variant in VARIANTS:
        vals=[]
        for seed in SEEDS:
            p=ROOT/folder/variant/f'seed_{seed}'/'fields.npz'
            hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            z=np.load(p); pred=z['prediction'].astype(np.float64); u=z['truth'].astype(np.float64)
            if reference is None: reference=u
            assert np.array_equal(u,reference)
            err=pred-u; n=u.shape[-2]*u.shape[-1]
            field=np.sqrt(np.mean(err**2,axis=(1,2,3)))
            fft=np.fft.fft2(err,norm='ortho')
            k=np.fft.fftfreq(u.shape[-1])*u.shape[-1]
            mask=(np.abs(k[:,None])>=12)|(np.abs(k[None,:])>=12)
            high=np.sqrt(np.sum(np.abs(fft[...,mask])**2,axis=(1,2))/n)
            # Parseval: equivalent to RMS of the high-pass filtered spatial error.
            filtered=np.fft.ifft2(fft*mask,norm='ortho')
            assert np.allclose(high,np.sqrt(np.mean(np.abs(filtered)**2,axis=(1,2,3))),rtol=1e-12)
            m=json.loads((p.parent/'metrics.json').read_text())
            assert np.isclose(np.mean(err**2),m['selected']['mse'],rtol=2e-6)
            rms=np.sqrt(np.mean(u**2,axis=(1,2,3)))
            assert np.isclose(np.mean(field/rms),m['selected']['relative_l2'],rtol=2e-6)
            assert np.isclose(np.mean(high/rms),m['selected']['high_band_error_over_total_l2'],rtol=2e-6)
            row={'mode':mode,'variant':variant,'seed':seed,'field_rmse':float(field.mean()),'high_rmse':float(high.mean())}
            rows.append(row); vals.append(row)
        groups[mode][variant]={key:{'mean':float(np.mean(a:=[v[key] for v in vals])), 'sd':float(np.std(a,ddof=1)), 'values':a} for key in ['field_rmse','high_rmse']}
pairs={}
for mode in groups:
    pairs[mode]={}
    for a,b in [('Core','Band'),('Core','Local'),('Core','HFB'),('Local','HFB'),('Band','HFB'),('FNO12_matched','HFB'),('FNO44_matched','HFB'),('HFS_Core','HFB')]:
        pairs[mode][a+'__'+b]={}
        for key in ['field_rmse','high_rmse']:
            x=np.array(groups[mode][a][key]['values']);y=np.array(groups[mode][b][key]['values']); diff=x-y
            indices=np.random.default_rng(91024).integers(0,len(diff),(20000,len(diff)))
            ci=np.quantile(diff[indices].mean(1),[.025,.975])
            pairs[mode][a+'__'+b][key]={'difference':float(diff.mean()),'bootstrap95':ci.tolist(),'wins':int((diff>0).sum()),'relative_reduction_percent':float(100*(x.mean()-y.mean())/x.mean())}
rms=np.sqrt(np.mean(reference**2,axis=(1,2,3)))
result={'metric':'Mean of four per-function spatial RMSE values; no reference-field normalization. High-frequency RMSE uses an orthonormal FFT and divides retained squared coefficient error by 96*96 before square root.','reference':{'rms_per_function':rms.tolist(),'rms_mean':float(rms.mean()),'min':float(reference.min()),'max':float(reference.max()),'units':'dimensionless manufactured solution'},'groups':groups,'pairs':pairs,'runs':len(rows),'input_hashes':hashes}
(OUT/'absolute_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
with (OUT/'absolute_metrics.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for mode,g in groups.items():
    for v,a in g.items():print(mode,v,*(f'{a[k]["mean"]:.6f} +/- {a[k]["sd"]:.6f}' for k in ['field_rmse','high_rmse']))
print(json.dumps(pairs,indent=2))
