from pathlib import Path
import subprocess,sys,json,datetime
from kdv_ablation import VARIANTS,SEEDS,audit
root=Path(__file__).resolve().parent/'reruns'
root.mkdir(exist_ok=True)
for mode,folder in [('hybrid','KdV_Hybrid'),('pde','KdV_Physics')]:
 out=root/folder
 audit(out,mode)
jobs=[]
for i,seed in enumerate(SEEDS):
 order=VARIANTS[i%len(VARIANTS):]+VARIANTS[:i%len(VARIANTS)]
 for variant in order:
  for mode,folder in [('hybrid','KdV_Hybrid'),('pde','KdV_Physics')]:jobs.append((variant,seed,mode,folder))
for j,(variant,seed,mode,folder) in enumerate(jobs,1):
 out=root/folder; dest=out/variant/f'seed_{seed}'
 if (dest/'metrics.json').exists():continue
 dest.mkdir(parents=True,exist_ok=True)
 state={'job':j,'total':len(jobs),'variant':variant,'seed':seed,'mode':mode,'status':'running','updated':datetime.datetime.now().isoformat()}
 (root/'status.json').write_text(json.dumps(state,indent=2))
 print(f'START {j}/{len(jobs)} {mode} {variant} {seed}',flush=True)
 with (dest/'run.log').open('w',encoding='utf-8') as log:
  r=subprocess.run([sys.executable,'-u',str(root/'kdv_ablation.py'),'--out',str(out),'--variant',variant,'--seed',str(seed),'--epochs','1200','--mode',mode],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  state['status']='failed';(root/'status.json').write_text(json.dumps(state));print((dest/'run.log').read_text()[-3000:],flush=True);raise SystemExit(r.returncode)
 m=json.loads((dest/'metrics.json').read_text());print(f'DONE {j} L2={m["selected"]["relative_l2"]:.5f} train_s={m["training_s"]:.1f}',flush=True)
(root/'status.json').write_text(json.dumps({'status':'complete','runs':len(jobs)}))
