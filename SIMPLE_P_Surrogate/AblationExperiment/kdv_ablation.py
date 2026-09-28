"""Controlled ablation of the existing OtherPDEs legacy strongly nonlinear KdV model.

Historical files are read-only. No new PDE distribution or pump data are added.
The core retains BOTH legacy low-band paths, Fourier input features and GELUs.
Only the extra high-band weights and local high-pass path are ablated.
"""
from __future__ import annotations
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, argparse, copy, csv, hashlib, json, time, platform
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'sources'
sys.path.insert(0, str(LEGACY))
import run_legacy_hf_pde_suite as bench
import NeuroOperators as ops

VARIANTS = ['Core', 'Band', 'Local', 'HFB', 'FNO12_matched', 'FNO44_matched', 'HFS_Core']
SEEDS = list(range(20260924,20260934))

class PrunedLegacyBlock(nn.Module):
    def __init__(self, original, band: bool, local: bool):
        super().__init__()
        self.low_fno = copy.deepcopy(original.low_fno)
        self.band_spectral = copy.deepcopy(original.band_spectral)
        if not band:
            self.band_spectral.high_modes = 0
            for name in ['weights_high_pos','weights_high_neg','weights_high_y_pos',
                         'weights_high_y_neg','weights_high_xy_pos','weights_high_xy_neg']:
                self.band_spectral.register_parameter(name, None)
        self.local_high = copy.deepcopy(original.local_high) if local else None
        self.high_gate = nn.Parameter(original.high_gate.detach().clone())
        width = original.low_fno.in_channels
        self.fuse = nn.Conv2d(width * (3 if local else 2), width, 1)
        with torch.no_grad():
            self.fuse.weight.copy_(original.fuse.weight[:, :width*(3 if local else 2)])
            self.fuse.bias.copy_(original.fuse.bias)

    def forward(self, x):
        low, band = self.low_fno(x), self.band_spectral(x)
        parts = [low, band]
        extra = band
        if self.local_high is not None:
            local = self.local_high(x)
            parts.append(local)
            extra = extra + local
        return low + self.high_gate.sigmoid() * extra + self.fuse(torch.cat(parts, 1))

class HFS(nn.Module):
    """Authors' featscale2 operation, transplanted to the common core.

    Source: SiaK4/HFS_ResUNet/Models/ResUnet_HFS.py, featscale2.
    Mean is across patches at matching intra-patch offsets, NOT a local mean.
    Authors initialize both residual scaling parameters to one (initially 2*x).
    This is a module-level adaptation, not a reproduction of the ResUNet study.
    """
    def __init__(self, channels, patch=4):
        super().__init__()
        self.patch = patch
        self.lambda_dc = nn.Parameter(torch.ones(1,channels,1,1,1))
        self.lambda_hfc = nn.Parameter(torch.ones(1,channels,1,1,1))
    def forward(self,x):
        b,c,h,w = x.shape
        p = self.patch
        z = x.unfold(2,p,p).unfold(3,p,p).reshape(b,c,-1,p,p)
        dc = z.mean(2,keepdim=True)
        z = z + self.lambda_dc*dc + self.lambda_hfc*(z-dc)
        return z.reshape(b,c,h//p,w//p,p,p).permute(0,1,2,4,3,5).reshape(b,c,h,w)

class ScaledBlock(nn.Module):
    def __init__(self,block,width):
        super().__init__()
        self.block,self.scaling=block,HFS(width)
    def forward(self,x):
        return self.scaling(self.block(x))

def spec_and_data():
    a=SimpleNamespace(grid=96,modes=12,high_modes=24,depth=4,target_params=500000,
        batch_size=16,lr=.001,attention_rank=8,attention_gate_init=-2.,
        profile='high_nonlinear',pdes='',cases='',train_samples=32,val_samples=8,test_samples=4)
    specs=bench.build_case_specs(a)
    index,spec=next((i,s) for i,s in enumerate(specs,1) if s.slug=='KdV_HighNonlinear_Steady2D')
    seed=20260705+index*1009
    data=bench.split_data(spec,a,seed)
    return spec,data,seed

def make_base(spec,variant,seed):
    torch.manual_seed(seed)
    if variant.startswith('FNO'):
        modes=12 if variant=='FNO12_matched' else 44
        width=22 if modes==12 else 6
        return ops.FNO2d_small(modes=modes,width=width,depth=4,input_features=4,
            output_features=1,fourier_feature_bands=(1,2,4,8),
            block_activation='gelu',head_activation='gelu'),width
    base=bench.model_factory('HF_FNO',width=5,spec=spec)
    band=variant in {'Band','HFB'}
    local=variant in {'Local','HFB'}
    base.blocks=nn.ModuleList([PrunedLegacyBlock(b,band,local) for b in base.blocks])
    if variant=='HFS_Core':
        base.blocks=nn.ModuleList([ScaledBlock(b,5) for b in base.blocks])
    return base,5

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def spectral_metrics(pred,truth):
    err=torch.fft.fft2(pred-truth,norm='ortho')
    ref=torch.fft.fft2(truth,norm='ortho')
    k=torch.fft.fftfreq(pred.shape[-1],device=pred.device)*pred.shape[-1]
    high=(k[:,None].abs()>=12)|(k[None,:].abs()>=12)
    num=(err.abs().square()[...,high]).sum((-1,-2))
    den=(ref.abs().square()[...,high]).sum((-1,-2)).clamp_min(1e-20)
    total=ref.abs().square().sum((-1,-2,-3)).clamp_min(1e-20)
    return {'high_band_relative_l2':float((num/den).sqrt().mean()),
            'high_band_error_over_total_l2':float((num/total).sqrt().mean()),
            'reference_high_energy_fraction':float((den/total).mean())}

def evaluate(model,spec,x,y,s,scale,mode='hybrid'):
    with torch.no_grad():
        pred=model(x)
        mse=F.mse_loss(pred,y)
        pde,raw=bench.pde_residual_mses(spec,pred,s,scale)
        res={'loss':float(mse+.05*pde if mode=='hybrid' else pde),'mse':float(mse),
             'relative_l2':float(bench.relative_l2(pred,y)),
             'pde_mse':float(pde),'pde_mse_absolute':float(raw)}
        res.update(spectral_metrics(pred,y))
    return res,pred

def audit(out,mode='hybrid'):
    spec,data,seed=spec_and_data()
    old=LEGACY/'LegacyHF_PDE_Benchmark_Hybrid_20260705/KdV/HighNonlinear_Steady2D/HF_FNO/sample_fields.npz'
    if not old.exists():old=ROOT/'sources/reference_sample_fields.npz'
    z=np.load(old)
    match=[]
    for key in z.files:
        arr=z[key]
        if arr.shape in {(96,96),(1,96,96)}:
            for i in range(4):
                if np.allclose(arr.squeeze(),data['y_test'][i,0].numpy(),atol=1e-7):match.append([key,i])
    assert match, 'Reconstructed dataset does not match the legacy test field'
    torch.manual_seed(777)
    full=bench.model_factory('HF_FNO',width=5,spec=spec).eval()
    converted=copy.deepcopy(full)
    converted.blocks=nn.ModuleList([PrunedLegacyBlock(b,True,True) for b in converted.blocks])
    x=data['x_train'][:2]
    with torch.no_grad():
        discrepancy=float((full(x)-converted(x)).abs().max())
    assert discrepancy < 1e-6
    hfs=HFS(4)
    assert torch.allclose(hfs(x),2*x,atol=2e-6)
    counts={}
    for name in VARIANTS:
        model,w=make_base(spec,name,SEEDS[0]); model=bench.HardConstraintWrapper(model,spec)
        y=model(x)
        loss=F.mse_loss(y,data['y_train'][:2])
        loss.backward()
        assert torch.isfinite(y).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.allclose(y[:,:,0],data['y_train'][:2,:,0],atol=1e-6) and torch.allclose(y[:,:,-1],data['y_train'][:2,:,-1],atol=1e-6)
        counts[name]={'width':w,'real_parameters':bench.count_parameters(model)}
    out.mkdir(parents=True,exist_ok=True)
    torch.save(data,out/'legacy_split.pt')
    report={'spec':asdict(spec),'data_seed':seed,'legacy_test_match':match,
        'data_sha256':sha(out/'legacy_split.pt'),'full_model_forward_max_difference':discrepancy,
        'counts':counts,'seeds':SEEDS,'epochs':1200,'training_mode':mode,'physics_weight':.05 if mode=='hybrid' else 1.,
        'validation_interval_epochs':20,'selection':'minimum validation hybrid loss' if mode=='hybrid' else 'minimum validation normalized PDE residual',
        'test_evaluation':'once after training at selected and final checkpoints',
        'source_hashes':{name:sha(LEGACY/name) for name in ['NeuroOperators.py','run_legacy_hf_pde_suite.py']},
        'hardware':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':sys.version,
        'hfs_source':'https://github.com/SiaK4/HFS_ResUNet/blob/main/Models/ResUnet_HFS.py',
        'notes':['Legacy fixed split only; no new independence experiment.',
          'Core retains duplicate low-band kernel, input Fourier features, GELU and common fusion.',
          'Band removes or restores ONLY the six extra frequency weight blocks.',
          'Legacy high_modes=24 is a slice count: positive rFFT columns 25..48 on the 96 grid, with a gap after the low modes.',
          'HFS_Core uses authors residual patch scaling with patch size 4 and two ones-initialized channel parameters per block.',
          'Matched FNO controls use identical Fourier input features, block and head GELU.',
          'Fixed width ablation isolates structure but does not equalize parameter counts. Matched FNO controls test capacity alternatives.']}
    (out/'protocol.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def train(out,variant,seed,epochs,mode='hybrid'):
    spec,_,_=spec_and_data()
    device=torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=torch.load(out/'legacy_split.pt',map_location='cpu')
    gpu={k:v.to(device) for k,v in data.items()}
    scale=gpu['source_train'].std().clamp_min(1.)
    base,width=make_base(spec,variant,seed)
    model=bench.HardConstraintWrapper(base,spec).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=5e-5)
    generator=torch.Generator().manual_seed(seed+98765)
    folder=out/variant/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_state=None;best_epoch=0;history=[]
    train_s=0.;eval_s=0.
    torch.cuda.synchronize();wall_start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();torch.cuda.synchronize();t=time.perf_counter()
        perm=torch.randperm(32,generator=generator).to(device)
        losses=[]
        for idx in perm.split(16):
            opt.zero_grad(set_to_none=True)
            pred=model(gpu['x_train'][idx])
            mse=F.mse_loss(pred,gpu['y_train'][idx])
            pde=bench.pde_residual_loss(spec,pred,gpu['source_train'][idx],scale)
            loss=mse+.05*pde if mode=='hybrid' else pde
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {variant} {seed} {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        scheduler.step();torch.cuda.synchronize();train_s+=time.perf_counter()-t
        if epoch==1 or epoch%20==0 or epoch==epochs:
            model.eval();t=time.perf_counter()
            val,_=evaluate(model,spec,gpu['x_val'],gpu['y_val'],gpu['source_val'],scale,mode)
            if mode=='pde':val['loss']=val['pde_mse']
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),**{'val_'+k:v for k,v in val.items()}}
            history.append(row)
            if val['loss']<best:
                best=val['loss'];best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.cuda.synchronize();eval_s+=time.perf_counter()-t
            if epoch==1 or epoch%200==0 or epoch==epochs:
                print(f'{variant} seed={seed} epoch={epoch}/{epochs} train={row["train_loss"]:.4g} val={val["loss"]:.4g} seconds={time.perf_counter()-wall_start:.1f}',flush=True)
    wall=time.perf_counter()-wall_start
    final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    final_metrics,final_pred=evaluate(model,spec,gpu['x_test'],gpu['y_test'],gpu['source_test'],scale,mode)
    model.load_state_dict(best_state)
    selected,pred=evaluate(model,spec,gpu['x_test'],gpu['y_test'],gpu['source_test'],scale,mode)
    with torch.no_grad():
        for _ in range(10):model(gpu['x_test'][:1])
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(100):model(gpu['x_test'][:1])
        torch.cuda.synchronize();infer=(time.perf_counter()-t)/100
    metrics={'variant':variant,'seed':seed,'training_mode':mode,'width':width,'params':bench.count_parameters(model),
        'epochs':epochs,'selected_epoch':best_epoch,'training_s':train_s,'validation_s':eval_s,
        'training_and_validation_wall_s':wall,'inference_s_single_field':infer,
        'selected':selected,'final':final_metrics,'data_sha256':sha(out/'legacy_split.pt')}
    torch.save({'state_dict':best_state,'final_state_dict':final_state,'metrics':metrics},folder/'checkpoint.pt')
    np.savez_compressed(folder/'fields.npz',prediction=pred.cpu().numpy(),
        final_prediction=final_pred.cpu().numpy(),truth=data['y_test'].numpy(),source=data['source_test'].numpy())
    with (folder/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'KdV_Hybrid')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--seed',type=int,default=SEEDS[0])
    parser.add_argument('--epochs',type=int,default=1200)
    parser.add_argument('--mode',choices=['hybrid','pde'],default='hybrid')
    args=parser.parse_args()
    if args.audit:audit(args.out,args.mode)
    else:train(args.out,args.variant,args.seed,args.epochs,args.mode)

