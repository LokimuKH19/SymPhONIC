from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from summarize import gather,VARIANTS,SEEDS,LABELS
ROOT=Path(__file__).resolve().parent
if Path(r'C:\Windows\Fonts\times.ttf').exists():
 font_manager.fontManager.addfont(r'C:\Windows\Fonts\times.ttf')
# 12 CSS px at 96 px/in corresponds to 9 typographic points.
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'})
r=json.loads((ROOT/'absolute_error_revision/absolute_metrics.json').read_text());colors=['#5C677D','#E69F00','#009E73','#CC3366','#0072B2','#7851A9','#8A5A44']
fig,axs=plt.subplots(3,2,figsize=(7.2,8.0),gridspec_kw={'height_ratios':[1,1,1.1]})
short=['Core','+ spectral','+ local','Full HFB','FNO 12','FNO 44','+ scaling']
for c,(mode,folder,title) in enumerate([('pde','KdV_Physics','Physics-only training'),('hybrid','KdV_Hybrid','Hybrid data–physics training')]):
 for row,k,ylab in [(0,'field_rmse','Field RMSE'),(1,'high_rmse','High-frequency RMSE')]:
  ax=axs[row,c]
  for j,v in enumerate(VARIANTS):
   vals=np.array(r['groups'][mode][v][k]['values'])
   ax.scatter(j+np.linspace(-.13,.13,len(vals)),vals,s=10,color=colors[j],alpha=.65,zorder=3)
   ax.errorbar(j,vals.mean(),yerr=vals.std(ddof=1),fmt='D',markersize=3.5,capsize=3,color=colors[j],elinewidth=1.2,zorder=4)
  ax.set_xticks(range(7),short,rotation=38,ha='right');ax.set_ylabel(ylab);ax.set_ylim(bottom=0)
  ax.set_title(f'({chr(97+row*2+c)}) {title}')
 ax=axs[2,c]
 for j,v in enumerate(VARIANTS):
  spectra=[]
  for seed in SEEDS:
   z=np.load(ROOT/folder/v/f'seed_{seed}'/'fields.npz')
   err=np.abs(np.fft.fft2(z['prediction']-z['truth'],norm='ortho')).mean((0,1))
   k=np.fft.fftfreq(err.shape[0])*err.shape[0];bins=np.floor(np.sqrt(k[:,None]**2+k[None,:]**2)).astype(int)
   vals=np.bincount(bins.ravel(),weights=err.ravel())/np.bincount(bins.ravel())
   spectra.append(vals)
  a=np.stack(spectra);x=np.arange(a.shape[1]);mean=a.mean(0);sd=a.std(0,ddof=1)
  ax.plot(x,mean,color=colors[j],lw=1,label=LABELS[v]);ax.fill_between(x,np.maximum(mean-sd,1e-12),mean+sd,color=colors[j],alpha=.1)
 ax.set_yscale('log');ax.set_xlabel('Radial Fourier index');ax.set_ylabel('Mean Fourier-coefficient error');ax.set_title(f'({chr(101+c)}) {title}')
for ax in axs.flat:
 ax.grid(alpha=.2,lw=.5);ax.spines[['top','right']].set_visible(False)
handles,labels=axs[2,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(left=.10,right=.99,top=.965,bottom=.135,wspace=.37,hspace=.70)
fig.savefig(ROOT/'absolute_error_revision/kdv_absolute_comparison.png',dpi=400)
fig.savefig(ROOT/'absolute_error_revision/kdv_absolute_comparison.pdf')
print('saved figure; Times New Roman 9 pt = 12 px at 96 dpi; raster 400 dpi')
