from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter
R=Path(__file__).resolve().parent
O=R/'six_panel_means_20260926';O.mkdir(exist_ok=True)
if Path(r'C:\Windows\Fonts\times.ttf').exists():
 font_manager.fontManager.addfont(r'C:\Windows\Fonts\times.ttf')
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
V=['Core','Band','Local','HFB','FNO12_matched','FNO44_matched','HFS_Core']
labels=['Core','+ spectral','+ local','Full HFB','FNO 12','FNO 44','+ scaling']
colors=['#5C677D','#E69F00','#009E73','#CC3366','#0072B2','#7851A9','#8A5A44']
data=json.loads((R/'absolute_error_revision/absolute_metrics.json').read_text())
fig,axs=plt.subplots(3,2,figsize=(7.2,8))
for c,(mode,folder,title) in enumerate([('pde','KdV_Physics','Physics-only training'),('hybrid','KdV_Hybrid','Hybrid data–physics training')]):
 for row,key,ylabel in [(0,'field_rmse','Field RMSE'),(1,'high_rmse','High-frequency RMSE'),(2,'pde_mse','Normalized PDE MSE (log scale)')]:
  ax=axs[row,c]
  for j,v in enumerate(V):
   if row<2:
    a=np.array(data['groups'][mode][v][key]['values'])
    ax.errorbar(j,a.mean(),yerr=a.std(ddof=1),fmt='none',capsize=3,color=colors[j],elinewidth=1.2,zorder=2)
   else:
    a=np.array([json.loads((R/folder/v/f'seed_{seed}'/'metrics.json').read_text())['selected']['pde_mse'] for seed in range(20260924,20260934)])
    ax.boxplot([a],positions=[j],widths=.48,patch_artist=True,showfliers=False,whis=1.5,boxprops={'facecolor':colors[j],'edgecolor':colors[j],'alpha':.25,'linewidth':1},medianprops={'color':colors[j],'linewidth':1.5},whiskerprops={'color':colors[j],'linewidth':1},capprops={'color':colors[j],'linewidth':1})
   ax.scatter(j+np.linspace(-.13,.13,10),a,s=10,color=colors[j],alpha=.65,zorder=3)
   ax.scatter(j,a.mean(),marker='D',s=22,facecolors='white',edgecolors=colors[j],linewidths=1.2,zorder=5)
  if row==2:
   ax.set_yscale('log');ax.set_ylim((.01,2) if c==0 else (.1,5))
   ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
   ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'));ax.yaxis.set_minor_formatter(NullFormatter())
  else:ax.set_ylim(bottom=0)
  ax.set_xticks(range(7),labels,rotation=38,ha='right');ax.set_ylabel(ylabel)
  ax.set_title(f'({chr(97+row*2+c)}) {title}')
  ax.grid(alpha=.2,lw=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(left=.10,right=.99,top=.965,bottom=.105,wspace=.37,hspace=.70)
for ext in ['png','pdf','svg']:fig.savefig(O/f'kdv_abcdef_open_means.{ext}',dpi=400)
(O/'caption.txt').write_text('Dots denote ten training seeds and open diamonds denote arithmetic means in all panels. In (a–d), error bars show sample standard deviations. In (e–f), boxes show interquartile ranges, horizontal lines show medians, and whiskers extend to observations within 1.5 times the interquartile range; all observations are shown. PDE MSE uses logarithmic axes without a display scaling factor.',encoding='utf8')
print(O)
