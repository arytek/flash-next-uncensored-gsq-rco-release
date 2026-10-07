"""Generate the two publication charts from recorded local measurements."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/release-chart-libs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
colors = ['#64748b', '#2563eb']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.labelcolor': '#334155', 'text.color': '#0f172a', 'xtick.color': '#334155', 'ytick.color': '#334155'})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
fig.suptitle('Faster generation, a larger model', fontsize=21, fontweight='bold', x=.07, ha='left')
for ax, values, title, unit, top in zip(axes, ([16.16,21.13],[75.839998528,83.144446848]),
                                      ('Generation speed · higher is faster', 'Total download · lower is smaller'),
                                      ('Tokens per second', 'GB, including both shards'), (26,100)):
    bars = ax.bar(['ISTA IQ3_XXS', 'This model'], values, width=.56, color=colors)
    ax.set_ylim(0, top); ax.set_ylabel(unit); ax.set_title(title, fontsize=13, pad=16)
    ax.grid(axis='y', color='#e2e8f0'); ax.set_axisbelow(True); ax.tick_params(length=0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+top*.02, f'{value:.2f}', ha='center', fontweight='bold', fontsize=15)
fig.text(.07,.08,'About 31% faster generation  ·  9.6% larger download', fontweight='bold', fontsize=14)
fig.text(.07,.025,'Local short-prompt tests; same hardware/settings, frozen llama.cpp 11199. Other workloads may differ.', fontsize=10, color='#475569')
fig.subplots_adjust(top=.78,bottom=.24,left=.08,right=.98,wspace=.35)
fig.savefig(OUT/'speed-size.png', dpi=170, facecolor='white'); plt.close(fig)

counts=[456,256,128,164]
ista=np.array([386,226,107,111])/counts*100
ours=np.array([377,222,109,109])/counts*100
fig,ax=plt.subplots(figsize=(11.5,5.5))
fig.suptitle('Local capability samples',fontsize=21,fontweight='bold',x=.07,ha='left')
x=np.arange(4); width=.32
for offset, values, color, label in ((-width/2,ista,colors[0],'ISTA IQ3_XXS'),(width/2,ours,colors[1],'This model')):
    bars=ax.bar(x+offset,values,width,color=color,label=label)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+1.2,f'{value:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.set_xticks(x,['Knowledge\nMMLU · 456','Maths\nGSM8K · 256','Instructions\nIFEval · 128','Coding\nHumanEval · 164'])
ax.set_ylim(0,100); ax.set_ylabel('Correct / passed (%) · higher is better')
ax.grid(axis='y',color='#e2e8f0'); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.legend(loc='upper right',bbox_to_anchor=(.96,.94),ncol=2,frameon=False,fontsize=11)
fig.text(.07,.08,'Similar sample scores do not establish equal quality.',fontweight='bold',fontsize=14)
fig.text(.07,.025,'Historical local samples, not full leaderboard runs. Coding uses raw completions. Sample sizes appear under tasks.',fontsize=10,color='#475569')
fig.subplots_adjust(top=.76,bottom=.26,left=.08,right=.98)
fig.savefig(OUT/'quality.png',dpi=170,facecolor='white'); plt.close(fig)
(OUT/'chart-data.json').write_text(json.dumps({'speed_tokens_per_second':{'ista':16.16,'model':21.13},
 'total_bytes':{'ista':75839998528,'model':83144446848},'quality_sample_counts':counts,
 'quality_ista_correct':[386,226,107,111],'quality_model_correct':[377,222,109,109]},indent=2)+'\n')
print('Created two PNG charts and their numerical source data.')
