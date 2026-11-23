#!/usr/bin/env python3
   
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUTPUT_ROOT=Path(__file__).resolve().parent / Path(__file__).stem
DATA=OUTPUT_ROOT/'data'
FIG=OUTPUT_ROOT/'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
                                                   
from s3_radiative import calculate

results={}
for U in (0.06,0.24,0.50):
    r,_=calculate(U=U,N=14,kappa=.18,sigma=.045,nf=161,step=.012,fast=True)
    results[U]=r
    print('U',U,'off-forward A,B:',r['results']['A']['off_forward'],r['results']['B']['off_forward'],flush=True)

with (DATA/'OT2a_Figure3_OffForward.csv').open('w',newline='') as fcsv:
    wcsv=csv.writer(fcsv);wcsv.writerow(['U','P_A_offforward','P_B_offforward','difference'])
    for U in (0.06,0.24,0.50):
        aa=results[U]['results']['A']['off_forward'];bb=results[U]['results']['B']['off_forward']
        wcsv.writerow([U,aa,bb,aa-bb])

fig,axes=plt.subplots(1,2,figsize=(10.6,4.05),layout='constrained')
Us=[.06,.24,.50];a=[results[u]['results']['A']['off_forward'] for u in Us];b=[results[u]['results']['B']['off_forward'] for u in Us]
x=np.arange(len(Us));w=.34
axes[0].bar(x-w/2,a,w,label='A');axes[0].bar(x+w/2,b,w,label='B')
axes[0].set_xticks(x,labels=[f'$U={u:.2f}$' for u in Us]);axes[0].set_ylabel('Integrated off-forward pair probability')
axes[0].set_title('(a) Normalized radiative coincidences');axes[0].legend(frameon=False)

base=results[.24];ks=np.array(base['k_reps']);order=np.argsort(ks)
k0=base['params']['k0']; theta=lambda k: .4+.7*np.cos(k)
ratio_born=np.cos(theta(ks)-theta(k0))**4
axes[1].plot(ks[order],ratio_born[order],color='k',ls='--',lw=2,label='Born contact fidelity')
for U in (.06,.24):
    res=results[U];pa=np.array(res['results']['A']['connected_angular_probabilities']);pb=np.array(res['results']['B']['connected_angular_probabilities'])
    mask=(pa>1e-13)
    axes[1].plot(ks[order][mask[order]],(pb/np.maximum(pa,1e-18))[order][mask[order]],'o-',label=f'Finite $U={U:.2f}$')
axes[1].set_title('(b) Geometry versus interaction dressing')
axes[1].set_xlabel('Outgoing photon momentum $p$')
axes[1].set_ylabel('B/A connected coincidence ratio')
axes[1].legend(frameon=False,fontsize=9)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(FIG/'fig3_radiative_coincidences.pdf',bbox_inches='tight')
fig.savefig(FIG/'fig3_radiative_coincidences.png',dpi=210,bbox_inches='tight')
plt.close(fig)
print('Created OT_2a regenerated Figure 3 (PNG and PDF).')
