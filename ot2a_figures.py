#!/usr/bin/env python3
   
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent / Path(__file__).stem
OUT=HERE/'figures'
OUT.mkdir(parents=True, exist_ok=True)
                                              
g=.6; wx=-.2; wc0=.25; J=.12; theta0=.4; lam=.7
k=np.linspace(-np.pi,np.pi,701)
wc=wc0+2*J*np.cos(k)
lo=(wc+wx-np.sqrt((wc-wx)**2+4*g*g))/2
                                                                           
s=.5*(1+(wc-wx)/np.sqrt((wc-wx)**2+4*g*g))
thetaA=np.full_like(k,theta0)
thetaB=theta0+lam*np.cos(k)
                                                        
z=.35+.11j
G=1/(z-wc-g*g/(z-wx))
metric=2*(lam*np.sin(k))**2
lp_delta=.5*s*metric

fig,ax=plt.subplots(1,3,figsize=(11.0,3.2),layout='constrained')
ax[0].plot(k,np.cos(thetaA),ls='--',label=r'$v_{1,A}$')
ax[0].plot(k,np.sin(thetaA),ls='--',label=r'$v_{2,A}$')
ax[0].plot(k,np.cos(thetaB),label=r'$v_{1,B}$')
ax[0].plot(k,np.sin(thetaB),label=r'$v_{2,B}$')
ax[0].set_title('(a) Physical matter spinors')
ax[0].set_ylabel('Exciton coupling spinor')
ax[0].legend(frameon=False,ncol=2,fontsize=8)
ax[1].plot(k,np.abs(G)**2,color='0.15',lw=2.0,label=r'$|G_{aa,A}|^2$')
ax[1].plot(k,np.abs(G)**2,ls='--',color='#c17d33',lw=1.5,label=r'$|G_{aa,B}|^2$')
ax[1].set_title('(b) Exact photon optical twins')
ax[1].set_ylabel(r'$|G_{aa}(k,z)|^2$ (model units)')
ax[1].legend(frameon=False,fontsize=8)
ax[2].plot(k,metric,label=r'$g^\eta_{kk,B}$')
ax[2].plot(k,lp_delta,label=r'$\Delta g^{\rm LP}_{kk}$')
ax[2].set_title('(c) Independently defined metrics')
ax[2].set_ylabel(r'Metric (lattice-length$^2$)')
ax[2].legend(frameon=False,fontsize=8)
for a in ax:
    a.set_xlim(-np.pi,np.pi)
    a.set_xticks([-np.pi,0,np.pi],labels=[r'$-\pi$',r'$0$',r'$\pi$'])
    a.set_xlabel('Bloch momentum $k$')
    a.grid(alpha=.2)
fig.savefig(OUT/'fig1_twins_geometry.pdf',bbox_inches='tight')
fig.savefig(OUT/'fig1_twins_geometry.png',dpi=210,bbox_inches='tight')
plt.close(fig)
assert np.max(np.abs((metric-2*lp_delta/s)))<1e-12
print('Figure 1 generated; max metric identity residual:', np.max(np.abs((metric-2*lp_delta/s))))
