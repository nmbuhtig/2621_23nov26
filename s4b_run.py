#!/usr/bin/env python3
   
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s4b_solver import Model,solve_two_photon,linear_bound
ROOT=Path(__file__).resolve().parent / Path(__file__).stem
DATA=ROOT/'data'
FIG=ROOT/'figures'
CHECK=ROOT/'checks'
for p in (DATA,FIG,CHECK):
    p.mkdir(parents=True, exist_ok=True)

def region(rows,kind):
    if kind=='all': return [r for r in rows if not r['input_pair']]
    if kind=='back_to_back':return [r for r in rows if abs(r['kq']+r['kr'])<1e-10 and not r['input_pair']]
    if kind=='forward':return [r for r in rows if r['input_pair']]
    raise ValueError(kind)

def aggregate(result,kind):
    rows=region(result['rows'],kind)
    return {name:float(sum(r[key] for r in rows)) for name,key in [('full','P_full'),('connected','P_connected'),('disconnected','P_disconnected'),('interference','P_interference')]}

def study(nf=81,step=.018,nseeds=12):
    strengths=[0.,.005,.01,.02,.04,.08,.12]
    rows=[]; examples={}
    for xi in strengths:
        for seed in range(nseeds):
            ms={t:Model(twin=t,strength=xi,seed=seed) for t in 'AB'}
            out={t:solve_two_photon(ms[t],U=.24,nf=nf,step=step,calculate_dark=True) for t in 'AB'}
            diff=linear_bound(out['A']['onephoton'],out['B']['onephoton'])
            assert diff<1e-10, (xi,seed,diff)
            assert np.max(abs(ms['A'].disorder-ms['B'].disorder))==0.
            for twin in 'AB':
                x=out[twin]
                rec=dict(xi=xi,seed=seed,twin=twin,N=ms[twin].N,nf=nf,step=step,
                   max_sampled_S1_difference=diff,P_two_photons=x['P_two_photon'],P_dark=x['P_dark'],
                   P_total_accounted=x['P_accounted'])
                for kind in ['all','back_to_back','forward']:
                    for key,val in aggregate(x,kind).items():rec[kind+'_'+key]=val
                rows.append(rec)
            if seed==0 and xi in (0.,.02,.04,.08):examples[str(xi)]={t:out[t] for t in 'AB'}
        av={t:np.array([r['back_to_back_full'] for r in rows if r['xi']==xi and r['twin']==t]) for t in 'AB'}
        print('xi',xi,'A mean',av['A'].mean(),'B mean',av['B'].mean(),'contrast',(av['A']-av['B']).mean(),flush=True)
    p=DATA/'S4b_Interacting_Disorder_Realizations.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    summary=[]
    for xi in strengths:
        a=[r for r in rows if r['xi']==xi and r['twin']=='A'];b=[r for r in rows if r['xi']==xi and r['twin']=='B']
        result=dict(xi=xi,nseed=nseeds)
        for region_name in ('back_to_back','all'):
            for key in ('full','connected','disconnected','interference'):
                va=np.array([r[region_name+'_'+key] for r in a]);vb=np.array([r[region_name+'_'+key] for r in b]);delta=va-vb
                result[region_name+'_'+key+'_A_mean']=float(va.mean())
                result[region_name+'_'+key+'_B_mean']=float(vb.mean())
                result[region_name+'_'+key+'_delta_mean']=float(delta.mean())
                result[region_name+'_'+key+'_delta_min']=float(delta.min())
                result[region_name+'_'+key+'_delta_max']=float(delta.max())
                result[region_name+'_'+key+'_delta_se']=float(delta.std(ddof=1)/np.sqrt(nseeds)) if nseeds>1 else 0.
        result['P_dark_A_mean']=float(np.mean([r['P_dark'] for r in a]))
        result['P_dark_B_mean']=float(np.mean([r['P_dark'] for r in b]))
        result['accounting_max_error']=float(max(abs(r['P_total_accounted']-1) for r in a+b))
        result['one_photon_max_sampled_difference']=float(max(r['max_sampled_S1_difference'] for r in a+b))
        summary.append(result)
    with (DATA/'S4b_Disorder_Summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)

                                                                                               
    x=np.array(strengths)
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.35))
    for twin in 'AB':
        ax[0].plot(x,[r[f'back_to_back_full_{twin}_mean'] for r in summary],'o-',label=twin)
    ax[0].set(xlabel='RMS photon-site disorder $\\xi$',ylabel='Back-to-back off-forward pair probability',title='Complete output: disconnected + connected + interference')
    contrasts=np.array([r['back_to_back_full_delta_mean'] for r in summary]);ses=np.array([r['back_to_back_full_delta_se'] for r in summary]);
    ax[1].errorbar(x,contrasts,yerr=ses,fmt='o-',capsize=2,label='Twin A minus B')
    ax[1].axhline(0,ls='--',color='grey',lw=1)
    ax[1].set(xlabel='RMS photon-site disorder $\\xi$',ylabel='Twin coincidence difference',title='Same disorder realization for both twins')
    for a in ax:a.grid(alpha=.25);a.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'S4b_Full_Interacting_Disorder.png',dpi=220);fig.savefig(FIG/'S4b_Full_Interacting_Disorder.pdf');plt.close(fig)

                                                                                                         
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.35))
    for twin in 'AB':
        ax[0].plot(x,[r[f'back_to_back_disconnected_{twin}_mean'] for r in summary],'o-',label=f'{twin} disconnected')
        ax[0].plot(x,[r[f'back_to_back_connected_{twin}_mean'] for r in summary],'s--',label=f'{twin} connected')
    ax[0].set(xlabel='RMS photon-site disorder $\\xi$',ylabel='Selected pair probability or component norm',title='Linear leakage versus nonlinear contribution')
    for key,label in [('connected','Connected norm difference'),('interference','Interference difference'),('disconnected','Linear-only difference')]:
        ax[1].plot(x,[r[f'back_to_back_{key}_delta_mean'] for r in summary],'o-',label=label)
    ax[1].set(xlabel='RMS photon-site disorder $\\xi$',ylabel='A minus B contribution',title='Observed contrast is not simply $|S_c|^2$')
    for a in ax:a.grid(alpha=.25);a.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'S4b_Disconnected_Connected_Interference.png',dpi=220);fig.savefig(FIG/'S4b_Disconnected_Connected_Interference.pdf');plt.close(fig)

                                                                         
    for strength in (.0,.04):
        results=examples[str(strength)]
        ra=results['A']['rows'];rb=results['B']['rows'];
        labels=[f'{r["kq"]/np.pi:+.2f},{r["kr"]/np.pi:+.2f}' for r in ra]
        ids=np.arange(len(ra));fig,ax=plt.subplots(figsize=(12,4.7))
        ax.bar(ids-.2,[r['P_full'] for r in ra],.4,label='A',alpha=.8)
        ax.bar(ids+.2,[r['P_full'] for r in rb],.4,label='B',alpha=.8)
        ax.set_xticks(ids,labels,rotation=75,fontsize=7)
        ax.set(ylabel='Integrated output-pair probability',xlabel='Output port momenta $(q/\\pi,r/\\pi)$',title=f'All momentum sectors included, photon disorder $\\xi={strength}$')
        ax.set_yscale('symlog',linthresh=1e-5);ax.legend(frameon=False);ax.grid(alpha=.2,axis='y')
        fig.tight_layout();stem='S4b_All_Output_Ports_'+('Ideal' if strength==0 else 'Disordered')
        fig.savefig(FIG/(stem+'.png'),dpi=210);fig.savefig(FIG/(stem+'.pdf'));plt.close(fig)

                                                                                                   
    loss_rows=[]
    for xi in (0.,.02,.04):
        for gam in (0.,.003,.01,.02):
            results={t:solve_two_photon(Model(twin=t,strength=xi,seed=0,gamma=gam),U=.24,nf=nf,step=step) for t in 'AB'}
            d=linear_bound(results['A']['onephoton'],results['B']['onephoton'])
            rec=dict(xi=xi,gamma=gam,s1_difference=d)
            for t in 'AB':
                rec['P2_'+t]=results[t]['P_two_photon']
                for kind in ('all','back_to_back'):
                    rec[kind+'_'+t]=aggregate(results[t],kind)['full']
            rec['back_to_back_delta']=rec['back_to_back_A']-rec['back_to_back_B']
            loss_rows.append(rec)
    with (DATA/'S4b_Loss_Disorder.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(loss_rows[0].keys()));w.writeheader();w.writerows(loss_rows)
    fig,ax=plt.subplots(figsize=(6.7,4.35))
    for xi in (0.,.02,.04):
        rr=[r for r in loss_rows if r['xi']==xi]
        ax.plot([r['gamma'] for r in rr],[r['back_to_back_delta'] for r in rr],'o-',label=f'$\\xi={xi}$')
    ax.set(xlabel='Exciton loss linewidth $\\gamma_X$',ylabel='Back-to-back A–B coincidence difference',title='Matter loss with photon-site disorder')
    ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(FIG/'S4b_Loss_Disorder.png',dpi=220);fig.savefig(FIG/'S4b_Loss_Disorder.pdf');plt.close(fig)

    report=dict(parameters=dict(N=6,U=.24,kappa=.18,lambda_B=.7,nf=nf,step=step,sigma=.045,nseed=nseeds),
        disorder_summary=summary,maximum_probability_deficit=float(max(abs(r['P_total_accounted']-1) for r in rows)),
        s1_max_sampled_diff=float(max(r['max_sampled_S1_difference'] for r in rows)),
        warning='Photon-only disorder common to both twins; all output photon port pairs counted; model-level values only. SE is over sampled realizations, not experimental uncertainty.')
    (DATA/'S4b_Report.json').write_text(json.dumps(report,indent=2))
    return report

def reset_model(P_dark=.0039,attempt_rate=1.,gamma_dark=.01):
           
    return attempt_rate/(1+attempt_rate*P_dark/gamma_dark)

def reset_study(Pdark):
    rows=[]
    rates=[.0001,.001,.01,.1,1,10]
    for R in rates:
        for gd in [1e-5,1e-4,1e-3,1e-2,.1]:
            rows.append(dict(attempt_rate=R,reset_rate=gd,shelving_probability=Pdark,
              effective_rate=reset_model(Pdark,R,gd),duty_factor=reset_model(Pdark,R,gd)/R,
              mean_wait_time=1/gd))
    with (DATA/'S4b_Reset_Renewal.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    vals=np.geomspace(1e-4,10,200)
    for gd in (1e-4,1e-3,1e-2,.1):
        ax.semilogx(vals,[reset_model(Pdark,v,gd)/v for v in vals],label=f'$\\gamma_d={gd:g}$')
    ax.set(xlabel='Attempt rate (inverse model-time units)',ylabel='Fraction of attempts accepted',ylim=(0,1.04),
           title=f'Illustrative single-dark-shelving renewal model ($p_d={Pdark:.4f}$)')
    ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(FIG/'S4b_Reset_Rate_Model.png',dpi=220);fig.savefig(FIG/'S4b_Reset_Rate_Model.pdf');plt.close(fig)
    return rows

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--fast',action='store_true');a=args.parse_args()
    report=study(nf=61 if a.fast else 81,step=.018,nseeds=4 if a.fast else 12)
    reset_study(report['disorder_summary'][0]['P_dark_B_mean'])
    print('MAX_ACCOUNTING_ERROR',report['maximum_probability_deficit'],'MAX_S1_ERR',report['s1_max_sampled_diff'])
