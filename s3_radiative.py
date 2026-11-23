#!/usr/bin/env python3
   
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent / Path(__file__).stem
DATA=ROOT/'data'
FIG=ROOT/'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
G=.6; WX=-.2; WC0=.25; J=.12; THETA0=.4; LAM=.7

def setup(N=14,kappa=.18,loss=0.,g=G,lam=LAM,theta0=THETA0,j=J):
    assert N%2==0
    kvals=(np.arange(N)+.5)*2*np.pi/N-np.pi
    pairs=[];used=set()
    for i,k in enumerate(kvals):
        m=int(np.argmin(abs(np.angle(np.exp(1j*(kvals+k))))))
        assert abs(np.angle(np.exp(1j*(kvals[m]+k))))<1e-10
        if i not in used:
            assert i!=m
            pairs.append((i,m));used.update([i,m])
    assert len(pairs)==N//2
                                                               
    W=np.zeros((9,3),complex)
    W[4,0]=np.sqrt(2);W[5,1]=1;W[7,1]=1;W[8,2]=np.sqrt(2)
                                                               
    assert np.max(abs(len(pairs)*W.conj().T@W/N-np.eye(3)))<1e-12
    hs={}; spectra={};pairs_all=[]
    for twin in 'AB':
        Hs=[]
        for i,m in pairs:
            l=[]
            for q in [kvals[i],kvals[m]]:
                ang=theta0+(lam*np.cos(q) if twin=='B' else 0.)
                coupl=g*np.array([np.cos(ang),np.sin(ang)])
                h=np.array([[WC0+2*j*np.cos(q)-.5j*kappa,*coupl],
                    [coupl[0],WX-.5j*loss,0],[coupl[1],0,WX-.5j*loss]],complex)
                l.append(h)
            Hs.append(tuple(l))
        hs[twin]=Hs
                                                 
    ik0=int(np.argmin([abs(abs(kvals[i])-np.pi/2) for i,m in pairs]))
    return dict(N=N,k=kvals,pairs=pairs,hs=hs,W=W,kappa=kappa,loss=loss,ik0=ik0)

def prep_legs(dat,ws):
    n=len(ws);out={}; E=np.eye(3);a=np.array([1,0,0],complex);kappa=dat['kappa'];W=dat['W'].reshape(3,3,3)
    for twin in 'AB':
        ins=[];outs=[];ss=[];s1err=0
        for h1,h2 in dat['hs'][twin]:
            legs_in=[];legs_out=[];slist=[]
            for h in [h1,h2]:
                M=ws[:,None,None]*E[None,:,:]-h[None,:,:]
                                             
                GI=np.linalg.solve(M,np.broadcast_to(a,(n,3))[...,None])[...,0]
                GO=np.linalg.solve(np.swapaxes(M,1,2),np.broadcast_to(a,(n,3))[...,None])[...,0]
                legs_in.append(GI);legs_out.append(GO)
                s=1-1j*kappa*GI[:,0];slist.append(s)
                if dat['loss']==0:s1err=max(s1err,float(np.max(abs(abs(s)**2-1))))
            cin=np.einsum('ai,bj,ijc->abc',legs_in[0],legs_in[1],W.conj())
            cout=np.einsum('ai,bj,ijc->abc',legs_out[0],legs_out[1],W)
            ins.append(cin);outs.append(cout);ss.append(slist)
        out[twin]=dict(ins=np.array(ins),outs=np.array(outs),s1=np.array(ss),single_unitarity_error=s1err)
    return out

def contact_tmatrix(dat,Es,U):
    d=dat['N'];W=dat['W'];num=len(Es);I9=np.eye(9);I3=np.eye(3)
    taus={}; loops={}
    for twin in 'AB':
        Pi=np.zeros((num,3,3),complex)
        for h1,h2 in dat['hs'][twin]:
            H2=np.kron(h1,np.eye(3))+np.kron(np.eye(3),h2)
            Ms=Es[:,None,None]*I9[None,:,:]-H2[None,:,:]
            X=np.linalg.solve(Ms,np.broadcast_to(W,(num,9,3)))
            Pi+=np.einsum('ia,eib->eab',W.conj(),X)/d
        loops[twin]=Pi
        if abs(U)<1e-15:taus[twin]=np.zeros_like(Pi)
        else:
            taus[twin]=np.linalg.solve((1/U)*I3[None,:,:]-Pi,np.broadcast_to(I3,(num,3,3)))
    return taus,loops

def gaussian_wavepacket(ws,center,sigma,step):
    f=np.exp(-((ws-center)**2)/(4*sigma**2)).astype(complex)
    f/=np.sqrt(step*np.vdot(f,f).real)
    return f

def calculate(N=14,kappa=.18,loss=0.,U=.3,sigma=.045,nf=161,step=.012,
              lam=LAM,theta0=THETA0,j=J,fast=False):
    dat=setup(N=N,kappa=kappa,loss=loss,lam=lam,theta0=theta0,j=j)
    ik0=dat['ik0'];i0,j0=dat['pairs'][ik0];k0=dat['k'][i0]
    h0=dat['hs']['A'][ik0][0].copy();h0[0,0]+=.5j*kappa;h0[1,1]+=.5j*loss;h0[2,2]+=.5j*loss
    center=float(np.linalg.eigvalsh(h0)[0]);ws=center+(np.arange(nf)-(nf-1)/2)*step
    f=gaussian_wavepacket(ws,center,sigma,step)
    legs=prep_legs(dat,ws)
    E=2*ws[0]+np.arange(2*nf-1)*step
    taus,loops=contact_tmatrix(dat,E,U)
    results={}; wavefunctions={}
    for twin in 'AB':
        ins=legs[twin]['ins'];outs=legs[twin]['outs'];tau=taus[twin]
                                                               
        conv=np.zeros((2*nf-1,3),complex)
        incident=ins[ik0]
        for a in range(nf):conv[a:a+nf]+=step*f[a]*f[:,None]*incident[a]
        propagated=np.einsum('eab,eb->ea',tau,conv)/dat['N']
                                                                           
        nc=len(dat['pairs']);connected=np.empty((nc,nf,nf),complex)
        for p in range(nc):
            for ia in range(nf):
                js=np.arange(nf);inds=ia+js
                connected[p,ia,:]=(-1j*kappa*kappa/(2*np.pi))*np.einsum('ja,ja->j',outs[p,ia,:,:],propagated[inds,:])
        full=connected.copy();ss1,ss2=legs[twin]['s1'][ik0]
        full[ik0]+=np.outer(ss1*f,ss2*f)
        pd=step**2*np.sum(abs(full)**2,axis=(1,2))
        pc=step**2*np.sum(abs(connected)**2,axis=(1,2))
        norm=float(np.sum(pd))
        off=float(sum(pd)-pd[ik0])
                                                                                  
                                                                       
        pdark=0.
        for pair_idx,(i,m) in enumerate(dat['pairs']) if loss==0 else []:
            for phot_side in [0,1]:
                hp,hm=dat['hs'][twin][pair_idx]
                if phot_side==1: hp,hm=hm,hp
                dark_k=dat['k'][m if phot_side==0 else i]
                angle=theta0+(lam*np.cos(dark_k) if twin=='B' else 0.)
                dv=np.array([0,-np.sin(angle),np.cos(angle)],complex)
                                                                 
                M=(E-WX)[:,None,None]*np.eye(3)[None,:,:]-hp[None,:,:]
                g_out=np.linalg.solve(np.swapaxes(M,1,2),
                    np.broadcast_to(np.array([1,0,0]),(len(E),3))[...,None])[...,0]
                outward=np.einsum('ea,b,abc->ec',g_out,dv,dat['W'].reshape(3,3,3))
                amplitude=(-1j*kappa**1.5/np.sqrt(2*np.pi))*np.einsum('ec,ec->e',outward,propagated)
                pdark+=step*float(np.sum(abs(amplitude)**2))
        results[twin]=dict(norm=norm,stable_dark_probability=(pdark if loss==0 else None), photon_plus_dark_closed_norm=(norm+pdark if loss==0 else None),
            connected_probability=float(sum(pc)),off_forward=off,
            forward=float(pd[ik0]),angular_probabilities=pd.tolist(),
            connected_angular_probabilities=pc.tolist(),single_unitarity_error=legs[twin]['single_unitarity_error'])
        wavefunctions[twin]=dict(full=full,connected=connected)
    diff1=float(np.max(abs(legs['A']['s1']-legs['B']['s1'])))
    result=dict(params=dict(N=N,U=U,kappa=kappa,loss=loss,sigma=sigma,nf=nf,step=step,lam=lam,j=j,center=center,k0=float(k0),input_pair_index=ik0),
        k_reps=[float(dat['k'][i]) for i,m in dat['pairs']],one_photon_difference=diff1,results=results)
    if not fast:print(json.dumps(result,indent=2))
    return result, wavefunctions

def scan_and_plot():
    cases=[('baseline',.24,0.,.045,.7),('weak_U',.06,0.,.045,.7),('moderate_U',.5,0.,.045,.7),
       ('loss',.24,.08,.045,.7),('broad_pulse',.24,0.,.10,.7),('zero_metric',.24,0.,.045,0.),('U_zero',0.,0.,.045,.7)]
    rows=[];saved={}
    for name,U,loss,sigma,lam in cases:
        r,psi=calculate(U=U,loss=loss,sigma=sigma,lam=lam,fast=True)
        saved[name]=r
        for twin in 'AB':
            d=r['results'][twin]
            rows.append([name,twin,U,loss,sigma,lam,r['one_photon_difference'],d['norm'],d['stable_dark_probability'],d['photon_plus_dark_closed_norm'],d['connected_probability'],d['off_forward'],d['forward'],d['single_unitarity_error']])
        print(name,'oneS',r['one_photon_difference'],'A',r['results']['A'],'B',r['results']['B'],flush=True)
    with (DATA/'S3_Radiative_Results.csv').open('w',newline='') as ff:
        w=csv.writer(ff);w.writerow(['case','twin','U','matter_loss','sigma','lam','max_S1_A_minus_B','norm_out','P_dark','P_accounted','P_connected','P_off_forward','P_forward','single_port_unitarity_error']);w.writerows(rows)
    with (DATA/'S3_Main_Report.json').open('w') as f:json.dump(saved,f,indent=2)
    base=saved['baseline'];k=np.array(base['k_reps']);ix=np.argsort(k)
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.1))
    for twin in 'AB':
        pp=np.array(base['results'][twin]['angular_probabilities'])
        ax[0].plot(k[ix],pp[ix],'o-',label=f'{twin}: all output')
        cc=np.array(base['results'][twin]['connected_angular_probabilities'])
        ax[1].plot(k[ix],cc[ix],'o-',label=f'{twin}: connected')
    ax[0].set(xlabel='Outgoing port momentum k',ylabel='Integrated pair probability',title='Full two-photon outgoing distribution')
    ax[1].set(xlabel='Outgoing port momentum k',ylabel='Connected pair probability',title='Interaction-generated component')
    for x in ax:x.legend(frameon=False);x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'S3_Radiative_Pair_Distribution.png',dpi=200);fig.savefig(FIG/'S3_Radiative_Pair_Distribution.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    names=[x[0] for x in cases];xx=np.arange(len(names));width=.34
    for t,shift in [('A',-width/2),('B',width/2)]:
        vals=[saved[n]['results'][t]['off_forward'] for n in names]
        ax.bar(xx+shift,vals,width,label=t)
    ax.set_xticks(xx,names,rotation=25,ha='right');ax.set_ylabel('Off-forward two-photon probability');ax.legend(frameon=False);ax.set_yscale('symlog',linthresh=1e-8)
    fig.tight_layout();fig.savefig(FIG/'S3_OffForward_Controls.png',dpi=200);fig.savefig(FIG/'S3_OffForward_Controls.pdf');plt.close(fig)
                                                                                        
    err=max(abs(saved[c]['results'][t]['photon_plus_dark_closed_norm']-1) for c in names if c!='loss' for t in 'AB')
    print('MAX LOSSLESS NORM ERROR',err)
    with (DATA/'S3_Norm_Audit.txt').open('w') as f:f.write('max_lossless_norm_error: '+str(err)+'\n')
    return err

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scan',action='store_true');parser.add_argument('--nf',type=int,default=161);parser.add_argument('--step',type=float,default=.012);parser.add_argument('--U',type=float,default=.24)
    a=parser.parse_args()
    if a.scan:scan_and_plot()
    else:calculate(nf=a.nf,step=a.step,U=a.U)

def dark_channel_control(N=14,kappa=.18,U=.24,lam=.7):
           
    dat=setup(N=N,kappa=kappa,lam=lam);ik0=dat['ik0'];W=dat['W']
    h0=dat['hs']['A'][ik0][0].copy();h0[0,0]+=.5j*kappa
    omega=float(np.linalg.eigvalsh(h0)[0]);E=2*omega
    tau,_=contact_tmatrix(dat,np.array([E]),U)
    results={}
    for twin in 'AB':
        h1,h2=dat['hs'][twin][ik0]
        g1=np.linalg.solve(omega*np.eye(3)-h1,np.array([1,0,0]))
        g2=np.linalg.solve(omega*np.eye(3)-h2,np.array([1,0,0]))
        vin=W.conj().T@np.kron(g1,g2)
        source=tau[twin][0]@vin
        values=[]
        for idx,(i,m) in enumerate(dat['pairs']):
            for phot_side in [0,1]:
                hp,hm=dat['hs'][twin][idx]
                if phot_side==1:hp,hm=hm,hp
                                                                   
                kval=dat['k'][dat['pairs'][idx][1-phot_side]]
                ang=THETA0+(lam*np.cos(kval) if twin=='B' else 0.)
                dark=np.array([0,-np.sin(ang),np.cos(ang)],complex)
                photon_out=np.linalg.solve(((E-WX)*np.eye(3)-hp).T,np.array([1,0,0]))
                wout=np.kron(photon_out,dark)@W/np.sqrt(N)
                values.append(abs(wout@source)**2)
        results[twin]={'dark_channel_vertex_sum_abs_squared':float(np.sum(values)),
            'maximum_dark_channel_vertex_abs_squared':float(np.max(values))}
    return results
