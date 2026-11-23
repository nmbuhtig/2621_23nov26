#!/usr/bin/env python3
   
from pathlib import Path
import numpy as np, csv, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import norm
ROOT=Path(__file__).resolve().parent / Path(__file__).stem
(ROOT/'figures').mkdir(parents=True, exist_ok=True)
(ROOT/'data').mkdir(parents=True, exist_ok=True)
G=.6; WX=-.2; WC0=.25; J=.12; TH=.4; LAM=.7; K0=np.pi/2

def v(k, tw='B'):
    t=TH + (LAM*np.cos(k) if tw=='B' else 0.)
    return np.array([np.cos(t),np.sin(t)])
def b(k,tw='B'):
    c,s=v(k,tw); return np.array([c*c,np.sqrt(2)*c*s,s*s])
def h(k,tw='B'):
    c,s=G*v(k,tw); return np.array([[WC0+2*J*np.cos(k),c,s],[c,WX,0],[s,0,WX]],complex)
def lower_fraction(k):
    wc=WC0+2*J*np.cos(k);d=wc-WX
    return .5*(1+d/np.sqrt(d*d+4*G*G))
def overlap_analytic(p,k):return np.cos(LAM*(np.cos(p)-np.cos(k)))**2
def amp_ratio(q,eps):
    p=K0+q; bp=b(p,'B'); bk=b(K0,'B'); ba=b(K0,'A')
    u=np.diag([1,1+eps,1]); return (bp@u@bk)/(ba@u@ba)
def predicted_anisotropy(eps):
    S=np.sin(2*TH); C=np.cos(2*TH); D=1+.5*eps*S*S
    linear=-2*LAM*eps*S*C/D
    curv=LAM**2*(2*(1+eps*S*S)/D-(eps*S*C/D)**2)
    return linear,curv

def fs(x,hstep=1e-5):
    y=b(x); der=(b(x+hstep)-b(x-hstep))/(2*hstep)
    return float(np.vdot(der,der).real - abs(np.vdot(y,der))**2)

                                                                                   
def fock_born_check(N=8, U=.08):
    kval=2*np.pi*np.arange(N)/N
    labels=[(ki,orb) for ki in range(N) for orb in range(3)]
    states=[]
    for i,(ki,oi) in enumerate(labels):
        for j in range(i,len(labels)):
            if (ki+labels[j][0])%N==0:states.append((i,j))
    lookup={q:i for i,q in enumerate(states)};n2=len(states)
                                                                                                    
    C=np.zeros((n2,3),complex)
    for ki in range(N):
        kj=(-ki)%N
        for channel,(oa,ob) in enumerate([(1,1),(1,2),(2,2)]):
            i,j=3*ki+oa,3*kj+ob
            t=tuple(sorted((i,j)))
            C[lookup[t],channel]+= (np.sqrt(2 if ki==kj else 1)/np.sqrt(2*N) if oa==ob else 1/np.sqrt(N))
    assert norm(C.conj().T@C-np.eye(3))<1e-12
    out={}
    for tw in 'AB':
        blocks=[h(k,tw)-np.diag([.02j,0,0]) for k in kval]
        H0=np.zeros((n2,n2),complex)
        for col,(i,j) in enumerate(states):
            occ={i:1,j:1} if i!=j else {i:2}
            for mode,count in occ.items():
                ik,orb=labels[mode]
                for neworb in range(3):
                    vmat=blocks[ik][neworb,orb]
                    if abs(vmat)<1e-15:continue
                    nxt=occ.copy();nxt[mode]-=1
                    if nxt[mode]==0:del nxt[mode]
                    newmode=3*ik+neworb
                    weight=np.sqrt(count*(nxt.get(newmode,0)+1))
                    nxt[newmode]=nxt.get(newmode,0)+1
                    dst=tuple(sorted(x for x,number in nxt.items() for _ in range(number)))
                    H0[lookup[dst],col]+=weight*vmat
        z=2*(-.8)+.28j
        G0=np.linalg.inv(z*np.eye(n2)-H0)
                                                                          
        V=U*C@C.conj().T
        Gfull=np.linalg.inv(z*np.eye(n2)-H0-V)
                                        
        con=Gfull-G0
        rank1=G0@C@np.linalg.inv(np.eye(3)-U*C.conj().T@G0@C)@(U*C.conj().T@G0)
        assert norm(con-rank1)<5e-12
        ports=[]; mats=[]
        for ki in range(1,N//2):
            kj=(-ki)%N;port=lookup[tuple(sorted((3*ki,3*kj)))]
            ports.append(port)
            mats.append(C.conj().T@G0[:,port])
        mat=np.stack(mats)
                                                                           
        max_leg=0.
        for ik,vec in enumerate(mat,start=1):
            ray=b(kval[ik],tw);scalar=np.vdot(ray,vec)
            max_leg=max(max_leg,norm(vec-scalar*ray))
        out[tw]={'born':G0@V@G0,'con':con,'ports':ports,'legs':mat,'max_contact_leg_error':max_leg}
    kidx=N//4-1                                           
    pidx=0; p=2*np.pi/N              
    ratio=abs(out['B']['born'][out['B']['ports'][pidx],out['B']['ports'][kidx]] / out['A']['born'][out['A']['ports'][pidx],out['A']['ports'][kidx]])**2
    target=np.cos(LAM*(np.cos(p)-np.cos(K0)))**4
    assert abs(ratio-target)<1e-11, (ratio,target)
    return {'N':N,'Hilbert_dimension':n2,'born_ratio_direct_fock':float(ratio),'born_formula':float(target), 'max_free_contact_leg_error':max(x['max_contact_leg_error'] for x in out.values()), 'finite_U_A_connected_abs':float(abs(out['A']['con'][out['A']['ports'][pidx],out['A']['ports'][kidx]])), 'finite_U_B_connected_abs':float(abs(out['B']['con'][out['B']['ports'][pidx],out['B']['ports'][kidx]]))}

def dipole_nogo_check():
                                                                                                    
    D=np.array([[1.,.25],[.1,.82]])
    a=.73; W=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    DB=D@W
    residue_err=norm(D@D.T-DB@DB.T)
                                                                                           
    probes=[np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]) for t in np.linspace(0,np.pi,71)]
    sigmaerr=max(norm(E.T@D@D.T@E-E.T@DB@DB.T@E) for E in probes)
                                                                                      
    c,s=np.cos(a),np.sin(a)
    T=np.array([[c*c,-np.sqrt(2)*c*s,s*s],
                [np.sqrt(2)*c*s,c*c-s*s,-np.sqrt(2)*c*s],
                [s*s,np.sqrt(2)*c*s,c*c]])
    irrep_err=norm(T.T@T-np.eye(3))
    return dict(full_polarization_residue_error=float(residue_err),sampled_complete_probe_error=float(sigmaerr),symmetric_square_unitarity_error=float(irrep_err))

def linear_mismatch_bound():
                                                                                                     
    rng=np.random.default_rng(346)
    worst_op=0.;worst_prob=0.;worst_violation=0.
    for i in range(1000):
        u,_,vh=np.linalg.svd(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        v,_,wh=np.linalg.svd(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        SA=u@np.diag(rng.uniform(0.3,1.,2))@vh
        SB=v@np.diag(rng.uniform(0.3,1.,2))@wh
        eps=norm(SA-SB,2)
        A=np.kron(SA,SA);B=np.kron(SB,SB)
        d=norm(A-B,2);worst_op=max(worst_op,d/(2*eps))
        psi=rng.normal(size=4)+1j*rng.normal(size=4); psi/=norm(psi)
        P=np.diag([1.,0.,0.,0.])
        pa=np.real(np.vdot(A@psi,P@A@psi));pb=np.real(np.vdot(B@psi,P@B@psi))
        worst_prob=max(worst_prob,abs(pa-pb)/(4*eps))
        worst_violation=max(worst_violation,max(0,d-2*eps,abs(pa-pb)-4*eps))
    assert worst_violation<1e-11
    return dict(worst_operator_bound_fraction=worst_op,worst_probability_bound_fraction=worst_prob,max_bound_violation=worst_violation, caveat='Applies only to disconnected two-photon S tensor and passive S1; epsilon must be certified over occupied wavepacket Hilbert space, not sampled points.')

def run():
    ks=np.linspace(.05,np.pi-.05,181)
    gh=[];err_green=0.;err_fs=0.;err_metric=0.
    for k in ks:
        z=.21+.18j
        ga=np.linalg.inv(z*np.eye(3)-h(k,'A'))[0,0]
        gb=np.linalg.inv(z*np.eye(3)-h(k,'B'))[0,0]
        err_green=max(err_green,abs(ga-gb))
        vv=abs(np.vdot(b(k,'B'),b(K0,'B')))**2
        p=abs(overlap_analytic(k,K0))**2
        err_fs=max(err_fs,abs(vv-p))
        err_metric=max(err_metric,abs(fs(k)-2*LAM**2*np.sin(k)**2))
        gh.append([float(k),float(vv),float(p),float(fs(k)),float(2*LAM**2*np.sin(k)**2),float(lower_fraction(k)*LAM**2*np.sin(k)**2)])
    assert err_green<1e-12 and err_fs<1e-12 and err_metric<1e-8
    with (ROOT/'data'/'S6_Contact_Fidelity.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['k','physical_born_fidelity','cos4','contact_metric_fd','contact_metric_formula','LP_metric_difference']);w.writerows(gh)
    epsarr=[0,.0093,.05,.15,.4]
    qs=np.linspace(-.25,.25,201)
    mismatch=[]
    for eps in epsarr:
        lin,curv=predicted_anisotropy(eps)
        arr=np.array([abs(amp_ratio(q,eps))**2 for q in qs])
        hh=2e-4
        numlinear=(abs(amp_ratio(hh,eps))**2-abs(amp_ratio(-hh,eps))**2)/(2*hh)
        numcurv=(2*abs(amp_ratio(0,eps))**2-abs(amp_ratio(hh,eps))**2-abs(amp_ratio(-hh,eps))**2)/(2*hh**2)
        assert abs(numlinear-lin)<1e-6 and abs(numcurv-curv)<3e-6,(eps,numcurv,curv)
        mismatch.append([eps,float(lin),float(numlinear),float(curv),float(numcurv),float((curv/(2*LAM**2)-1)*100)])
    with (ROOT/'data'/'S6_Interaction_Anisotropy.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['U12_over_U_minus_one','analytic_linear_coefficient','finite_difference_linear','analytic_sym_curvature','finite_difference_sym_curvature','curvature_relative_shift_percent']);w.writerows(mismatch)
    fock=fock_born_check();nogo=dipole_nogo_check();bound=linear_mismatch_bound()
                                                                                                                     
    mismatchcases=[('R=3',.0483),('R=4',.00236),('R=5',.000201),('R=6',.00000707),('S5 selected',.0963),('S5 complete',1.997)]
    misrows=[[x,eps,min(4*eps,1.0)] for x,eps in mismatchcases]
    with (ROOT/'data'/'S6_Linear_Mismatch_Diagnostics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','sampled_singlephoton_op_difference','4eps_disconnected_probability_upper_bound_if_operator_certified']);w.writerows(misrows)
    arr=np.array(gh);fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
    ax[0].plot(arr[:,0],arr[:,1],label='Physical contact fidelity',lw=2.6)
    ax[0].plot(arr[:,0],arr[:,2],'--',label='Symmetric-square identity',lw=1.4,color='black')
    ax[0].set(xlabel='Outgoing momentum $p$',ylabel='Born B/A probability',title='Identity is an ordinary fidelity law');ax[0].legend(frameon=False,fontsize=8)
    ax[1].plot(arr[:,0],arr[:,4],label='Contact metric',lw=2.3)
    ax[1].plot(arr[:,0],arr[:,5],label='Lower-polariton metric difference',lw=2)
    ax[1].set(xlabel='Momentum $k$',ylabel='Quantum metric',title='Related, but distinct, geometric objects');ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'S6_Novelty_Audit_Geometric_Identity.png',dpi=210);fig.savefig(ROOT/'figures'/'S6_Novelty_Audit_Geometric_Identity.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
    for eps in epsarr:
        ax[0].plot(qs,[abs(amp_ratio(q,eps))**2 for q in qs],label=f'$\\epsilon={eps:g}$',lw=1.8)
    ax[0].set(xlabel='Outgoing transfer $q$',ylabel='Normalized Born B/A probability',title='Equal optics, imperfect SU(2) interaction');ax[0].legend(frameon=False,fontsize=8)
    ax[1].plot([r[0] for r in mismatch],[r[3] for r in mismatch],'-o',label='Symmetrized curvature')
    ax[1].axhline(2*LAM**2,ls='--',color='black',label='SU(2) metric $=0.98$')
    ax[1].set(xlabel='Anisotropy $U_{12}/U-1$',ylabel='Even-in-$q$ Born curvature',title='Interaction corrections are predictable');ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'S6_SU2_Anisotropy_Audit.png',dpi=210);fig.savefig(ROOT/'figures'/'S6_SU2_Anisotropy_Audit.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8));labels=[r[0] for r in misrows];vals=[r[2] for r in misrows]
    ax.bar(range(len(labels)),vals);ax.set_xticks(range(len(labels)),labels);ax.set_yscale('log')
    ax.axhline(.03178,ls='--',color='black',label='S4 ideal $\\Delta P\\approx0.03178$')
    ax.set(ylabel='Conditional $4\\epsilon$ bound on disconnected probability difference',title='Certificate requires full occupied-band operator norm')
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures'/'S6_Linear_FalsePositive_Bounds.png',dpi=210);fig.savefig(ROOT/'figures'/'S6_Linear_FalsePositive_Bounds.pdf');plt.close(fig)
    report={'max_one_photon_green_error':err_green,'max_contact_fidelity_error':err_fs,'max_FS_metric_fd_error':err_metric,'contact_metric_at_k0':float(2*LAM**2),
            'fock':fock,'full_polarization_unitary_nogo':nogo,'disconnected_probability_bound':bound,'anisotropy':mismatch,'mismatch_diagnostics':misrows,
            'scopes':['The S2b contact metric identity is a standard tensor-product fidelity relation; originality alone is low.','The one-photon equivalence plus fixed interaction and radiative distinction is a candidate distinct conjunction, not novelty-certified.','The S5b Rb comparison is an interaction-only analog, not a demonstrated light–matter device.']}
    (ROOT/'data'/'S6_Audit_Results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
