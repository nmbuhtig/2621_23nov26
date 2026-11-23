#!/usr/bin/env python3
   
from pathlib import Path
import csv, numpy as np
from scipy.special import jv
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent / Path(__file__).stem
DATA=ROOT/'data'
FIG=ROOT/'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LAM=.7; THETA0=.4; G=.6; WX=-.2; WC0=.25; J=.12

def theta(k, twin='B', lam=LAM, theta0=THETA0):
    return theta0+(lam*np.cos(k) if twin=='B' else 0.*np.asarray(k))

def v(k,twin='B',lam=LAM,theta0=THETA0):
    t=theta(k,twin,lam,theta0)
    return np.stack([np.cos(t),np.sin(t)],axis=-1)

def contact(k,twin='B',lam=LAM,theta0=THETA0):
    w=v(k,twin,lam,theta0)
                                                                                
    return np.stack([w[...,0]**2,np.sqrt(2)*w[...,0]*w[...,1],w[...,1]**2],axis=-1)

def fs_metric(k,lam=LAM):
    return 2*(lam*np.sin(k))**2

def born_ratio(p,k=LAM,twin='B',lam=LAM):
    pass

def lower_exciton_fraction(k,g=G,wx=WX,wc0=WC0,J=J):
    wc=wc0+2*J*np.cos(k)
    det=wc-wx
    return .5*(1+det/np.sqrt(det*det+4*g*g))

def poly_coeff(d,lam=LAM,th=THETA0,g=G):
                                                            
    if d%2==0:
        q=abs(d)//2; even=(-1)**q*jv(abs(d),lam)
        return g*np.array([np.cos(th),np.sin(th)])*even
    q=(abs(d)-1)//2; odd=(-1)**q*jv(abs(d),lam)
    return g*np.array([-np.sin(th),np.cos(th)])*odd

def finite_range_g(k,R,lam=LAM,th=THETA0,g=G):
    k=np.asarray(k)
    ans=np.zeros(k.shape+(2,),dtype=complex)
    for d in range(-R,R+1):
        ans+=np.exp(1j*d*k)[...,None]*poly_coeff(d,lam,th,g)
    return ans

def physical_metric_fd(k,twin='B',h=1e-5):
    z=v(k,twin); zp=(v(k+h,twin)-v(k-h,twin))/(2*h)
    return float(np.real(np.vdot(zp,zp)-np.abs(np.vdot(z,zp))**2))

def save_csv(fn,headers,rows):
    with (DATA/fn).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)

def run():
    k0=np.pi/2
    momenta=np.linspace(.2,np.pi-.2,211)
    c0=contact(k0,'B');cA=contact(k0,'A')
    assert abs(np.linalg.norm(c0)-1)<2e-15
    rows=[];max_norm=0.; max_analytic=0.;max_metric=0.;max_lp=0.
    for p in momenta:
        a=contact(p,'A');b=contact(p,'B')
        max_norm=max(max_norm,abs(np.vdot(a,a)-1),abs(np.vdot(b,b)-1))
        pred=np.cos(theta(p,'B')-theta(k0,'B'))**4
        born=abs(np.vdot(b,c0)/np.vdot(a,cA))**2
        max_analytic=max(max_analytic,abs(born-pred))
        h=1e-5;cp=(contact(p+h)-contact(p-h))/(2*h)
        metric=np.real(np.vdot(cp,cp)-abs(np.vdot(b,cp))**2)
        ma=fs_metric(p);max_metric=max(max_metric,abs(ma-metric))
        sp=lower_exciton_fraction(p)
                                                                            
        max_lp=max(max_lp,abs(sp*(LAM*np.sin(p))**2-sp*ma/2))
        rows.append([p,p-k0,theta(p,'B'),born,pred,metric,ma,sp,sp*(LAM*np.sin(p))**2,np.linalg.norm(b)])
    assert max_norm<1e-13 and max_analytic<1e-13 and max_metric<1e-8
    save_csv('S2b_Born_Geometric_Identity.csv',
        ['p','q_from_pi_over_2','theta_B','numerical_born_ratio','analytic_cos4','contact_FS_metric_fd','contact_metric_formula','matter_fraction','full_LP_metric_difference','contact_norm'],rows)

                                                                             
    H=np.array([[.23,.21+.16j,0.09],[.21-.16j,-.12,.17],[.09,.17,.31]],complex)
    max_cov=0.;max_metric_cov=0.;bad_gap=0.
    for k in [.35,.88,1.63,2.5]:
        for p in [.27,.92,1.45,2.4]:
            Wk=expm(1j*k*H);Wp=expm(1j*p*H)
            ck=contact(k);cp=contact(p)
            inv=np.vdot(cp,ck)
            via=np.vdot(Wp.conj().T@cp,(Wp.conj().T@Wk)@(Wk.conj().T@ck))
            max_cov=max(max_cov,abs(inv-via))
        eps=1e-5;W=expm(1j*k*H);vk=W.conj().T@contact(k)
        def f(x): return expm(-1j*x*H)@contact(x)
        derivative=(f(k+eps)-f(k-eps))/(2*eps)
        covderivative=derivative+1j*H@vk
        gm=np.real(np.vdot(covderivative,covderivative)-abs(np.vdot(vk,covderivative))**2)
        gn=np.real(np.vdot(derivative,derivative)-abs(np.vdot(vk,derivative))**2)
        max_metric_cov=max(max_metric_cov,abs(gm-fs_metric(k)))
        bad_gap=max(bad_gap,abs(gn-fs_metric(k)))
    assert max_cov<3e-14 and max_metric_cov<1e-8 and bad_gap>.03
                                                 
    ks=np.linspace(-np.pi,np.pi,4097)
    hrows=[]
    for R in range(0,9):
        u=finite_range_g(ks,R)
        ideal=G*v(ks)
        err_g=np.max(np.linalg.norm(u-ideal,axis=1))/G
        err_norm=np.max(np.abs(np.sum(np.abs(u)**2,axis=1)/G**2-1))
                                                                                        
        z=.43+.21j
        wc=WC0+2*J*np.cos(ks)
        GA=1/(z-wc-G**2/(z-WX))
        GB=1/(z-wc-np.sum(np.abs(u)**2,axis=1)/(z-WX))
        green_max=float(np.max(abs(GB-GA)))
        hrows.append([R,err_g,err_norm,green_max])
    save_csv('S2b_Finite_Range_Approximation.csv',['max_real_space_range','max_g_relative_error','max_total_coupling_intensity_relative_error','max_photon_green_difference_at_z'],hrows)

                                       
    green_err=0.;tr_err=0.;symerr=0.
    for k in np.linspace(-np.pi,np.pi,75):
        for twin in ['A','B']:
            coupl=G*v(k,twin)
            h=np.array([[WC0+2*J*np.cos(k),*coupl],[coupl[0],WX,0],[coupl[1],0,WX]],complex)
            cm=G*v(-k,twin)
            hm=np.array([[WC0+2*J*np.cos(-k),*cm],[cm[0],WX,0],[cm[1],0,WX]],complex)
            tr_err=max(tr_err,np.max(abs(hm-h.conj())))
            for z in [.3+.12j,-.6+.23j,1.2+.3j]:
                photon=np.linalg.solve(z*np.eye(3)-h,[1,0,0])[0]
                direct=1/(z-h[0,0]-G*G/(z-WX))
                green_err=max(green_err,abs(photon-direct))
    assert green_err<5e-14 and tr_err<1e-13

                                                                              
    step=1e-4
    def rr(q): return abs(np.vdot(contact(k0+q),contact(k0)))**2
    curvature=(2*rr(0)-rr(step)-rr(-step))/step**2/2
    assert abs(curvature-2*LAM**2)<1e-7,(curvature,2*LAM**2)

    fig,axs=plt.subplots(1,3,figsize=(13,3.9))
    rows=np.array(rows)
    axs[0].plot(rows[:,0],rows[:,3],lw=2.7,label='Independent 3-channel Born')
    axs[0].plot(rows[:,0],rows[:,4],ls='--',c='black',lw=1.5,label=r'$\cos^4[\theta_B(p)-\theta_B(k_0)]$')
    axs[0].set(xlabel=r'Outgoing pair momentum $p$',ylabel='B/A normalized Born probability',title='Pure fidelity contrast, no radial term')
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot(rows[:,0],rows[:,5],lw=2.8,label=r'$g^{\eta}_{pp}$')
    axs[1].plot(rows[:,0],rows[:,8],lw=2,label=r'$\Delta g^{LP}_{pp}$')
    axs[1].set(xlabel=r'$p$',ylabel='Physical quantum metric',title='Independent geometric predictors')
    axs[1].legend(frameon=False,fontsize=9)
    axs[2].plot(rows[:,0],rows[:,9],lw=2.5,label='A and B')
    axs[2].set(xlabel=r'$p$',ylabel='Contact norm',ylim=(.93,1.07),title='Interaction-access norm is constant')
    axs[2].legend(frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'S2b_Constant_Norm_Metric_Scattering.png',dpi=220);fig.savefig(FIG/'S2b_Constant_Norm_Metric_Scattering.pdf');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(10.5,3.9));arr=np.array(hrows)
    axs[0].semilogy(arr[:,0],arr[:,2],'o-',lw=2,label='Total coupling-strength mismatch')
    axs[0].semilogy(arr[:,0],arr[:,1],'s-',lw=1.5,label='Bloch coupling-vector truncation')
    axs[0].set(xlabel='Maximum real-space hopping range (cells)',ylabel='Worst-case normalized error',title='Finite-range optical near twins')
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot(ks,np.sum(abs(finite_range_g(ks,1))**2,axis=1)/G**2,lw=1.7,label='Range 1')
    axs[1].plot(ks,np.sum(abs(finite_range_g(ks,3))**2,axis=1)/G**2,lw=1.8,label='Range 3')
    axs[1].plot(ks,np.sum(abs(finite_range_g(ks,5))**2,axis=1)/G**2,lw=1.8,label='Range 5')
    axs[1].axhline(1,ls='--',c='black',label='Exact optical twin')
    axs[1].set(xlabel=r'$k$',ylabel=r'$\sum_j |g_j(k)|^2/g^2$',title='Finite-range approximation to exact coupling')
    axs[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'S2b_Finite_Range_Cost.png',dpi=220);fig.savefig(FIG/'S2b_Finite_Range_Cost.pdf');plt.close(fig)

    report={
        'lambda':LAM,'theta0':THETA0,'g':G,'k0':k0,
        'exact_pair_contact_norm':1.,'metric_contact_at_k0':fs_metric(k0),
        'metric_full_LP_difference_at_k0':float(lower_exciton_fraction(k0)*LAM**2),
        'metric_analytic_born_curvature_at_k0':2*LAM**2,
        'independent_fd_born_curvature_at_k0':curvature,
        'max_analytic_born_error':max_analytic,
        'max_contact_metric_fd_error':max_metric,
        'max_spinless_time_reversal_deviation':tr_err,
        'max_one_photon_green_difference':green_err,
        'max_moving_frame_overlap_error':max_cov,
        'max_moving_frame_metric_error':max_metric_cov,
        'naive_frame_metric_spurious_deviation':bad_gap,
        'finite_range_errors':hrows,
        'caution':'Exact optical twins use exponentially decaying hopping; not finite-range Maxwell mode design. Pure Born identity, not full photon S.'
    }
    (DATA/'S2b_Analytical_Report.txt').write_text('\n'.join(f'{k}: {x}' for k,x in report.items())+'\n')
    print((DATA/'S2b_Analytical_Report.txt').read_text())
if __name__=='__main__':run()
