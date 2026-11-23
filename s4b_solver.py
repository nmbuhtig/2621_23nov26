#!/usr/bin/env python3
   
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import scipy.linalg as sla

G=.6; WX=-.2; WC0=.25; J=.12; THETA0=.4; LAM=.7

@dataclass
class Model:
    N:int=6
    twin:str='A'
    strength:float=0.
    seed:int=0
    kappa:float=.18
    gamma:float=0.
    lam:float=LAM
    theta0:float=THETA0
    g:float=G
    detuning:float=0.
    def __post_init__(self):
        assert self.N%2==0 and self.twin in ('A','B')
        self.k=(np.arange(self.N)+.5)*2*np.pi/self.N-np.pi
        self.F=np.exp(-1j*np.outer(self.k,np.arange(self.N)))/np.sqrt(self.N)
        assert np.max(abs(self.F.conj().T@self.F-np.eye(self.N)))<1e-12
        rng=np.random.default_rng(self.seed)
        x=rng.standard_normal(self.N);x-=x.mean();x/=x.std()
        self.disorder=self.strength*x
        self.M=3*self.N
        self.angles=self.theta0+ (self.lam*np.cos(self.k) if self.twin=='B' else np.zeros_like(self.k))
        self.H=np.zeros((self.M,self.M),complex)
        self.H[:self.N,:self.N]=np.diag(WC0+2*J*np.cos(self.k))+self.F@np.diag(self.disorder)@self.F.conj().T
        self.H[self.N:2*self.N,self.N:2*self.N]=(WX+self.detuning/2)*np.eye(self.N)
        self.H[2*self.N:,2*self.N:]=(WX-self.detuning/2)*np.eye(self.N)
        for i in range(self.N):
            self.H[i,self.N+i]=self.H[self.N+i,i]=self.g*np.cos(self.angles[i])
            self.H[i,2*self.N+i]=self.H[2*self.N+i,i]=self.g*np.sin(self.angles[i])
        assert np.max(abs(self.H-self.H.conj().T))<2e-14
        self.Heff=self.H.copy();self.Heff[:self.N,:self.N]-=.5j*self.kappa*np.eye(self.N)
        self.Heff[self.N:,self.N:]-=.5j*self.gamma*np.eye(2*self.N)
                                                                   
        B=np.zeros((self.M,self.M,3*self.N),complex)
        for r in range(self.N):
            a=np.zeros(self.M,complex);b=a.copy()
            a[self.N:2*self.N]=self.F[:,r]
            b[2*self.N:]=self.F[:,r]
            B[:,:,3*r]=np.outer(a,a)
            B[:,:,3*r+1]=(np.outer(a,b)+np.outer(b,a))/np.sqrt(2)
            B[:,:,3*r+2]=np.outer(b,b)
        self.C=B.reshape(self.M*self.M,3*self.N)
        assert np.max(abs(self.C.conj().T@self.C-np.eye(3*self.N)))<3e-14
        self.C3=B
        pairs=[]
        for q in range(self.N):
            for r in range(q,self.N):pairs.append((q,r))
        self.pairs=pairs
        self.ip=self._find_pair_near_pi2()
        self.input=self.pairs[self.ip]
        assert self.input[0]!=self.input[1]
        self.center=float(np.linalg.eigvalsh(np.array([[WC0+2*J*np.cos(self.k[self.input[0]]),self.g],[self.g,WX]]))[0])

    def _find_pair_near_pi2(self):
        for i,(q,r) in enumerate(self.pairs):
            if abs(self.k[q]+self.k[r])<1e-12 and abs(abs(self.k[q])-np.pi/2)<1e-10:
                return i
        raise ValueError('N must be 4m+2 to include k0=+-pi/2 on antiperiodic grid')

    def S1(self,ws):
        mats=np.linalg.inv(ws[:,None,None]*np.eye(self.M)[None,:,:]-self.Heff)
        s=np.eye(self.N)[None,:,:]-1j*self.kappa*mats[:,:self.N,:self.N]
        return mats,s

    @cached_property
    def contact_spectral(self):
        eig,V=sla.eig(self.Heff)
        iv=sla.inv(V)
        C=self.C3
        left=np.einsum('im,jn,mna->ija',iv,iv,C,optimize=True).reshape(-1,3*self.N)
        right=np.einsum('mna,mi,nj->aij',C.conj(),V,V,optimize=True).reshape(3*self.N,-1)
        return eig,left,right,np.linalg.cond(V)

    def contact_loop(self,Es):
        eig,left,right,cond=self.contact_spectral
        den=1/(Es[:,None]-(eig[:,None]+eig[None,:]).reshape(-1)[None,:])
        Pi=(right[None,:,:]*den[:,None,:])@np.broadcast_to(left,(len(Es),*left.shape))
        return Pi

    def tau(self,Es,U):
        Pi=self.contact_loop(Es)
        if U==0:return np.zeros_like(Pi),Pi
        eye=np.eye(3*self.N)
        tau=np.linalg.solve(eye[None,:,:]-U*Pi,U*np.broadcast_to(eye,(len(Es),3*self.N,3*self.N)))
        return tau,Pi

    def input_legs(self,greens):
        p1,p2=self.input
        u=greens[:,:,p1];v=greens[:,:,p2]
        return np.sqrt(2)*np.einsum('im,jn,mna->ija',u,v,self.C3.conj(),optimize=True)

    def output_legs(self,greens,q,r):
        u=greens[:,q,:];v=greens[:,r,:]
        return np.sqrt(2)*np.einsum('im,jn,mna->ija',u,v,self.C3,optimize=True)


def gaussian(ws,center,sigma,step):
    f=np.exp(-.25*((ws-center)/sigma)**2).astype(complex)
    f/=np.sqrt(step*np.vdot(f,f).real)
    return f

def solve_two_photon(model,U=.24,nf=81,step=.018,sigma=.045,
                     output_pairs=None,include_wavefunctions=False,calculate_dark=False):
    ws=model.center+(np.arange(nf)-(nf-1)/2)*step
    f=gaussian(ws,model.center,sigma,step)
    greens,s=model.S1(ws)
    Es=2*ws[0]+np.arange(2*nf-1)*step
    tau,Pi=model.tau(Es,U)
    p1,p2=model.input
    Lin=model.input_legs(greens)
    conv=np.zeros((2*nf-1,3*model.N),complex)
    for a in range(nf):
        conv[a:a+nf]+=step*f[a]*f[:,None]*Lin[a]
    propagated=np.einsum('eab,eb->ea',tau,conv,optimize=True)
    idx=np.add.outer(np.arange(nf),np.arange(nf))
    pairs=model.pairs if output_pairs is None else output_pairs
    rows=[];wv={}
    for q,r in pairs:
        Lout=model.output_legs(greens,q,r)
        connected=(-1j*model.kappa**2/(2*np.pi))*np.einsum('ija,ija->ij',Lout,propagated[idx],optimize=True)
                                                                       
        disconnected=np.outer(s[:,q,p1]*f,s[:,r,p2]*f)+np.outer(s[:,q,p2]*f,s[:,r,p1]*f)
        if q==r:
            disconnected/=np.sqrt(2);connected/=np.sqrt(2)
        full=disconnected+connected
        pd=float(step**2*np.vdot(full,full).real)
        pc=float(step**2*np.vdot(connected,connected).real)
        pdis=float(step**2*np.vdot(disconnected,disconnected).real)
        interference=float(2*step**2*np.vdot(disconnected,connected).real)
        rows.append(dict(q=q,r=r,kq=float(model.k[q]),kr=float(model.k[r]),input_pair=(q,r)==model.input,
           P_full=pd,P_connected=pc,P_disconnected=pdis,P_interference=interference))
        if include_wavefunctions:wv[(q,r)]=dict(full=full,connected=connected,disconnected=disconnected)
    dark_probability=None
    if calculate_dark:
        if model.gamma != 0 or model.detuning != 0:
            raise ValueError('Stable dark-asymptotic state requires zero exciton decay and degeneracy')
                                                                                      
        g_dark,_=model.S1(Es-WX)
        dark_probability=0.
        for kd in range(model.N):
            dv=np.zeros(model.M,complex)
            dv[model.N+kd]=-np.sin(model.angles[kd])
            dv[2*model.N+kd]=np.cos(model.angles[kd])
                                                                                   
            assert np.linalg.norm(model.Heff@dv-WX*dv)<5e-13
            cd=np.einsum('mna,n->ma',model.C3,dv.conj(),optimize=True)
            for q in range(model.N):
                lout=np.sqrt(2)*(g_dark[:,q,:]@cd)
                amp=(-1j*model.kappa**1.5/np.sqrt(2*np.pi))*np.einsum('ea,ea->e',lout,propagated,optimize=True)
                dark_probability+=step*float(np.vdot(amp,amp).real)
    out=dict(model=model,rows=rows,wavefunctions=wv,onephoton=s,ws=ws,Es=Es,
       P_dark=dark_probability, P_accounted=(dark_probability+sum(r['P_full'] for r in rows) if dark_probability is not None and output_pairs is None else None),
       loop=Pi,complex_condition=model.contact_spectral[3],
       P_two_photon=float(sum(r['P_full'] for r in rows)) if output_pairs is None else None,
       P_offforward=float(sum(r['P_full'] for r in rows if not r['input_pair'])) if output_pairs is None else None,
       P_connected=float(sum(r['P_connected'] for r in rows)) if output_pairs is None else None,
       P_disconnected=float(sum(r['P_disconnected'] for r in rows)) if output_pairs is None else None,
       P_interference=float(sum(r['P_interference'] for r in rows)) if output_pairs is None else None)
    return out


def linear_bound(sA,sB):
           
    return max(float(np.linalg.norm(a-b,2)) for a,b in zip(sA,sB))

if __name__=='__main__':
    x=Model();print('center',x.center,'input',x.input,x.k[x.input[0]],'contact condition',x.contact_spectral[3])
    a=solve_two_photon(x,nf=61); print('P2',a['P_two_photon'],'off',a['P_offforward'],'connected',a['P_connected'],'disc',a['P_disconnected'],'int',a['P_interference'])
