import sys; import numpy as np
import os
                                                                           
SOLVER_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOLVER_DIR)
import s3_radiative as s3
orig = s3.contact_tmatrix
def born_tm(dat, Es, U):                                               
    t, l = orig(dat, Es, U)
    return {k: U*np.broadcast_to(np.eye(3), v.shape).copy() for k, v in t.items()}, l
COMMON = dict(kappa=.18, sigma=.045, nf=161, step=.012, fast=True)
def chan(N, U, born=False):
    s3.contact_tmatrix = born_tm if born else orig
    r, _ = s3.calculate(N=N, U=U, **COMMON)
    s3.contact_tmatrix = orig
    ks = np.array(r['k_reps']); o = np.argsort(ks)
    pa = np.array(r['results']['A']['connected_angular_probabilities'])[o]
    pb = np.array(r['results']['B']['connected_angular_probabilities'])[o]
    i0 = int(np.argmin(abs(ks[o] - r['params']['k0'])))
    return pa, pb, i0
def est(r, i0, q):
    r1 = (r[i0+1]+r[i0-1])/2; r2 = (r[i0+2]+r[i0-2])/2; rho = r1/r2
    return (rho-1)/(3*q*q), (rho-1)/(q*q*(4*rho-1))
for N in (14, 30, 62, 126):
    q = 2*np.pi/N
    W, _, i0 = chan(N, 1.0, born=True)                                      
    _, pbB, _ = chan(N, 1.0, born=True)
    line = [f'N={N:3d}']
    for lab, U in (('Born', None), ('0.24', .24), ('0.50', .50)):
        pb = pbB if U is None else chan(N, U)[1]
        r = pb / W
        e1, e2 = est(r, i0, q)
        line.append(f'{lab}: lin={e1:.4f} exact={e2:.4f}')
                                      
    lam = .7; rr = lambda x: np.cos(lam*np.sin(x))**4
    rho = rr(q)/rr(2*q); line.append(f'analytic exact-inv={(rho-1)/(q*q*(4*rho-1)):.4f}')
    print('  '.join(line), flush=True)
