import sys, time, csv
import numpy as np
sys.path.insert(0, sys.argv[1])                            
from s3_radiative import calculate, setup, contact_tmatrix, LAM, THETA0

SIZES = (14, 30, 46, 62, 126)
US = (0.001, 0.06, 0.12, 0.24, 0.50)
COMMON = dict(kappa=0.18, loss=0.0, sigma=0.045, nf=161, step=0.012, lam=LAM, fast=True)

def symcurv(r, i0, Q): return (1 - (r[i0-1] + r[i0+1]) / (2*r[i0])) / Q**2

rows, loops_rows = [], []
for N in SIZES:
    Q = 2*np.pi/N
    Rborn = None
    for U in US:
        t = time.time()
        res, _ = calculate(N=N, U=U, **COMMON)
        ks = np.array(res['k_reps']); k0 = res['params']['k0']
        A, B = res['results']['A'], res['results']['B']
        pa = np.array(A['connected_angular_probabilities']); pb = np.array(B['connected_angular_probabilities'])
        o = np.argsort(ks); ks_s = ks[o]; ratio = (pb/pa)[o]
        born = np.cos(LAM*(np.cos(ks_s)-np.cos(k0)))**4
        i0 = int(np.argmin(abs(ks_s-k0)))
        R = B['off_forward']/A['off_forward']
        if U == US[0]: Rborn = R
        c, cb = symcurv(ratio, i0, Q), symcurv(born, i0, Q)
        rows.append(dict(N=N, U=U, P_A=A['off_forward'], P_B=B['off_forward'], NP_A=N*A['off_forward'],
            NP_B=N*B['off_forward'], R=R, R_Born=Rborn, D=R/Rborn, curv=c, curv_born_grid=cb,
            rel_dev=c/cb-1, max_abs_ratio_minus_born=float(np.max(abs(ratio-born))),
            odd=(ratio[i0+1]-ratio[i0-1])/(2*Q), secs=time.time()-t))
        print({k: (round(v,5) if isinstance(v,float) else v) for k,v in rows[-1].items()}, flush=True)
                                          
    dat = setup(N=N, kappa=0.18, lam=LAM)
    h0 = dat['hs']['A'][dat['ik0']][0].copy(); h0[0,0] += .5j*0.18
    E0 = 2*float(np.linalg.eigvalsh(h0)[0])
    _, loops = contact_tmatrix(dat, np.array([E0]), 0.24)
    for tw in 'AB':
        Pi = loops[tw][0]
        sv = np.linalg.svd(Pi, compute_uv=False)
        ev = np.linalg.eigvals(Pi)
        loops_rows.append(dict(N=N, twin=tw, E0=E0, norm_Pi=sv[0],
            UPi_norm_024=0.24*sv[0], UPi_norm_050=0.50*sv[0],
            eig=';'.join(f'{e.real:+.4f}{e.imag:+.4f}j' for e in sorted(ev, key=lambda z: -abs(z)))))
        print(loops_rows[-1], flush=True)

for name, data in (('rerun1_scan.csv', rows), ('rerun2_loops.csv', loops_rows)):
    with open(name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
