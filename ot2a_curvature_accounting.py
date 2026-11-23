#!/usr/bin/env python3
   
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_radiative import calculate, LAM

OUT = Path(__file__).resolve().parent / Path(__file__).stem / 'data'
OUT.mkdir(parents=True, exist_ok=True)

N, KAPPA, SIGMA = 14, .18, .045
Q = 2 * np.pi / N


def sym_curvature(r, i0):
                                                                   
    return (1 - (r[i0 - 1] + r[i0 + 1]) / (2 * r[i0])) / Q**2


rows = []
for U in (0.005, 0.02, 0.06, 0.12, 0.24, 0.50):
    res, _ = calculate(U=U, N=N, kappa=KAPPA, sigma=SIGMA, nf=161, step=.012, fast=True)
    ks = np.array(res['k_reps']); k0 = res['params']['k0']
    pa = np.array(res['results']['A']['connected_angular_probabilities'])
    pb = np.array(res['results']['B']['connected_angular_probabilities'])
    o = np.argsort(ks); ks, ratio = ks[o], (pb / pa)[o]
    born = np.cos(LAM * (np.cos(ks) - np.cos(k0)))**4
    i0 = int(np.argmin(abs(ks - k0)))
    c, cb = sym_curvature(ratio, i0), sym_curvature(born, i0)
    rows.append([U, c, cb, c / cb - 1, float(np.max(abs(ratio - born)))] + list(ratio))
    print(f'U={U}: curvature={c:.4f} born_grid={cb:.4f} rel={c/cb-1:+.3f} '
          f'max|ratio-born|={np.max(abs(ratio-born)):.4f}', flush=True)

with (OUT / 'Curvature_vs_U.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['U', 'sym_curvature', 'born_grid_curvature', 'relative_change',
                'max_abs_ratio_minus_born'] + [f'ratio_p={k:.6f}' for k in ks])
    w.writerows(rows)

acc = []
for nf, step in ((161, .012), (241, .012)):
    res, _ = calculate(U=.24, N=N, kappa=KAPPA, sigma=SIGMA, nf=nf, step=step, fast=True)
    for t in 'AB':
        d = res['results'][t]
        acc.append([nf * step, t, d['norm'], d['stable_dark_probability'],
                    1 - d['photon_plus_dark_closed_norm'], d['off_forward']])
        print(f'window={nf*step:.2f} {t}: pairs={d["norm"]:.6f} '
              f'dark={d["stable_dark_probability"]:.3e} '
              f'residual={1-d["photon_plus_dark_closed_norm"]:.2e} '
              f'off_forward={d["off_forward"]:.6f}', flush=True)

with (OUT / 'Accounting.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frequency_window', 'twin', 'P_photon_pairs', 'P_photon_plus_dark',
                'residual', 'P_off_forward'])
    w.writerows(acc)