#!/usr/bin/env python3
   

import csv
from pathlib import Path

from s3_radiative import calculate

OUTPUT = Path(__file__).resolve().parent / "P3_Finite_Size.csv"
SIZES = (14, 30, 46, 62, 126)
INTERACTIONS = (0.24, 0.50)
COMMON = dict(kappa=0.18, loss=0.0, sigma=0.045,
              nf=161, step=0.012, lam=0.7, fast=True)


def main():
    rows = []
    for N in SIZES:
        measurements = {}
        for U in INTERACTIONS:
            result, _ = calculate(N=N, U=U, **COMMON)
            p_a = result["results"]["A"]["off_forward"]
            p_b = result["results"]["B"]["off_forward"]
            if p_a <= 0:
                raise ValueError(f"Nonpositive P_A for N={N}, U={U}")
            measurements[U] = (p_a, p_b, p_b / p_a)

        a024, b024, r024 = measurements[0.24]
        a050, b050, r050 = measurements[0.50]
                                                                         
        rows.append([N, a024, b024, r024, a050, b050, r050])
        print(f"N={N:3d}: P_A(0.24)={a024:.8f}, "
              f"P_B(0.24)={b024:.8f}, R(0.24)={r024:.8f}, "
              f"R(0.50)={r050:.8f}", flush=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "P_A_U024", "P_B_U024", "R_U024",
                         "P_A_U050", "P_B_U050", "R_U050"])
        for row in rows:
            writer.writerow([row[0], *(f"{value:.12f}" for value in row[1:])])

    print(f"Saved {len(rows)} lattice sizes to {OUTPUT}")


if __name__ == "__main__":
    main()
