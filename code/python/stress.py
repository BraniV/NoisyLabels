"""Boundary-noise stress test for MLShrinkNoisyLabels (Section 7.3).

True label noise is one-sided and attains the analyst's bound: rho0 = 0, rho1 = 0.30 for every x,
so the clean inclusion probability sits at the upper end of the admissible bracket and the
interval midpoint is biased. Compared: SURE alone, interval teacher, midpoint point teacher, and
the rate-as-inclusion point teacher (noisy label rate treated as the inclusion probability).
Bounds: admissible (0.15, 0.30) and too narrow (0.075, 0.15). Slab-scale misspecification:
the analyst maps inclusion bounds to the gate scale with tau_a in {1.5, 3} instead of tau = 2.
Usage: python3 stress.py [R]  -> stress_results.npy, stress_summary.txt
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', '..', 'paper', 'Figs')
RES = os.path.join(HERE, '..', '..', 'results')
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
import numpy as np, sys, sim7
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
RHO0, RHO1 = 0.0, 0.30
sim7.flip_rates = lambda d, rb0, rb1: (np.full_like(d, RHO0), np.full_like(d, RHO1))
NS = (128, 512, 2048, 8192); inf = np.inf; K = sim7.KCELL
m = sim7.Model(2.0); T = sim7.tables(m); q = sim7.cell_quadratics(T, K)
pstar = np.clip(q[1] / q[0], 0, 1); ctx = sim7.risk_cells(pstar, q)
Ta = {1.5: sim7.tables(sim7.Model(1.5)), 3.0: sim7.tables(sim7.Model(3.0))}

# closed form of zeta: t V_hi / ((1 - t) V_lo + t V_hi)
d = sim7.dgrid(m); dd = d[1] - d[0]; A, C, _ = m.endpoints(d); B = C - A
from scipy.stats import norm
def V(pi): return np.sum(B * B * (pi * norm.pdf(d, 0, m.s1) + (1 - pi) * norm.pdf(d, 0, sim7.SIG))) * dd
Vlo, Vhi = V(sim7.PLO), V(sim7.PHI); pg = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 0.7])
t = (pg - sim7.PLO) / (sim7.PHI - sim7.PLO)
closed = t * Vhi / ((1 - t) * Vlo + t * Vhi); tab = sim7.zeta_of(T, pg)
print('closed-form zeta', np.round(closed, 4), 'quadrature', np.round(tab, 4), 'max diff', np.abs(closed - tab).max())

def teachers(bu, Cu, bounds, Tmap):
    qlo, qhi = sim7.u_teacher(bu, Cu, sim7.UG, True); qpl, _ = sim7.u_teacher(bu, Cu, sim7.UG, False)
    a, b = sim7.cell_intervals(Tmap, K, qlo, qhi, *bounds, sim7.UG)
    ar, br = sim7.cell_intervals(Tmap, K, qpl, qpl, 0.0, 0.0, sim7.UG)       # rate-as-inclusion
    return a, b, (a + b) / 2, (ar + br) / 2

res = {}
for r in range(R):
    rng = np.random.default_rng([sim7.SEED, 555, r])
    bu, Cu, bx = sim7.pilot_fit(rng, m, 0.0, 0.0)
    tch = {('adm', 2.0): teachers(bu, Cu, (0.15, 0.30), T), ('narrow', 2.0): teachers(bu, Cu, (0.075, 0.15), T),
           ('adm', 1.5): teachers(bu, Cu, (0.15, 0.30), Ta[1.5]), ('adm', 3.0): teachers(bu, Cu, (0.15, 0.30), Ta[3.0])}
    for key, (a, b, mid, rate) in tch.items():
        res.setdefault((key, 'cov'), []).append(np.mean((pstar >= a - 1e-12) & (pstar <= b + 1e-12)))
        res.setdefault((key, 'width'), []).append(np.mean(b - a))
        res.setdefault((key, 'middist'), []).append(np.mean(np.abs(mid - pstar)))
    for n in NS:
        u, Y, Vv = sim7.stein_scores(m, rng, n); c = sim7.cell_index(u, K)
        Ys, Vs = np.bincount(c, Y, K), np.bincount(c, Vv, K)
        res.setdefault((n, 'sure'), []).append(sim7.risk_cells(sim7.cell_fit(Ys, Vs, np.zeros(K), np.ones(K), 0), q) - ctx)
        for key, (a, b, mid, rate) in tch.items():
            for lab, lo, hi in (('int', a, b), ('mid', mid, mid), ('rate', rate, rate)):
                for om in (4.0, inf):
                    if lab != 'int' and om == inf: continue
                    res.setdefault((n, key, lab, om), []).append(sim7.risk_cells(sim7.cell_fit(Ys, Vs, lo, hi, om), q) - ctx)
                omc = sim7.cv_omega(u, Y, Vv, K, lo, hi, rng)
                res.setdefault((n, key, lab, 'cv'), []).append(sim7.risk_cells(sim7.cell_fit(Ys, Vs, lo, hi, omc), q) - ctx)
    if r % 100 == 99: print('rep', r + 1, flush=True)
res = {k: np.array(v) for k, v in res.items()}
np.save(os.path.join(RES, 'stress_results.npy'), res, allow_pickle=True)
L = []
def ms(x): return f'{1e3*x.mean():.2f} ({1e3*x.std(ddof=1)/np.sqrt(len(x)):.2f})'
for key in (('adm', 2.0), ('narrow', 2.0), ('adm', 1.5), ('adm', 3.0)):
    L.append(f'{key}: coverage {100*res[(key,"cov")].mean():.1f}%  width {res[(key,"width")].mean():.3f}  |midpoint - target| {res[(key,"middist")].mean():.3f}')
    for n in NS:
        s = res[(n, 'sure')]; row = [f'n={n} SURE {ms(s)}']
        for lab, om in (('int', inf), ('int', 'cv'), ('mid', 4.0), ('mid', 'cv'), ('rate', 4.0), ('rate', 'cv')):
            x = res[(n, key, lab, om)]; row.append(f'{lab}{om} {ms(x)} worse {100*np.mean(x > s + 1e-12):.0f}%')
        L.append('   ' + ' | '.join(row))
open(os.path.join(RES, 'stress_summary.txt'), 'w').write('\n'.join(L)); print('\n'.join(L))
