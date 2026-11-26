"""Known-prior simulation of MLShrinkNoisyLabels (Section 7): Stein-calibrated context gates
with point and interval teachers built from noisy membership labels.

Model: U ~ U(0,1), logit pi(u) = -4 + 5u, H ~ Bern(pi(U)), theta | H=1 ~ N(0, tau^2), D = theta + Z.
Endpoints: Gaussian spike-and-slab posterior means at pi_lo = 0.01 and pi_hi = 0.9.
All risks are exact (quadrature); fitted gates are evaluated on the population.
Usage: python3 sim7.py [R]   -> results/sim7_results.npy
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', '..', 'paper', 'Figs')
RES = os.path.join(HERE, '..', '..', 'results')
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
import numpy as np, sys, time, itertools
from scipy.stats import norm
from scipy.optimize import minimize

R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 20261125
PLO, PHI, SIG = 0.01, 0.9, 1.0
TAUS = (2.0, 4.0); NS = (128, 512, 2048, 8192); FLIPS = ((0.05, 0.20), (0.10, 0.30)); KS_MULT = (0.0, 0.5, 1.0, 1.5)
OMEGAS = (0.0, 0.25, 1.0, 4.0, np.inf); NPILOT = 5000; KCELL = 10

def logistic(z): return 1 / (1 + np.exp(-z))
def pi_of_u(u): return logistic(-4 + 5 * u)

class Model:
    def __init__(s, tau):
        s.tau = tau; s.s = tau**2 / (tau**2 + SIG**2); s.s1 = np.sqrt(SIG**2 + tau**2)
        s.c = 0.5 * np.log(s.s1**2 / SIG**2); s.k = 0.5 * (1 / SIG**2 - 1 / s.s1**2)
    def lr(s, d): return -s.c + s.k * d * d                    # log phi_s1(d)/phi_sig(d)
    def w(s, pi, d): return logistic(np.log(pi / (1 - pi)) + s.lr(d))
    def rule(s, pi, d): return s.s * s.w(pi, d) * d
    def drule(s, pi, d):
        w = s.w(pi, d); return s.s * (w + d * w * (1 - w) * 2 * s.k * d)
    def endpoints(s, d):
        A = s.rule(PLO, d); C = s.rule(PHI, d)
        dB = s.drule(PHI, d) - s.drule(PLO, d)
        return A, C, dB

# ------------------------------------------------------------------ quadrature tables
UG = (np.arange(400) + 0.5) / 400                    # u grid (density 1)
def dgrid(m): L = 8 * m.s1; return np.linspace(-L, L, 1601)
def tables(m):
    d = dgrid(m); dd = d[1] - d[0]; A, C, _ = m.endpoints(d); B = C - A
    T = dict(Vb=np.zeros(len(UG)), Yb=np.zeros(len(UG)), Eb=np.zeros(len(UG)), Ebayes=0.0)
    for i, u in enumerate(UG):
        pi = pi_of_u(u); f = (pi * norm.pdf(d, 0, m.s1) + (1 - pi) * norm.pdf(d, 0, SIG)) * dd
        w = m.w(pi, d); mm = m.s * w * d; pv = w * (m.s * SIG**2 + (m.s * d)**2) - mm**2
        T['Vb'][i] = np.sum(B * B * f); T['Yb'][i] = np.sum(B * (mm - A) * f)
        T['Eb'][i] = np.sum((pv + (A - mm)**2) * f); T['Ebayes'] += np.sum(pv * f) / len(UG)
    # zeta(pi) table
    pg = np.linspace(PLO, PHI, 2001); Z = np.zeros_like(pg)
    for i, pi in enumerate(pg):
        f = (pi * norm.pdf(d, 0, m.s1) + (1 - pi) * norm.pdf(d, 0, SIG)) * dd
        Z[i] = np.sum(B * (m.rule(pi, d) - A) * f) / np.sum(B * B * f)
    T['zeta_pi'] = pg; T['zeta'] = np.maximum.accumulate(Z)
    return T

def cell_index(u, K): return np.minimum((u * K).astype(int), K - 1)
def cell_quadratics(T, K):
    c = cell_index(UG, K); n = len(UG)
    a = np.bincount(c, T['Vb'], K) / n; b = np.bincount(c, T['Yb'], K) / n; e = np.bincount(c, T['Eb'], K) / n
    return a, b, e                                 # risk contribution a beta^2 - 2 b beta + e
def risk_cells(beta, q): a, b, e = q; return np.sum(a * beta**2 - 2 * b * beta + e)

def zeta_of(T, pi): return np.interp(np.clip(pi, PLO, PHI), T['zeta_pi'], T['zeta'])

# ------------------------------------------------------------------ teacher from noisy membership labels
KN = np.array([0.25, 0.5, 0.75])
def spline_u(u): return np.column_stack([np.ones_like(u), u, u**2, u**3] + [np.maximum(u - k, 0)**3 for k in KN])
def fit_logit(X, y, ridge=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = logistic(X @ b); W = p * (1 - p)
        H = (X * W[:, None]).T @ X + ridge * np.eye(len(b)); g = X.T @ (y - p) - ridge * b
        st = np.linalg.solve(H, g); b += st
        if np.max(np.abs(st)) < 1e-9: break
    p = logistic(X @ b); H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b))
    return b, np.linalg.inv(H)

def flip_rates(d, rb0, rb1):
    e = np.exp(-d**2 / 8); return rb0 * (1 - e), rb1 * e          # rho0(x), rho1(x)

def pilot_fit(rng, m, rb0, rb1):
    u = rng.random(NPILOT); H = rng.random(NPILOT) < pi_of_u(u)
    th = np.where(H, m.tau * rng.standard_normal(NPILOT), 0.0); d = th + SIG * rng.standard_normal(NPILOT)
    r0, r1 = flip_rates(d, rb0, rb1); flip = rng.random(NPILOT) < np.where(H, r1, r0)
    Lt = np.where(flip, ~H, H).astype(float)
    bu, Cu = fit_logit(spline_u(u), Lt)                           # P(L~ = 1 | u)
    Xx = np.column_stack([spline_u(u), d**2]); bx, _ = fit_logit(Xx, Lt)   # P(L~ = 1 | d, u)
    return bu, Cu, bx

def u_teacher(bu, Cu, ug, ci=True):
    X = spline_u(ug); eta = X @ bu
    if not ci: q = logistic(eta); return q, q
    se = np.sqrt(np.einsum('ij,jk,ik->i', X, Cu, X)); return logistic(eta - 1.96 * se), logistic(eta + 1.96 * se)

def cell_intervals(T, K, qlo, qhi, rb0, rb1, ug):
    pim = (qlo - rb0) / (1 - rb0); pip = qhi / (1 - rb1)
    zl = zeta_of(T, np.maximum(pim, PLO)); zh = zeta_of(T, np.minimum(pip, PHI))
    c = cell_index(ug, K); a = np.full(K, np.inf); b = np.full(K, -np.inf)
    np.minimum.at(a, c, zl); np.maximum.at(b, c, zh)
    return np.clip(a, 0, 1), np.clip(b, 0, 1)

# ------------------------------------------------------------------ gates
def proj(z, a, b): return np.minimum(np.maximum(z, a), b)
def cell_fit(Ysum, Vsum, a, b, om):
    z = np.where(Vsum > 0, Ysum / np.maximum(Vsum, 1e-300), 0.5)
    if om == np.inf: return np.clip(proj(z, a, b), 0, 1)
    return np.clip((z + om * proj(z, a, b)) / (1 + om), 0, 1)

def stein_scores(m, rng, n):
    u = rng.random(n); H = rng.random(n) < pi_of_u(u)
    th = np.where(H, m.tau * rng.standard_normal(n), 0.0); d = th + SIG * rng.standard_normal(n)
    A, C, dB = m.endpoints(d); B = C - A
    return u, B * (d - A) - SIG**2 * dB, B * B

def cv_omega(u, Y, V, K, a, b, rng):
    fold = rng.permutation(len(u)) % 2; c = cell_index(u, K); best, bom = np.inf, 0.0
    for om in OMEGAS:
        tot = 0.0
        for f in (0, 1):
            tr = fold == f; te = ~tr
            beta = cell_fit(np.bincount(c[tr], Y[tr], K), np.bincount(c[tr], V[tr], K), a, b, om)
            tot += np.sum(np.bincount(c[te], V[te], K) * beta**2 - 2 * np.bincount(c[te], Y[te], K) * beta)
        if tot < best: best, bom = tot, om
    return bom

# ------------------------------------------------------------------ label-based pointwise rules (exact risk)
def pointwise_rule_risk(m, bx, rb0, rb1, mode, true_flip=None):
    d = dgrid(m); dd = d[1] - d[0]; A, C, _ = m.endpoints(d); tot = 0.0
    for u in UG:
        pi = pi_of_u(u); f = (pi * norm.pdf(d, 0, m.s1) + (1 - pi) * norm.pdf(d, 0, SIG)) * dd
        w = m.w(pi, d); mm = m.s * w * d; pv = w * (m.s * SIG**2 + (m.s * d)**2) - mm**2
        X = np.column_stack([spline_u(np.full_like(d, u)), d**2]); et = logistic(X @ bx)
        if mode == 'midpoint':
            em = np.clip((et - rb0) / (1 - rb0), 0, 1); ep = np.clip(et / (1 - rb1), 0, 1); eta = (em + ep) / 2
        else:                                                    # forward correction with known flip functions
            r0, r1 = flip_rates(d, *true_flip); eta = np.clip((et - r0) / (1 - r0 - r1), 0, 1)
        act = np.clip(m.s * eta * d, np.minimum(A, C), np.maximum(A, C))
        tot += np.sum((pv + (act - mm)**2) * f)
    return tot / len(UG)

# ------------------------------------------------------------------ main experiment
def run_main():
    rng0 = np.random.default_rng(SEED); out = {}; t0 = time.time()
    for tau in TAUS:
        m = Model(tau); T = tables(m); q = cell_quadratics(T, KCELL); a_, b_, e_ = q
        pstar = np.clip(b_ / a_, 0, 1); r_ctx = risk_cells(pstar, q)
        out[(tau, 'const')] = dict(bayes=T['Ebayes'], ctx=r_ctx, ctx_pointwise=np.mean(T['Eb'] - T['Yb']**2 / T['Vb'] * (T['Yb'] / T['Vb'] <= 1) * (T['Yb'] >= 0)),
                                   A=risk_cells(np.zeros(KCELL), q), C=risk_cells(np.ones(KCELL), q), mid=risk_cells(np.full(KCELL, .5), q), pstar=pstar)
        for fl in FLIPS:
            res = {}
            for r in range(R):
                rng = np.random.default_rng([SEED, int(tau), int(fl[0] * 100), r])
                bu, Cu, bx = pilot_fit(rng, m, *fl)
                ug = UG; qlo, qhi = u_teacher(bu, Cu, ug, True); qpl, _ = u_teacher(bu, Cu, ug, False)
                teach = {}
                for km in KS_MULT:
                    rb0, rb1 = km * fl[0], km * fl[1]
                    teach[km] = cell_intervals(T, KCELL, qlo, qhi, rb0, rb1, ug)
                    teach[('pl', km)] = cell_intervals(T, KCELL, qpl, qpl, rb0, rb1, ug)
                    if km in (0.5, 1.0):
                        res.setdefault(('wmid', km), []).append(pointwise_rule_risk(m, bx, rb0, rb1, 'midpoint'))
                res.setdefault('fwd', []).append(pointwise_rule_risk(m, bx, 0, 0, 'forward', fl))
                for n in NS:
                    u, Y, V = stein_scores(m, rng, n); c = cell_index(u, KCELL)
                    Ys, Vs = np.bincount(c, Y, KCELL), np.bincount(c, V, KCELL)
                    b0 = cell_fit(Ys, Vs, np.zeros(KCELL), np.ones(KCELL), 0.0)
                    res.setdefault((n, 'sure'), []).append(risk_cells(b0, q))
                    for km in KS_MULT:
                        a, b = teach[km]; mid = (a + b) / 2
                        valid = (pstar >= a - 1e-12) & (pstar <= b + 1e-12)
                        res.setdefault((n, 'cov', km), []).append(valid.mean())
                        res.setdefault((n, 'width', km), []).append(np.mean(b - a))
                        ap, bp = teach[('pl', km)]
                        res.setdefault((n, 'covpl', km), []).append(((pstar >= ap - 1e-12) & (pstar <= bp + 1e-12)).mean())
                        for om in OMEGAS:
                            bi = cell_fit(Ys, Vs, a, b, om); bq = cell_fit(Ys, Vs, mid, mid, om)
                            res.setdefault((n, 'int', km, om), []).append(risk_cells(bi, q))
                            res.setdefault((n, 'pt', km, om), []).append(risk_cells(bq, q))
                            # pathwise check of Theorem 1 on valid cells: per-cell risk never larger
                            if om in (1.0, np.inf):
                                rc_i = a_ * bi**2 - 2 * b_ * bi; rc_0 = a_ * b0**2 - 2 * b_ * b0
                                res.setdefault((n, 'viol', km, om), []).append(int(np.any((rc_i > rc_0 + 1e-15) & valid)))
                        omi = cv_omega(u, Y, V, KCELL, a, b, rng); omp = cv_omega(u, Y, V, KCELL, mid, mid, rng)
                        res.setdefault((n, 'int', km, 'cv'), []).append(risk_cells(cell_fit(Ys, Vs, a, b, omi), q))
                        res.setdefault((n, 'pt', km, 'cv'), []).append(risk_cells(cell_fit(Ys, Vs, mid, mid, omp), q))
                        if km == 0.5 and n == 8192:
                            # Corollary 1 check (population): predicted vs observed gate bias at omega = 1
                            dist = np.maximum(a - pstar, 0) + np.maximum(pstar - b, 0)
                            bi = cell_fit(Ys, Vs, a, b, 1.0)
                            res.setdefault('cor_pred', []).append(np.mean(dist / 2)); res.setdefault('cor_obs', []).append(np.mean(np.abs(bi - pstar)))
                            res.setdefault('cor_sure', []).append(np.mean(np.abs(b0 - pstar)))
                if r % 50 == 49: print(f'tau {tau} flips {fl} rep {r+1} {time.time()-t0:.0f}s', flush=True)
            out[(tau, fl)] = {k: np.array(v) for k, v in res.items()}
    return out

def run_secondary(R2):
    """tau = 2, n = 512, flips (0.05, 0.20), k = 1: K in {5, 20}, n_pilot = 500, and an affine hat-basis gate."""
    global NPILOT
    m = Model(2.0); T = tables(m); fl = FLIPS[0]; out = {}
    for label, K, npil in (('K5', 5, 5000), ('K10', 10, 5000), ('K20', 20, 5000), ('pilot500', 10, 500)):
        NPILOT = npil; q = cell_quadratics(T, K); pstar = np.clip(q[1] / q[0], 0, 1); ctx = risk_cells(pstar, q); rs = {}
        for r in range(R2):
            rng = np.random.default_rng([SEED, 99, K, npil, r])
            bu, Cu, bx = pilot_fit(rng, m, *fl); qlo, qhi = u_teacher(bu, Cu, UG, True)
            a, b = cell_intervals(T, K, qlo, qhi, *fl, UG); mid = (a + b) / 2
            u, Y, V = stein_scores(m, rng, 512); c = cell_index(u, K); Ys, Vs = np.bincount(c, Y, K), np.bincount(c, V, K)
            rs.setdefault('sure', []).append(risk_cells(cell_fit(Ys, Vs, np.zeros(K), np.ones(K), 0), q) - ctx)
            rs.setdefault('int', []).append(risk_cells(cell_fit(Ys, Vs, a, b, cv_omega(u, Y, V, K, a, b, rng)), q) - ctx)
            rs.setdefault('pt', []).append(risk_cells(cell_fit(Ys, Vs, mid, mid, cv_omega(u, Y, V, K, mid, mid, rng)), q) - ctx)
            rs.setdefault('cov', []).append(np.mean((pstar >= a - 1e-12) & (pstar <= b + 1e-12)))
        out[label] = {k: np.array(v) for k, v in rs.items()}
    NPILOT = 5000
    # affine gate on a linear hat basis (partition of unity), knots at cell centres, K = 10
    K = 10; knots = (np.arange(K) + 0.5) / K
    def hat(u):
        P = np.maximum(0, 1 - np.abs(u[:, None] - knots[None, :]) * K); P[u < knots[0], 0] = 1; P[u > knots[-1], -1] = 1
        return P / P.sum(1, keepdims=True)
    Phi = hat(UG); Qm = (Phi * T['Vb'][:, None]).T @ Phi / len(UG); bv = Phi.T @ T['Yb'] / len(UG); e0 = np.mean(T['Eb'])
    def risk_aff(beta): return beta @ Qm @ beta - 2 * bv @ beta + e0
    bo = minimize(lambda x: risk_aff(x), np.full(K, .5), jac=lambda x: 2 * Qm @ x - 2 * bv, bounds=[(0, 1)] * K, method='L-BFGS-B').x
    ctx_aff = risk_aff(bo); rs = {}
    q10 = cell_quadratics(T, 10)
    for r in range(R2):
        rng = np.random.default_rng([SEED, 77, r])
        bu, Cu, bx = pilot_fit(rng, m, *fl); qlo, qhi = u_teacher(bu, Cu, UG, True)
        pim = np.maximum((qlo - fl[0]) / (1 - fl[0]), PLO); pip = np.minimum(qhi / (1 - fl[1]), PHI)
        u, Y, V = stein_scores(m, rng, 512); P = hat(u)
        al = np.interp(u, UG, zeta_of(T, pim)); bl = np.interp(u, UG, zeta_of(T, pip))
        def J(x, om):
            p = P @ x; dl = np.maximum(al - p, 0); du = np.maximum(p - bl, 0)
            val = np.sum(V * p * p - 2 * Y * p) + om * np.sum(V * (dl**2 + du**2)) + 1e-6 * x @ x
            gp = 2 * V * p - 2 * Y + om * 2 * V * (du - dl)
            return val, P.T @ gp + 2e-6 * x
        for om, key in ((0.0, 'sure'), (1.0, 'int1'), (4.0, 'int4')):
            x = minimize(J, np.full(K, .5), args=(om,), jac=True, bounds=[(0, 1)] * K, method='L-BFGS-B').x
            rs.setdefault(key, []).append(risk_aff(x) - ctx_aff)
    out['affine'] = {k: np.array(v) for k, v in rs.items()}; out['affine_ctx'] = ctx_aff
    return out

if __name__ == '__main__':
    main = run_main(); sec = run_secondary(max(100, R // 2))
    np.save(os.path.join(RES, 'sim7_results.npy'), dict(main=main, sec=sec), allow_pickle=True)
    print('saved')
