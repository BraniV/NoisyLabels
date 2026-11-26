"""Recompute the label-based pointwise rules on the full u-grid (same pilot draws as sim7.py)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', '..', 'paper', 'Figs')
RES = os.path.join(HERE, '..', '..', 'results')
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
import numpy as np, sys, sim7
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = {}
for tau in sim7.TAUS:
    m = sim7.Model(tau)
    for fl in sim7.FLIPS:
        res = {}
        for r in range(R):
            rng = np.random.default_rng([sim7.SEED, int(tau), int(fl[0] * 100), r])
            bu, Cu, bx = sim7.pilot_fit(rng, m, *fl)
            for km in (0.5, 1.0):
                res.setdefault(('wmid', km), []).append(sim7.pointwise_rule_risk(m, bx, km * fl[0], km * fl[1], 'midpoint'))
            res.setdefault('fwd', []).append(sim7.pointwise_rule_risk(m, bx, 0, 0, 'forward', fl))
        out[(tau, fl)] = {k: np.array(v) for k, v in res.items()}
        print(tau, fl, {k: round(float(v.mean()), 4) for k, v in out[(tau, fl)].items()}, flush=True)
np.save(os.path.join(RES, 'sim7_pointwise.npy'), out, allow_pickle=True)
