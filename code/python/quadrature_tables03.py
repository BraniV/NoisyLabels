"""Reproduces Table 1 (zeta(pi)) and Table 2 (population risks) of MLShrinkNoisyLabels03."""
import numpy as np
from scipy.stats import norm
sig=1.0; plo,phi=0.01,0.9
for tau in (2.0,4.0):
    s=tau**2/(tau**2+sig**2); s1=np.sqrt(sig**2+tau**2)
    d=np.linspace(-30,30,120001); dd=d[1]-d[0]
    w=lambda pi: pi*norm.pdf(d,0,s1)/(pi*norm.pdf(d,0,s1)+(1-pi)*norm.pdf(d,0,sig))
    A=s*w(plo)*d; C=s*w(phi)*d; B=C-A
    def mix(pi): return pi*norm.pdf(d,0,s1)+(1-pi)*norm.pdf(d,0,sig)
    zeta=[np.sum(B*(s*w(p)*d-A)*mix(p))/np.sum(B*B*mix(p)) for p in (0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9)]
    print('tau',tau,'zeta',np.round(zeta,3))
    R=dict(bayes=0,ctx=0,mid=0,A=0,C=0); us=(np.arange(200)+0.5)/200
    for u in us:
        pi=1/(1+np.exp(4-5*u)); f=mix(pi); m=s*w(pi)*d
        pv=w(pi)*(s*sig**2+(s*d)**2)-m**2
        z=np.clip(np.sum(B*(m-A)*f)/np.sum(B*B*f),0,1)
        for k,a in (('bayes',m),('ctx',A+B*z),('mid',A+B/2),('A',A),('C',C)):
            R[k]+=np.sum((pv+(a-m)**2)*f)*dd/len(us)
    print('tau',tau,{k:round(v,3) for k,v in R.items()})
