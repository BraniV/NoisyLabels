import os
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', '..', 'paper', 'Figs')
RES = os.path.join(HERE, '..', '..', 'results')
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf','/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':'serif','font.serif':['TeX Gyre Termes','Times New Roman','Times','Nimbus Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'pdf.fonttype':42})
D=np.load(os.path.join(RES, 'sim7_results.npy'),allow_pickle=True).item(); M=D['main']; S=D['sec']; PW=np.load(os.path.join(RES, 'sim7_pointwise.npy'),allow_pickle=True).item()
NS=(128,512,2048,8192); FL=((0.05,0.2),(0.1,0.3)); inf=np.inf
def ms(x): return 1e3*x.mean(), 1e3*x.std(ddof=1)/np.sqrt(len(x))
L=[]
# Table A: main, k=1
L.append('% Table A: excess risk over context oracle (x1e3), k=1')
for fl in FL:
  for tau in (2.0,4.0):
    o=M[(tau,fl)]; ctx=M[(tau,'const')]['ctx']
    for name,key in (('SURE alone',lambda n:(n,'sure')),('Interval teacher, cross-fitted $\\omega$',lambda n:(n,'int',1.0,'cv')),('Interval teacher, $\\omega=\\infty$',lambda n:(n,'int',1.0,inf)),('Point teacher, cross-fitted $\\omega$',lambda n:(n,'pt',1.0,'cv')),('Point teacher, $\\omega=4$',lambda n:(n,'pt',1.0,4.0))):
      cells=[]
      for n in NS:
        m_,s_=ms(o[key(n)]-ctx); cells.append(f'{m_:.2f} ({s_:.2f})')
      L.append(f'{fl} tau={tau} & {name} & '+' & '.join(cells)+r' \\')
# viol count
tot=0; nv=0
for k,o in M.items():
  if isinstance(k[1],tuple):
    for kk,v in o.items():
      if isinstance(kk,tuple) and len(kk)==4 and kk[1]=='viol': tot+=len(v); nv+=v.sum()
L.append(f'% pathwise violations {nv} of {tot}')
# Table B: k sweep at n=512 and 128, tau=2 and 4
L.append('% Table B: k sweep, n=128 and 512')
for fl in FL:
  for tau in (2.0,4.0):
    o=M[(tau,fl)]; ctx=M[(tau,'const')]['ctx']
    for km in (0.0,0.5,1.0,1.5):
      c=o[(512,'cov',km)].mean()*100; cp=o[(512,'covpl',km)].mean()*100; w=o[(512,'width',km)].mean()
      i128=ms(o[(128,'int',km,'cv')]-ctx)[0]; p128=ms(o[(128,'pt',km,'cv')]-ctx)[0]; i512=ms(o[(512,'int',km,'cv')]-ctx)[0]; p512=ms(o[(512,'pt',km,'cv')]-ctx)[0]
      L.append(f'{fl} tau={tau} k={km} & {c:.1f} & {cp:.1f} & {w:.3f} & {i128:.2f} & {p128:.2f} & {i512:.2f} & {p512:.2f} \\\\  SURE128 {ms(o[(128,"sure")]-ctx)[0]:.2f} SURE512 {ms(o[(512,"sure")]-ctx)[0]:.2f}')
# Table C: pointwise label rules vs Bayes and ctx
L.append('% Table C: pointwise rules, risk minus Bayes (x1e3); ctx - Bayes')
for tau in (2.0,4.0):
  c=M[(tau,'const')]
  for fl in FL:
    p=PW[(tau,fl)]
    L.append(f'tau={tau} {fl}: ctx-Bayes {1e3*(c["ctx"]-c["bayes"]):.2f}; wmid k1 {ms(p[("wmid",1.0)]-c["bayes"])}; wmid k.5 {ms(p[("wmid",0.5)]-c["bayes"])}; fwd {ms(p["fwd"]-c["bayes"])}')
L.append('% Secondary'); 
for k,v in S.items():
  if isinstance(v,dict): L.append(k+' '+str({kk:tuple(round(z,2) for z in ms(vv)) for kk,vv in v.items()}))
open(os.path.join(RES, 'sim7_summary.txt'),'w').write('\n'.join(L)); print('\n'.join(L))
# Figure
fig,ax=plt.subplots(1,4,figsize=(7.0,2.7),sharey=True)
st=[('SURE alone',lambda n:(n,'sure'),dict(color='0.55',marker='o',ls='-')),('Interval, $\\omega=\\infty$',lambda n:(n,'int',1.0,inf),dict(color='k',marker='s',ls='-')),('Interval, cross-fitted',lambda n:(n,'int',1.0,'cv'),dict(color='k',marker='s',ls='-',mfc='w')),('Midpoint, cross-fitted',lambda n:(n,'pt',1.0,'cv'),dict(color='k',marker='^',ls='--')),('Midpoint, $\\omega=4$',lambda n:(n,'pt',1.0,4.0),dict(color='k',marker='v',ls=':'))]
for a,(tau,fl) in zip(ax,[(2.0,FL[0]),(2.0,FL[1]),(4.0,FL[0]),(4.0,FL[1])]):
  o=M[(tau,fl)]; ctx=M[(tau,'const')]['ctx']
  for lab,key,kw in st:
    y=[1e3*(o[key(n)]-ctx).mean() for n in NS]; a.plot(NS,y,ms=3.5,lw=1.1,label=lab,**kw)
  a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xticks(NS); a.set_xticklabels(['128','512','2048','8192'])
  a.set_xlabel('training coefficients $n$'); a.text(0.04,0.04,rf'$\tau={tau:g}$, $\bar\rho=({fl[0]:g},{fl[1]:g})$',transform=a.transAxes,fontsize=8)
ax[0].set_ylabel(r'excess risk over context oracle ($\times10^{-3}$)')
fig.legend(*ax[0].get_legend_handles_labels(),frameon=False,fontsize=8,ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig(os.path.join(FIGS, 'sim7_excess.pdf')); fig.savefig(os.path.join(FIGS, 'sim7_excess.png'),dpi=300)
