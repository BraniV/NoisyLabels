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
a,b=0.3,0.6; q=(a+b)/2; z=np.linspace(-0.2,1.2,1401)
clip=lambda x: np.clip(x,0,1); P=lambda x: np.clip(x,a,b)
fig,ax=plt.subplots(1,2,figsize=(6.5,2.9),sharey=True)
for k,(axx,om) in enumerate(zip(ax,(1.0,4.0))):
    axx.axhspan(a,b,color='0.9',lw=0); axx.axvspan(a,b,color='0.95',lw=0)
    axx.plot(z,clip(z),color='0.55',lw=1.0,label=r'SURE alone, $\omega=0$')
    axx.plot(z,clip((z+om*P(z))/(1+om)),'k-',lw=1.5,label=r'interval teacher $[a,b]$')
    axx.plot(z,clip((z+om*q)/(1+om)),'k--',lw=1.3,label=r'point teacher $q=(a+b)/2$')
    axx.set_xlabel(r'unpenalized Stein estimate $\widehat z_c$'); axx.set_xlim(-0.2,1.2); axx.set_ylim(-0.02,1.02); axx.set_xticks([0,0.3,0.6,1.0])
    axx.text(0.03,0.95,rf'$\omega={om:g}$',transform=axx.transAxes,va='top')
ax[0].set_ylabel(r'fitted gate $\widehat\beta_c(\omega)$'); ax[1].legend(frameon=False,loc='lower right',fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'theorem1_schematic.pdf')); fig.savefig(os.path.join(FIGS, 'theorem1_schematic.png'),dpi=300)
