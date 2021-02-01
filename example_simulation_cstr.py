import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
 
from functools import partial
from dynamic_systems.simulation import CSTR
 
# For figure aesthetics 
plt.rcParams['mathtext.fontset'] = 'custom'   
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'   
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'   
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'   
plt.rcParams['font.size'] = 16   
plt.rcParams['mathtext.fontset'] = 'stix'   
plt.rcParams['font.family'] = 'STIXGeneral'  


def cte_Qc(t, obs, Qc=0.0):
    return Qc


t = 0.0; tf = 25; dt = 1e-3; t = np.arange(t, tf, dt)  
x0 = np.array([0.1, 430.882699002514, 416.723084574301])
cstr = CSTR(process_noise=0.0, measurement_noise=0.0)

Qcs = [0, 25, 50, 75, 100, 125, 150, 175, 200]
state_vars = []

for Qc in Qcs:
    u, x, y = cstr(t, x0, u_fn=partial(cte_Qc, Qc=Qc))
    state_vars.append(x)

fig, axes = plt.subplots(3, 1, figsize=(15, 5))
norm = mpl.colors.Normalize(vmin=np.min(Qcs), vmax=np.max(Qcs))
cmap = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
cmap.set_array([])
for v, Qc in zip(state_vars, Qcs):
    axes[0].plot(t, v[:, 0], c=cmap.to_rgba(Qc))
    axes[1].plot(t, v[:, 1], c=cmap.to_rgba(Qc))
    im = axes[2].plot(t, v[:, 2], c=cmap.to_rgba(Qc))
axes[0].set_yticks([i / 10 for i in range(0, 12, 2)])  
axes[1].set_yticks([350, 400, 450, 500, 550])
axes[2].set_yticks([350, 400, 450, 500, 550])
axes[0].set_ylabel(r'$C$')
axes[1].set_ylabel(r'$T$')
axes[2].set_ylabel(r'$T_{c}$')
axes[2].set_xlabel(r'$t$')
fig.colorbar(cmap, ax=axes.ravel().tolist())
plt.show()