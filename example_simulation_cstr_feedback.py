import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
 
from functools import partial
from dynamic_systems.controller import PID
from dynamic_systems.simulation import CSTR
 
# For figure aesthetics 
plt.rcParams['mathtext.fontset'] = 'custom'   
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'   
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'   
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'   
plt.rcParams['font.size'] = 16   
plt.rcParams['mathtext.fontset'] = 'stix'   
plt.rcParams['font.family'] = 'STIXGeneral'  


t = 0.0; tf = 25; dt = 1e-3; t = np.arange(t, tf, dt)  
x0 = np.array([0.1, 430.882699002514, 416.723084574301])
cstr = CSTR(process_noise=0.0, measurement_noise=0.0)
pid = PID(Kp=1, Ki=5, Kd=0, dt=1e-3, ref=430.9, min_action=20, max_action=200, pos_ref=False)
u, x, y = cstr(t, x0, u_fn=pid)

plt.figure()
plt.plot(t, y)
plt.show()
