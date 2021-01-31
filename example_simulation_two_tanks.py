import numpy as np
import matplotlib.pyplot as plt

from dynamic_systems.controller import PID
from dynamic_systems.simulation import TwoTanks


# For figure aesthetics
plt.rcParams['mathtext.fontset'] = 'custom'  
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'  
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'  
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'  
plt.rcParams['font.size'] = 16  
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['font.family'] = 'STIXGeneral' 

t0 = 0.0
tf = 150.0
dt = 0.2
t = np.arange(t0, tf, dt)
Kp, Ki, Kd = 2.1611152842430346, 0.1915584042916885, 0

two_tanks = TwoTanks(fault_type=0, measurement_noise=0.0, process_noise=0.0)
controller = PID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, ref=0.75)

u, x, y = two_tanks(t, [0.0, 0.0], controller)
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t, x[:, 0], label=r'$h_{1}(t)$')
axes[0].plot(t, x[:, 1], label=r'$h_{2}(t)$')
axes[0].plot(t, [0.75] * len(t), 'k--', label=r'r(t)')
axes[0].legend()
axes[1].plot(t, u, label=r'$u(t)$')
# axes[0].set_ylabel(r'$h(t)$')
axes[1].legend()
axes[1].set_xlabel(r'Time $t$')
# axes[1].set_ylabel(r'$q_{in}(t)$')
plt.show()
#plt.savefig('./Figures/PID_Simulation.pdf')
controller = PID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, ref=0.75)

inputs = []
observations = []
for fault in range(5):
    two_tanks = TwoTanks(fault_type=fault, measurement_noise=0.01, process_noise=0.01)
    u, x, y = two_tanks(t, [0.0, 0.0], controller)
    inputs.append(u)
    observations.append(y)
    controller.reset()

fig, axes = plt.subplots(2, 5, sharex='col')
fault_names = ['Normal Operation', 'Sensor Bias', r'$a_{13}$ decay', r'$a_{23}$ decay', r'$a_{12}$ decay']
axes[0, 0].set_ylabel(r'$h_{2}(t)$')
axes[1, 0].set_ylabel(r'$q_{in}(t)$')
for i in range(5):
    axes[0, i].plot(t, observations[i])
    axes[1, i].plot(t, inputs[i])
    axes[0, i].set_title(fault_names[i])
    axes[1, i].set_xlabel(r'$t$')