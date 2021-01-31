import numpy as np

from scipy.constants import g
from .abstract_sys import AbstractSystem


class TwoTanks(AbstractSystem):
    r"""Two tanks system simulator. This class simulates the two-tanks nonlinear system using the standard euler method
    for solving the following ODE,

    .. math::
        \dot{x}_{1} &= \dfrac{1}{A_{1}}q_{in}(t) - \dfrac{\sqrt{2g}(a_{12}+a_{13})}{A_{1}}\sqrt{x_{1}},
        \dot{x}_{2} &= \dfrac{a_{12}\sqrt{g}}{A_{2}}\sqrt{x_{1}} - \dfrac{a_{23}\sqrt{2g}}{A_{2}}\sqrt{x_{2}}.


    Attributes
    ----------
    A1: float
        Area of Tank 1
    A2: float
        Area of Tank 2
    a12: float
        Orifice area connecting tanks 1 and 2
    a13: float
        Orifice area connecting tank 1 with the environment
    a23: float
        Orifice area connecting tank 2 with the environment
    fault_type: int
        Fault type occuring in the system. 0 corresponds to normal operation.
    """
    def __init__(self, A1=1, A2=1, a12=0.1, a13=0.1, a23=0.1,
                 h1_max=1.5, h2_max=1.5, fault_type=0,
                 measurement_noise=0.01, process_noise=0.01):
        self.A1 = A1
        self.A2 = A2
        self.a12 = a12
        self.a13 = a13
        self.a23 = a23
        self.fault_type = fault_type
        self.h1_max = h1_max
        self.h2_max = h2_max
        super().__init__(process_noise, measurement_noise)
        #self.measurement_noise = measurement_noise
        #self.process_noise = process_noise

    def dynamics(self, t, x, u):
        r"""Implements the mathematical function,

        .. math::
            \dot{\mathbf{x}} &= \mathbf{f}(\mathbf{x}, \mathbf{u}, t)

        Parameters
        ----------
        x : tuple of floats
            Tuple containing the system state. More specifically, x = (h1, h2).
        u : float
            Action imposed on the system. In the system description, u(t) = q_{in}(t).
        t : float
            time.

        Returns
        dx : tuple of floats
            Tuple containing the derivative of h1 and h2.
        """
        if self.fault_type == 2:
            a13 = self.a13 * np.exp(- 3e-2 * t)
        else:
            a13 = self.a13
        if self.fault_type == 3:
            a23 = self.a23 * np.exp(- 3e-2 * t)
        else:
            a23 = self.a23
        if self.fault_type == 4:
            a12 = self.a12 * np.exp(- 3e-2 * t)
        else:
            a12 = self.a12

        C = np.array([
            [1 / self.A1, - np.sqrt(2 * g) * (a12 + a13) / self.A1],
            [a12 * np.sqrt(2 * g) / self.A2, - a23 * np.sqrt(2 * g) / self.A2]
        ])
        h1, h2 = x

        dh1 = C[0, 0] * u + C[0, 1] * np.sqrt(h1) + self.process_noise * np.random.randn()
        dh2 = C[1, 0] * np.sqrt(h1) + C[1, 1] * np.sqrt(h2) + self.process_noise * np.random.randn()
        dx = np.array([dh1, dh2])
        return dx

    def observe(self, t, x):
        """Observation function. Transform states into observations. If the system is in normal operation, simply
        returns h2 + sensor noise. If the system is in fault f1, adds sensor bias (trend).

        Parameters
        ----------
        x : tuple of floats
            Tuple containing the system states.
        t : time of observation

        Returns
        -------
        y : float
            Observation.

        """
        if self.fault_type == 1:
            y = x[1] + self.measurement_noise * np.random.randn() + (np.sqrt(2) / 100) * t
        else:
            y = x[1] + self.measurement_noise * np.random.randn()
        return y

    def saturate(self, x):
        h1, h2 = x
        h1 = h1 if h1 > 0 else 0
        h1 = h1 if h1 < self.h1_max else self.h1_max
        h2 = h2 if h2 > 0 else 0
        h2 = h2 if h2 < self.h2_max else self.h2_max

        return np.array([h1, h2])


    def __call__(self, t, x0, u_fn):
        dt = t[1] - t[0]
        X = [x0]
        states = []
        observations = []
        actions = []
        for ti in t:
            state = np.array(X[-1])
            observation = self.observe(ti, state)
            action = u_fn(ti, observation)

            states.append(state)
            observations.append(observation)
            actions.append(action)

            mid_state = X[-1] + self.dynamics(ti, state, action) * (dt / 2)
            mid_state = self.saturate(mid_state)
            new_state = mid_state + self.dynamics(ti, mid_state, action) * dt
            new_state = self.saturate(mid_state)

            X.append(new_state)
        return np.array(actions), np.array(states), np.array(observations)
