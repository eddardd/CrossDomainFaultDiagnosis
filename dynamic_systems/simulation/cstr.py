import numpy as np
from .abstract_sys import AbstractSystem


class CSTR(AbstractSystem):
    def __init__(self,
                 V=150.0,
                 Q=100.0,
                 Vc=10.0,
                 dHr=-2e+5,
                 UA=7e+5,
                 k0=7.2e+10,
                 a0=1.0,
                 b0=1.0,
                 ER=1e+4,
                 rho=1e+3,
                 rhoc=1e+3,
                 Cp=1.0,
                 Cpc=1.0,
                 N=1.0,
                 process_noise=1e-2,
                 measurement_noise=0.00,
                 fault_type=0):
        # Disturbances
        self.Ci = 1.0
        self.Ti = 350.0
        self.Tci = 350.0
        self.V = V
        self.Q = Q
        self.Vc = Vc
        self.dHr = dHr
        self.UA = UA
        self.k0 = k0
        self.a0 = a0
        self.b0 = b0
        self.ER = ER
        self.rho = rho
        self.rhoc = rhoc
        self.Cp = Cp
        self.Cpc = Cpc
        self.N = N
        self.fault_type = fault_type
        super().__init__(process_noise, measurement_noise)

    def dynamics(self, t, x, u):
        QV = self.Q / self.V
        k = self.k0 * np.exp(- self.ER / x[1])
        if self.fault_type == 1:
            a = self.a0 * np.exp(- 0.004 * t)
        else:
            a = self.a0
        if self.fault_type == 2:
            b = self.b0 * np.exp(- 0.005 * t)
        else:
            b = self.b0
        alpha = a * self.dHr * k / (self.rho * self.Cp)
        beta = b * self.UA / (self.rho * self.Cp * self.V)
        betac = b * self.UA / (self.rhoc * self.Cpc * self.Vc)

        dxdt = np.array([
            QV * (self.Ci - x[0]) - a * k * (x[0] ** self.N),
            QV * (self.Ti - x[1]) - alpha * (x[0] ** self.N) - beta * (x[1] - x[2]),
            (u / self.Vc) * (self.Tci - x[2]) + betac * (x[1] - x[2])
        ])
        dxdt += self.process_noise * np.random.randn(*dxdt.shape)

        return dxdt

    def observe(self, t, x):
        return x[1]

    def saturate(self, x):
        x[0] = np.maximum(0, x[0])

        return x

    def __call__(self, t, x0, u_fn):
        dt = t[1] - t[0]
        X = [x0]
        states = []
        observations = []
        actions = []
        for ti in t:
            if self.fault_type == 10 and ti % 60 == 0:
                self.Ci += 0.005 * np.random.randn()
            elif self.fault_type == 11 and ti % 60 == 0:
                self.Ti += 5 * np.random.randn()
            elif self.fault_type == 12 and ti % 60 == 0:
                self.Tci += 5 * np.random.randn()
            state = np.array(X[-1])
            action = u_fn(ti, self.observe(ti, state))
            # Note: observation vector for the CSTR system is quite different from
            # the two-tanks system.
            observation = [self.Ci, self.Ti, self.Tci, *state, action]

            states.append(state)
            observations.append(observation)
            actions.append(action)

            k1 = self.dynamics(ti, state, action)
            state_2 = state + k1 * (dt / 2)
            state_2 = self.saturate(state_2)

            k2 = self.dynamics(ti + (dt / 2), state_2, action) 
            state_3 = state + k2 * (dt / 2)
            state_3 = self.saturate(state_2)

            k3 = self.dynamics(ti + (dt / 2), state_3, action)
            state_4 = state + k3 * dt
            state_4 = self.saturate(state_4)

            k4 = self.dynamics(ti + dt, state_4, action)
            new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            new_state = self.saturate(new_state)

            X.append(new_state)
        actions, states, observations = np.array(actions), np.array(states), np.array(observations)
        # Note: Values set according to the simulink plant of Pilario and Cao
        # Adds measurement noise to Ci
        observations[:, 0] += np.sqrt(0.000001) * np.random.randn(*observations[:, 0].shape)
        # Adds measurement noise to Ti
        observations[:, 1] += 0.05 * np.random.randn(*observations[:, 1].shape)
        # Adds measurement noise to Tci
        observations[:, 2] += 0.05 * np.random.randn(*observations[:, 2].shape)
        # Adds measurement noise to C
        observations[:, 3] += np.sqrt(0.000001) * np.random.randn(*observations[:, 3].shape)
        # Adds measurement noise to T
        observations[:, 4] += 0.05 * np.random.randn(*observations[:, 4].shape)
        # Adds measurement noise to Tc
        observations[:, 5] += 0.05 * np.random.randn(*observations[:, 5].shape)
        # Adds measurement noise to Qc
        observations[:, 6] += 0.2 * np.random.randn(*observations[:, 5].shape)

        if self.fault_type == 3:
            observations[:, 0] += 0.005 * t
        elif self.fault_type == 4:
            observations[:, 1] += 0.1 * t
        elif self.fault_type == 5:
            observations[:, 2] += 0.1 * t
        elif self.fault_type == 6:
            observations[:, 3] += 0.005 * t
        elif self.fault_type == 7:
            observations[:, 4] += 0.1 * t
        elif self.fault_type == 8:
            observations[:, 5] += 0.1 * t
        elif self.fault_type == 9:
            observations[:, 6] -= 0.2 * t
        return actions, states, observations