class PID:
    def __init__(self, Kp, Ki, Kd, dt, ref, max_action=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.memory_err = 0.0
        self.history_err = []
        self.history_obs = []
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.Dterm = 0.0
        self.max_action = max_action
        self.ref = ref

    def __call__(self, t, observation):
        err = self.ref - observation

        self.Pterm = self.Kp * err
        self.Iterm += self.Ki * err * self.dt
        self.Dterm = self.Kd * (err - self.memory_err) / self.dt

        self.history_err.append(err)
        self.history_obs.append(observation)
        self.memory_err = err
        action = self.Pterm + self.Iterm + self.Dterm
        action = action if action > 0.0 else 0.0
        action = action if action < self.max_action else self.max_action
        return action

    def reset(self):
        self.history_err = []
        self.Pterm = self.Iterm = self.Dterm = self.memory_err = 0.0