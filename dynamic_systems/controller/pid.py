class PID:
    def __init__(self, Kp, Ki, Kd, dt, ref,
                 min_action=0.0,
                 max_action=1.0,
                 pos_ref=True):
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
        self.min_action = min_action
        self.max_action = max_action
        self.ref = ref
        self.last_t = 0.0
        self.pos_ref = pos_ref

    def __call__(self, t, observation):
        if self.pos_ref:
            err = self.ref - observation
        else:
            err = observation - self.ref
        dt = t - self.last_t

        self.Pterm = self.Kp * err
        self.Iterm += self.Ki * err * self.dt
        if dt > 0.0:
            self.Dterm = self.Kd * (err - self.memory_err) / self.dt
        else:
            self.Dterm = 0.0

        self.history_err.append(err)
        self.history_obs.append(observation)
        self.memory_err = err
        action = self.Pterm + self.Iterm + self.Dterm
        action = self.saturate(action)
        self.last_t = t
        return action

    def saturate(self, action):
        action = action if action > self.min_action else self.min_action
        action = action if action < self.max_action else self.max_action

        return action

    def reset(self):
        self.history_err = []
        self.Pterm = self.Iterm = self.Dterm = self.memory_err = 0.0