class PID:
    def __init__(self, Kp, Ki, Kd, dt, ref,
                 min_action=0.0,
                 max_action=1.0,
                 pos_ref=True,
                 initial_action=0.0,
                 implementation="direct"):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.history_err = []
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.Dterm = 0.0
        self.min_action = min_action
        self.max_action = max_action
        self.ref = ref
        self.last_t = 0.0
        self.pos_ref = pos_ref
        self.last_action = initial_action
        self.last_err = 0.0
        self.implementation = implementation

    def __call__(self, t, observation):
        if self.pos_ref:
            err = self.ref - observation
        else:
            err = observation - self.ref
        self.history_err.append(err)
        dt = t - self.last_t

        if self.implementation == "direct":
            self.Pterm = self.Kp * self.history_err[-1]
            self.Iterm += self.Ki * self.history_err[-1] * self.dt
            if dt > 0.0 and len(self.history_err) > 3:
                self.Dterm = self.Kd * (self.history_err[-1] - self.history_err[-2]) / self.dt
            else:
                self.Dterm = 0.0
            
            action = self.Pterm + self.Iterm + self.Dterm
            action = self.saturate(action)
        elif self.implementation == "recursive":
            self.Iterm = self.Ki * dt * self.history_err[-1]
            if len(self.history_err) > 2:
                self.Pterm = self.Kp * (self.history_err[-1] - self.history_err[-2])
            else:
                self.Pterm = 0.0
            if len(self.history_err) > 3 and dt > 0.0:
                self.Dterm = self.Kd * (self.history_err[-1] - 2 * self.history_err[-2] + self.history_err[-3]) / dt
            else:
                self.Dterm = 0.0
            action = self.last_action + self.Pterm + self.Iterm + self.Dterm
            action = self.saturate(action)
        else:
            raise ValueError("Invalid implementation. Should be either 'direct' or 'recursive', but got {}".format(self.implementation))
        self.last_action = action
        self.last_t = t
        return action

    def saturate(self, action):
        action = action if action > self.min_action else self.min_action
        action = action if action < self.max_action else self.max_action

        return action

    def reset(self):
        self.history_err = []
        self.Pterm = self.Iterm = self.Dterm = self.memory_err = 0.0