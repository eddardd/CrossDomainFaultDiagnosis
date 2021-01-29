from abc import ABC, abstractmethod


class AbstractSystem(ABC):
    def __init__(self, process_noise, measurement_noise):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        super().__init__()

    @abstractmethod
    def dynamics(self, t, x, u):
        pass

    @abstractmethod
    def observe(self, t, x):
        pass

    @abstractmethod
    def saturate(self, t, x):
        pass

    @abstractmethod
    def __call__(self, t, x0, u_fn):
        pass