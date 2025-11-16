import math
from .base import Metric

class ConsistencyDeviationMetric(Metric):
    """
    Computes D_v = |v_measured_avg - v_implied_avg|
    about the last learning cycle.

    You should call update() at the end of a cycle with:
      - eta_0: plasticity at cycle start
      - eta_T: plasticity at cycle end
      - v_measured_avg: average v_t over the cycle
      - T: number of steps
    """

    def __init__(self, tau_base: float, alpha: float, eps: float = 1e-12):
        self.tau_base = tau_base
        self.alpha = alpha
        self.eps = eps
        self.v_measured_avg = 0.0
        self.v_implied_avg = 0.0
        self.D_v = 0.0

    def update(
        self,
        eta_0: float | None = None,
        eta_T: float | None = None,
        v_measured_avg: float | None = None,
        T: int | None = None,
        **kwargs,
    ):
        if None in (eta_0, eta_T, v_measured_avg, T):
            return

        self.v_measured_avg = v_measured_avg

        # Avoid log(0) or negative ratios
        ratio = max(eta_0 / (eta_T + self.eps), self.eps)
        self.v_implied_avg = (
            (self.tau_base / (self.alpha * T)) * math.log(ratio)
            - (1.0 / self.alpha)
        )
        self.D_v = abs(self.v_measured_avg - self.v_implied_avg)

    def compute(self):
        return self.D_v

    def reset(self):
        self.v_measured_avg = 0.0
        self.v_implied_avg = 0.0
        self.D_v = 0.0

