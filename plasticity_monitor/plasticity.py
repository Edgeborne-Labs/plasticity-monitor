import math
from .state import PlasticityState

class PlasticityController:
    """
    Implements velocity-modulated plasticity half-life:
      - tracks η_t, v_t across steps
      - decays η_t each step based on v_t
      - triggers freeze when η_t < η_min or v_t > v_max
    """

    def __init__(
        self,
        eta0: float = 1.0,
        tau_base: float = 50.0,
        alpha: float = 10.0,
        beta: float = 0.9,
        eta_min: float = 0.2,
        v_max: float = 0.05,
        eps: float = 1e-12,
    ):
        self.tau_base = tau_base
        self.alpha = alpha
        self.beta = beta
        self.eta_min = eta_min
        self.v_max = v_max
        self.eps = eps

        self.state = PlasticityState(
            eta=eta0,
            v=0.0,
            step=0,
            cycle=0,
            eta_0=eta0,
        )

    def reset_cycle(self, eta0: float | None = None):
        """
        Reset for a new learning cycle (called after freeze and audit).
        """
        s = self.state
        if eta0 is None:
            eta0 = s.eta
        s.cycle += 1
        s.step = 0
        s.eta = eta0
        s.eta_0 = eta0
        s.v = 0.0

    def step(self, delta_theta_norm: float, theta_norm: float) -> dict:
        """
        Perform one learning step update for plasticity.

        Args:
            delta_theta_norm: ||Δθ_t||_2
            theta_norm: ||θ_t||_2

        Returns:
            dict with updated eta, v, step, cycle, freeze flag, reason.
        """
        s = self.state

        # 1. Compute normalized update u_t
        u_t = delta_theta_norm / (theta_norm + self.eps)

        # 2. Update EMA velocity v_t
        v_t = self.beta * s.v + (1.0 - self.beta) * u_t

        # 3. Velocity-modulated plasticity decay
        decay_rate = (1.0 + self.alpha * v_t) / self.tau_base
        eta_next = s.eta * math.exp(-decay_rate)

        # 4. Update state
        s.v = v_t
        s.eta = eta_next
        s.step += 1

        # 5. Freeze conditions
        freeze_plasticity = eta_next < self.eta_min
        freeze_velocity = v_t > self.v_max
        freeze = freeze_plasticity or freeze_velocity

        reason = None
        if freeze_plasticity:
            reason = "eta_min"
        elif freeze_velocity:
            reason = "v_max"

        return {
            "eta": s.eta,
            "v": s.v,
            "step": s.step,
            "cycle": s.cycle,
            "freeze": freeze,
            "freeze_reason": reason,
        }

    def get_state(self) -> PlasticityState:
        return self.state

