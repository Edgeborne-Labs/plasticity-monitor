from dataclasses import dataclass

@dataclass
class PlasticityState:
    """Holds the current plasticity and velocity state for a learning cycle."""
    eta: float      # current plasticity η_t
    v: float        # current learning velocity v_t
    step: int       # step index within current cycle
    cycle: int      # cycle index
    eta_0: float    # plasticity at cycle start

