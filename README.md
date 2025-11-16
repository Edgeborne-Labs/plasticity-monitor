# plasticity-monitor
# Plasticity Monitor

**Plasticity Half-Life + Safety Monitor for Safe Continual Learning**

Plasticity Monitor is an open-source reference implementation of a **bounded continual-learning mechanism** and **tripwire safety monitor** for modern AI models (LLMs, RL agents, etc.).

It implements:

- **Plasticity Half-Life** – a scalar “learning capacity” η that decays over training steps and is **modulated by learning velocity**.
- **Freeze Conditions** – automatic triggers that stop learning when plasticity is low or learning is too aggressive.
- **Safety Monitor** – a pluggable monitoring framework that tracks:
  - parameter velocity
  - parameter drift
  - (optionally) consistency deviation between measured and implied learning velocity

The goal of this library is to provide a **clear, minimal, research-friendly reference** for **safe continual learning**, while an enterprise-grade monitoring and drift-analysis system is developed separately.

> ⚠️ This library is **not** a full safety solution. It is a foundational mechanism and monitoring primitive that can be integrated into larger systems.

---

## ✨ Core Concepts

### Plasticity Half-Life

We represent the model’s learning capacity as a scalar `eta_t` (ηₜ), which decays over time according to:

\[
\eta_{t+1} = \eta_t \cdot \exp\left(-\frac{1 + \alpha v_t}{\tau_{\text{base}}}\right)
\]

Where:
- `v_t` is an exponential moving average of normalized parameter updates,
- `tau_base` is the base half-life,
- `alpha` controls how sensitively plasticity reacts to learning velocity.

### Learning Velocity

We track how aggressively the model is updating its parameters:

\[
u_t = \frac{\|\Delta \theta_t\|}{\|\theta_t\| + \epsilon}, \quad
v_t = \beta v_{t-1} + (1-\beta)u_t
\]

- `u_t`: relative update size at step t  
- `v_t`: smoothed “learning speed”  

### Freeze Conditions

A learning cycle is **frozen** when:
- `eta_t < eta_min` (plasticity budget exhausted), or
- `v_t > v_max` (learning velocity too high)

This triggers a checkpoint and a drift audit.

---

## 🎯 Project Goals

- Provide a **minimal, understandable reference implementation** of Plasticity Half-Life.
- Give researchers and engineers a way to **experiment with bounded continual learning**.
- Offer a **lightweight safety monitor** that can be integrated into existing training pipelines.
- Serve as the **open specification** for more advanced enterprise implementations.

This repository is **intentionally simple** and focused on clarity.

---

## 📦 Installation

> ⚠️ Note: v0.1 is pre-release and expects Python 3.9+.

Clone the repository:

```bash
git clone https://github.com/<your-username>/plasticity-monitor.git
cd plasticity-monitor
Install in editable mode:

bash
Copy code
pip install -e .
(Once published to PyPI, users would be able to do pip install plasticity-monitor.)

🚀 Quickstart: Plasticity Controller
Here’s a minimal example showing how to use the PlasticityController with synthetic parameter updates:

python
Copy code
from plasticity_monitor.plasticity import PlasticityController
import random

plasticity = PlasticityController(
    eta0=1.0,
    tau_base=50.0,
    alpha=10.0,
    beta=0.9,
    eta_min=0.2,
    v_max=0.05,
)

theta_norm = 1.0  # pretend ||theta||≈1.0 for demo

for step in range(1, 51):
    # fake parameter change
    delta_theta_norm = random.uniform(0.0, 0.05)
    out = plasticity.step(delta_theta_norm, theta_norm)

    print(
        f"step={out['step']:3d}  "
        f"eta={out['eta']:.4f}  "
        f"v={out['v']:.4f}  "
        f"freeze={out['freeze']}  reason={out['freeze_reason']}"
    )

    if out["freeze"]:
        print(">> Freeze triggered, end of cycle.")
        break
This prints ηₜ, vₜ, and freeze events over steps.

🔍 Quickstart: Safety Monitor (Conceptual)
The Safety Monitor aggregates metrics and computes a risk score.

Pseudocode:

python
Copy code
from plasticity_monitor.monitor.safety_monitor import SafetyMonitor
from plasticity_monitor.metrics.param_drift import ParamDriftMetric
from plasticity_monitor.metrics.consistency import ConsistencyDeviationMetric

config = ...  # load YAML or dict with risk scoring, thresholds, actions
actions_executor = ...  # object that knows how to execute actions (log, freeze, notify)

monitor = SafetyMonitor(config, actions_executor)

# During training:
monitor.step(
    param_velocity=v_t,
    plasticity=eta_t,
    # Optionally add drift metrics, eval results, etc.
)
Actual integration will depend on your training setup (PyTorch, RL, etc.), and example scripts will be added under examples/.

🤝 Scope of This OSS Version
Included in the open-source version:

Plasticity scalar η and learning velocity v

Velocity-modulated exponential decay

Freeze conditions (eta_min, v_max)

Minimal Safety Monitor

Basic metrics:

parameter velocity

parameter drift

consistency deviation (measured vs implied average velocity)

Simple PyTorch integration examples

Not included (these are part of future enterprise/pro versions):

Large-scale distributed monitoring

GPU-accelerated activation drift (SVCCA/CCA)

Web-based dashboards and consoles

Multi-agent governance layers

Production integrations with RLHF pipelines, vLLM, etc.

📚 Background
Continual learning methods (EWC, SI, etc.) focus on performance retention, not hard safety bounds.

Alignment methods (RLHF, Constitutional AI) adjust behavior, but not rate-limits on internal change.

Drift detection and activation similarity methods exist, but lack bounded learning control.

Plasticity Monitor aims to provide a primitive building block for safe adaptation, not an end-to-end solution.

📜 License & Patent Notice
This project is released under the Apache 2.0 License (see LICENSE).

⚠ Patent Notice
The Plasticity Half-Life mechanism and Safety Monitor described in this repository are the
subject of a pending patent application held by Edgeborne Labs LLC (inventor: Jeanpaul Powell).
This open-source implementation is provided primarily for research and evaluation.
Commercial use of this mechanism may require a separate patent license from Edgeborne Labs.
For licensing inquiries, please contact: info@edgebornelabs.com.

🧪 Status
 Design and mathematical formulation

 v0.1 PlasticityController implementation

 v0.1 basic SafetyMonitor implementation

 PyTorch integration example

 Initial tests and CI

 v0.1 release

🧾 Citation
If you use this library in academic work, please cite:

Powell, J. (2025). Plasticity Half-Life: Bounded Continual Learning with Velocity-Modulated Decay and Safety Monitoring. Edgeborne Labs LLC.

(BibTeX snippet to be added once an arXiv paper is posted.)

🧭 Roadmap
v0.1 — Core plasticity and minimal Safety Monitor

v0.2 — Consistency deviation metric and basic drift tracking

v0.3 — Behavior drift eval integration

v0.4+ — (subject to roadmap): activation drift, improved integration, more metrics

💬 Contributing
Contributions, suggestions, and issues are welcome.
This is an early-stage project — please open a GitHub issue to discuss ideas.

For serious collaboration inquiries or integration into safety-critical systems, please reach out to:
📧 info@edgebornelabs.com

