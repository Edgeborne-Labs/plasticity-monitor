class SafetyMonitor:
    """
    Aggregates metrics, computes risk, and triggers actions.
    """

    def __init__(self, config: dict, actions_executor):
        self.cfg = config
        self.actions_executor = actions_executor
        self.metrics = self._init_metrics(config.get("metrics", {}))
        self.weights = config["risk_scoring"]["weights"]
        self.transformers = config["risk_scoring"]["transformers"]
        self.thresholds = config["risk_scoring"]["thresholds"]
        self.actions_cfg = config["actions"]

    def _init_metrics(self, metrics_cfg: dict):
        # instantiate Metric subclasses based on config
        # stub for now
        return {}

    def update_metrics(self, **kwargs):
        for metric in self.metrics.values():
            metric.update(**kwargs)

    def compute_risk(self):
        raw = {name: m.compute() for name, m in self.metrics.items()}
        components = {
            name: self._transform_metric(name, val)
            for name, val in raw.items()
        }
        total = 0.0
        for name, val in components.items():
            w = self.weights.get(name, 0.0)
            total += w * val
        return total, components

    def _transform_metric(self, name, value):
        cfg = self.transformers.get(name)
        if cfg is None:
            return max(0.0, min(1.0, float(value)))

        ttype = cfg["type"]
        if ttype == "linear":
            min_v = cfg["min"]
            max_v = cfg["max"]
            if value <= min_v:
                return 0.0
            if value >= max_v:
                return 1.0
            return (value - min_v) / (max_v - min_v)
        # add other transformers as needed
        return max(0.0, min(1.0, float(value)))

    def decide_band(self, risk_score: float) -> str:
        th = self.thresholds
        if risk_score < th["green_max"]:
            return "green"
        elif risk_score < th["yellow_max"]:
            return "yellow"
        elif risk_score < th["orange_max"]:
            return "orange"
        else:
            return "red"

    def step(self, **kwargs):
        self.update_metrics(**kwargs)
        risk, components = self.compute_risk()
        band = self.decide_band(risk)
        actions = self.actions_cfg[f"on_{band}"]
        self.actions_executor.execute(actions, risk, components)
        return {"risk": risk, "band": band, "components": components}

