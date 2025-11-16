from abc import ABC, abstractmethod

class Metric(ABC):
    """Base class for all metrics used in the SafetyMonitor."""

    @abstractmethod
    def update(self, **kwargs):
        """Ingest new data relevant to this metric."""
        pass

    @abstractmethod
    def compute(self):
        """Return the current metric value(s) as a scalar or dict."""
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state."""
        pass

