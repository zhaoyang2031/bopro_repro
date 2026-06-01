import torch
from torch.distributions import Normal
from torch.nn import Softplus
from abc import abstractmethod
from typing import Callable


class Utility:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, observations: torch.Tensor):
        pass

    @staticmethod
    @abstractmethod
    def _expected_utility(
        f: torch.Tensor, stddev: torch.Tensor | float, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()

    def expected_utility(self, f: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Expected utility given true (or predicted) function values

        Args:
            f (torch.Tensor): Function values.

        Returns:
            torch.Tensor: Expected utility values.
        """
        return self._expected_utility(f, *args, **kwargs)

    @abstractmethod
    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate utility given observation values

        Args:
            y (torch.Tensor): Observation values

        Returns:
            torch.Tensor: Utility values at corresponding locations
        """
        return self._utility(y)


class SimpleRegret(Utility):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @staticmethod
    def _expected_utility(f, *args, **kwargs):
        return f


class ThresholdedUtility(Utility):
    """Base class for utility functions which use an improvement threshold"""

    def __init__(self, observations: torch.Tensor, percentile: float | torch.Tensor):
        super().__init__()
        if percentile is None or percentile < 0 or percentile > 1:
            raise ValueError("Percentile must be between 0 and 1")
        if not torch.is_tensor(percentile):
            percentile = torch.as_tensor(percentile).to(observations)
        self._percentile = percentile
        self.update(observations)

    def update(self, observations: torch.Tensor):
        self.threshold = torch.quantile(observations, q=self._percentile)

    @property
    def percentile(self) -> torch.Tensor:
        return self._percentile

    @abstractmethod
    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _expected_utility(
        f: torch.Tensor, stddev: torch.Tensor | float, threshold: torch.Tensor | float
    ) -> torch.Tensor:
        raise NotImplementedError()

    def expected_utility(
        self,
        f: torch.Tensor,
        stddev: torch.Tensor | float,
        threshold: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Expected utility given true (or predicted) function values assuming Gaussian noise.

        Args:
            f (torch.Tensor): Function values.
            stddev (torch.Tensor or float): Noise standard deviation.
            threshold (torch.Tensor or float or None, optional): Threshold value for utility calculation.
                If None, current threshold is used. Defaults to None.

        Returns:
            torch.Tensor: Expected utility values.
        """
        if threshold is None:
            threshold = self.threshold
        return self._expected_utility(f, stddev, threshold)


class ProbabilityOfImprovement(ThresholdedUtility):
    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        u = (y >= self.threshold).to(torch.get_default_dtype())
        return u

    @staticmethod
    def _expected_utility(
        f: torch.Tensor, stddev: torch.Tensor | float, threshold: torch.Tensor | float
    ) -> torch.Tensor:
        normal = Normal(0, stddev)
        return normal.cdf(f - threshold)


class ExpectedImprovement(ThresholdedUtility):
    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        u = (y - self.threshold).maximum(torch.zeros_like(y))
        return u

    @staticmethod
    def _expected_utility(
        f: torch.Tensor, stddev: torch.Tensor | float, threshold: torch.Tensor | float
    ) -> torch.Tensor:
        normal = Normal(0, stddev)
        diff = f - threshold
        ei = diff * normal.cdf(diff) + stddev * normal.log_prob(diff).exp()
        return ei


class SoftExpectedImprovement(ExpectedImprovement):
    def __init__(
        self,
        observations: torch.Tensor,
        percentile: float | torch.Tensor,
        beta: float = 10.0,
        **kwargs,
    ):
        super().__init__(observations, percentile)
        self._fun = Softplus(beta, **kwargs)

    def _utility(self, y: torch.Tensor) -> torch.Tensor:
        u = self._fun(y - self.threshold)
        return u


utilities_registry: dict[str, Callable[[torch.Tensor, float], Utility]] = {
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "SR": SimpleRegret,
    "sEI": SoftExpectedImprovement,
}
