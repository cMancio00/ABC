from abc import ABC, abstractmethod
from typing import override

from torch import Tensor
from torch.distributions import Binomial, Bernoulli, MixtureSameFamily, Categorical
import torch


class GenerativeProcess:
    def __init__(self, theta1: float, theta2: float, rate: float, support: int = 4):
        r"""
        A generative process for a mixture of two binomial distributions.

        The process follows the model:
        .. math::

            \lambda \cdot Binomial(x_i | \theta_1, N = 4) + (1 - \lambda) \cdot Binomial(x_i | \theta_2, N = 4)

        :param theta1: Probability of success for the first Binomial distribution.
        :param theta2: Probability of success for the second Binomial distribution.
        :param rate:  The mixing weight (lambda) for the first Binomial distribution.
        :param support: The number of trials for the Binomial distribution (default is 4).
        """

        self.theta1 = theta1
        self.theta2 = theta2
        self.rate = rate
        self.support = support

    @property
    def mixture(self) -> MixtureSameFamily:
        weights = Categorical(probs=torch.tensor([self.rate, 1.0 - self.rate]))

        components = Binomial(
            total_count=self.support,
            probs=torch.tensor([self.theta1, self.theta2]),
        )
        return MixtureSameFamily(weights, components)

    @property
    def density(self) -> tuple[Tensor, Tensor]:
        domain: Tensor = torch.arange(start=0, end=self.support + 1, step=1)
        probs = self.mixture.log_prob(domain).exp()
        return domain, probs


class Simulator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> Tensor: ...

    @abstractmethod
    def propose_parameters(
        self, n_params: int = 100
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class BinomialMixtureSimulator(Simulator):
    @override
    def generate(
        self,
        theta1: Tensor,
        theta2: Tensor,
        rate: Tensor,
        support: int = 4,
        times: int = 100,
    ):
        z = Bernoulli(rate).sample((times,))
        y1 = Binomial(support, theta1).sample((times,))
        y2 = Binomial(support, theta2).sample((times,))

        y = torch.where(z == 1, y1, y2)

        return y

    @override
    def propose_parameters(self, n_params: int = 100):
        theta2 = torch.empty((n_params,)).uniform_()
        theta1 = theta2 + (1 - theta2) * torch.rand((n_params,))
        rate = torch.empty((n_params,)).uniform_()
        return theta1, theta2, rate
