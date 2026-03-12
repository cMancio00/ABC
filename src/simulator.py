from torch import Tensor
from torch.distributions import Binomial, Bernoulli, MixtureSameFamily, Categorical
import torch

import seaborn as sns
import matplotlib.pyplot as plt


class GenerativeProcess:
    def __init__(self, theta1: float, theta2: float, rate: float, support: int = 4):
        """
        A generative process for a mixture of two binomial distributions.

        The process follows the model:
        .. math::

            \\lambda \\cdot Binomial(x_i | \\theta_1, N = 4) + (1 - \\lambda) \\cdot Binomial(x_i | \\theta_2, N = 4)

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

    def generate(self, times: int = 100):
        z = Bernoulli(self.rate).sample((times,))
        y1 = Binomial(self.support, torch.tensor(self.theta1)).sample((times,))
        y2 = Binomial(self.support, torch.tensor(self.theta2)).sample((times,))

        y = torch.where(z == 1, y1, y2)

        return y


x, p = GenerativeProcess(0.6,0.2, .7).density
print(p.sum())

plt.figure(figsize=(12,7))

sns.barplot(
    x=x.numpy(),
    y=p.numpy(),
    color="royalblue",
    edgecolor="black",
    label=r"$\lambda Bin(4,\theta_1) + (1-\lambda)Bin(4,\theta_2)$"
)

plt.xlabel(r"$x$")
plt.ylabel(r"$P(X=x)$")

plt.legend()
plt.show()
