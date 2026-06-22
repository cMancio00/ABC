from typing import Literal

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from sampler.base_abc import BaseABC
from simulator import Simulator
from utils.statistics import calculate_frequency
from torch.distributions import Beta, Uniform


class ImportanceSampling(BaseABC):
    r"""
    Importance Sampling will weight each proposed parameters instead of only accepting some of them.

    Given a prior :math:`p` and a proposal :math:`g` for a parameter :math:`\theta`, to get the unnormalized weight :math:`\tilde{w}_i` of a specific :math:`\theta_i \sim p`,
    we calculate the following ratio:

    .. math::

        \frac{p(\theta_i)}{g(\theta_i)}

    To get the normalized weight :math:`w_i` we simply calculate all the :math:`N` unnormalized weights and the total sum is the normalizing constant

    .. math::

        w_i = \frac{\tilde{w}_i}{\sum^N_{i=1 } w_i}

    The parameters that would be discarded in the previous algorithm are now weighted and used to calculate the expectation.
    """

    def __init__(
        self,
        observations: Tensor,
        simulator: Simulator,
        summary_statistic: Literal["frequency"],
        threshold: float = 0.1,
    ):
        super().__init__(observations, simulator, summary_statistic, threshold)
        self.theta_proposal = Beta(2, 2)
        self.rate_proposal = Beta(4, 3)

    def plot_proposals(self):
        x = torch.linspace(0, 1, 1000)
        theta = torch.exp(self.theta_proposal.log_prob(x))
        rate = torch.exp(self.rate_proposal.log_prob(x))

        fig, ax = plt.subplots(2, figsize=(12, 10))
        ax[0].plot(x.numpy(), theta.numpy())
        ax[0].set_title(
            f"Theta Proposal: Beta({self.theta_proposal.concentration1},{self.theta_proposal.concentration0})"
        )

        ax[1].plot(x.numpy(), rate.numpy())
        ax[1].set_title(
            f"Rate Proposal: Beta({self.rate_proposal.concentration1},{self.rate_proposal.concentration0})"
        )

        plt.show()

    @torch.no_grad()
    def compute(self, n_samples: int = 1000) -> tuple[Tensor, Tensor, Tensor]:

        samples_theta1 = self.theta_proposal.sample((n_samples,))
        samples_theta2 = self.theta_proposal.sample((n_samples,)) * samples_theta1
        samples_rate = self.rate_proposal.sample((n_samples,))

        importance_weights = []
        for _theta1, _theta2, _rate in zip(
            samples_theta1, samples_theta2, samples_rate
        ):
            sim = self.simulator.generate(
                theta1=_theta1,
                theta2=_theta2,
                rate=_rate,
                times=self.observations.size(0),
            )

            obs_v, obs_p = calculate_frequency(self.observations, min_length=0)
            _, sim_p = calculate_frequency(sim, min_length=obs_v.size(0))

            prior_weights = (
                Uniform(0, 1).log_prob(_rate)
                + Uniform(0, 1).log_prob(_theta1)
                + Uniform(0, 1).log_prob(_theta2)
            )
            proposal_weights = (
                self.rate_proposal.log_prob(_rate)
                * self.theta_proposal.log_prob(_theta1)
                * self.theta_proposal.log_prob(_theta2)
            )

            weight = (prior_weights - proposal_weights).exp()

            distance = (obs_p - sim_p).abs().sum()
            kernel = torch.exp(-distance / self.threshold)
            importance_weights.append(kernel * weight)

        normalized_weights = torch.stack(importance_weights)
        normalized_weights = normalized_weights / normalized_weights.sum()
        normalized_weights.size()

        # is_rate = (samples_rate * normalized_weights)
        # is_theta1 = (samples_theta1 * normalized_weights)
        # is_theta2 = (samples_theta2 * normalized_weights)

        return samples_theta1, samples_theta2, samples_rate, normalized_weights
