from typing import Literal

import torch
from rich.progress import track
from torch import Tensor

from simulator import Simulator
from utils.statistics import calculate_frequency


class BaseABC:
    def __init__(
        self,
        observations: Tensor,
        simulator: Simulator,
        summary_statistic: Literal["frequency"],
        threshold: float,
    ):
        self.observations = observations
        self.simulator = simulator
        self.summary_statistic = summary_statistic
        self.threshold = threshold

    @torch.no_grad()
    def compute(self, n_iteration: int) -> list[tuple[float, float, float]]:
        accepted: list[tuple[float, float, float]] = []

        theta1s, theta2s, rates = self.simulator.propose_parameters(n_iteration)

        for theta1, theta2, rate in track(
            zip(theta1s, theta2s, rates),
            total=len(theta1s),
            description="Basic ABC Sampling",
        ):
            sim = self.simulator.generate(
                theta1=theta1, theta2=theta2, rate=rate, times=self.observations.size(0)
            )

            obs_v, obs_p = calculate_frequency(self.observations, min_length=0)
            _, sim_p = calculate_frequency(sim, min_length=obs_v.size(0))

            distance = (obs_p - sim_p).abs()

            if not distance.greater_equal(torch.tensor([self.threshold])).sum():
                accepted.append((theta1.item(), theta2.item(), rate.item()))

        return accepted
