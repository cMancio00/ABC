import click
from sampler.base_abc import BaseABC
from simulator import GenerativeProcess, BinomialMixtureSimulator
from rich.console import Console
from rich.table import Table

@click.command()
@click.option(
    "-t1",
    "--theta1",
    default=0.5,
    type=float,
    help="Parameter of the first Binomial from the Mixture",
)
@click.option(
    "-t2",
    "--theta2",
    default=0.3,
    type=float,
    help="Parameter of the second Binomial from the Mixture",
)
@click.option(
    "-r",
    "--rate",
    default=0.6,
    type=float,
    help="Lambda of the Bernoulli that gives weights to the Mixture",
)
@click.option("--samples", default=10**2, type=int, help="Number of observed samples")
@click.option(
    "-e", "--epsilon", default=1e-1, type=float, help="Tolerance of acceptance"
)
@click.option(
    "-i",
    "--iterations",
    default=10**3,
    type=int,
    help="Number of iterations, corresponds to the number of proposed parameters from the prior",
)
def compute(theta1: float, theta2: float, rate: float, samples: int, epsilon: float, iterations: int):

    process = GenerativeProcess(theta1, theta2, rate)
    observations = process.mixture.sample((samples,))

    simulator = BinomialMixtureSimulator()

    algo = BaseABC(
        observations=observations,
        simulator=simulator,
        summary_statistic="frequency",
        threshold=epsilon,
    )

    accepted = algo.compute(iterations)
    # print(accepted)

    console = Console()

    table = Table(title="ABC Results")

    table.add_column("Parameter", justify="center", style="cyan", no_wrap=True)
    table.add_column("True Value", justify="center", style="green")
    table.add_column(f"Accepted Value(s): {(len(accepted)/iterations * 100):.2f}%", justify="center", style="magenta")

    if len(accepted) == 0:
        accepted = [(None, None, None)]

    table.add_row("theta1", f"{theta1:.3f}",
                  ", ".join(f"{a[0]:.3f}" for a in accepted if a[0] is not None))
    table.add_row("theta2", f"{theta2:.3f}",
                  ", ".join(f"{a[1]:.3f}" for a in accepted if a[1] is not None))
    table.add_row("rate", f"{rate:.3f}", ", ".join(f"{a[2]:.3f}" for a in accepted if a[2] is not None))

    console.print(table)

