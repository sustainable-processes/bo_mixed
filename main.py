from pathlib import Path

import matplotlib.pyplot as plt
import typer
from summit import *

from benchmark import MixedBenchmark
from optimize import MOBO


def main(
    repeats: int = 1,
    max_iterations: int = 20,
    num_initial_experiments: int = 10,
    save_dir: str = "results",
    show_plot: bool = False,
):
    # Setup experiment
    exp = MixedBenchmark()

    for i in range(repeats):
        # Reset experiment
        exp.reset()

        # Run optimization
        strategy = MOBO(exp.domain)
        r = Runner(
            strategy=strategy,
            experiment=exp,
            max_iterations=max_iterations,
            num_initial_experiments=num_initial_experiments,
        )
        r.run()

        # Plot results
        fig, ax = exp.pareto_plot(colorbar=True)
        if show_plot:
            plt.show()
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"pareto_plot_repeat_{i}.png")


if __name__ == "__main__":
    typer.run(main)
