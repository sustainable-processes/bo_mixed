from pathlib import Path

import matplotlib.pyplot as plt
import typer
from summit import *

import wandb
from benchmark import MixedBenchmark
from optimize import MOBO
from utils import WandbRunner


def main(
    repeats: int = 1,
    max_iterations: int = 20,
    num_initial_experiments: int = 10,
    # noise_level: float = 0.0,
    save_dir: str = "results",
    show_plot: bool = False,
    wandb_tracking: bool = True,
    wandb_project: str = "bo_mixed",
    wandb_entity: str = "ceb-sre",
    wandb_artifact_name: str = "mixed_benchmark",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Setup experiment
    exp = MixedBenchmark()

    # Runner class
    if wandb_tracking:
        runner_cls = WandbRunner
    else:
        runner_cls = Runner

    for i in range(repeats):
        # Reset experiment
        exp.reset()

        # Setup optimization
        strategy = MOBO(exp.domain)
        r = runner_cls(
            strategy=strategy,
            experiment=exp,
            max_iterations=max_iterations,
            num_initial_experiments=num_initial_experiments,
        )

        # Setup wandb
        wandb_run = None
        if wandb_tracking:
            config = r.to_dict()
            del config["experiment"]["data"]
            wandb_run = wandb.init(
                project=wandb_project, entity=wandb_entity, config=config
            )

        # Run optimization
        r.run(skip_wandb_initialization=True)

        # Plot results
        fig, ax = exp.pareto_plot(colorbar=True)
        if show_plot:
            plt.show()
        fig.savefig(save_dir / f"pareto_plot_repeat_{i}.png")
        if wandb_tracking:
            wandb.log({"pareto_plot": wandb.Image(fig)})

        # Save results
        r.save(save_dir / f"repeat_{i}.json")
        if wandb_tracking:
            artifact = wandb.Artifact(wandb_artifact_name, type="optimization_result")
            artifact.add_file(save_dir / f"repeat_{i}.json")
            # artifact.add_file(output_path / f"repeat_{i}_model.pth")
            wandb_run.log_artifact(artifact)
            wandb.finish()


if __name__ == "__main__":
    typer.run(main)
