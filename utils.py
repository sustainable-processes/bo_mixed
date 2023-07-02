import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pkg_resources
from summit import *
from summit.utils.multiobjective import hypervolume, pareto_efficient
from tqdm import trange
from wandb.apis.public import Run

import wandb
from wandb import Artifact


class WandbRunner(Runner):
    """Run a closed-loop strategy and experiment cycle with logging to Wandb



    Parameters
    ----------
    strategy : :class:`~summit.strategies.base.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : :class:`~summit.experiment.Experiment`
        The experiment or benchmark class to use for running experiments
    neptune_project : str
        The name of the Neptune project to log data to
    neptune_experiment_name : str
        A name for the neptune experiment
    netpune_description : str, optional
        A description of the neptune experiment
    files : list, optional
        A list of filenames to save to Neptune
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is 100.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments. Default is 1.
    f_tol : float, optional
        How much difference between successive best objective values will be tolerated before stopping.
        This is generally useful for nonglobal algorithms like Nelder-Mead. Default is None.
    max_same : int, optional
        The number of iterations where the objectives don't improve by more than f_tol. Default is max_iterations.
    max_restarts : int, optional
        Number of restarts if f_tol is violated. Default is 0.
    hypervolume_ref : array-like, optional
        The reference for the hypervolume calculation if it is a multiobjective problem.
        Should be an array of length the number of objectives. Default is at the origin.
    """

    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        wandb_entity: str = None,
        wandb_project: str = None,
        wandb_run_name: Optional[str] = None,
        wandb_notes: str = None,
        wandb_tags: List[str] = None,
        wandb_save_code: bool = True,
        wandb_artifact: str = None,
        hypervolume_ref=None,
        **kwargs,
    ):
        super().__init__(strategy, experiment, **kwargs)

        # Hypervolume reference for multiobjective experiments
        n_objs = len(self.experiment.domain.output_variables)
        self.ref = hypervolume_ref if hypervolume_ref is not None else n_objs * [0]

        # Check that Neptune-client is installed
        installed = {pkg.key for pkg in pkg_resources.working_set}
        if "wandb" not in installed:
            raise RuntimeError(
                "Wandb is not installed. Use pip install summit[experiments] to add extra dependencies."
            )

        # Set up Neptune variables
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_notes = wandb_notes
        self.wandb_tags = wandb_tags
        self.wandb_save_code = wandb_save_code
        self.wandb_artifact = wandb_artifact

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """Run the closed loop experiment cycle

        Parameters
        ----------
        prev_res: DataSet, optional
            Previous results to initialize the optimization
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to `~/.summit/runner`.
        callback : callable
            A callback to run on each iteration.
        """
        # Set parameters
        prev_res = None
        self.restarts = 0
        n_objs = len(self.experiment.domain.output_variables)
        fbest_old = np.zeros(n_objs)
        fbest = np.zeros(n_objs)
        prev_res = kwargs.get("prev_res")
        suggest_kwargs = kwargs.get("suggest_kwargs", {})

        # Serialization
        save_freq = kwargs.get("save_freq")
        save_dir = kwargs.get("save_dir", str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get("save_at_end", True)
        if self.wandb_artifact:
            artifact = wandb.Artifact(self.wandb_artifact)
            artifact.add_dir(save_dir)

        # Callback
        callback = kwargs.get("callback")

        # Create wandb run
        skip_wandb_intialization = kwargs.get("skip_wandb_intialization", False)
        if not skip_wandb_intialization:
            wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.wandb_run_name,
                tags=self.wandb_tags,
                notes=self.wandb_notes,
                save_code=self.wandb_save_code,
            )
        else:
            wandb_run = None
        wandb_start_iteration = kwargs.get("wandb_start_iteration", 0)

        # Run optimization loop
        if kwargs.get("progress_bar", True):
            bar = trange(self.max_iterations)
        else:
            bar = range(self.max_iterations)
        wi = wandb_start_iteration
        i = 0
        for i in bar:
            # Get experiment suggestions
            if i == 0 and prev_res is None:
                k = self.n_init if self.n_init is not None else self.batch_size
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=k, **suggest_kwargs
                )
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res, **suggest_kwargs
                )

            prev_res = self.experiment.run_experiments(next_experiments)

            # Send parameter values to wandb
            for j in range(prev_res.shape[0]):
                log_dict = {}
                for v in self.experiment.domain.input_variables:
                    log_dict.update(
                        {
                            "iteration": wi + j,
                            v.name: prev_res.iloc[j][v.name].values[0],
                        }
                    )

                for v in self.experiment.domain.output_variables:
                    log_dict.update(
                        {
                            "iteration": wi + j,
                            v.name: prev_res.iloc[j][v.name].values[0],
                        }
                    )

                if j == len(prev_res) - 1:
                    for k, v in enumerate(self.experiment.domain.output_variables):
                        if i > 0:
                            fbest_old[k] = fbest[k]
                        if v.maximize:
                            fbest[k] = self.experiment.data[v.name].max()
                        elif not v.maximize:
                            fbest[k] = self.experiment.data[v.name].min()
                        log_dict.update(
                            {
                                "iteration": wi + j,
                                f"{v.name}_best": fbest[k],
                            }
                        )

                # Send hypervolume for multiobjective experiments
                if n_objs > 1:
                    output_names = [
                        v.name for v in self.experiment.domain.output_variables
                    ]
                    data = self.experiment.data[output_names].copy()
                    for v in self.experiment.domain.output_variables:
                        if v.maximize:
                            data[(v.name, "DATA")] = -1.0 * data[v.name]
                    y_pareto, _ = pareto_efficient(data.to_numpy(), maximize=False)
                    hv = hypervolume(y_pareto, self.ref)
                    log_dict.update({"hypervolume": hv})

                # Callback
                if callback is not None:
                    log_dict.update(callback(self, prev_res, i, j))

                # Actually log
                wandb.log(log_dict, step=wi + j, commit=True)
            wi += prev_res.shape[0]

            # Save state
            if save_freq is not None:
                file = save_dir / f"iteration_{i}.json"
                if i % save_freq == 0:
                    self.save(file)
                    wandb.log_artifact(artifact)
                if not save_dir:
                    os.remove(file)

            # Stop if no improvement
            compare = np.abs(fbest - fbest_old) > self.f_tol
            if all(compare) or i <= 1:
                nstop = 0
            else:
                nstop += 1

            if self.max_same is not None:
                if nstop >= self.max_same and self.restarts >= self.max_restarts:
                    self.logger.info(
                        f"{self.strategy.__class__.__name__} stopped after {i+1} iterations and {self.restarts} restarts."
                    )
                    break
                elif nstop >= self.max_same:
                    nstop = 0
                    prev_res = None
                    self.strategy.reset()
                    self.restarts += 1

        # Save at end
        if save_at_end:
            file = save_dir / f"iteration_{i}.json"
            self.save(file)
            if self.wandb_artifact:
                wandb.log_artifact(artifact)
            if not save_dir:
                os.remove(file)
        return wandb_run

    def to_dict(
        self,
    ):
        d = super().to_dict()
        d["runner"].update(
            dict(
                hypervolume_ref=self.ref,
                wandb_entity=self.wandb_entity,
                wandb_project=self.wandb_project,
                wandb_run_name=self.wandb_run_name,
                wandb_notes=self.wandb_notes,
                wandb_tags=self.wandb_tags,
                wandb_save_code=self.wandb_save_code,
                wandb_artifact=self.wandb_artifact,
            )
        )
        return d


def get_ds_from_json(filepath: str) -> DataSet:
    path = Path(filepath)
    with open(path, "r") as f:
        data = json.load(f)
    data = data["experiment"]["data"]
    return DataSet.from_dict(data)


def download_with_retries(artifact: Artifact, n_retries: int = 5):
    logger = logging.getLogger(__name__)
    for i in range(n_retries):
        try:
            return artifact.download()
        except ConnectionError as e:
            logger.error(e)
            logger.info(f"Retrying download of {artifact.name}")
            time.sleep(2**i)


def download_runs_wandb(
    api: wandb.Api,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "distill",
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> List[Run]:
    """Download runs from wandb


    Parameters
    ----------
    api : wandb.Api
        The wandb API object.
    wandb_entity : str, optional
        The wandb entity to search, by default "ceb-sre"
    wandb_project : str, optional
        The wandb project to search, by default "multitask"
    include_tags : Optional[List[str]], optional
        A list of tags that the run must have, by default None
    filter_tags : Optional[List[str]], optional
        A list of tags that the run must not have, by default None
    extra_filters : Optional[Dict[str, Any]], optional
        A dictionary of extra filters to apply to the wandb search, by default None

    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading runs from wandb")

    # Filters
    filters = {}
    tag_query = []
    if include_tags is not None and len(include_tags) > 0:
        for include_tag in include_tags:
            tag_query.append({"tags": {"$in": [include_tag]}})
        # filters["tags"] = {"$infilt": include_tags}
    if filter_tags is not None and len(filter_tags) > 0:
        tag_query += [{"tags": {"$nin": filter_tags}}]
    if len(tag_query) > 0:
        filters["$and"] = tag_query
    if only_finished_runs:
        filters["state"] = "finished"
    if extra_filters is not None:
        filters.update(extra_filters)

    # Get runs
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters=filters,
    )
    return runs


def get_wandb_run_dfs(
    api,
    wandb_entity: str,
    wandb_project: str,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    extra_filters: Optional[Dict] = None,
    num_iterations: Optional[int] = None,
    commit: Optional[str] = None,
    limit: Optional[int] = 20,
) -> List[DataSet]:
    """Get data from wandb"""
    runs = download_runs_wandb(
        api,
        wandb_entity,
        wandb_project,
        only_finished_runs=only_finished_runs,
        include_tags=include_tags,
        filter_tags=filter_tags,
        extra_filters=extra_filters,
    )

    if commit is not None:
        runs = [run for run in runs if run.commit == commit]

    dfs = []
    for i, run in enumerate(runs):
        if limit is not None and i >= limit:
            continue
        artifacts = run.logged_artifacts()
        path = None
        for artifact in artifacts:
            if artifact.type == "optimization_result":
                path = download_with_retries(artifact)
                break
        if path is not None:
            path = list(Path(path).glob("repeat_*.json"))[0]
            ds = get_ds_from_json(path)
            # ds["yld_best", "DATA"] = ds["yld"].astype(float).cummax()
            dfs.append(ds)

    # if columns is None:
    #     columns = ["yld_best"]
    # dfs = [run.history(x_axis="iteration", keys=columns) for run in tqdm(runs)]
    if num_iterations is not None:
        dfs = [
            df.iloc[:num_iterations, :] for df in dfs if df.shape[0] >= num_iterations
        ]
    if len(dfs) == 0:
        raise ValueError(f"No runs found")
    return dfs


def calculate_hypervolume(data, domain: Domain):
    hypervolumes = []
    output_names = [v.name for v in domain.output_variables]
    y = data.copy()
    reference_point = [
        -y[v.name].min() if v.maximize else y[v.name].max()
        for v in domain.output_variables
    ]
    # Hypervolume assumes a minimization problem
    for v in domain.output_variables:
        if v.maximize:
            y[v.name] *= -1

    for i in range(data.shape[0]):
        yi = y.iloc[:i]

        # Get pareto front
        pareto, indices = pareto_efficient(yi[output_names].to_numpy(), maximize=False)
        # Calculate hypervolume
        hv = hypervolume(pareto, reference_point)
        hypervolumes.append(hv)
    return hypervolumes
