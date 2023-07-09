import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement as qEHVI,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement as qNEHVI,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import FilterFeatures
from botorch.optim import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood
from summit import *

dtype = torch.double


class CategoricalqNEHVI(qNEHVI):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], torch.Tensor],
        X_baseline: torch.Tensor,
        domain: Domain,
        descriptor_tensors: Optional[Dict[str, torch.Tensor]] = None,
        sampler: Optional[MCSampler] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, ref_point, X_baseline, sampler, **kwargs)
        self._domain = domain
        self.skip = True if domain.num_categorical_variables() == 0 else False
        self.descriptor_tensors = descriptor_tensors

    def forward(self, X):
        if not self.skip:
            X = self.round_categorical(X, self._domain, self.descriptor_tensors)
        return super().forward(X)

    @staticmethod
    def round_categorical(
        X, domain: Domain, descriptor_tensors: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Round all categorical variables to a one-hot encoding"""
        num_experiments = X.shape[1]
        X = X.clone()

        for q in range(num_experiments):
            c = 0
            for v in domain.input_variables:
                if isinstance(v, CategoricalVariable) and v.ds is None:
                    n_levels = len(v.levels)
                    levels_selected = X[:, q, c : c + n_levels].argmax(axis=1)
                    X[:, q, c : c + n_levels] = 0
                    for j, l in zip(range(X.shape[0]), levels_selected):
                        X[j, q, int(c + l)] = 1

                    check = int(X[:, q, c : c + n_levels].sum()) == X.shape[0]
                    if not check:
                        raise ValueError(
                            (
                                f"Rounding to a one-hot encoding is not properly working. Please report this bug at "
                                f"https://github.com/sustainable-processes/summit/issues. Tensor: \n {X[:, :, c : c + n_levels]}"
                            )
                        )
                    c += n_levels
                # Choose the closest option by Euclidean distance
                elif isinstance(v, CategoricalVariable) and v.ds is not None:
                    # Get embeddings
                    num_descriptors = v.ds.shape[1]
                    embeddings = X[:, q, c : c + num_descriptors]
                    option_embeddings = descriptor_tensors[v.name]
                    # Calculated distances
                    distances = torch.cdist(embeddings, option_embeddings)
                    # Select the closest option
                    levels_selected = distances.argmin(axis=1)
                    # Update embeddings
                    new_embeddings = option_embeddings[levels_selected, :]
                    X[:, q, c : c + num_descriptors] = new_embeddings
                    c += num_descriptors
                else:
                    c += 1
        return X


class MOBO(Strategy):
    """Multiobjective Bayesian Optimisation using BOtorch


    Parameters
    ----------
    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.

    Examples
    --------

    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import NelderMead
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name="yld", description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> strategy = STBO(domain)
    >>> next_experiments  = strategy.suggest_experiments()
    >>> print(next_experiments)
    NAME temperature flowrate_a             strategy
    TYPE        DATA       DATA             METADATA
    0          0.500      0.500  Nelder-Mead Simplex
    1          0.625      0.500  Nelder-Mead Simplex
    2          0.500      0.625  Nelder-Mead Simplex

    """

    def __init__(
        self,
        domain: Domain,
        transform: Transform = None,
        input_groups: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        Strategy.__init__(self, domain, transform, **kwargs)
        if len(self.domain.output_variables) < 2:
            raise DomainError("MOBO only works with multi-objective problems")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Acqusition function and reference point
        self.static_reference_point = {}
        for v in self.domain.output_variables:
            self.static_reference_point[v.name] = (
                v.bounds[1] if v.maximize else -v.bounds[0]
            )

        # Input grouping
        self.input_groups = input_groups
        if self.input_groups:
            variable_names = [v.name for v in self.domain.input_variables]
            self.input_group_indices = {
                output_name: [variable_names.index(input) for input in g]
                for output_name, g in input_groups.items()
            }
        else:
            self.input_group_indices = None
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None and self.all_experiments is None:
            lhs = LHS(self.domain, transform=self.transform)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            conditions = lhs.suggest_experiments(k)
            for v in self.domain.input_variables:
                if isinstance(v, CategoricalVariable) and v.ds is not None:
                    indices = conditions[v.name].values
                    descriptors = v.ds.loc[indices]
                    descriptors.index = conditions.index
                    conditions = conditions.join(descriptors, how="inner")
                    var_descriptor_names = v.ds.data_columns
                    for var in var_descriptor_names:
                        descriptor = conditions[var].copy()
                        conditions = conditions.drop(columns=var, level=0)
                        conditions[var, "METADATA"] = descriptor
            return conditions
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = pd.concat([self.all_experiments, prev_res], axis=0)
        self.iterations += 1
        data = self.all_experiments

        # Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            data,
            min_max_scale_inputs=True,
            standardize_outputs=True,
            categorical_method="mixed",
        )

        # Make it always a maximization problem
        for v in self.domain.output_variables:
            if not v.maximize:
                output[v.name] = -output[v.name]

        # Convert to torch tensors
        X = torch.tensor(
            inputs.data_to_numpy().astype(float),
            device=self.device,
            dtype=dtype,
        )
        y = torch.tensor(
            output.data_to_numpy().astype(float),
            device=self.device,
            dtype=dtype,
        )

        # Fit independent GP models
        self.model = self.fit_model(X, y)

        # Optimize acquisition function
        reference_point = self._get_transformed_reference_point(output)
        results = self.optimize_acquisition_function(
            self.model,
            X=X,
            num_experiments=num_experiments,
            reference_point=reference_point,
            num_restarts=kwargs.get("num_restarts", 100),
            raw_samples=kwargs.get("raw_samples", 2000),
            # mc_samples=kwargs.get("mc_samples", 128),
        )

        # Convert result to datset
        result = DataSet(
            results.cpu().detach().numpy(),
            columns=inputs.data_columns,
        )

        # Untransform
        result = self.transform.un_transform(
            result,
            min_max_scale_inputs=True,
            standardized_outputs=True,
            categorical_method="mixed",
        )

        # Add metadata
        result[("strategy", "METADATA")] = "MOBO"
        return result

    def fit_model(self, X, y):
        models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, v in enumerate(self.domain.output_variables):
                transform = None
                if self.input_group_indices:
                    feature_indices = torch.tensor(
                        self.input_group_indices[v.name], device=self.device
                    )
                    transform = FilterFeatures(feature_indices=feature_indices)
                model = SingleTaskGP(X, y[:, [i]], input_transform=transform)
                models.append(model)
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll, max_retries=20)
        return model

    def optimize_acquisition_function(
        self,
        model,
        X: torch.Tensor,
        reference_point: torch.Tensor,
        num_experiments: int,
        num_restarts: int = 100,
        raw_samples: int = 2000,
        mc_samples: int = 128,
    ):
        # Sampler
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        # Normalize any descriptor dfs
        descriptors_tensors = {}
        for v in self.domain.input_variables:
            if isinstance(v, CategoricalVariable) and v.ds is not None:
                new_ds = v.ds.copy()
                for descriptor in new_ds.data_columns:
                    var_max = v.ds[descriptor].max()
                    var_min = v.ds[descriptor].min()
                    new_ds[descriptor, "DATA"] = (new_ds[descriptor] - var_min) / (
                        var_max - var_min
                    )
                descriptors_tensors[v.name] = torch.tensor(
                    new_ds.to_numpy(), device=self.device, dtype=dtype
                )

        # Acquisition function
        acq = CategoricalqNEHVI(
            model=model,
            ref_point=reference_point,
            domain=self.domain,
            sampler=sampler,
            X_baseline=X,
            prune_baseline=True,
            descriptor_tensors=descriptors_tensors,
        )

        # Optimize acquisition function
        results, _ = optimize_acqf(
            acq_function=acq,
            bounds=self._get_input_bounds(),
            num_restarts=num_restarts,
            q=num_experiments,
            raw_samples=raw_samples,
        )

        return results

    def _get_input_bounds(self):
        bounds = []
        for v in self.domain.input_variables:
            if isinstance(v, ContinuousVariable):
                # Because of min max scaling
                bounds += [[0, 1]]
            elif isinstance(v, CategoricalVariable) and v.ds is None:
                # Because of one-hot encoding
                bounds += [[0, 1] for _ in v.levels]
            elif isinstance(v, CategoricalVariable) and v.ds is not None:
                # Because of min-max scaling
                bounds += [[0, 1] for _ in v.ds.columns]
        return torch.tensor(np.array(bounds), dtype=dtype, device=self.device).T

    def _get_transformed_reference_point(self, output: DataSet):
        transformed_reference_point = []
        for v in self.domain.output_variables:
            # Get the worst point in each output
            # Should always be transformed to a maximization problem
            val = output.sort_values(v.name, ascending=True)[v.name].iloc[0]
            transformed_reference_point.append(val)
        return torch.tensor(
            np.array(transformed_reference_point), dtype=dtype, device=self.device
        )

    def reset(self):
        """Reset MTBO state"""
        self.all_experiments = None
        self.iterations = 0
        self.fbest = (
            float("inf") if self.domain.output_variables[0].maximize else -float("inf")
        )

    def to_dict(self, **strategy_params):
        strategy_params.update({"input_groups": self.input_groups})
        return super().to_dict(**strategy_params)
