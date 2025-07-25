"""
ElpdPairwise compares the fit of models using expected log pointwise posterior density (ELPD).
See Vehtari, Gelman, and Gabry (2017) for more details.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from . import logger

PARETO_WARNING_FILTER = {
    "action": "ignore",
    "message": "Estimated shape parameter of Pareto distribution is greater than 0.67.*",
    "category": UserWarning,
}


class ElpdPairwise:
    """
    Calculates the expected log pointwise posterior density (ELPD) for each posterior sample
    """

    def __init__(self, log_lik: Callable[[np.ndarray], np.ndarray]):
        """
        log_lik: (n_draws, n_dim) -> (n_draws, n_data_point)
        """

        self.log_lik = log_lik
        self.models: dict[str, xr.DataArray] = {}
        self.az_datasets: dict[str, az.InferenceData] = {}
        self._log_lik_generator = {}

    def add_model_post(
        self,
        post_samples: np.ndarray | xr.DataArray,
        name: Optional[str] = None,
        log_lik: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Add posterior samples to the comparison.
        post_samples: (n_draws, n_param_dim)
        """

        name = name or str(len(self.models))

        while name in self.models:
            logger.warning("Model name `%s` already exists. Adding underscore.", name)
            name += "_"

        if post_samples.ndim != 2:
            raise ValueError(
                f"Posterior samples must have 2 dimensions, but got {post_samples.ndim}."
            )

        if isinstance(post_samples, np.ndarray):
            n_draws, n_param_dim = post_samples.shape
            post_samples = xr.DataArray(
                post_samples,
                dims=("draw", "param"),
                coords={"draw": range(n_draws), "param": range(n_param_dim)},
            )
        self.models[name] = post_samples

        if log_lik is not None:
            self._log_lik_generator[name] = log_lik

    def pop_model(self, name: str):
        """
        Remove a model from the comparison.
        """
        self.models.pop(name)
        self.az_datasets.pop(name, None)
        self._log_lik_generator.pop(name, None)

    def evaluate_log_lik(
        self,
        model_name: Optional[str] = None,
        reevaluate: bool = False,
    ):
        """
        Evaluate the log likelihood and LOO for a given model.
        """

        if not model_name:
            for name in self.models:
                self.evaluate_log_lik(name, reevaluate)
            return

        if model_name not in self.models:
            raise ValueError(f"Model `{model_name}` not found.")

        if (model_name not in self.az_datasets) or reevaluate:

            log_lik: xr.DataArray = xr.apply_ufunc(
                self._log_lik_generator.get(model_name, self.log_lik),
                self.models[model_name],
                input_core_dims=[("param",)],
                output_core_dims=[("data_point",)],
                vectorize=True,
                output_dtypes=[float],
            )

            log_lik = (
                log_lik.assign_coords({"data_point": range(log_lik.data_point.size)})
                .expand_dims({"chain": [0]})
                .to_dataset(name="obs")
            )

            az_ds: az.InferenceData = az.convert_to_inference_data(
                log_lik, group="log_likelihood"
            )
            az_ds.add_groups(
                posterior=self.models[model_name].expand_dims({"chain": [0]})
            )

            self.az_datasets[model_name] = az_ds

    def compare_elpd(
        self,
        model_names: Optional[Sequence[str]] = None,
        reevaluate: bool = False,
        suppress_pareto_warning: bool = False,
    ) -> pd.DataFrame:
        """
        Compare the models using ELPD.
        """
        if not self.models:
            raise ValueError("No models to compare.")

        if model_names is None:
            self.evaluate_log_lik(model_names, reevaluate)
            return _az_compare(
                self.az_datasets,
                ic="loo",
                suppress_pareto_warning=suppress_pareto_warning,
            )

        for model_name in model_names:
            self.evaluate_log_lik(model_name, reevaluate)

        return _az_compare(
            {k: v for k, v in self.az_datasets.items() if k in model_names},
            ic="loo",
            suppress_pareto_warning=suppress_pareto_warning,
        )


def _az_compare(
    compare_dict: dict[str, az.InferenceData],
    ic: str = "loo",
    suppress_pareto_warning: bool = False,
) -> pd.DataFrame:
    """
    A thin wrapper around arviz.compare to handle the Pareto warning.
    """

    if not suppress_pareto_warning:
        return az.compare(compare_dict, ic=ic)

    with warnings.catch_warnings():
        warnings.filterwarnings(**PARETO_WARNING_FILTER)
        return az.compare(compare_dict, ic=ic)
