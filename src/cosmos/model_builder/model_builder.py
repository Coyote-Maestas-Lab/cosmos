"""
Generate samples and summary for all models
"""

import logging
import os
import pickle
from collections.abc import Callable, Sequence
from functools import partial
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from cosmos.adapt_grid.grid import Grid, GridMargin, MarginSummary
from cosmos.likelihood.double import ModelFull, ModelSkeleton
from cosmos.model_comparator.elpd_pairwise import ElpdPairwise
from cosmos.prior_factory import PriorFactory

from .model_loglik import gen_model_log_likelihood

HALF_INTERVAL = 3  # Half of the initial interval length of the grid
ADAPT_BOUNDARY_THRES = -5  # See Grid.adapt_boundaries
ADAPT_SPLIT_THRES = -7  # See Grid.adapt_split
N_SAMPLES = 1000  # Default number of samples to generate


class ModelBuilder:
    """
    DMS 2 model sample generation class.
    """

    _prior: PriorFactory

    def __init__(self, prior: PriorFactory, data_path: str):

        if prior.prior is None:
            raise ValueError("Prior is not generated yet.")
        self._prior = prior

        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)

    def to_pickle(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @property
    def data(self) -> pd.DataFrame:
        return self._prior.data

    @property
    def prior(self):
        return self._prior.prior

    ############################ Run Cosmos  ############################
    def run_cosmos(
        self,
        group_new_idx: int,
        no_s_hat: bool = True,
        ref_indices: Optional[int] = None,
        suppress_pareto_warning: bool = True,
    ) -> None:
        """
        Run Cosmos for one specific group.
        """

        # initialize model
        model_full, model_skeleton = self.init_model(group_new_idx, no_s_hat)

        # generate samples for all models and summarize
        dict_summary, dict_samples = self.gen_samples_for_all_models(
            group_new_idx, model_full, model_skeleton, no_s_hat
        )

        # model comparison
        compare_res = self.compare_models(
            dict_samples, model_full, model_skeleton, no_s_hat, suppress_pareto_warning
        )

        # save summary and comparison
        with open(f"{self.data_path}/group_new_{group_new_idx}_summary.pkl", "wb") as f:
            pickle.dump(dict_summary, f)
        compare_res.to_pickle(
            f"{self.data_path}/group_new_{group_new_idx}_comparison.pkl"
        )

        # save sample of best model
        # data_path, f"sample_group_new_{group_new_idx}", "samples", f"model_{model_idx}.pkl"
        top_model = compare_res[compare_res["rank"] == 0].index[0]
        top_model_idx = int(top_model.split("_")[-1])
        samples_top_model = ModelBuilder.save_sample_selected_model(
            group_new_idx, top_model_idx, dict_samples, self.data_path
        )

        # cross comparison: reference group likelihood on top model
        # data_path, f"sample_group_new_{group_new_idx}", "az_datasets",
        # f"ref_{ref_index}_model_{model_index}.nc"
        if ref_indices is not None:
            for ref_index in ref_indices:
                model_full_ref, model_skeleton_ref = self.init_model(
                    ref_index, no_s_hat
                )
                ModelBuilder.cross_comparison_ref_loglik(
                    ref_index,
                    model_full_ref,
                    model_skeleton_ref,
                    group_new_idx,
                    top_model_idx,
                    samples_top_model,
                    self.data_path,
                )

    def init_model(
        self, group_new_idx: int, no_s_hat: bool = True
    ) -> tuple[ModelFull, Optional[ModelSkeleton]]:
        sub_df = self.data[self.data["group_new"] == group_new_idx]

        beta_m_hat = sub_df["beta_hat_1"].values
        beta_y_hat = sub_df["beta_hat_2"].values

        if no_s_hat:
            s2_m = np.zeros_like(sub_df["se_hat_1"].values)
            s2_y = np.zeros_like(sub_df["se_hat_2"].values)
        else:
            s2_m = np.square(sub_df["se_hat_1"].values)
            s2_y = np.square(sub_df["se_hat_2"].values)

        sigma2_m = (
            self.prior["sigma2_m"] if no_s_hat else self.prior["sigma2_m_adjusted"]
        )

        model_full = ModelFull(
            mu_m=self.prior["mu_m"],
            mu_theta=self.prior["mu_theta"],
            sigma2_m=sigma2_m,
            sigma2_theta=self.prior["sigma2_theta"],
            pi=self.prior["pi"][group_new_idx],
            c_gamma_hat=self.prior["c_gamma_hat"],
            c_tau_hat=self.prior["c_tau_hat"],
            beta_m_hat=beta_m_hat,
            beta_y_hat=beta_y_hat,
            s2_m=s2_m,
            s2_y=s2_y,
        )

        if no_s_hat:
            return model_full, None

        model_skeleton = ModelSkeleton(
            mu_m=self.prior["mu_m"],
            mu_theta=self.prior["mu_theta"],
            sigma2_m=sigma2_m,
            sigma2_theta=self.prior["sigma2_theta"],
            pi=self.prior["pi"][group_new_idx],
            c_gamma_hat=self.prior["c_gamma_hat"],
            c_tau_hat=self.prior["c_tau_hat"],
            beta_m_hat=beta_m_hat,
            beta_y_hat=beta_y_hat,
            s2_m=s2_m,
            s2_y=s2_y,
        )
        return model_full, model_skeleton

    @staticmethod
    def linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # linear regression of beta_hat_2 on beta_hat_1
        reg = LinearRegression().fit(X, y)
        # return slope and intercept
        return reg.coef_[0], reg.intercept_

    def generate_boundary(self, group_new_idx: int) -> tuple[float, float]:
        sub_df = self.data[self.data["group_new"] == group_new_idx]
        beta_m_hat = sub_df["beta_hat_1"].values
        beta_y_hat = sub_df["beta_hat_2"].values

        boundary_gamma, boundary_tau = ModelBuilder.linear_regression(
            beta_m_hat.reshape(-1, 1), beta_y_hat
        )
        # print(f"boundary_gamma: {boundary_gamma}, boundary_tau: {boundary_tau}")
        return boundary_gamma, boundary_tau

    @staticmethod
    def _generate_model_sample_impl(
        param_names: Sequence[str],
        log_lik_all: Callable,
        center_gamma: float,
        center_tau: float,
        c_gamma_hat: float,
        c_tau_hat: float,
        n: int = 1000,
        quantiles: Sequence[float] = (),
    ) -> tuple[Optional[np.ndarray], Optional[dict[str, MarginSummary]]]:
        """
        Generate samples for a given model and given parameters
        """

        def gaussian_pdf(mu, sigma):
            return partial(norm.pdf, loc=mu, scale=sigma)

        axes_lst = []
        for name, variance, center in zip(
            ["gamma", "tau"],
            [c_gamma_hat, c_tau_hat],
            [center_gamma, center_tau],
        ):
            if name in param_names:
                # Initial grid: center +- HALF_INTERVAL
                # with a resolution of 20 * HALF_INTERVAL
                axes_lst.append(
                    GridMargin(
                        -HALF_INTERVAL + center,
                        HALF_INTERVAL + center,
                        int(20 * HALF_INTERVAL),
                        gaussian_pdf(0, variance),
                        name=name,
                    )
                )

        if not axes_lst:
            return None, None

        grid = Grid(axes_lst, log_lik_all)
        grid.adapt_boundaries(ADAPT_BOUNDARY_THRES)
        grid.adapt_split(ADAPT_SPLIT_THRES, 2)  # Bisect

        summary: dict[str, MarginSummary] = grid.marginal_posterior_summary(
            quantiles=quantiles
        )
        if any(np.isnan(param.mean) for param in summary.values()):
            # Model failed - posterior is 0 everywhere
            return np.full((n, len(param_names)), np.nan), summary
        return grid.sample(n), summary

    def gen_samples_for_all_models(
        self,
        group_new_idx: int,
        model_full: ModelFull,
        model_skeleton: Optional[ModelSkeleton],
        no_s_hat: bool = True,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    ) -> tuple[dict[str, dict[str, MarginSummary]], dict[str, np.ndarray]]:
        """
        Generate samples of all six/four models for one specific group.
        """
        boundary_gamma, boundary_tau = self.generate_boundary(group_new_idx)
        c_gamma_hat = self.prior["c_gamma_hat"]
        c_tau_hat = self.prior["c_tau_hat"]

        samples_6, summary_6 = ModelBuilder._generate_model_sample_impl(
            ["gamma", "tau"],
            gen_model_log_likelihood(has_gamma=True, has_tau=True, model=model_full),
            center_gamma=boundary_gamma,
            center_tau=boundary_tau,
            c_gamma_hat=c_gamma_hat,
            c_tau_hat=c_tau_hat,
            n=N_SAMPLES,
            quantiles=quantiles,
        )

        samples_5, summary_5 = ModelBuilder._generate_model_sample_impl(
            ["gamma"],
            gen_model_log_likelihood(has_gamma=True, has_tau=False, model=model_full),
            center_gamma=boundary_gamma,
            center_tau=boundary_tau,
            c_gamma_hat=c_gamma_hat,
            c_tau_hat=c_tau_hat,
            n=N_SAMPLES,
            quantiles=quantiles,
        )
        samples_5 = np.concatenate([samples_5, np.zeros_like(samples_5)], axis=1)

        samples_4, summary_4 = ModelBuilder._generate_model_sample_impl(
            ["tau"],
            gen_model_log_likelihood(has_gamma=False, has_tau=True, model=model_full),
            center_gamma=boundary_gamma,
            center_tau=boundary_tau,
            c_gamma_hat=c_gamma_hat,
            c_tau_hat=c_tau_hat,
            n=N_SAMPLES,
            quantiles=quantiles,
        )
        samples_4 = np.concatenate([np.zeros_like(samples_4), samples_4], axis=1)

        samples_3 = np.zeros_like(samples_6)
        summary_3 = {}

        dict_summary: dict[str, dict[str, MarginSummary]] = {
            "model_6": summary_6,
            "model_5": summary_5,
            "model_4": summary_4,
            "model_3": summary_3,
        }
        dict_samples: dict[str, np.ndarray] = {
            "model_6": samples_6,
            "model_5": samples_5,
            "model_4": samples_4,
            "model_3": samples_3,
        }

        if no_s_hat:
            return dict_summary, dict_samples

        samples_2, summary_2 = ModelBuilder._generate_model_sample_impl(
            ["tau"],
            gen_model_log_likelihood(
                has_gamma=False, has_tau=True, model=model_skeleton
            ),
            center_gamma=boundary_gamma,
            center_tau=boundary_tau,
            c_gamma_hat=c_gamma_hat,
            c_tau_hat=c_tau_hat,
            n=N_SAMPLES,
            quantiles=quantiles,
        )
        if any(np.isnan(param.mean) for param in summary_2.values()):
            # Model failed - posterior is 0 everywhere
            samples_2 = np.full_like(samples_6, np.nan)
        else:
            samples_2 = np.concatenate([np.zeros_like(samples_2), samples_2], axis=1)

        samples_1 = np.zeros_like(samples_6)
        summary_1 = {}

        dict_summary["model_2"] = summary_2
        dict_summary["model_1"] = summary_1

        dict_samples["model_2"] = samples_2
        dict_samples["model_1"] = samples_1

        return dict_summary, dict_samples

    @staticmethod
    def compare_models(
        dict_samples: dict[str, Optional[np.ndarray]],
        model_full: ModelFull,
        model_skeleton: Optional[ModelSkeleton],
        no_s_hat: bool = True,
        suppress_pareto_warning: bool = False,
    ) -> pd.DataFrame:

        def full_model_loglik_individual_array(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_full.log_lik_individual(gamma, tau)

        def skeleton_model_loglik_individual_array(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_skeleton.log_lik_individual(gamma, tau)

        comparator = ElpdPairwise(full_model_loglik_individual_array)
        for idx in [6, 5, 4, 3]:
            comparator.add_model_post(dict_samples[f"model_{idx}"], f"model_{idx}")

        if not no_s_hat:
            for idx in [2, 1]:
                comparator.add_model_post(
                    dict_samples[f"model_{idx}"],
                    f"model_{idx}",
                    skeleton_model_loglik_individual_array,
                )

        compare_res = comparator.compare_elpd(
            suppress_pareto_warning=suppress_pareto_warning
        )
        return compare_res

    @staticmethod
    def save_sample_selected_model(
        group_new_idx: int,
        model_idx: int,
        dict_samples: dict[str, np.ndarray],
        data_path: str,
    ) -> np.ndarray:
        """
        Save samples for the selected model.
        """

        output_dir = os.path.join(
            data_path, f"sample_group_new_{group_new_idx}", "samples"
        )
        os.makedirs(output_dir, exist_ok=True)

        samples = dict_samples[f"model_{model_idx}"]

        logging.info(
            "Saving samples for group %s model %s...", group_new_idx, model_idx
        )
        file_name = f"model_{model_idx}.pkl"
        with open(os.path.join(output_dir, file_name), "wb") as f:
            pickle.dump(samples, f)

        return samples

    def run_cosmos_ref_comparison(
        self,
        ref_indices: list[int],
        data_path: Optional[str] = None,
        no_s_hat: bool = True,
    ) -> None:
        """
        Run Cosmos for reference group comparison.
        Need the results of run_cosmos on all positions.
        """
        if data_path is None:
            data_path = self.data_path

        _, combined_data_comparison = ModelBuilder.summary_cosmos(data_path)
        for ref_index in ref_indices:
            model_full_ref, model_skeleton_ref = self.init_model(ref_index, no_s_hat)
            ModelBuilder.cross_comparison_ref_compare(
                ref_index,
                model_full_ref,
                model_skeleton_ref,
                combined_data_comparison,
                data_path,
            )

    ############################ Cross Comparison ############################
    @staticmethod
    def cross_comparison_ref_loglik(
        ref_index: int,
        model_full_ref: ModelFull,
        model_skeleton_ref: ModelSkeleton,
        group_new_idx: int,
        model_index: int,
        samples_model: np.ndarray,
        data_path: str,
    ) -> None:
        """
        Compute log likelihood of reference group on the top model of the group index.
        """

        def log_lik_full_ref(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_full_ref.log_lik_individual(gamma, tau)

        def log_lik_skeleton_ref(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_skeleton_ref.log_lik_individual(gamma, tau)

        comparator = ElpdPairwise(None)
        match model_index:
            case 1 | 2:
                comparator.add_model_post(
                    post_samples=samples_model,
                    name=f"group_new_{group_new_idx}",
                    log_lik=log_lik_skeleton_ref,
                )
            case 3 | 4 | 5 | 6:
                comparator.add_model_post(
                    post_samples=samples_model,
                    name=f"group_new_{group_new_idx}",
                    log_lik=log_lik_full_ref,
                )
            case _:
                raise ValueError(f"Invalid model index {model_index}")

        az_dataset_dir = os.path.join(
            data_path, f"sample_group_new_{group_new_idx}", "az_datasets"
        )
        os.makedirs(az_dataset_dir, exist_ok=True)
        az_dataset_path = os.path.join(
            az_dataset_dir, f"ref_{ref_index}_model_{model_index}.nc"
        )

        logging.info("Evaluating log likelihood for group %s...", group_new_idx)
        comparator.evaluate_log_lik(model_name=f"group_new_{group_new_idx}")
        logging.info(
            "Finished evaluating log likelihood for group %s, saving...",
            group_new_idx,
        )
        comparator.az_datasets[f"group_new_{group_new_idx}"].to_netcdf(az_dataset_path)
        logging.info("Saved az dataset for group %s", group_new_idx)

    @staticmethod
    def cross_comparison_ref_compare(
        ref_index: int,
        model_full_ref: ModelFull,
        model_skeleton_ref: ModelSkeleton,
        combined_data_comparison: dict[int, pd.DataFrame],
        data_path: str,
        suppress_pareto_warning: bool = False,
    ) -> None:

        def log_lik_full_ref(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_full_ref.log_lik_individual(gamma, tau)

        def log_lik_skeleton_ref(params: np.ndarray) -> np.ndarray:
            gamma, tau = params
            return model_skeleton_ref.log_lik_individual(gamma, tau)

        comparator = ElpdPairwise(log_lik_full_ref)
        best_models: dict[int, str] = {
            k: v.index[0] for k, v in combined_data_comparison.items()
        }  # group index -> best model name
        for group_new_idx, top_model_name in best_models.items():
            group_new_name = f"group_new_{group_new_idx}"
            top_model_index = int(top_model_name.split("_")[-1])

            # load samples
            sample_path = os.path.join(
                data_path,
                f"sample_group_new_{group_new_idx}",
                "samples",
                f"model_{top_model_index}.pkl",
            )
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as _handle:
                    _posterior_sample: np.ndarray = pickle.load(_handle)
            else:
                raise FileNotFoundError(
                    f"Posterior sample for group new {group_new_name} "
                    "on reference group {ref_index} not found.\n"
                )

            # add posterior to comparator
            match top_model_index:
                case 1 | 2:
                    comparator.add_model_post(
                        post_samples=_posterior_sample,
                        name=f"group_new_{group_new_idx}",
                        log_lik=log_lik_skeleton_ref,
                    )
                case 3 | 4 | 5 | 6:
                    comparator.add_model_post(
                        post_samples=_posterior_sample,
                        name=f"group_new_{group_new_idx}",
                        log_lik=log_lik_full_ref,
                    )
                case _:
                    raise ValueError(f"Invalid model index {top_model_index}")

            # load az dataset
            az_dataset_path = os.path.join(
                data_path,
                f"sample_group_new_{group_new_idx}",
                "az_datasets",
                f"ref_{ref_index}_model_{top_model_index}.nc",
            )
            if os.path.exists(az_dataset_path):
                az_ds = az.InferenceData.from_netcdf(az_dataset_path)
                comparator.az_datasets[group_new_name] = az_ds
                logging.info("Loaded existing az dataset for group %s", group_new_name)
            else:
                logging.error(
                    "Az sample for group new %s on reference group %s not found.\n",
                    group_new_name,
                    ref_index,
                )
                raise FileNotFoundError(
                    f"Az sample for group new {group_new_name} "
                    f"on reference group {ref_index} not found.\n"
                )

        logging.info("Comparing models using group %s data...", ref_index)
        compare_res = comparator.compare_elpd(
            suppress_pareto_warning=suppress_pareto_warning
        )

        res_dir = os.path.join(data_path, "comparison_results")
        os.makedirs(res_dir, exist_ok=True)
        compare_res.to_pickle(os.path.join(res_dir, f"ref_{ref_index}.pkl"))
        logging.info(
            "Saved comparison results to %s",
            os.path.join(res_dir, f"ref_{ref_index}.pkl"),
        )

    ############################ Summary Cosmos ############################

    @staticmethod
    def summary_cosmos(
        data_path: str,
    ) -> tuple[
        dict[int, dict[str, dict[str, MarginSummary]]],
        dict[int, pd.DataFrame],
    ]:
        """
        Summary the pkl results of Cosmos across position groups.
        """
        combined_data_summary: dict[int, dict[str, dict[str, MarginSummary]]] = {}
        combined_data_comparison: dict[int, pd.DataFrame] = {}

        # list all compairson files in data_path
        for pkl_file in os.listdir(data_path):
            if not pkl_file.startswith("group_new_"):
                continue
            parts = pkl_file.split("_")
            group_new_idx = parts[2]

            file_path = os.path.join(data_path, pkl_file)
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if pkl_file.endswith("_summary.pkl"):
                combined_data_summary[int(group_new_idx)] = data
            elif pkl_file.endswith("_comparison.pkl"):
                combined_data_comparison[int(group_new_idx)] = data

        # check the keys of combined_data_comparison and combined_data_summary is the same
        keys_comparison = set(combined_data_comparison)
        keys_summary = set(combined_data_summary)
        if keys_comparison != keys_summary:
            raise ValueError(
                "The keys of combined_data_comparison and combined_data_summary is not the same."
            )

        # check if the dict is empty
        if not combined_data_summary or not combined_data_comparison:
            raise ValueError(
                "Either the summary and comparision dictionary is empty."
                "Check if the provided data path is valid."
            )

        return combined_data_summary, combined_data_comparison
