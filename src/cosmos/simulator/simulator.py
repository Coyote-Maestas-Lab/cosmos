"""
A generatative model for Cosmos
"""

import copy
from collections.abc import Sequence
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from cosmos.dms_data import DMSData
from cosmos.model_builder import ModelBuilder
from cosmos.prior_factory import PriorFactory

from .sim_config import DEFAULT_CONFIG, Config


class Simulator:
    """
    Generates data for a given configuration, following the Cosmos model.
    """

    df_variant: pd.DataFrame
    df_position: pd.DataFrame

    data: DMSData
    prior: PriorFactory
    model: ModelBuilder

    def __init__(self, config: Optional[Config] = None):
        # Set up configurations
        self.config: Config = self.default_config() if config is None else config

    @staticmethod
    def default_config() -> Config:
        """
        Returns the default configuration for the simulator.
        """
        return copy.deepcopy(DEFAULT_CONFIG)

    ####### Simulation Functions #######

    def _simulate_x_class(self) -> pd.DataFrame:
        """
        For each position, generate position class from a multinomial distribution (labels: config.class_x.class_type)
        """
        # Read from config
        n_samples = self.config.simulation.n_position
        n_components = len(self.config.class_x.pi)
        weights = self.config.class_x.pi

        # sample component indices
        components = np.random.choice(n_components, size=n_samples, p=weights)
        position_class = [self.config.class_x.class_type[k] for k in components]

        # concat into a dataframe with two columns: components and gamma
        df_class = pd.DataFrame({"x_class": position_class})
        df_class["position"] = range(n_samples)

        return df_class

    @staticmethod
    def get_true_model(
        tau_group: int, gamma_group: int, x_class: Literal["null", "mixed"]
    ) -> str:
        """
        Get the model name based on tau_group, gamma_group, and x_class.
        """

        match tau_group, gamma_group, x_class:
            case 0, _, "null":
                model_idx = 1
            case 1, _, "null":
                model_idx = 2
            case 0, 0, "mixed":
                model_idx = 3
            case 1, 0, "mixed":
                model_idx = 4
            case 0, 1, "mixed":
                model_idx = 5
            case 1, 1, "mixed":
                model_idx = 6
            case _:
                raise ValueError(
                    f"Unrecognized combination of tau_group={tau_group}, gamma_group={gamma_group}, x_class={x_class}"
                )

        return f"model_{model_idx}"

    def _simulate_position(self) -> pd.DataFrame:
        """
        For each position, generate gamma, tau, and position class.

        Result columns: position, gamma_group, gamma, tau_group, tau, class
        """
        # For each position, generate gamma and tau from a gaussian mixture model
        df_gamma = self._simulate_gamma()
        df_tau = self._simulate_tau()

        # For each position, generate position class from a multinomial distribution
        df_class = self._simulate_x_class()
        df_position = pd.merge(df_gamma, df_tau, on="position")
        df_position = df_position[
            ["position", "gamma_group", "gamma", "tau_group", "tau"]
        ]
        df_position = pd.merge(df_position, df_class, on="position")

        df_position["model"] = df_position.apply(
            lambda row: self.get_true_model(
                tau_group=row["tau_group"],
                gamma_group=row["gamma_group"],
                x_class=row["x_class"],
            ),
            axis=1,
        )

        return df_position

    def _simulate_variant(self, df_position: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position * n_variant_per_position variants.
        For each variant, based on its position information, generate beta_x,
        theta, beta_y,
        beta_x_hat, beta_y_hat

        Result columns:
        position, mutation, variant (variant id)
        class_x, beta_x (from _simulate_beta_x)
        [self.position.columns]
        theta, beta_y (from _simulate_beta_y)
        beta_x_hat, beta_y_hat (from _simulate_beta_x_y_hat)
        sigma_x, sigma_y (from config)
        """
        df_variant = self._init_position_variant_id()
        df_beta_x = self._simulate_beta_x(df_position)
        df_variant = pd.merge(df_variant, df_beta_x, on="variant")
        df_variant = pd.merge(
            df_variant, df_position, on="position"
        )  # merge into variant

        df_variant = self._simulate_beta_y(df_variant)
        df_variant = self._simulate_beta_x_y_hat(df_variant)
        df_variant["sigma_x"] = self.config.observation.sigma_x
        df_variant["sigma_y"] = self.config.observation.sigma_y

        return df_variant

    def simulate(self):
        """
        Run simulation
        """

        np.random.seed(self.config.simulation.seed)

        self.df_position = self._simulate_position()
        self.df_variant = self._simulate_variant(df_position=self.df_position)

    def build_cosmos(self, model_path) -> None:
        """
        Build Cosmos objects with the simulated data.
        """
        if self.df_variant is None or self.df_position is None:
            raise ValueError("Simulation not run.")

        data = self.df_variant[
            [
                "variant",
                "position",
                "mutation",
                "beta_x_hat",
                "sigma_x",
                "beta_y_hat",
                "sigma_y",
            ]
        ].copy()
        data["mutation"] = "missense"
        data.columns = [
            "variants",
            "group",
            "type",
            "beta_hat_1",
            "se_hat_1",
            "beta_hat_2",
            "se_hat_2",
        ]

        self.data = DMSData(
            data,
            ["pheno1", "pheno2"],
            include_type=["missense"],
            exclude_type=None,
            min_num_variants_per_group=10,
        )
        self.prior = PriorFactory(
            self.data,
            x_name="beta_hat_1",
            y_name="beta_hat_2",
            x_se_name="se_hat_1",
            x_gmm_n_components=2,
        )
        self.model = ModelBuilder(
            prior=self.prior,
            data_path=model_path,
        )

    ########## Helper Functions for Simulation ##########

    def _init_position_variant_id(self) -> pd.DataFrame:
        """
        Initialize position and variant_id
        """
        # read config
        n_position = self.config.simulation.n_position
        n_variant_per_position = self.config.simulation.n_variant_per_position

        # generate label
        position = np.repeat(range(n_position), n_variant_per_position)
        mutation_id = np.tile(range(n_variant_per_position), n_position)

        # concat into a dataframe with two columns: position and mutation_id
        df_position_variant_id = pd.DataFrame(
            {"position": position, "mutation": mutation_id}
        )
        df_position_variant_id["variant"] = range(len(df_position_variant_id))

        return df_position_variant_id

    def _simulate_beta_x(self, df_position: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate beta_x from a gaussian mixture model
        """
        # read config
        nvar = self.config.simulation.n_variant_per_position
        weights = self.config.mixed_x.pi  # parameters of gmm
        means = self.config.mixed_x.mu  # parameters of gmm
        variances = self.config.mixed_x.omega  # parameters of gmm
        n_components = len(weights)  # parameters of gmm

        # sample based on df_position
        # if "null", beta_x is 0
        # if "mixed", sample from a gaussian mixture model
        beta_x = []
        model_x = []
        for _, row in df_position.iterrows():
            if row["x_class"] == "null":
                model_x.extend(["null"] * nvar)
                beta_x.extend([0] * nvar)
            elif row["x_class"] == "mixed":
                components = np.random.choice(n_components, size=nvar, p=weights)
                beta_x.extend(
                    [
                        np.random.normal(means[k], np.sqrt(variances[k]))
                        for k in components
                    ]
                )
                model_x.extend(components)
            else:
                raise ValueError(f"Unrecognized class {row['class']}")

        # concat into a dataframe with two columns: compoenents and beta_x
        df_beta_x = pd.DataFrame({"class_x": model_x, "beta_x": beta_x})
        df_beta_x["variant"] = range(len(df_beta_x))

        return df_beta_x

    def _simulate_beta_y(self, df_variant: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate beta_y from beta_x, gamma, and tau
        """
        # simulate theta
        df_variant["theta"] = np.random.normal(
            0, self.config.observation.sigma_theta, len(df_variant)
        )

        # generate beta_y
        df_variant["beta_y"] = (
            df_variant["beta_x"] * df_variant["gamma"]
            + df_variant["tau"]
            + df_variant["theta"]
        )

        return df_variant

    def _simulate_beta_x_y_hat(self, df_variant: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate beta_x_hat and beta_y_hat from beta_x and beta_y
        """
        # simulate beta_x_hat
        df_variant["beta_x_hat"] = df_variant["beta_x"] + np.random.normal(
            0, self.config.observation.sigma_x, len(df_variant)
        )

        # simulate beta_y_hat
        df_variant["beta_y_hat"] = df_variant["beta_y"] + np.random.normal(
            0, self.config.observation.sigma_y, len(df_variant)
        )

        return df_variant

    def _simulate_gamma(self) -> pd.DataFrame:
        """
        For each position, generate gamma from a gaussian mixture model
        """
        # read config
        n_samples = self.config.simulation.n_position
        n_components = len(self.config.causal_gamma.pi)
        weights = self.config.causal_gamma.pi
        means = self.config.causal_gamma.mean
        variances = self.config.causal_gamma.sd

        # sample component indices
        components = np.random.choice(n_components, size=n_samples, p=weights)

        # sample from the chosen Gaussians
        gamma = np.array(
            [np.random.normal(means[k], np.sqrt(variances[k])) for k in components]
        )

        # concat into a dataframe with two columns: components and gamma
        df_gamma = pd.DataFrame({"gamma_group": components, "gamma": gamma})
        df_gamma["position"] = range(n_samples)

        return df_gamma

    def _simulate_tau(self) -> pd.DataFrame:
        """
        For each position, simulate tau from a gaussian mixture model
        """
        # read config
        n_samples = self.config.simulation.n_position
        n_components = len(self.config.causal_tau.pi)
        weights = self.config.causal_tau.pi
        means = self.config.causal_tau.mean
        variances = self.config.causal_tau.sd

        # sample component indices
        components = np.random.choice(n_components, size=n_samples, p=weights)

        # sample from the chosen Gaussians
        tau = np.array(
            [np.random.normal(means[k], np.sqrt(variances[k])) for k in components]
        )

        # concat into a dataframe with two columns: components and tau
        df_tau = pd.DataFrame({"tau_group": components, "tau": tau})
        df_tau["position"] = range(n_samples)

        return df_tau

    ####### Run analysis #######
    def run_cosmos(
        self,
        group_new_idx: int,
        no_s_hat: bool = False,
        suppress_pareto_warning: bool = True,
    ) -> None:
        """
        Run Cosmos for one specific group_new on the simulated data.
        """
        if self.df_variant is None or self.df_position is None:
            raise ValueError("Simulation not run.")

        self.model.run_cosmos(
            group_new_idx=group_new_idx,
            no_s_hat=no_s_hat,
            suppress_pareto_warning=suppress_pareto_warning,
        )

    ####### Plotting Functions #######

    def plot_beta_x_y_hat(
        self, axes: Optional[Sequence[Axes]] = None
    ) -> Sequence[Axes]:
        """
        Plot beta_x_hat and beta_y_hat from the simulated data.
        """
        if self.df_variant is None:
            raise ValueError("Simulation not run.")

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

        return self._plot_beta_x_y_hat(self.df_variant, axes=axes)  # type: ignore[assign]

    @staticmethod
    def _plot_beta_x_y_hat(
        df_variant: pd.DataFrame,
        axes: Sequence[Axes],
    ) -> Sequence[Axes]:
        """
        Plot beta_x_hat and beta_y_hat
        """
        # plot beta_x_hat
        _ = sns.histplot(
            data=df_variant,
            x="beta_x_hat",
            bins=50,
            alpha=0.6,
            kde=True,
            ax=axes[0],
        )
        _ = axes[0].set_title(r"Simulated $\hat{\beta}_x$")
        _ = axes[0].set_xlabel(r"$\hat{\beta}_{x, ij}$")
        _ = axes[0].set_ylabel("Frequency")

        # plot beta_y_hat
        _ = sns.histplot(
            data=df_variant,
            x="beta_y_hat",
            bins=50,
            alpha=0.6,
            kde=True,
            ax=axes[1],
        )
        _ = axes[1].set_title(r"Simulated $\hat{\beta}_y$")
        _ = axes[1].set_xlabel(r"$\hat{\beta}_{y, ij}$")
        _ = axes[1].set_ylabel("Frequency")

        # plot scatterplot of beta_x_hat and beta_y_hat
        _ = sns.scatterplot(
            data=df_variant,
            x="beta_x_hat",
            y="beta_y_hat",
            alpha=0.6,
            zorder=1,
            ax=axes[2],
        )
        _ = axes[2].axhline(0.0, color="lightgray", linestyle="--", zorder=0)
        _ = axes[2].axvline(0.0, color="lightgray", linestyle="--", zorder=0)
        _ = axes[2].set_title(r"Scatterplot of $\hat{\beta}_x$ and $\hat{\beta}_y$")
        _ = axes[2].set_xlabel(r"$\hat{\beta}_{x, ij}$")
        _ = axes[2].set_ylabel(r"$\hat{\beta}_{y, ij}$")

        return axes

    def plot_beta_x(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot beta_x from the simulated data.
        """
        if self.df_variant is None:
            raise ValueError("Simulation not run.")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)

        return self._plot_beta_x(self.df_variant, ax=ax)

    @staticmethod
    def _plot_beta_x(df_beta_x: pd.DataFrame, ax: Axes) -> Axes:
        """
        Frequency plot of beta_x
        """
        _ = sns.histplot(
            data=df_beta_x,
            x="beta_x",
            bins=50,
            alpha=0.6,
            hue="class_x",
            kde=True,
            ax=ax,
        )
        _ = ax.set_title(r"Simulated $\beta_x$ from Gaussian Mixture Model")
        _ = ax.set_xlabel(r"$\beta_{x, ij}$")
        _ = ax.set_ylabel("Frequency")

        return ax

    def plot_beta_y(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot beta_y from the simulated data.
        """
        if self.df_variant is None:
            raise ValueError("Simulation not run.")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)

        return self._plot_beta_y(self.df_variant, ax=ax)

    @staticmethod
    def _plot_beta_y(df_variant: pd.DataFrame, ax: Axes) -> Axes:
        """
        Frequency plot of beta_y
        """
        _ = sns.histplot(
            data=df_variant,
            x="beta_y",
            bins=50,
            alpha=0.6,
            kde=True,
            ax=ax,
        )
        _ = ax.set_title(r"Simulated $\beta_y$")
        _ = ax.set_xlabel(r"$\beta_{y, ij}$")
        _ = ax.set_ylabel("Frequency")

        return ax

    def plot_gamma(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot gamma from the simulated data.
        """
        if self.df_position is None:
            raise ValueError("Simulation not run.")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)

        return self._plot_gamma(self.df_position, ax=ax)

    @staticmethod
    def _plot_gamma(df_gamma: pd.DataFrame, ax: Axes) -> Axes:
        _ = sns.histplot(
            data=df_gamma,
            x="gamma",
            bins=50,
            alpha=0.6,
            hue="gamma_group",
            kde=True,
            ax=ax,
        )
        _ = ax.set_title(r"Simulated $\gamma$ from Gaussian Mixture Model")
        _ = ax.set_xlabel(r"\gamma_i")
        _ = ax.set_ylabel("Frequency")

        return ax

    def plot_tau(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot tau from the simulated data.
        """
        if self.df_position is None:
            raise ValueError("Simulation not run.")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)

        return self._plot_tau(self.df_position, ax=ax)

    @staticmethod
    def _plot_tau(df_tau: pd.DataFrame, ax: Axes) -> Axes:
        """
        Frequency plot of tau
        """
        _ = sns.histplot(
            data=df_tau,
            x="tau",
            bins=50,
            alpha=0.6,
            hue="tau_group",
            kde=True,
            ax=ax,
        )
        _ = ax.set_title(r"Simulated $\tau$ from Gaussian Mixture Model")
        _ = ax.set_xlabel(r"\tau_i")
        _ = ax.set_ylabel("Frequency")

        return ax
