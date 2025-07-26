"""
A generatative model for Cosmos
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        self.config = DEFAULT_CONFIG if config is None else config

    def _simulate_position_class(self) -> pd.DataFrame:
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

        # concat into a dataframe with two columns: compoenents and gamma
        df_class = pd.DataFrame({"class": position_class})
        df_class["position"] = range(n_samples)

        return df_class

    def _simulate_position(self, plot: bool) -> pd.DataFrame:
        """
        For each position, generate gamma, tau, and position class.

        Result columns: position, group_gamma, gamma, group_tau, tau, class
        """
        # For each position, generate gamma and tau from a gaussian mixture model
        df_gamma = self._simulate_gamma(plot=plot)
        df_tau = self._simulate_tau(plot=plot)

        # For each position, generate position class from a multinomial distribution
        df_class = self._simulate_position_class()
        df_position = pd.merge(df_gamma, df_tau, on="position")
        df_position = df_position[
            ["position", "group_gamma", "gamma", "group_tau", "tau"]
        ]
        df_position = pd.merge(df_position, df_class, on="position")

        return df_position

    def _simulate_variant(self, df_position: pd.DataFrame, plot: bool) -> pd.DataFrame:
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
        df_beta_x = self._simulate_beta_x(df_position, plot=plot)
        df_variant = pd.merge(df_variant, df_beta_x, on="variant")
        df_variant = pd.merge(
            df_variant, df_position, on="position"
        )  # merge into variant

        df_variant = self._simulate_beta_y(df_variant, plot=plot)
        df_variant = self._simulate_beta_x_y_hat(df_variant, plot=True)
        df_variant["sigma_x"] = self.config.observation.sigma_x
        df_variant["sigma_y"] = self.config.observation.sigma_y

        return df_variant

    def simulate(self, plot: bool = False):
        """
        Run simulation
        """

        np.random.seed(self.config.simulation.seed)

        self.df_position = self._simulate_position(plot=plot)
        self.df_variant = self._simulate_variant(
            df_position=self.df_position, plot=plot
        )

    def run_cosmos(self, model_path) -> None:
        """
        Run Cosmos model with the simulated data.
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

    def _simulate_beta_x(
        self, df_position: pd.DataFrame, plot: bool = False
    ) -> pd.DataFrame:
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
            if row["class"] == "null":
                model_x.extend(["null"] * nvar)
                beta_x.extend([0] * nvar)
            elif row["class"] == "mixed":
                components = np.random.choice(n_components, size=nvar, p=weights)
                beta_x.extend(
                    [
                        np.random.normal(means[k], np.sqrt(variances[k]))
                        for k in components
                    ]
                )
                model_x.extend(components)

        # concat into a dataframe with two columns: compoenents and beta_x
        df_beta_x = pd.DataFrame({"class_x": model_x, "beta_x": beta_x})
        df_beta_x["variant"] = range(len(df_beta_x))

        if plot:
            self._plot_beta_x(df_beta_x)

        return df_beta_x

    def _simulate_beta_y(
        self, df_variant: pd.DataFrame, plot: bool = False
    ) -> pd.DataFrame:
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

        if plot:
            self._plot_beta_y(df_variant)

        return df_variant

    def _simulate_beta_x_y_hat(
        self, df_variant: pd.DataFrame, plot: bool = False
    ) -> pd.DataFrame:
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

        if plot:
            self._plot_beta_x_y_hat(df_variant)

        return df_variant

    def _simulate_gamma(self, plot: bool = False) -> pd.DataFrame:
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
        df_gamma = pd.DataFrame({"group_gamma": components, "gamma": gamma})
        df_gamma["position"] = range(n_samples)

        # plot
        if plot:
            self._plot_gamma(df_gamma)

        return df_gamma

    def _simulate_tau(self, plot: bool = False) -> pd.DataFrame:
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
        df_tau = pd.DataFrame({"group_tau": components, "tau": tau})
        df_tau["position"] = range(n_samples)

        # plot
        if plot:
            self._plot_tau(df_tau)

        return df_tau

    @staticmethod
    def _plot_beta_x_y_hat(df_variant: pd.DataFrame):
        """
        Plot beta_x_hat and beta_y_hat
        """
        # plot three figures in a row
        _, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

        # plot beta_x_hat
        _ = sns.histplot(
            data=df_variant,
            x="beta_x_hat",
            bins=50,
            alpha=0.6,
            kde=True,
            ax=axs[0],
        )
        axs[0].set_title(r"Simulated $\hat{beta}_x$")
        axs[0].set_xlabel("Value")
        axs[0].set_ylabel("Density")

        # plot beta_y_hat
        _ = sns.histplot(
            data=df_variant,
            x="beta_y_hat",
            bins=50,
            alpha=0.6,
            kde=True,
            ax=axs[1],
        )
        axs[1].set_title("Simulated beta_y_hat")
        axs[1].set_xlabel("Value")
        axs[1].set_ylabel("Density")

        # plot scatterplot of beta_x_hat and beta_y_hat
        _ = sns.scatterplot(
            data=df_variant,
            x="beta_x_hat",
            y="beta_y_hat",
            alpha=0.6,
            ax=axs[2],
        )
        axs[2].set_title(r"Scatterplot of $\hat{beta}_x$ and $\hat{beta}_y$")
        axs[2].set_xlabel(r"$\hat{beta}_x$")
        axs[2].set_ylabel(r"$\hat{beta}_y$")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_beta_y(df_variant: pd.DataFrame):
        """
        Density plot of beta_y
        """
        _ = plt.figure(figsize=(5, 4), dpi=200)
        _ = sns.histplot(
            data=df_variant,
            x="beta_y",
            bins=50,
            alpha=0.6,
            kde=True,
        )
        _ = plt.title(r"Simulated $\beta_y$")
        _ = plt.xlabel("Value")
        _ = plt.ylabel("Density")
        plt.show()

    @staticmethod
    def _plot_gamma(df_gamma: pd.DataFrame):
        _ = plt.figure(figsize=(5, 4), dpi=200)
        _ = sns.histplot(
            data=df_gamma,
            x="gamma",
            bins=50,
            alpha=0.6,
            hue="group_gamma",
            kde=True,
        )
        _ = plt.title(r"Simulated $\gamma$ from Gaussian Mixture Model")
        _ = plt.xlabel("Value")
        _ = plt.ylabel("Density")
        plt.show()

    @staticmethod
    def _plot_tau(df_tau: pd.DataFrame):
        """
        Density plot of tau
        """
        _ = plt.figure(figsize=(5, 4), dpi=200)
        _ = sns.histplot(
            data=df_tau,
            x="tau",
            bins=50,
            alpha=0.6,
            hue="group_tau",
            kde=True,
        )
        _ = plt.title(r"Simulated $\tau$ from Gaussian Mixture Model")
        _ = plt.xlabel("Value")
        _ = plt.ylabel("Density")
        plt.show()

    @staticmethod
    def _plot_beta_x(df_beta_x: pd.DataFrame):
        """
        Density plot of beta_x
        """
        _ = plt.figure(figsize=(5, 4), dpi=200)
        _ = sns.histplot(
            data=df_beta_x,
            x="beta_x",
            bins=50,
            alpha=0.6,
            hue="class_x",
            kde=True,
        )
        _ = plt.title(r"Simulated $\beta_x$ from Gaussian Mixture Model")
        _ = plt.xlabel("Value")
        _ = plt.ylabel("Density")
        plt.show()
