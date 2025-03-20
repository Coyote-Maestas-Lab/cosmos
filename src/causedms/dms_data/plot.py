"""
Plotting functions for DMSData.

TODO: Decouple from DMSData class.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from .dms_data import DMSData


def plot_histogram(data: DMSData, pheno: int, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pheno_name = data.phenotypes[pheno - 1]

    _ = sns.histplot(
        data.data, x=f"beta_hat_{pheno}", bins=30, kde=True, hue="type", ax=ax
    )
    _ = ax.set_title(f"Histogram of {pheno_name} beta_hat")

    return ax


def plot_histogram_with_gmm(data: DMSData, component: int = 2, ax=None):

    if data.prior is None:
        raise ValueError("Must generate prior first.")
    mu_m = data.prior["mu_m"]
    sigma2_m = data.prior["sigma2_m"]

    # use snsplot the histgoram with gmm estiamtes
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    # group by mutation type
    _ = sns.histplot(data.data, x="beta_hat_1", bins=30, kde=True, hue="type", ax=ax)
    _ = ax.set_title(f"Histogram of {data.phenotypes[0]} beta_hat")
    # add gmm estimates as vertical lines
    for i in range(component):
        mu_i = mu_m[i]
        sigma2_m_i = sigma2_m[i]
        _ = ax.axvline(
            mu_i, color="red" if i == 0 else "blue", linestyle="--", label="mu_m"
        )
        _ = ax.axvline(
            mu_i - 3 * sigma2_m_i,
            color="pink" if i == 0 else "aqua",
            linestyle="--",
            label="mu_m-3sigma2_m",
        )
        _ = ax.axvline(
            mu_i + 3 * sigma2_m_i,
            color="pink" if i == 0 else "aqua",
            linestyle="--",
            label="mu_m+3sigma2_m",
        )

    return ax


def plot_scatterplot(data: DMSData, type_col_dict: dict = None, ax=None):
    """
    Plot scatterplot of beta_hat_1 vs beta_hat_2.
    """
    if type_col_dict is None:
        type_col_dict = {
            "synonymous": "green",
            "deletion": "orange",
            "missense": "grey",
            "nonsense": "red",
            "insertion": "purple",
        }
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    _ = sns.scatterplot(
        data=data.data,
        x="beta_hat_1",
        y="beta_hat_2",
        s=20,
        alpha=0.5,
        hue="type",
        palette=type_col_dict,
    )
    _ = plt.xlabel(f"{data.phenotypes[0]} beta_hat")
    _ = plt.ylabel(f"{data.phenotypes[1]} beta_hat")
    _ = plt.title(f"{data.phenotypes[0]} beta_hat vs {data.phenotypes[1]} beta_hat")

    return ax
