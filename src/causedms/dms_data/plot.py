"""
Plotting functions for DMSData.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from .dms_data import DMSData


def _plot_histogram(data: "DMSData", pheno: int, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pheno_name = data.phenotypes[pheno - 1]

    _ = sns.histplot(
        data.data, x=f"beta_hat_{pheno}", bins=30, kde=True, hue="type", ax=ax
    )
    _ = ax.set_title(f"Histogram of {pheno_name} beta_hat")

    return ax


def _plot_scatterplot(data: "DMSData", type_col_dict: dict = None, ax=None):
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
