"""
Plotting functions for the prior of the model.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from .prior_factory import PriorFactory


def plot_histogram_with_gmm(prior: PriorFactory, component: int = 2, ax=None):

    if prior.prior is None:
        raise ValueError("Must generate prior first.")
    mu_m = prior.prior["mu_m"]
    sigma2_m = prior.prior["sigma2_m"]

    # use snsplot the histgoram with gmm estiamtes
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    # group by mutation type
    _ = sns.histplot(prior.data, x="beta_hat_1", bins=30, kde=True, hue="type", ax=ax)
    _ = ax.set_title(f"Histogram of {prior.phenotypes[0]} beta_hat")
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
