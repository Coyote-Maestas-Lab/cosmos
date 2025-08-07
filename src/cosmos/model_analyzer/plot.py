"""
Plotting utilities for ModelAnalyzer
"""

from itertools import product
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.axes import Axes

from cosmos.hgvs import Missense

from .model_analyzer import ModelAnalyzer


def _parse_mutant(hgvs: str) -> Optional[str]:
    try:
        return Missense.from_hgvs(hgvs).mutant
    except AssertionError:
        return None


def _get_global_lims(
    data: pd.DataFrame, extra_margin: float = 0.02
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get the global limits for the plot.

    lower = min - extra_margin * (max - min)
    upper = max + extra_margin * (max - min)
    """

    def get_lim(min_val: float, max_val: float) -> tuple[float, float]:
        lower = min_val - extra_margin * (max_val - min_val)
        upper = max_val + extra_margin * (max_val - min_val)
        return lower, upper

    xlim = get_lim(data["beta_hat_1"].min(), data["beta_hat_1"].max())
    ylim = get_lim(data["beta_hat_2"].min(), data["beta_hat_2"].max())

    return xlim, ylim


def plot_position(
    analyzer: ModelAnalyzer, position: int, ax: Optional[Axes] = None, **kwargs
) -> Axes:
    """
    Plot the tau and gamma values for a specific position.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.75, 3.4))

    xlim, ylim = _get_global_lims(analyzer.data)

    raw_data = analyzer.data[analyzer.data["group"] == position]
    raw_data = raw_data[raw_data["type"].isin(["missense"])]

    aa_label = "mutant_aa"
    raw_data[aa_label] = raw_data["variants"].apply(_parse_mutant)

    res = analyzer.best_models[analyzer.best_models["position"] == position]

    dot_size = 30
    dot_color = "#4169E1"

    _ = sns.scatterplot(
        data=raw_data,
        x="beta_hat_1",
        y="beta_hat_2",
        s=dot_size,
        color=dot_color,
        edgecolor="black",
        zorder=1,
        alpha=0.8,
        ax=ax,
        **kwargs,
    )

    # Add repel text labeling
    texts = []
    for _, row in raw_data.iterrows():
        # wt = row["wildtype"]
        texts.append(
            ax.text(
                row["beta_hat_1"],
                row["beta_hat_2"],
                row[aa_label],
                fontsize=13,
                ha="center",
                va="center",
                zorder=2,
            )
        )
    # Adjust text to avoid overlaps
    _ = adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="grey"),
        expand=(1.2, 1.4),
        force_text=1.0,
    )

    _ = ax.axhline(0, color="grey", linestyle="--", zorder=-1)
    _ = ax.axvline(0, color="grey", linestyle="--", zorder=-1)

    pheno1, pheno2 = analyzer.phenotypes[0], analyzer.phenotypes[1]

    _ = ax.set_xlabel(pheno1)
    _ = ax.set_ylabel(pheno2)

    _ = ax.set_xlim(*xlim)
    _ = ax.set_ylim(*ylim)

    _ = ax.set_title(f"Position {position} ({pheno1} vs {pheno2})")

    plot_x = np.linspace(*ax.get_xlim(), 100)
    gamma, tau = res["gamma_mean"].fillna(0).values, res["tau_mean"].fillna(0).values
    _ = ax.plot(
        plot_x,
        gamma * plot_x + tau,  # type: ignore
        color="black",
        linestyle="-",
        zorder=0,
    )

    return ax


def _format_position_ax(ax: Axes, max_pos: int):
    sns.despine(ax=ax, left=True, bottom=True)
    _ = ax.grid(axis="both", color="lightgray", linestyle="--", alpha=0.5)
    _ = ax.set_xlabel("Position")
    _ = ax.set_xlim(-1, max_pos + 1)

    ticks = np.arange(0, max_pos + 1, 5)[1:]
    _ = ax.set_xticks(ticks)
    _ = ax.set_xticklabels(ticks)


def plot_best_models(analyzer: ModelAnalyzer, ax: Optional[Axes] = None) -> Axes:
    """
    Plot the best models for each position.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(36, 3))

    best_models = analyzer.best_models.copy()
    best_models["position"] = best_models["position"].astype(int)
    best_models["model"] = best_models["model_rank1"].str[-1].astype(int)

    _ = sns.lineplot(
        data=best_models,
        x="position",
        y="model",
        color="grey",
        zorder=1,
        ax=ax,
    )
    _ = sns.scatterplot(
        data=best_models,
        x="position",
        y="model",
        hue="model",
        palette=sns.color_palette(n_colors=6)[::-1],
        zorder=2,
        ax=ax,
        legend=False,
    )

    _ = ax.set_ylabel("Best model")
    _format_position_ax(ax, best_models["position"].max())

    return ax


def plot_best_params(
    analyzer: ModelAnalyzer, param: Literal["tau", "gamma"], ax: Optional[Axes] = None
) -> Axes:
    """
    Plot the best tau values for each position.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(36, 3))

    best_models = analyzer.best_models.copy()
    best_models["position"] = best_models["position"].astype(int)
    best_models = best_models.fillna(
        {"_".join(col): 0 for col in product(["tau", "gamma"], ["mean", "std"])}
    )

    _ = sns.lineplot(
        data=best_models,
        x="position",
        y=f"{param}_mean",
        color="grey",
        marker="o",
        markerfacecolor="black",
        zorder=1,
        ax=ax,
    )
    _ = ax.errorbar(
        x=best_models["position"],
        y=best_models[f"{param}_mean"],
        yerr=best_models[f"{param}_std"],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        zorder=0,
    )

    _ = ax.set_ylabel(f"$\\{param}$ estimate")
    _ = ax.axhline(0, color="grey", linestyle="--", zorder=0)
    _format_position_ax(ax, best_models["position"].max())

    return ax
