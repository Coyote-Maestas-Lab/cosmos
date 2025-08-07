"""
Global plotting functions.
For more specific plotting functions, see individual modules.
"""

from collections import ChainMap
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from cosmos.dms_data import DMSData
from cosmos.hgvs import Missense
from cosmos.model_analyzer import ModelAnalyzer
from cosmos.model_builder import ModelBuilder

PALETTE = {
    "deletion": "#e41a1c",  # red
    "synonymous": "#90ee90",  # green
    "missense": "#999999",  # gray
    "insertion": "#377eb8",  # blue
}

ZORDERS = {
    "deletion": 2,
    "synonymous": 3,
    "missense": 1,
    "insertion": 2,
}

ALPHAS = {
    "missense": 0.3,
    "synonymous": 1.0,
    "deletion": 0.8,
    "insertion": 0.8,
}


def plot_global_scatter(
    obj: ModelAnalyzer | ModelBuilder | DMSData,
    types: Iterable[str] = ("missense", "synonymous"),
    palette: Optional[dict[str, str]] = None,
    zorders: Optional[dict[str, int]] = None,
    alphas: Optional[dict[str, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a global scatter plot of the phenotypes.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    df = obj.data

    _palette = ChainMap(palette or {}, PALETTE)
    _zorders = ChainMap(zorders or {}, ZORDERS)
    _alphas = ChainMap(alphas or {}, ALPHAS)

    for label in types:
        subset = df[df["type"].str.startswith(label)]
        _ = sns.scatterplot(
            data=subset,
            x="beta_hat_1",
            y="beta_hat_2",
            label=label,
            color=_palette[label],
            s=25,
            alpha=_alphas[label],
            edgecolor="k",
            linewidth=0.5,
            zorder=_zorders[label],
            ax=ax,
        )

    _ = ax.legend(title="Mutation Type", fontsize=14, title_fontsize=14, loc="best")

    pheno_x, pheno_y = obj.phenotypes[0], obj.phenotypes[1]

    _ = ax.set_xlabel(pheno_x, fontsize=24)
    _ = ax.set_ylabel(pheno_y, fontsize=24)
    _ = ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    return ax


def _get_missense_variants(df: pd.DataFrame) -> pd.DataFrame:
    df_sel = df[df["type"].isin(["missense", "synonymous"])].reset_index(drop=True)
    variants = [Missense.from_hgvs(hgvs) for hgvs in df_sel["variants"]]
    variants_df = pd.DataFrame(
        {
            "wildtype": [v.wildtype for v in variants],
            "mutant": [v.mutant for v in variants],
            "position": [v.position for v in variants],
        },
    )
    df_with_variants = pd.concat([df_sel, variants_df], axis=1)

    return df_with_variants


def plot_global_heatmap(
    obj: ModelAnalyzer | ModelBuilder | DMSData,
    pheno: str,
    pos_range: Optional[tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    norm: Optional[Normalize] = None,
) -> Axes:
    """
    Plot a global heatmap of the phenotypes.
    """

    df = obj.data
    df_missense = _get_missense_variants(df)

    pheno_idx = obj.phenotypes.index(pheno)
    var_name = f"beta_hat_{pheno_idx + 1}"

    plot_df = df_missense.pivot_table(
        index=["mutant"], columns="group", values=var_name
    )

    if pos_range is not None:
        range_min, range_max = pos_range
        plot_df_sel = plot_df.loc[:, range_min:range_max]
    else:
        plot_df_sel = plot_df

    if ax is None:
        width = plot_df_sel.shape[1] * 0.2
        _, ax = plt.subplots(figsize=(width, 4), dpi=300)

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["darkblue", "white", "darkred"], N=256
    )

    norm = norm or TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    _ = sns.heatmap(
        plot_df_sel,
        mask=plot_df_sel.isnull(),
        cmap=cmap,
        norm=norm,
        # center=0,
        # vmin=plot_df.min().min(),
        # vmax=plot_df.max().max(),
        vmin=-3,
        vmax=2,
        linecolor="black",
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    return ax
