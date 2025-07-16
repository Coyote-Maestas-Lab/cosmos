"""
Data preprocessing and prior generation for causal inference.
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from .plot import _plot_histogram, _plot_scatterplot


class DMSData:
    """
    Holds the DMS data and generates the prior for the DMS model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        phenotypes: Sequence[str],
        include_type: Optional[list[str]] = None,
        exclude_type: Optional[list[str]] = None,
        min_num_variants_per_group: int = 10,
    ):
        self._check_cols(data, phenotypes)

        if include_type is None:
            include_type = ["missense"]
        if exclude_type is None:
            exclude_type = ["synonymous"]

        self.data = self.relabel_data_group(
            data, include_type, exclude_type, min_num_variants_per_group
        )
        self.include_type = include_type
        self.exclude_type = exclude_type

        self.phenotypes = list(phenotypes)

        self.prior = None

    @staticmethod
    def _check_cols(data: pd.DataFrame, phenotypes: Sequence[str]) -> None:
        """
        Check if the required columns are present in the data.
        """
        required_cols = (
            {
                "variants",
                "group",  # TODO: change the name?
                "type",
            }
            | {f"beta_hat_{i+1}" for i in range(len(phenotypes))}
            | {f"se_hat_{i+1}" for i in range(len(phenotypes))}
        )

        if not required_cols.issubset(data.columns):
            missing_cols = required_cols - set(data.columns)
            raise ValueError(f"Missing required columns in data: {missing_cols}")

    @staticmethod
    def relabel_data_group(
        df: pd.DataFrame,
        include_type: list[str],
        exclude_type: list[str],
        min_num_variants_per_group: int = 10,
    ) -> pd.DataFrame:
        """
        Create column group_new by relabeling the group column.
        """

        def relabel_data_group_helper(
            df_include: pd.DataFrame, min_num_variants_per_group: int
        ) -> pd.DataFrame:
            # Merge groups with less than min_num_variants_per_group into the next group
            df = df_include.copy()
            df["group_new"] = np.nan
            group_counts = df["group"].value_counts().sort_index()
            curr_index = 1
            curr_count = 0
            for group in df["group"].unique():
                curr_count += group_counts[group]
                df.loc[df["group"] == group, "group_new"] = curr_index
                if curr_count >= min_num_variants_per_group:
                    curr_index += 1
                    curr_count = 0
            df["group_new"] = (
                df["group_new"].astype("int").astype("category")
            )  # TODO: change the name?
            return df

        df_include_all = df[df["type"].isin(include_type)].reset_index(drop=True)
        df_exclude_all = df[df["type"].isin(exclude_type)].reset_index(drop=True)

        # relabel include group, starting from 1
        df_include_all = relabel_data_group_helper(
            df_include_all, min_num_variants_per_group
        )

        # relabel exclude group, starting from 0 and going down
        # One index per exclude type
        df_exclude_all["group_new"] = np.nan
        exc_idx = 0
        for exc in exclude_type:
            df_exclude_all.loc[df_exclude_all["type"] == exc, ["group_new"]] = exc_idx
            exc_idx -= 1

        # combine the two dataframes
        df_res = pd.concat([df_include_all, df_exclude_all], axis=0)
        df_res["group_new"] = df_res["group_new"].astype("int").astype("category")

        return df_res

    def plot_histogram(self, pheno: int, ax=None):
        return _plot_histogram(self, pheno, ax=ax)

    def plot_scatterplot(self, type_col_dict: dict = None, ax=None):
        return _plot_scatterplot(self, type_col_dict=type_col_dict, ax=ax)
