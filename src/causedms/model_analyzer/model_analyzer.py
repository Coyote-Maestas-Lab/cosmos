import logging
import os

import pandas as pd

from causedms.adapt_grid.grid import MarginSummary
from causedms.model_builder import ModelBuilder


class ModelAnalyzer:
    """
    Analyze results from a CausalDMS model.
    """

    def __init__(self, model: ModelBuilder, data_path: str, has_position: bool = True):
        self.model = model
        self.protein_feaure = None

        self.data_path = data_path

        if self.data_path == model.data_path:
            logging.warning(
                "data_path is the same as model.data_path."
                "A subfolder 'analysis' will be created in data_path."
            )
            self.data_path = os.path.join(self.data_path, "analysis")

        os.makedirs(self.data_path, exist_ok=True)

        self.position_group_mapping: pd.DataFrame = self.get_position_group_mapping(
            self.data, has_position=has_position
        )

        self.data_summary, combined_data_comparison = ModelBuilder.summary_causal_dms(
            model.data_path
        )

        self.data_comparison = self.process_data_comparison(
            combined_data_comparison, self.position_group_mapping
        )

        # if has_position:

    @property
    def data(self) -> pd.DataFrame:
        return self.model.data

    @staticmethod
    def get_position_group_mapping(
        data: pd.DataFrame, has_position: bool = True
    ) -> pd.DataFrame:
        """
        Map between group and group_new index in dataDMS2
        """
        # filter out data with "exclude" group in dataDMS2
        data_include = data[data["group_new"].astype(int) > 0]

        # select columns group and group_new
        data_group = data_include[["group", "group_new"]]
        data_group = data_group.drop_duplicates().reset_index(drop=True)

        # group_new are integer
        if has_position:
            data_group["group"] = data_group["group"].astype(int)
        data_group["group_new"] = data_group["group_new"].astype(int)

        return data_group

    @staticmethod
    def process_data_comparison(
        comparison: dict[int, pd.DataFrame], data_group: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine model comparison data from ModelBuilder
        """
        df_comparison = pd.concat(comparison).reset_index(level=0)
        df_comparison.reset_index(inplace=True)
        df_comparison.rename(
            columns={"level_0": "group_new", "index": "model"}, inplace=True
        )

        # sort by group_new
        df_comparison["group_new"] = pd.to_numeric(
            df_comparison["group_new"], downcast="integer", errors="coerce"
        )
        df_comparison.sort_values(by=["group_new", "rank"], inplace=True)
        df_comparison.reset_index(drop=True, inplace=True)

        # modify rank
        df_comparison["rank"] = df_comparison["rank"] + 1

        # map group_new to group
        df_comparison = pd.merge(df_comparison, data_group, on="group_new", how="left")
        df_comparison["group"] = df_comparison["group"].fillna(
            df_comparison["group_new"]
        )

        return df_comparison

    @staticmethod
    def extract_tau_gamma_value_from_model(
        summary_sub: dict[str, dict[str, MarginSummary]],
        model: str,
        n_quantiles: int = 3,
    ) -> tuple[MarginSummary, MarginSummary]:
        """
        Extract tau and gamma marginal info, given the model name.
        """
        tau_margin = summary_sub[model].get("tau", MarginSummary.null(n_quantiles))
        gamma_margin = summary_sub[model].get("gamma", MarginSummary.null(n_quantiles))

        return tau_margin, gamma_margin

    @staticmethod
    def _get_param_summary(
        position_mapping: pd.DataFrame,
        data_summary: dict[int, dict[str, dict[str, MarginSummary]]],
        data_comparison: pd.DataFrame,
        rank: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate summary of tau and gamma posterior values from the x'th best model at each position
        """
        df_tau = pd.DataFrame()
        df_gamma = pd.DataFrame()

        gamma_margin_rows = []
        tau_margin_rows = []

        for group_new_idx, summary_sub in data_summary.items():

            comparison_sub = data_comparison[
                data_comparison["group_new"] == group_new_idx
            ]

            # f"model_{index}" where index in range(1, 7)
            model = comparison_sub[comparison_sub["rank"] == rank]["model"].values[0]

            tau_margin, gamma_margin = ModelAnalyzer.extract_tau_gamma_value_from_model(
                summary_sub, model
            )

            tau_margin_row = tau_margin.to_dict()
            gamma_margin_row = gamma_margin.to_dict()

            gamma_margin_row["group_new"] = group_new_idx
            tau_margin_row["group_new"] = group_new_idx

            gamma_margin_rows.append(gamma_margin_row)
            tau_margin_rows.append(tau_margin_row)

        df_gamma = pd.DataFrame(gamma_margin_rows)
        df_tau = pd.DataFrame(tau_margin_rows)

        # map group_new to group
        df_gamma = pd.merge(df_gamma, position_mapping, on="group_new", how="left")
        df_tau = pd.merge(df_tau, position_mapping, on="group_new", how="left")

        # sort by group
        df_gamma.sort_values(by="group", inplace=True)
        df_tau.sort_values(by="group", inplace=True)
        df_gamma.reset_index(drop=True, inplace=True)
        df_tau.reset_index(drop=True, inplace=True)

        return df_tau, df_gamma

    def get_param_summary(self, rank: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate summary of tau and gamma posterior values from the best model at each position
        """
        return self._get_param_summary(
            self.position_group_mapping,
            self.data_summary,
            self.data_comparison,
            rank=rank,
        )

    @staticmethod
    def _select_model_by_rank(df: pd.DataFrame, rank: int) -> pd.DataFrame:
        """
        Given the data_comparison DataFrame, for each position, select the model by rank.
        The rank is 1-indexed and in descending order (1 = best).
        """
        df_rank = df[df["rank"] == rank].reset_index(drop=True)
        model_order = sorted(df_rank["model"].unique(), reverse=True)
        df_rank["model"] = pd.Categorical(
            df_rank["model"], categories=model_order, ordered=True
        )
        return df_rank

    def select_model_by_rank(self, rank: int = 1) -> pd.DataFrame:
        """
        For each position, select the model by rank (default to the best).
        The rank is 1-indexed and in descending order (1 = best).
        """
        return self._select_model_by_rank(self.data_comparison, rank)

    def summary(self, rank: int = 1) -> pd.DataFrame:
        """
        Get the summary of each parameter from the x'th best model at each position.

        If `save`, the summary will be saved to a CSV file in `analysis_path`.

        NOTE: Summarizing the non-best model doesn't make much sense.
        """

        df_rank = self.select_model_by_rank(rank=rank)
        df_rank = df_rank[["group", "group_new", "model"]]
        df_rank.columns = ["group", "group_new", f"model_rank{rank}"]

        summary_tau, summary_gamma = self.get_param_summary(rank=rank)
        summary_tau = summary_tau[["group", "group_new", "mean", "std"]]
        summary_tau.columns = ["group", "group_new", "tau_mean", "tau_std"]
        summary_gamma = summary_gamma[["group", "group_new", "mean", "std"]]
        summary_gamma.columns = ["group", "group_new", "gamma_mean", "gamma_std"]

        df_combined = df_rank.merge(summary_tau, on=["group", "group_new"]).merge(
            summary_gamma, on=["group", "group_new"]
        )

        df_combined.rename(
            columns={"group": "position", "group_new": "group"}, inplace=True
        )

        return df_combined
