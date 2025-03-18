"""
Adaptive grid approximation for Bayesian posterior approximation, 
  with adaptive boundary extension and hypercube refinement.
"""

import itertools
from collections.abc import Hashable
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.special import logsumexp

from . import GRID_TOL, GridMargin, logger

type HypercubeKey = Iterable[tuple[float, float]]


class Grid:  # pylint: disable=too-many-instance-attributes
    """
    Collection of multiple GridMargin objects, representing a grid in multiple dimensions.
    Allows adaptive boundary extension and hypercube refinement.
    Only need to specify the log likelihood up to an additive constant.
    """

    TOL = GRID_TOL

    def __init__(
        self, axes: Iterable[GridMargin], log_lik: Callable[[np.ndarray], float]
    ):

        self.log_lik = log_lik
        self.axes: Iterable[GridMargin] = axes

        # Store the data for each hypercube, indexed by the midpoints
        self.grid_data = {}

        # At construction, take the Cartesian product of intervals of all axes
        for entries in itertools.product(*list(self.axes)):
            key, vals = Grid._parse_axes_entries(*entries)
            self.grid_data[key] = vals

            # Calculate log prior, log likelihood, log hypercube volume, and log posterior
            self._calculate_logs(key, self.grid_data[key])

        all_log_prior_log_lik = np.array(
            [entry["_log_prior_log_lik"] for entry in self.grid_data.values()]
        )

        self._max_log_prior_log_lik = np.max(
            all_log_prior_log_lik
        )  # For numerical stability in future splits

        self.log_c = -logsumexp(  # pylint: disable=invalid-unary-operand-type
            all_log_prior_log_lik
        )
        self._shifted_prior_x_lik_sum = np.exp(
            -self.log_c - self._max_log_prior_log_lik
        )

        self._boundary_adapted = False

        self._rng = np.random.default_rng()

    def get_log_post(self, key: HypercubeKey) -> float:
        """
        Lazily calculate the log posterior
        """
        if not isinstance(key, Hashable):
            key = tuple(key)
        hypercube = self.grid_data[key]
        return hypercube["_log_prior_log_lik"] + self.log_c - hypercube["log_volume"]

    @staticmethod
    def _parse_axes_entries(*entries):
        """
        Reorders the entries of each axis into :
        - A tuple of (begin, end), key of self.grid_data
        - A dictionary of begin, end, mid, and prior values (values of self.grid_data)
        """
        return (
            tuple((entry["begin"], entry["end"]) for entry in entries),
            {
                col: np.array([entry[col] for entry in entries])
                for col in ("begin", "end", "mid", "prior")
            },
        )

    def __str__(self) -> str:
        return "Grid (" + " X ".join(self.names) + ")"

    def names(self) -> Iterable[str]:
        """
        Names of the axes, in order.
        """
        return [axis.name for axis in self.axes]

    def _get_axis_idx(self, axis: int | str) -> int:
        """
        Convert axis names to indices, if necessary.
        """
        if isinstance(axis, str):
            return self.names().index(axis)
        return axis

    def _update_normalizer(self):
        """
        Update the normalizing constant of the posterior.
        """
        self.log_c = -(
            np.log(self._shifted_prior_x_lik_sum) + self._max_log_prior_log_lik
        )

        if np.isposinf(self.log_c):
            raise ValueError(
                "Normalizing constant is positive infinite. Unnormalized posterior is zero everywhere."  # pylint: disable=line-too-long
            )

    def extend(self, axis: int | str, step: float, extend_end: bool = True):
        """
        Extend the grid on either side of the specified axis.
        """
        # Convert axis names to indices
        axis = self._get_axis_idx(axis)

        if extend_end:
            new_margin_entry = self.axes[axis].extend_end(step)
        else:
            new_margin_entry = self.axes[axis].extend_begin(step)

        new_entries = [iter(axis) for axis in self.axes]
        new_entries[axis] = [new_margin_entry]

        # Calculate the new hypercubes
        for entries in itertools.product(*new_entries):
            key, vals = Grid._parse_axes_entries(*entries)
            self.grid_data[key] = vals

            # Calculate log prior, log likelihood, log hypercube volume, and log posterior
            self._calculate_logs(key, self.grid_data[key])
            self._shifted_prior_x_lik_sum += np.exp(
                self.grid_data[key]["_log_prior_log_lik"] - self._max_log_prior_log_lik
            )

        self._update_normalizer()  # Update self.log_c, the normalizing constant

    def _calculate_logs(self, key, vals):
        """
        Calculate log prior, log likelihood, log hypercube volume, and log posterior
        and store them in `vals` in place.
        """
        vals["log_lik"] = self.log_lik(vals["mid"])
        vals["log_prior"] = np.sum(np.log(vals["prior"]))
        vals["log_volume"] = np.sum(np.log(vals["end"] - vals["begin"]))
        if 0 in vals["end"] - vals["begin"]:
            logger.warning("Zero volume hypercube at %s", key)
            logger.warning("Begin: %s", vals["begin"])
            logger.warning("End: %s", vals["end"])
        vals["_log_prior_log_lik"] = vals["log_lik"] + vals["log_prior"]

    def approx_log_posterior(self, point: Iterable[float]) -> float:
        """
        Approximate log posterior with the midpoint log posterior of the corresponding hypercube

        TODO: Iterates over all hypercubes. Implement a more efficient algorithm.
        """
        assert len(point) == len(self.axes), "Dimension mismatch."
        for key in self.grid_data:
            if all((key[i][0] <= point[i] <= key[i][1]) for i in range(len(point))):
                return self.get_log_post(key)

        raise ValueError("Point not in the grid.")

    def _get_hypercubes_from_margin(
        self, axis: int | str, idx: int
    ) -> Iterable[HypercubeKey]:
        """
        Get the hypercubes that contain the indexed interval on an axis.
        """
        axis = self._get_axis_idx(axis)

        interval_begin, interval_end = (
            self.axes[axis].begin[idx],
            self.axes[axis].end[idx],
        )

        hypercubes = []
        for key in self.grid_data:
            begin, end = key[axis]
            if (begin - interval_begin <= GRID_TOL) and (
                interval_end - end <= GRID_TOL
            ):
                hypercubes.append(key)

        return hypercubes

    def approx_marginal_log_posterior(self, axis: int | str, idx: int) -> float:
        """
        Approximate marginal posterior for the specified axis at the indexed interval.
        Note that the indexed interval does not have subintervals by construction.
        However, there could be hypercubes that contain the indexed interval.
        """
        axis = self._get_axis_idx(axis)

        log_interval_size = np.log(
            self.axes[axis].end[idx] - self.axes[axis].begin[idx]
        )

        post_lst = []
        for key in self._get_hypercubes_from_margin(axis, idx):
            hypercube = self.grid_data[key]
            log_hypercube_edge = np.log(
                hypercube["end"][axis] - hypercube["begin"][axis]
            )
            post_lst.append(hypercube["_log_prior_log_lik"] - log_hypercube_edge)

        return logsumexp(np.array(post_lst)) + log_interval_size + self.log_c

    def marginal_log_posterior(self):
        """
        Calculate the marginal log posterior for each axis.
        """
        for axis_idx, axis in enumerate(self.axes):
            marginal_log_post = []
            for idx in range(len(axis)):
                marginal_log_post.append(
                    self.approx_marginal_log_posterior(axis_idx, idx)
                )
            axis.log_post_val = marginal_log_post

        return {axis.name: axis.log_post_val for axis in self.axes}

    def marginal_posterior_summary(
        self, reevaluate: bool = False, quantiles: Iterable[float] = (0.25, 0.5, 0.75)
    ):
        """
        Summarize the marginal posterior for each axis.
        """
        if reevaluate or not hasattr(self.axes[0], "post_val"):
            self.marginal_log_posterior()

        res = {}
        for axis in self.axes:
            try:
                res[axis.name] = self._get_axis_marginal_posterior_summary(
                    axis, quantiles
                )
            except ValueError as e:
                if "Log posterior is -inf everywhere." in str(e):
                    res[axis.name] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "quantiles": [np.nan] * len(quantiles),
                        "map": np.nan,
                    }
                else:
                    raise e
        return res

    @staticmethod
    def _get_axis_marginal_posterior_summary(
        axis: GridMargin, quantiles: Iterable[float]
    ):
        """
        Summarize the marginal posterior for a single axis.
        Weighted mean, standard deviation, and quantiles.
        """
        if not hasattr(axis, "log_post_val"):
            raise ValueError("Marginal posterior not calculated.")

        try:
            weighted_mean = np.average(axis.mid, weights=np.exp(axis.log_post_val))
        except ZeroDivisionError as e:
            raise ValueError("Log posterior is -inf everywhere.") from e

        weighted_std = np.sqrt(
            np.average(
                (axis.mid - weighted_mean) ** 2, weights=np.exp(axis.log_post_val)
            )
        )

        weighted_quantiles = weighted_quantile(
            axis.begin, axis.end, np.exp(axis.log_post_val), quantiles
        )

        max_post_idx = np.argmax(axis.log_post_val)
        max_post = axis.mid[max_post_idx]

        return {
            "mean": weighted_mean,
            "std": weighted_std,
            "quantiles": weighted_quantiles,
            "map": max_post,
        }

    def adapt_boundaries(
        self,
        log_post_thres: float,
        steps: Optional[dict[int | str, float]] = None,
        fixed_axes: Optional[list[int | str]] = None,
    ):
        """
        Adaptively expand the grid boundaries,
        so that the boundary marginal log posterior is below the threshold.
        """
        if fixed_axes is None:
            fixed_axes = []

        # Convert axis names to indices
        fixed_axes = [self._get_axis_idx(axis) for axis in fixed_axes]

        if steps is None:
            # Default extension step to half the original step size
            steps = {idx: axis.step * 0.5 for idx, axis in enumerate(self.axes)}
        else:
            for idx, axis in enumerate(self.axes):
                if idx in fixed_axes:
                    continue
                if idx not in steps:
                    steps[idx] = steps.get(axis.name, axis.step * 0.5)

        is_ready = False
        while not is_ready:
            is_ready = True
            for axis_idx, axis in enumerate(self.axes):
                if axis_idx in fixed_axes:
                    continue

                # Check the left boundary
                if self.approx_marginal_log_posterior(axis_idx, 0) > log_post_thres:
                    logger.debug(
                        "%s head, %.4f",
                        axis.name,
                        self.approx_marginal_log_posterior(axis_idx, 0),
                    )
                    self.extend(axis_idx, steps[axis_idx], extend_end=False)
                    is_ready = False
                    logger.debug(
                        "Extended %s on the left by %.4f", axis.name, steps[axis_idx]
                    )

                # Check the right boundary
                if self.approx_marginal_log_posterior(axis_idx, -1) > log_post_thres:
                    logger.debug(
                        "%s tail, %.4f",
                        axis.name,
                        self.approx_marginal_log_posterior(axis_idx, -1),
                    )
                    self.extend(axis_idx, steps[axis_idx], extend_end=True)
                    is_ready = False
                    logger.debug(
                        "Extended %s on the right by %.4f", axis.name, steps[axis_idx]
                    )

        self._boundary_adapted = True

    def split(self, key: HypercubeKey, n: int):
        """
        Split the specified hypercube into n^d smaller hypercubes, in place.
        """

        # First, split the entries of each axis, if necessary
        hypercube_data = self.grid_data[key]
        split_entries = []
        for axis, begin, end in zip(
            self.axes, hypercube_data["begin"], hypercube_data["end"]
        ):
            finer_entries = axis.split_interval(begin, end, n)

            split_entries.append(finer_entries)

        # Drop the old hypercube
        old_hypercube = self.grid_data.pop(key)
        self._shifted_prior_x_lik_sum -= np.exp(
            old_hypercube["_log_prior_log_lik"] - self._max_log_prior_log_lik
        )

        # Calculate the new hypercubes
        for entries in itertools.product(*split_entries):
            key, vals = Grid._parse_axes_entries(*entries)
            self.grid_data[key] = vals

            # Calculate log prior, log likelihood, log hypercube volume, and log posterior
            self._calculate_logs(key, self.grid_data[key])
            self._shifted_prior_x_lik_sum += np.exp(
                self.grid_data[key]["_log_prior_log_lik"] - self._max_log_prior_log_lik
            )

        self._update_normalizer()  # Update self.log_c, the normalizing constant

        return split_entries

    def adapt_split(self, log_post_thres: float, n: int = 2, min_size: float = 1e-7):
        """
        Adaptively split the grid so that each hypercube has a log posterior below the threshold.
        Order independent.
        """

        if not self._boundary_adapted:
            logger.warning("Boundary adaptation not performed.")

        min_log_vol = len(self.axes) * np.log(min_size)

        self._adapt_split_global(log_post_thres, n, min_log_vol)

    def _adapt_split_global(self, log_post_thres: float, n: int, min_log_vol: float):
        """
        Globally scan for not-fine-enough hypercubes, if any, split them.
        """

        _hypercubes_to_be_split = []

        # Global scan
        for key, data in self.grid_data.items():

            # Check if the log posterior + log vol is above the threshold
            # Limit the minimum size of the hypercube
            if (self.log_c + data["_log_prior_log_lik"] <= log_post_thres) or (
                data["log_volume"] < min_log_vol
            ):
                continue

            _hypercubes_to_be_split.append(key)

        if not _hypercubes_to_be_split:
            # All hypercubes are fine enough
            logger.info("Split complete.")
            return

        # Split troublesome hypercubes
        logger.info("%d hypercubes to be split", len(_hypercubes_to_be_split))
        for key in _hypercubes_to_be_split:
            logger.debug("Splitting hypercube at %s", key)
            self.split(key, n)

        # Scan again
        self._adapt_split_global(log_post_thres, n, min_log_vol)

    def _sample_hcube(self, n: int) -> Iterable[HypercubeKey]:
        """
        Sample from the posterior distribution.

        Hypercubes are chosen based on the posterior over themselves,
        then samples are drawn uniformly from the hypercube.

        samples: n * d * 2 (start, end)
        """

        population = self.grid_data.keys()
        p = np.exp(
            np.array([hcube["_log_prior_log_lik"] for hcube in self.grid_data.values()])
            + self.log_c
        )
        samples = self._rng.choice(list(population), p=p, size=n, replace=True)

        return samples

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample from the posterior distribution.

        res: n * d
        """

        if rng is not None:
            self._rng = rng

        selected_cubes = self._sample_hcube(n)  # n * d * 2

        res = self._rng.uniform(selected_cubes[..., 0], selected_cubes[..., 1])

        return res

    def seed(self, seed: int):
        """
        Seed the random number generator.
        """
        self._rng = np.random.default_rng(seed)


def weighted_quantile(
    begin: Iterable[float],
    end: Iterable[float],
    weights: Iterable[float],
    quantile: float | Iterable[float],
) -> float:
    """
    Calculate the weighted median of a list of intervals.
    Quantiles will be sorted automatically.
    """
    if isinstance(quantile, float):
        quantile = [quantile]
    else:
        assert len(quantile) > 0, "Quantiles must be non-empty."
        quantile = sorted(quantile, reverse=True)

    if (quantile[-1] < 0) or (quantile[0] > 1):
        raise ValueError("Quantiles must be in [0, 1].")

    intervals = sorted(zip(begin, end), key=lambda x: x[0])
    weights = weights / sum(weights)

    res = []

    curr_quantile = quantile.pop()
    cum_weight = 0
    for (b, e), w in zip(intervals, weights):
        cum_weight += w
        while curr_quantile <= cum_weight:
            eta = (cum_weight - curr_quantile) / w
            res.append(eta * b + (1 - eta) * e)
            if quantile:
                curr_quantile = quantile.pop()
            else:
                return res

    logger.warning("This should not happen.")
    return res
