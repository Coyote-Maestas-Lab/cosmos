"""
GridMargin is an axis with intervals and a prior over each interval.
Building blocks for Grid.
"""

import bisect
from collections.abc import Callable, Generator, Sequence

import numpy as np
import pandas as pd
from scipy import integrate

from . import GRID_TOL, logger


class GridMargin:
    """
    One-dimensional grid of intervals with a prior on each interval.
    """

    TOL = GRID_TOL

    def __init__(  # pylint: disable=too-many-arguments
        self,
        begin: float,
        end: float,
        n_intervals: int,
        prior: Callable[[float], float],
        name: str = "x",
    ):
        """
        Build a grid of intervals and calculate the integrated prior on each interval.
        """

        self.name = name
        self.prior = prior
        begin_arr, self.step = np.linspace(
            begin, end, n_intervals, endpoint=False, retstep=True
        )

        if self.step < GridMargin.TOL:
            raise ValueError(
                f"Step size {self.step} is smaller than `GridMargin.TOL` {GridMargin.TOL}."
            )

        self.begin = list(begin_arr)
        self.end = list(begin_arr + self.step)
        self.mid = list(begin_arr + self.step * 0.5)

        self.prior_val = [
            integrate.quad(prior, grid_begin, grid_end)[0]
            for grid_begin, grid_end in zip(self.begin, self.end)
        ]

    def __len__(self) -> int:
        return len(self.mid)

    def __str__(self) -> str:
        return f"{self.name} axis ({self.begin[0]}, {self.end[-1]})"

    def __iter__(self) -> Generator[dict[str, float]]:
        for begin, end, mid, prior_val in zip(
            self.begin, self.end, self.mid, self.prior_val
        ):
            yield {"begin": begin, "end": end, "mid": mid, "prior": prior_val}

    def __getitem__(self, idx: int) -> dict[str, float]:
        return {
            "begin": self.begin[idx],
            "end": self.end[idx],
            "mid": self.mid[idx],
            "prior": self.prior_val[idx],
        }

    def extend_begin(self, step: float) -> dict[str, float]:
        """
        Extend the grid on the left side.
        """
        self.end = [self.begin[0]] + self.end
        self.begin = [self.begin[0] - step] + self.begin
        self.mid = [self.begin[0] + step * 0.5] + self.mid
        self.prior_val = [
            integrate.quad(self.prior, self.begin[0], self.end[0])[0]
        ] + self.prior_val

        return self[0]

    def extend_end(self, step: float) -> dict[str, float]:
        """
        Extend the grid on the right side.
        """
        self.begin.append(self.end[-1])
        self.end.append(self.end[-1] + step)
        self.mid.append(self.end[-1] - step * 0.5)
        self.prior_val.append(
            integrate.quad(self.prior, self.begin[-1], self.end[-1])[0]
        )

        return self[-1]

    @staticmethod
    def _bisect_exact_with_tol(x: float, arr: Sequence[float]) -> int:
        """
        Find the index of x in the sorted array arr, exact match with tolerance.
        """
        idx = bisect.bisect_left(arr, x)

        if idx > 0 and (x - arr[idx - 1] < GridMargin.TOL):
            idx -= 1

        return idx

    @staticmethod
    def _round_to_arr_with_tol(
        x: float | Sequence[float], arr: Sequence[float]
    ) -> float | list[float]:
        """
        Round x to the nearest value in arr.
        """
        if isinstance(x, float):
            idx = GridMargin._bisect_exact_with_tol(x, arr)

            if abs(arr[idx] - x) >= GridMargin.TOL:
                raise ValueError(f"{x} not found in array.")

            return arr[idx]

        return [GridMargin._round_to_arr_with_tol(xi, arr) for xi in x]

    def _find_begin(self, begin: float, precise: bool = True) -> int:
        """
        Find which index `begin` is in self.begin
        """
        idx = GridMargin._bisect_exact_with_tol(begin, self.begin)

        if abs(self.begin[idx] - begin) >= GridMargin.TOL:
            if precise:
                raise ValueError(f"{begin} not found in the grid interval begins.")
            logger.warning("%s not found in the grid interval begins.", begin)

        return idx

    def _find_end(self, end: float, precise: bool = True) -> int:
        """
        Find which index `end` is in self.end
        """
        idx = GridMargin._bisect_exact_with_tol(end, self.end)

        if abs(self.end[idx] - end) >= GridMargin.TOL:
            if precise:
                raise ValueError(f"{end} not found in the grid interval ends.")
            logger.warning("%s not found in the grid interval ends.", end)

        return idx

    def _find_mid(self, mid: float, precise: bool = True) -> int:
        """
        Find which index `mid` is in self.mid
        """
        idx = GridMargin._bisect_exact_with_tol(mid, self.mid)

        if abs(self.mid[idx] - mid) >= GridMargin.TOL:
            if precise:
                raise ValueError(f"{mid} not found in the grid interval mids.")
            logger.warning("%s not found in the grid interval mids.", mid)

        return idx

    def calc_prior(self, begin: float, end: float) -> float:
        """
        Calculate the prior probability of the float interval (begin, end).
        """
        idx_begin = self._find_begin(begin)
        idx_end = self._find_end(end)

        return self._calc_prior(idx_begin, idx_end)

    def _calc_prior(self, idx_begin: int, idx_end: int) -> float:
        """
        Calculate the prior probability of the index interval [idx_begin, idx_end].
        """
        return sum(self.prior_val[idx_begin : idx_end + 1])

    def get_interval(self, begin: float, end: float) -> dict[str, float]:
        """
        Get the interval with the specified begin and end.
        """
        idx_begin = self._find_begin(begin)
        idx_end = self._find_end(end)

        if idx_begin == idx_end:
            return self[idx_begin]

        return {
            "begin": begin,
            "end": end,
            "mid": (begin + end) * 0.5,
            "prior": self._calc_prior(idx_begin, idx_end),
        }

    def split_interval(
        self, begin: float, end: float, n: int
    ) -> list[dict[str, float]]:
        """
        If the interval (begin, end) is not already split, split it into n smaller intervals.

        TODO: Test if both branches are identical.
        """
        begin_idx, end_idx = self._find_begin(begin), self._find_end(end)

        # The interval hasn't been split yet
        if begin_idx == end_idx:
            return self._split_interval_impl(begin_idx, n)

        # If already split, do nothing but return split entries
        split_begins = GridMargin._round_to_arr_with_tol(
            np.linspace(begin, end, n, endpoint=False),
            self.begin,
        )
        split_ends = GridMargin._round_to_arr_with_tol(
            np.linspace(begin, end, n + 1, endpoint=True)[1:],
            self.end,
        )
        return [self.get_interval(b, e) for b, e in zip(split_begins, split_ends)]

    def _split_interval_impl(self, idx: int, n: int) -> list[dict[str, float]]:
        """
        Split the interval at index idx into n smaller intervals.
        """

        old_start, old_end = self.begin[idx], self.end[idx]
        finer_margin = GridMargin(old_start, old_end, n, self.prior)

        def insert_at_index(original: list, i: int, other: list) -> list:
            if (i < 0) or (i >= len(original)):
                raise IndexError("Index out of range.")
            return original[:i] + other + original[i + 1 :]

        self.begin = insert_at_index(self.begin, idx, finer_margin.begin)

        self.end = insert_at_index(self.end, idx, finer_margin.end)
        self.mid = insert_at_index(self.mid, idx, finer_margin.mid)
        self.prior_val = insert_at_index(self.prior_val, idx, finer_margin.prior_val)

        return list(finer_margin)

    def interval_idx(self, x: float) -> int:
        """
        Find the index of the interval containing x.
        x in (begin[res], end[res]]
        """
        if x <= self.begin[0]:
            return -1
        # Use bisection
        idx: int = np.searchsorted(self.end, x)

        return idx

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the grid to a pandas DataFrame.
        """

        df = pd.DataFrame({"begin": self.begin, "end": self.end, "mid": self.mid})

        return df

    @staticmethod
    def from_pandas(
        df: pd.DataFrame, prior: Callable[[float], float], name: str = "x"
    ) -> "GridMargin":
        """
        Convert a pandas DataFrame to a GridMargin object.
        """
        grid = GridMargin(0, 1, 1, prior, name)
        grid.begin = df["begin"].tolist()
        grid.end = df["end"].tolist()
        grid.mid = df["mid"].tolist()

        grid.step = (df["end"] - df["begin"]).max()

        grid.prior_val = [
            integrate.quad(prior, grid_begin, grid_end)[0]
            for grid_begin, grid_end in zip(grid.begin, grid.end)
        ]

        return grid
