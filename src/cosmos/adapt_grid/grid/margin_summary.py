"""
Summary of the marginal distribution of an axis
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MarginSummary:
    mean: float
    std: float
    quantiles: list[float]
    max_post: float  # Maximum a posteriori estimate

    @classmethod
    def null(cls, n_quantiles: int = 0):
        return cls(
            mean=np.nan,
            std=np.nan,
            quantiles=[np.nan] * n_quantiles,
            max_post=np.nan,
        )

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "mean": self.mean,
            "std": self.std,
            "quantiles": self.quantiles,
            "max_post": self.max_post,
        }
