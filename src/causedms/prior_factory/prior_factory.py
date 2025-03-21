"""
Implements prior generation for the causalDMS model
"""

from causedms.dms_data import DMSData

from .prior_generator import generate_prior
from .plot import plot_histogram_with_gmm


class PriorFactory:
    def __init__(self, data: DMSData):
        self.data = data

    @property
    def prior(self, *args, regenerate: bool = False, **kwargs):
        """
        x_name: str = "beta_hat_1",
        y_name: str = "beta_hat_2",
        x_se_name: str = "se_hat_1",
        x_gmm_n_components: int = 2,
        """
        if self._prior is None or regenerate:
            self._prior = generate_prior(*args, **kwargs)

        return self._prior

    def plot_histogram_with_gmm(self, component: int = 2, ax=None):
        return plot_histogram_with_gmm(self, component, ax)
