"""
Implements prior generation for the Cosmos model
"""

from causedms.dms_data import DMSData

from .plot import _plot_histogram_with_gmm
from .prior_generator import generate_prior


class PriorFactory:
    def __init__(self, data: DMSData):
        self._data = data

    def gen_prior(self, *args, regenerate: bool = False, **kwargs):
        """
        x_name: str = "beta_hat_1",
        y_name: str = "beta_hat_2",
        x_se_name: str = "se_hat_1",
        x_gmm_n_components: int = 2,
        """
        if not hasattr(self, "_prior") or regenerate:
            self._prior = generate_prior(self.data, *args, **kwargs)

        return self._prior

    @property
    def prior(self):
        return self._prior

    @property
    def data(self):
        return self._data.data

    @property
    def phenotypes(self):
        return self._data.phenotypes

    def plot_histogram_with_gmm(self, component: int = 2, ax=None):
        return _plot_histogram_with_gmm(self, component, ax)
