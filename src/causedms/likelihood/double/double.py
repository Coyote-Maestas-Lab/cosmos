"""
Log likelihood functions for double-phenotype models.
"""

from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class DoublePhenotypeModel(ABC):  # pylint: disable=too-many-instance-attributes
    """
    Log likelihood functions for double-phenotype models.
    """

    def __init__(self, **kwargs):
        """
        Possible kwargs:
        beta_m_hat: J, data
        beta_y_hat: J, data
        mu_m: K, observed quantity
        mu_theta: K, observed quantity
        sigma2_m: K, observed quantity
        sigma2_theta: K, observed quantity
        s2_m: J, observed quantity
        s2_y: J, observed quantity
        pi: K, observed quantity
        """

        self.beta_m_hat = kwargs.get("beta_m_hat")
        self.beta_y_hat = kwargs.get("beta_y_hat")
        self.mu_m = kwargs.get("mu_m")
        self.mu_theta = kwargs.get("mu_theta")
        self.sigma2_m = kwargs.get("sigma2_m")
        self.sigma2_theta = kwargs.get("sigma2_theta")
        self.s2_m = kwargs.get("s2_m")
        self.s2_y = kwargs.get("s2_y")
        self.pi = kwargs.get("pi")

    @abstractmethod
    def log_lik_individual(self, gamma: float, tau: float) -> np.ndarray:
        """
        Log likelihood for a single individual.
        """

    def log_lik_all(self, gamma: float, tau: float) -> float:
        """
        Log likelihood for all individuals.
        """

        return self.log_lik_individual(gamma, tau).sum()


class ModelFull(DoublePhenotypeModel):
    """
    Full model
    """

    def __init__(self, **kwargs):
        logger.info("Full model, params:\ngamma, tau")
        super().__init__(**kwargs)

    def log_lik_individual(self, gamma: float, tau: float) -> np.ndarray:
        """
        Return a series of log likelihoods, one for each individual.
        """

        mean_term = np.array([self.mu_m, gamma * self.mu_m + tau + self.mu_theta]).T
        cov_term = np.array(
            [
                [
                    [sigma2_m_k, gamma * sigma2_m_k],
                    [
                        gamma * sigma2_m_k,
                        gamma * gamma * sigma2_m_k + sigma2_theta_k,
                    ],
                ]
                for sigma2_m_k, sigma2_theta_k in zip(self.sigma2_m, self.sigma2_theta)
            ]
        )

        def single_call(
            beta_m_j: float, beta_y_j: float, s2_m_j: float, s2_y_j: float
        ) -> float:
            gauss_mix_cumulator = 0.0
            for pi_k, mean, cov in zip(self.pi, mean_term, cov_term):
                gauss_mix_cumulator += pi_k * multivariate_normal.pdf(
                    [beta_m_j, beta_y_j],
                    mean=mean,
                    cov=cov + np.diag([s2_m_j, s2_y_j]),
                )
            return np.log(gauss_mix_cumulator)

        return np.array(
            [
                single_call(*args)
                for args in zip(self.beta_m_hat, self.beta_y_hat, self.s2_m, self.s2_y)
            ]
        )


class ModelSkeleton(DoublePhenotypeModel):
    """
    Skeleton model
    """

    def __init__(self, **kwargs):
        logger.info("Skeleton model, params:\ngamma, tau")
        super().__init__(**kwargs)

    def log_lik_individual(self, _gamma: float, tau: float) -> np.ndarray:
        """
        Return a series of log likelihoods, one for each individual.
        """

        mean_term = np.array([0.0, tau])

        def single_call(
            beta_m_j: float, beta_y_j: float, s2_m_j: float, s2_y_j: float
        ) -> float:
            return multivariate_normal.logpdf(
                [beta_m_j, beta_y_j],
                mean=mean_term,
                cov=np.diag([s2_m_j, s2_y_j]),
            )

        return np.array(
            [
                single_call(*args)
                for args in zip(self.beta_m_hat, self.beta_y_hat, self.s2_m, self.s2_y)
            ]
        )
