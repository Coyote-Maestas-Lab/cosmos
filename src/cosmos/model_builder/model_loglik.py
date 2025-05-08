"""
Generate model log likelihoods
"""

import numpy as np

from cosmos.likelihood.double import DoublePhenotypeModel


def gen_model_log_likelihood(
    has_gamma: bool, has_tau: bool, model: DoublePhenotypeModel
):
    """
    Given a model and parameters to mask, return a function that computes the log likelihood
    """
    match (has_gamma, has_tau):
        case (True, True):

            def loglik(params: np.ndarray) -> float:
                gamma, tau = params
                return model.log_lik_all(gamma, tau)

        case (True, False):

            def loglik(params: np.ndarray) -> float:
                gamma = params[0]
                return model.log_lik_all(gamma, 0.0)

        case (False, True):

            def loglik(params: np.ndarray) -> float:
                tau = params[0]
                return model.log_lik_all(0.0, tau)

        case (False, False):

            def loglik(params: np.ndarray) -> float:
                return model.log_lik_all(0.0, 0.0)

    return loglik
