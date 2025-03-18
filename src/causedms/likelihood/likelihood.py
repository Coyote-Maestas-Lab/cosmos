"""
Log likelihood function for the model, 
"""

import numpy as np
from scipy.stats import multivariate_normal


def gen_log_lik_individual(
    beta_m_hat, beta_y_hat, mu_m, mu_theta, sigma2_m, sigma2_theta, s2_m, s2_y, pi
):
    """
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

    def log_lik_individual_impl(gamma: float, tau: float) -> float:

        mean_term = np.array([mu_m, gamma * mu_m + tau + mu_theta]).T
        cov_term = np.array(
            [
                [
                    [sigma2_m_k, gamma * sigma2_m_k],
                    [
                        gamma * sigma2_m_k,
                        gamma * gamma * sigma2_m_k + sigma2_theta_k,
                    ],
                ]
                for sigma2_m_k, sigma2_theta_k in zip(sigma2_m, sigma2_theta)
            ]
        )

        def single_call(
            beta_m_j: float, beta_y_j: float, s2_m_j: float, s2_y_j: float
        ) -> float:
            gauss_mix_cumulator = 0.0
            for pi_k, mean, cov in zip(pi, mean_term, cov_term):
                gauss_mix_cumulator += pi_k * multivariate_normal.pdf(
                    [beta_m_j, beta_y_j],
                    mean=mean,
                    cov=cov + np.diag([s2_m_j, s2_y_j]),
                )
            return np.log(gauss_mix_cumulator)

        return np.array(
            [single_call(*args) for args in zip(beta_m_hat, beta_y_hat, s2_m, s2_y)]
        )

    return log_lik_individual_impl


# For each i:
def gen_log_lik_all(
    beta_m_hat, beta_y_hat, mu_m, mu_theta, sigma2_m, sigma2_theta, s2_m, s2_y, pi
):
    """
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
    log_lik_individual = gen_log_lik_individual(
        beta_m_hat, beta_y_hat, mu_m, mu_theta, sigma2_m, sigma2_theta, s2_m, s2_y, pi
    )

    def log_lik_all_impl(gamma: float, tau: float) -> float:

        return log_lik_individual(gamma, tau).sum()

    return log_lik_all_impl
