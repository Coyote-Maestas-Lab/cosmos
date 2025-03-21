"""
Functions to generate prior parameters.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

PRIOR_MULTIPLIER = 50


def parameter_priors(df: pd.DataFrame, x_name: str, y_name: str) -> dict[str, float]:
    """
    Generate prior parameters for the DMS model,
    linear regression based.
    """
    # Get the mean and standard deviation of the beta_hat columns
    x = df[[x_name]]
    y = df[y_name]
    reg = LinearRegression().fit(x, y)

    # print("Intercept:", reg.intercept_)
    # print("Slope:", reg.coef_[0])
    # print("R^2:", reg.score(X, y))

    c_gamma_hat = np.abs(reg.coef_[0] * PRIOR_MULTIPLIER)
    residual = y - reg.predict(x)
    c_tau_hat = np.sqrt(np.var(residual)) * PRIOR_MULTIPLIER

    return {"c_gamma_hat": c_gamma_hat, "c_tau_hat": c_tau_hat}


def mixture_params(
    x: np.ndarray | pd.DataFrame, n_components: int = 2, seed: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Gaussian Mixture parameters and predicted labels.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(x)
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    means = np.squeeze(means)
    covariances = np.squeeze(covariances)
    weights = np.squeeze(weights)

    if n_components == 1:
        means = np.array([means])
        covariances = np.array([covariances])
        weights = np.array([weights])

    # put the estimated parameters into a dictionary
    gmm_params = {
        "mu_k_hat": means,
        "sigma2_k_hat": covariances,
        "pi": weights,
    }
    return gmm_params, gmm.predict(x)


def integrated_params_prior(
    df: pd.DataFrame, x_gmm_n_components: int, x_name: str, y_name: str, x_se_name: str
) -> dict[str, np.ndarray]:
    """
    Prior for parameters after integration
    """
    # TODO: Test if mixture matters in simulatoin
    # for both abundance and surface expresssion, we use a mixture of two normal distributions
    gmm_params_M, df[f"{x_name}_gmm_class"] = mixture_params(
        df[[x_name]], x_gmm_n_components
    )
    sigma2_k_hat = gmm_params_M["sigma2_k_hat"]

    
    # Adjust by square of geometric mean of se_hat_1
    if x_se_name not in df.columns:
        sigma2_k_hat_adjust = np.zeros_like(sigma2_k_hat)
    sigma2_k_hat_adjust = np.exp(np.mean(np.log(df[x_se_name])) * 2)
    if (sigma2_k_hat <= sigma2_k_hat_adjust).any():
        logging.warning("sigma2_k_hat adjustment failed. Variance set to 0.")
    gmm_params_M["sigma2_k_hat_adj"] = sigma2_k_hat - sigma2_k_hat_adjust
    # If any is negative, set to 0
    gmm_params_M["sigma2_k_hat_adj"][gmm_params_M["sigma2_k_hat_adj"] < 0] = 0

    gmm_params_theta = {
        "mu_k_hat": np.array([0]),
        "sigma2_k_hat": np.array([np.var(df[y_name]) / 4]),
        "pi": np.array([1]),
    }

    mu_m = gmm_params_M["mu_k_hat"]
    mu_theta = np.full_like(mu_m, gmm_params_theta["mu_k_hat"])
    sigma2_m = gmm_params_M["sigma2_k_hat"]
    sigma2_m_adjusted = gmm_params_M["sigma2_k_hat_adj"]
    sigma2_theta = np.full_like(sigma2_m, gmm_params_theta["sigma2_k_hat"])

    # compute pi for each group
    pi = {}
    for group_idx in df["group_new"].unique():
        sub_df = df[df["group_new"] == group_idx]
        pi[group_idx] = [
            np.sum(sub_df[f"{x_name}_gmm_class"] == val) / len(sub_df)
            for val in range(x_gmm_n_components)
        ]

    return {
        "mu_m": mu_m,
        "sigma2_m": sigma2_m,
        "mu_theta": mu_theta,
        "sigma2_theta": sigma2_theta,
        "pi": pi,
        "sigma2_m_adjusted": sigma2_m_adjusted,
    }


def generate_prior(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    x_se_name: str,
    x_gmm_n_components: int,
) -> dict[str, np.ndarray | float]:
    """
    Generate prior for the causalDMS model:
    - priors for gamma and tau - c_gamma_hat, c_tau_hat
    - priors for beta_1
    """
    return parameter_priors(data, x_name, y_name) | integrated_params_prior(
        data, x_gmm_n_components, x_name, y_name, x_se_name
    )
