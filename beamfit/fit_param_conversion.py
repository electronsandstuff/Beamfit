import numpy as np
from .utils import SuperGaussianResult


def get_mu_sigma(h, pixel_size):  # For backwards compatibility
    r = SuperGaussianResult(h=h)
    return r.get_mean() * pixel_size, r.get_covariance_matrix() * pixel_size**2


def get_mu_sigma_std(h, c, pixel_size, pixel_size_std):  # For backwards compatibility
    r = SuperGaussianResult(h=h, c=c)
    mu = r.get_mean()
    sigma = r.get_covariance_matrix()
    mu_var = r.get_mean_std() ** 2
    sigma_var = r.get_covariance_matrix_std() ** 2

    # Scale by the pixel size and calculate variances
    pixel_size_var = pixel_size_std**2
    mu_scaled_var = (
        mu_var * pixel_size_var + pixel_size_var * mu**2 + mu_var * pixel_size**2
    )
    pixel_size_squared_var = 4 * pixel_size**2 * pixel_size_var
    sigma_scaled_var = (
        sigma_var * pixel_size_squared_var
        + pixel_size_squared_var * sigma**2
        + sigma_var * pixel_size**4
    )

    # Return them
    return np.sqrt(mu_scaled_var), np.sqrt(sigma_scaled_var)
