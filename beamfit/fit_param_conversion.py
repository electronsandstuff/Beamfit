import numpy as np
import scipy.special as special


def get_mu_sigma(h, pixel_size):
    # Pull out the parameters
    mu = np.array([h[0], h[1]]) * pixel_size

    # Get sigma
    sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
    n = h[5]
    scaling_factor = special.gamma((2 + n) / n) / 2 / special.gamma(1 + 1 / n)
    sigma = sigma * scaling_factor * pixel_size ** 2

    # Return them
    return mu, sigma


def get_mu_sigma_std(h, C, pixel_size, pixel_size_std):
    # Pull out the parameters
    mu = np.array([h[0], h[1]])

    # Get sigma
    sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
    n = h[5]
    scaling_factor = special.gamma((2 + n) / n) / 2 / special.gamma(1 + 1 / n)
    sigma = sigma * scaling_factor

    # Calculate mu's variance
    mu_var = np.array([C[0, 0], C[1, 1]])

    # Calculate Sigma's variance (hack for now to deal w/ uncertainty)
    sigma_var = np.array([[C[2, 2] * 2 * np.sqrt(h[2]), C[3, 3]], [C[3, 3], C[2, 2] * 2 * np.sqrt(h[2])]])

    n = h[5]
    n_var = C[5, 5]
    scaling_factor_deriv = special.gamma((2 + n) / n) / 2 / n ** 2 * special.polygamma(0, 1 + 1 / n)
    scaling_factor_deriv += (1 / n - (2 + n) / n ** 2) * special.gamma((2 + n) / n) * special.polygamma(0,
                                                                                                        (2 + n) / n) / 2
    scaling_factor_deriv /= special.gamma(1 + 1 / n)
    scaling_factor_var = n_var * scaling_factor_deriv ** 2
    sigma_var = sigma_var * scaling_factor_var + sigma ** 2 * scaling_factor_var + scaling_factor ** 2 * sigma_var

    # Scale by the pixel size and calculate variances
    pixel_size_var = pixel_size_std ** 2
    mu_scaled_var = mu_var * pixel_size_var + pixel_size_var * mu ** 2 + mu_var * pixel_size ** 2
    pixel_size_squared_var = 4 * pixel_size ** 2 * pixel_size_var
    sigma_scaled_var = sigma_var * pixel_size_squared_var + pixel_size_squared_var * sigma ** 2 + sigma_var * pixel_size ** 4

    # Return them
    return np.sqrt(mu_scaled_var), np.sqrt(sigma_scaled_var)
