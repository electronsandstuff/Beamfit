import numpy as np
from scipy import special, optimize as opt


def fit_gaussian_1d(image):
    """
    Integrates the image across each axis and fits a Gaussian function with offset to each axis.  Predicts the 2D
    Gaussian of best fit from the resulting data.

    Note:  This function only works for Gaussians with positive amplitude at the moment.  Invert any images where the
    Gaussian points downward.
    """
    # Make our fit function
    def fitfun(x, mu, sigma, a, c):
        return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2) + c

    # Perform the fit to each axis
    fits = []
    for axis in [0, 1]:
        # Integrate the profile onto one axis and make the x values
        y = np.sum(image, axis=axis)
        x = np.arange(y.size)

        # Run the nonlinear fit
        ymax = y.max()
        ymin = y.min()
        p0 = np.array([np.argmax(y), np.sum(y > ((ymax - ymin) * np.exp(-1/8) + ymin)), ymax - ymin, ymin])
        fits.append(opt.curve_fit(fitfun, x, y, p0))

    # Convert to estimates of 2D parameters
    xfit = fits[1][0]
    yfit = fits[0][0]
    s2 = np.sqrt(2)
    x_amplitude_scale = 2/(special.erf((image.shape[1] - yfit[0])/yfit[1]/s2) - special.erf(-yfit[0]/yfit[1]/s2))
    x_amplitude_scale /= yfit[1] * np.sqrt(2 * np.pi)
    y_amplitude_scale = 2/(special.erf((image.shape[0] - xfit[0])/xfit[1]/s2) - special.erf(-xfit[0]/xfit[1]/s2))
    y_amplitude_scale /= xfit[1] * np.sqrt(2 * np.pi)
    h = np.array([
        xfit[0], yfit[0], xfit[1] ** 2, 0.0, yfit[1] ** 2, 1.0,
        (xfit[2] * x_amplitude_scale + yfit[2] * y_amplitude_scale) / 2,
        (xfit[3] / image.shape[1] + yfit[3] / image.shape[0]) / 2
    ])

    # Return it
    return h
