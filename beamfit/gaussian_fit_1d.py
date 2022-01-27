import numpy as np
from scipy import special, optimize as opt

from .utils import AnalysisMethod, SuperGaussianResult


# TODO: integrate pixel weights
class GaussianProfile1D(AnalysisMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __fit__(self, image, image_sigmas=None):
        """
        Integrates the image across each axis and fits a Gaussian function with offset to each axis.  Predicts the 2D
        Gaussian of best fit from the resulting data.

        Note:  This function only works for Gaussians with positive amplitude at the moment.  Invert any images where the
        Gaussian points downward.
        """
        def fitfun(x, mu, sigma, a, c):
            return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2) + c

        # Fit the projection on each axis
        fits = []
        for axis in range(2):
            # Integrate the profile onto one axis and make the x values
            y = np.sum(image, axis=axis)

            # Estimate the fit parameters
            ymax = y.max()
            ymin = y.min()
            p0 = np.array([np.argmax(y), np.sum(y > ((ymax - ymin) * np.exp(-1 / 8) + ymin)), ymax - ymin, ymin])

            # Run the fit
            fits.append(opt.curve_fit(fitfun, np.arange(y.size), y, p0)[0])
        yfit, xfit = fits

        # Calculate scaling values for x/y for amplitude. This is required since gaussian can clip on the edge of the
        # image meaning the integral involves the error function
        sy, sx = [2/(special.erf((s-f[0])/f[1]/np.sqrt(2)) - special.erf(-f[0]/f[1]/np.sqrt(2)))/(f[1]*np.sqrt(2*np.pi))
                  for s, f in zip(image.shape, reversed(fits))]

        # Return it
        return SuperGaussianResult(
            mu=np.array([xfit[0], yfit[0]]),
            sigma=np.array([[xfit[1]**2, 0.0], [0.0, yfit[1]**2]]),
            a=(xfit[2]*sx + yfit[2]*sy)/2,
            o=(xfit[3]/image.shape[1] + yfit[3]/image.shape[0])/2
        )


def fit_gaussian_1d(image):  # Backwards compatibility
    return GaussianProfile1D().fit(image).h
