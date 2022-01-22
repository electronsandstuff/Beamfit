import numpy as np
import scipy.optimize as opt
from gaussufunc import supergaussian, supergaussian_grad

from . import factory
from .utils import AnalysisMethod, SuperGaussianResult


def h_to_hb(h):
    return np.array([
        h[0],
        h[1],
        np.sqrt(h[2]),
        h[3],
        np.sqrt(h[2] * h[4] - h[3] ** 2),
        np.sqrt(h[5]),
        h[6],
        h[7]
    ])


def h_to_hb_grad(hb, grad):
    return np.array([
        grad[0, :],
        grad[1, :],
        2 * hb[2] * grad[2, :] - ((2 * (hb[3] ** 2 + hb[4] ** 2)) / hb[2] ** 3) * grad[4, :],
        grad[3, :] + (2 * hb[3]) / hb[2] ** 2 * grad[4, :],
        (2 * hb[4]) / hb[2] ** 2 * grad[4, :],
        2 * hb[5] * grad[5, :],
        grad[6, :],
        grad[7, :]
    ])


def hb_to_h(hb):
    return np.array([
        hb[0],
        hb[1],
        hb[2] ** 2,
        hb[3],
        (hb[3] ** 2 + hb[4] ** 2) / hb[2] ** 2,
        hb[5] ** 2,
        hb[6],
        hb[7]
    ])


def fitfun_bounded(xdata, mux, muy, sxx, vxy, sdt, sn, a, o):
    h = hb_to_h(np.array([mux, muy, sxx, vxy, sdt, sn, a, o]))
    return supergaussian(xdata[0], xdata[1], *h)


def fitjac_bounded(xdata, mux, muy, sxx, vxy, sdt, sn, a, o):
    hb = np.array([mux, muy, sxx, vxy, sdt, sn, a, o])
    h = hb_to_h(hb)
    return h_to_hb_grad(hb, np.array(supergaussian_grad(xdata[0], xdata[1], *h))).T


class SuperGaussian(AnalysisMethod):
    def __init__(self, predfun="GaussianProfile1D", predfun_args=None, maxfev=100, **kwargs):
        super().__init__(**kwargs)
        if predfun_args is None:
            predfun_args = {}
        self.predfun = predfun
        self.predfun_args = predfun_args
        self.maxfev = maxfev

    def get_name(self):
        return 'SuperGaussian'

    def __fit__(self, image):
        # Find an initial guess of the parameters with a fast method
        h0 = factory.create(self.predfun, **self.predfun_args).fit(image).h

        # Get the x and y data for the fit
        m, n = np.mgrid[:image.shape[0], :image.shape[1]]
        x = np.vstack((m[~image.mask], n[~image.mask]))
        y = np.array(image[~image.mask])

        # TODO: add different options for bounded parameter fit: log
        hb, _ = opt.curve_fit(fitfun_bounded, x, y, h_to_hb(h0), jac=fitjac_bounded, maxfev=self.maxfev)
        h = hb_to_h(hb)

        # Return the fit and the covariance variance matrix
        return SuperGaussianResult(
            mu=np.array([h[0], h[1]]),
            sigma=np.array([[h[2], h[3]], [h[3], h[4]]]),
            n=h[5],
            a=h[6],
            o=h[7]
        )


def fit_supergaussian(image, image_weights=None, prediction_func="2D_linear_Gaussian", sigma_threshold=3,
                      sigma_threshold_guess=1, smoothing=5, maxfev=100):  # Backwards compatibility
    predfun = {'2D_linear_Gaussian': 'GaussianLinearLeastSquares', '1D_Gaussian': 'GaussianProfile1D'}[prediction_func]
    return SuperGaussian(predfun=predfun, predfun_args={'sigma_threshold': sigma_threshold_guess},
                         sigma_threshold=sigma_threshold, maxfev=maxfev).fit(image).h, np.identity(8)
