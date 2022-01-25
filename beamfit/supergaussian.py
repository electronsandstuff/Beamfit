import numpy as np
import scipy.optimize as opt
from gaussufunc import supergaussian, supergaussian_grad

from . import factory
from .utils import AnalysisMethod, SuperGaussianResult


class SigmaTrans:
    def __init__(self):
        pass

    def forward(self, h):
        raise NotImplementedError

    def reverse(self, h):
        raise NotImplementedError

    def forward_grad(self, h, grad):
        raise NotImplementedError


class SuperGaussian(AnalysisMethod):
    def __init__(self, predfun="GaussianProfile1D", predfun_args=None, sig_param='LogCholesky', sig_param_args=None,
                 maxfev=100, **kwargs):
        super().__init__(**kwargs)
        if sig_param_args is None:
            sig_param_args = {}
        if predfun_args is None:
            predfun_args = {}
        self.predfun = factory.create('analysis', predfun, **predfun_args)
        self.predfun_args = predfun_args
        self.maxfev = maxfev
        self.sig_param = factory.create('sig_param', sig_param, **sig_param_args)

    def __fit__(self, image):
        lo, hi = image.min(), image.max()  # Normalize image
        image = (image - lo)/(hi - lo)

        # Get the x and y data for the fit
        m, n = np.mgrid[:image.shape[0], :image.shape[1]]
        x = np.vstack((m[~image.mask], n[~image.mask]))
        y = np.array(image[~image.mask])

        # Setup the fitting functions
        def h_to_theta(h):
            # Break out the variables
            mu = h[:2]
            sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
            n = h[5]
            a = h[6]
            o = h[7]

            # Transform parameters
            st = self.sig_param.forward(sigma)  # The sigma parameterization
            nt = np.log(n)  # n is positive
            return np.array([mu[0], mu[1], st[0], st[1], st[2], nt, a, o])

        def theta_to_h(theta):
            # Break out the variables
            mu = theta[:2]
            st = theta[2:5]
            nt = theta[5]
            a = theta[6]
            o = theta[7]

            # Transform sigma and n back
            sigma = self.sig_param.reverse(st)
            n = np.exp(nt)
            return np.array([mu[0], mu[1], sigma[0, 0], sigma[0, 1], sigma[1, 1], n, a, o])

        def theta_to_h_grad(theta):
            # Break out the parameters
            st = theta[2:5]
            nt = theta[5]

            # Construct the jacobian
            j = np.identity(8)
            j[2:5, 2:5] = self.sig_param.reverse_grad(st)  # Add the sigma parameterization gradient
            j[5, 5] = np.exp(nt)
            return j

        def fitfun(xdata, *theta):
            return supergaussian(xdata[0], xdata[1], *theta_to_h(theta))

        def fitfun_grad(xdata, *theta):
            # TODO: I should really change supergaussian_grad's output to be the transpose of what it currently is
            # TODO: this would make it line up with the convention for jacobian functions
            jacf = theta_to_h_grad(theta)
            jacg = np.array(supergaussian_grad(xdata[0], xdata[1], *theta_to_h(theta))).T
            return jacg @ jacf  # Chain rule

        theta_opt, _ = opt.curve_fit(fitfun, x, y, h_to_theta(self.predfun.fit(image).h), jac=fitfun_grad,
                                     maxfev=self.maxfev)
        h_opt = theta_to_h(theta_opt)

        # Return the fit and the covariance variance matrix
        return SuperGaussianResult(
            mu=np.array([h_opt[0], h_opt[1]]),
            sigma=np.array([[h_opt[2], h_opt[3]], [h_opt[3], h_opt[4]]]),
            n=h_opt[5],
            a=h_opt[6]*(hi - lo),
            o=h_opt[7] + lo
        )


def fit_supergaussian(image, image_weights=None, prediction_func="2D_linear_Gaussian", sigma_threshold=3,
                      sigma_threshold_guess=1, smoothing=5, maxfev=100):  # Backwards compatibility
    predfun = {'2D_linear_Gaussian': 'GaussianLinearLeastSquares', '1D_Gaussian': 'GaussianProfile1D'}[prediction_func]
    return SuperGaussian(predfun=predfun, predfun_args={'sigma_threshold': sigma_threshold_guess},
                         sigma_threshold=sigma_threshold, maxfev=maxfev).fit(image).h, np.identity(8)
