import os
import pickle
import unittest
import numpy as np
import beamfit


def get_mu_sigma_numerical(h):
    """
    Evaluates the mean and variance/co-variance of the super-Gaussian numerically.  Generates an image with 10 sigma
    worth of tails.  Sums up the values X*rho, Y*rho, X^2*rho, X*Y*rho, and Y^2*rho.
    """
    # Estimate what the mean and standard deviation of the thing should be and make a test image to numerically
    # integrate at 10 sigma on each side
    mu, std = beamfit.get_mu_sigma(h, 1.0)
    sigma = 10
    xmin = int(mu[0] - sigma * np.sqrt(std[0, 0]))
    xmax = int(mu[0] + sigma * np.sqrt(std[0, 0]))
    ymin = int(mu[1] - sigma * np.sqrt(std[1, 1]))
    ymax = int(mu[1] + sigma * np.sqrt(std[1, 1]))
    Xn, Yn = np.mgrid[xmin:xmax, ymin:ymax]
    sgn = beamfit.supergaussian(Xn, Yn, *h) - h[-1]

    # Numerically integrate to find mean and variance
    norm = np.sum(sgn)
    mux = np.sum(Xn * sgn) / norm
    muy = np.sum(Xn * sgn) / norm
    Vxx = np.sum(Xn**2 * sgn) / norm
    Vxy = np.sum(Xn * Yn * sgn) / norm
    Vyy = np.sum(Yn**2 * sgn) / norm
    mu = np.array([mux, muy])
    sigma = np.array([[Vxx, Vxy], [Vxy, Vyy]]) - mu[:, None] * mu[None, :]

    return mu, sigma


def calc_gradient_central_difference(
    fn, x0=np.array([0, 0, 0]), h=1e-5, atol=1e-9, fn_type="sigma_mat"
):
    # Coefficients for the finite differences schemes
    coef = [
        np.array([0.0, -1 / 2, 0.0, 1 / 2, 0.0]),
        np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]),
    ]

    # Find the x values to evaluate at
    n = coef[-2].size
    x = (
        np.arange(-(n // 2), n // 2 + 1)[None, None, :]
        * np.identity(len(x0))[:, :, None]
        * h
    )
    x = np.reshape(x, (len(x0), len(x0) * x.shape[2])).T
    x = x + x0[None, :]

    # Evaluate the Jacobian
    if fn_type == "sigma_mat":

        def fn_internal(x):
            return fn(x).ravel()[[True, True, False, True]]
    elif fn_type == "vector":
        fn_internal = fn
    elif fn_type == "scalar":

        def fn_internal(x):
            return np.array([fn(x)])
    else:
        raise ValueError(f'Unrecognized value for "fn_type": "{fn_type}"')

    y = np.array([fn_internal(xx) for xx in x])
    y = np.reshape(y.T, (y.shape[1], len(x0), y.shape[0] // len(x0)))
    j = np.array([np.sum(y * c[None, None, :] / h, axis=2) for c in coef])

    # Get the error estimate and set places where Jacobian is zero to nan manually
    err = np.abs((j[1] - j[0]) / j[1])
    err[np.abs(j[1]) < atol] = float("nan")

    if fn_type == "scalar":
        return j[1][0, 0], err[0, 0]
    return j[1], err  # Return the Jacobian and error estimates


class TestBeamfit(unittest.TestCase):
    def setUp(self):
        """Loads the data for tests"""
        # Get our path and the path of the data file
        this_path = os.path.dirname(__file__)
        filename = os.path.join(this_path, "test_data.pickle")

        # Load the test data
        with open(filename, "rb") as f:
            self.test_data = pickle.load(f)

    def test_fit_supergaussian(self):
        # Pull out the test image and validation data
        test_image = self.test_data["supergaussian_fit_data"]["images"][0]
        valid_h = self.test_data["supergaussian_fit_data"]["labels"][0][0]
        # valid_C = self.test_data["supergaussian_fit_data"]["labels"][0][1]

        # Fit it and Compare
        res = beamfit.SuperGaussian().fit(test_image)
        # test_h, test_C = beamfit.fit_supergaussian(test_image)
        np.testing.assert_allclose(res.h, valid_h, rtol=0.2)
        res = beamfit.SuperGaussian().fit(test_image, np.ones_like(test_image))
        np.testing.assert_allclose(res.h, valid_h, rtol=0.2)

    def test_supergaussian(self):
        # Pull out the test data
        X = self.test_data["gaussufunc"]["X"]
        Y = self.test_data["gaussufunc"]["Y"]
        h = self.test_data["gaussufunc"]["h"]
        valid = self.test_data["gaussufunc"]["supergaussian"]

        # Compute the test function
        test = beamfit.supergaussian(X, Y, *h)

        # Test it
        np.testing.assert_allclose(test, valid)

    def test_supergaussian_grad(self):
        # Pull out the test data
        X = self.test_data["gaussufunc"]["X"]
        Y = self.test_data["gaussufunc"]["Y"]
        h = self.test_data["gaussufunc"]["h"]
        valid = self.test_data["gaussufunc"]["supergaussian_grad"]

        # Compute the test function
        test = beamfit.supergaussian_grad(X, Y, *h).T

        # Test it
        for t, v in zip(valid, test):
            np.testing.assert_allclose(t, v, atol=1e-9)

    def test_get_mu_sigma(self):
        # Numerically find the reference values for the image
        h_test = np.array([128, 128, 50**2, 0.0, 40**2, 0.8, 1.0, 0.05])
        mu_ref, sigma_ref = get_mu_sigma_numerical(h_test)

        # Find the test values from the library
        mu_test, sigma_test = beamfit.get_mu_sigma(h_test, 1.0)

        # Compare them
        np.testing.assert_allclose(mu_test, mu_ref, atol=1e-9)
        np.testing.assert_allclose(sigma_test, sigma_ref, atol=1e-9)

    def internal_gaussian_test(self, p):
        h_refs = [
            np.array([128, 256, 32**2, 0.0, 8**2, 1, 1.0, 0.06]),
            np.array([115, 345, 32**2, 0.0, 16**2, 1, 10.0, 0.15]),
            np.array([132, 375, 16**2, 0.0, 16**2, 1, 432.0, -23]),
            np.array([142, 128, 25**2, 0.0, 19**2, 1, 253.0, 432]),
        ]

        for h_ref in h_refs:
            # Generate the image
            X, Y = np.mgrid[:256, :512]
            sg = beamfit.supergaussian(X, Y, *h_ref)
            np.testing.assert_allclose(p.fit(sg).h, h_ref, rtol=0.005, atol=1e-6)

    def test_gaussian_profile_1d(self):
        self.internal_gaussian_test(beamfit.GaussianProfile1D())

    def test_gaussian_linear_least_squares(self):
        self.internal_gaussian_test(beamfit.GaussianLinearLeastSquares())

    def test_rms_integration(self):
        self.internal_gaussian_test(beamfit.RMSIntegration())

    def test_supergaussian_scaling_grad(self):
        for x0 in [0.5, 0.75, 1.0, 1.25, 1.5]:
            j, _ = calc_gradient_central_difference(
                beamfit.super_gaussian_scaling_factor,
                x0=np.array([x0]),
                fn_type="scalar",
            )
            np.testing.assert_allclose(
                beamfit.super_gaussian_scaling_factor_grad(x0), j, atol=1e-9
            )

    def test_gaussian_linear_least_squares_trans_grad(self):
        x0s = [
            np.array([1, 1, 1, 1, 1, 1, 1, 0]),
            np.array([1, 2, 3, 4, 5, 6, 1, 0]),
            np.array([6, 5, 4, 3, 2, 1, 1, 0]),
            np.array([6, 5, 4, 4, 5, 6, 1, 0]),
        ]
        for x0 in x0s:
            j, _ = calc_gradient_central_difference(
                beamfit.gaussian_lls_trans, x0=x0, fn_type="vector"
            )
            np.testing.assert_allclose(
                beamfit.gaussian_lls_trans_grad(x0), j, atol=1e-10
            )
