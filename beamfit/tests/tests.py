################################################################################
# Unit tests for beamfit - Christopher M. Pierce (chris@chris-pierce.com)
################################################################################

import os
import pickle
import unittest

import numpy as np

################################################################################
# Imports
################################################################################
import beamfit
import gaussufunc


################################################################################
# Helper functions
################################################################################
def get_mu_sigma_numerical(h):
    """
    Evaluates the mean and variance/co-variance of the super-Gaussian numerically.  Generates an image with 10 sigma
    worth of tails.  Sums up the values X*rho, Y*rho, X^2*rho, X*Y*rho, and Y^2*rho.
    """
    # Estimate what the mean and standard deviation of the thing should be and make a test image to numerically
    # integrate at 10 sigma on each side
    mu, std = beamfit.get_mu_sigma(h, 1.0)
    sigma = 10
    xmin = int(mu[0] - sigma*np.sqrt(std[0,0]))
    xmax = int(mu[0] + sigma*np.sqrt(std[0,0]))
    ymin = int(mu[1] - sigma*np.sqrt(std[1,1]))
    ymax = int(mu[1] + sigma*np.sqrt(std[1,1]))
    Xn, Yn = np.mgrid[xmin:xmax, ymin:ymax]
    sgn = beamfit.supergaussian(Xn, Yn, *h) - h[-1]

    # Numerically integrate to find mean and variance
    norm = np.sum(sgn)
    mux = np.sum(Xn * sgn) / norm
    muy = np.sum(Xn * sgn) / norm
    Vxx = np.sum(Xn ** 2 * sgn) / norm
    Vxy = np.sum(Xn * Yn * sgn) / norm
    Vyy = np.sum(Yn ** 2 * sgn) / norm
    mu = np.array([mux, muy])
    sigma = np.array([[Vxx, Vxy], [Vxy, Vyy]]) - mu[:, None]*mu[None, :]

    return mu, sigma


################################################################################
# The tests
################################################################################
class TestBeamfit(unittest.TestCase):
    def setUp(self):
        '''Loads the data for tests'''
        # Get our path and the path of the data file
        this_path = os.path.dirname(__file__)
        filename = os.path.join(this_path, 'test_data.pickle')

        # Load the test data
        with open(filename, 'rb') as f:
            self.test_data = pickle.load(f)

    def test_fit_supergaussian(self):
        # Pull out the test image and validation data
        test_image = self.test_data['supergaussian_fit_data']['images'][0]
        valid_h = self.test_data['supergaussian_fit_data']['labels'][0][0]
        valid_C = self.test_data['supergaussian_fit_data']['labels'][0][1]

        # Fit it
        test_h, test_C = beamfit.fit_supergaussian(test_image,
                                                   np.ones_like(test_image))

        # Compare
        self.assertTrue(np.isclose(test_h, valid_h).all())
        self.assertTrue(np.isclose(test_C, valid_C).all())

    def test_supergaussian(self):
        # Pull out the test data
        X = self.test_data['gaussufunc']['X']
        Y = self.test_data['gaussufunc']['Y']
        h = self.test_data['gaussufunc']['h']
        valid = self.test_data['gaussufunc']['supergaussian']

        # Compute the test function
        test = gaussufunc.supergaussian(X, Y, *h)

        # Test it
        self.assertTrue(np.isclose(valid, test).all())

    def test_supergaussian_grad(self):
        # Pull out the test data
        X = self.test_data['gaussufunc']['X']
        Y = self.test_data['gaussufunc']['Y']
        h = self.test_data['gaussufunc']['h']
        valid = self.test_data['gaussufunc']['supergaussian_grad']

        # Compute the test function
        test = gaussufunc.supergaussian_grad(X, Y, *h)

        # Test it
        for t, v in zip(valid, test):
            self.assertTrue(np.isclose(v, t).all())

    def test_get_mu_sigma(self):
        # Numerically find the reference values for the image
        h_test = np.array([128, 128, 50**2, 0.0, 40**2, 0.8, 1.0, 0.05])
        mu_ref, sigma_ref = get_mu_sigma_numerical(h_test)

        # Find the test values from the library
        mu_test, sigma_test = beamfit.get_mu_sigma(h_test, 1.0)

        # Compare them
        self.assertTrue(np.isclose(mu_ref, mu_test).all())
        self.assertTrue(np.isclose(sigma_ref, sigma_test).all())
