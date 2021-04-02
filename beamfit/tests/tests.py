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
        test = beamfit.gaussufunc.supergaussian(X, Y, *h)

        # Test it
        self.assertTrue(np.isclose(valid, test).all())

    def test_supergaussian_grad(self):
        # Pull out the test data
        X = self.test_data['gaussufunc']['X']
        Y = self.test_data['gaussufunc']['Y']
        h = self.test_data['gaussufunc']['h']
        valid = self.test_data['gaussufunc']['supergaussian_grad']

        # Compute the test function
        test = beamfit.gaussufunc.supergaussian_grad(X, Y, *h)

        # Test it
        for t, v in zip(valid, test):
            self.assertTrue(np.isclose(v, t).all())

    def test_get_mu_sigma(self):
        # Get the test data
        h = self.test_data['mu_sigma']['h']
        C = self.test_data['mu_sigma']['C']
        pixel_size = self.test_data['mu_sigma']['pixel_size']
        pixel_size_std = self.test_data['mu_sigma']['pixel_size_std']
        mu = self.test_data['mu_sigma']['mu']
        mu_std = self.test_data['mu_sigma']['mu_std']
        sigma = self.test_data['mu_sigma']['sigma']
        sigma_std = self.test_data['mu_sigma']['sigma_std']

        # Compute the test
        test_mu, test_sigma = beamfit.get_mu_sigma(h, pixel_size)

        # Compare them
        self.assertTrue(np.isclose(mu, test_mu).all())
        self.assertTrue(np.isclose(sigma, test_sigma).all())

    def test_get_mu_sigma_std(self):
        # Get the test data
        h = self.test_data['mu_sigma']['h']
        C = self.test_data['mu_sigma']['C']
        pixel_size = self.test_data['mu_sigma']['pixel_size']
        pixel_size_std = self.test_data['mu_sigma']['pixel_size_std']
        mu = self.test_data['mu_sigma']['mu']
        mu_std = self.test_data['mu_sigma']['mu_std']
        sigma = self.test_data['mu_sigma']['sigma']
        sigma_std = self.test_data['mu_sigma']['sigma_std']

        # Compute the test
        test_mu_std, test_sigma_std = beamfit.get_mu_sigma_std(h, C,
                                                               pixel_size, pixel_size_std)

        # Compare them
        self.assertTrue(np.isclose(mu_std, test_mu_std).all())
        self.assertTrue(np.isclose(sigma_std, test_sigma_std).all())

    def test_fit_supergaussian_bad_input(self):
        for obj in ["i", 0, 0.0, True, [], {}, np.zeros((1,)), np.zeros((0, 0))]:
            with self.assertRaises(ValueError) as context:
                beamfit.fit_supergaussian(obj)
