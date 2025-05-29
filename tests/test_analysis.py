import unittest
import numpy as np
import beamfit


class TestAnalysisMethod(unittest.TestCase):
    def test_threshold(self):
        m, n = np.mgrid[:64, :64]
        image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
        o = beamfit.AnalysisMethodDebugger(sigma_threshold=2).fit(image)
        self.assertEqual(np.sum(o.mask), np.sum(image < np.exp(-(2**2))))

    def test_median_filter(self):
        m, n = np.mgrid[:64, :64]
        image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
        image[4, 24] = 100  # Set some random hot pixels
        image[32, 15] = 100
        image[54, 22] = 100
        o = beamfit.AnalysisMethodDebugger(median_filter_size=3).fit(image)
        self.assertAlmostEqual(o.max(), 1, places=1)

    def test_masked(self):
        m, n = np.mgrid[:64, :64]
        image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
        o = beamfit.AnalysisMethodDebugger(sigma_threshold=2, median_filter_size=3).fit(
            image
        )
        self.assertTrue(np.ma.isMaskedArray(o))
