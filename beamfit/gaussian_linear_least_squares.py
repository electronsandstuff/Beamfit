import numpy as np
from .utils import AnalysisMethod, SuperGaussianResult


class GaussianLinearLeastSquares(AnalysisMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __fit__(self, image):
        # Normalize image z axis
        hi = image.max()
        lo = image.min()
        image_norm = ((image - lo)/(hi - lo) + np.exp(-10))/(1 + np.exp(-10))

        # Create coefficient matrices for fit
        m, n = np.mgrid[:image_norm.shape[0], :image_norm.shape[1]]
        mm = m[~image_norm.mask]
        nn = n[~image_norm.mask]
        x = np.array([np.ones_like(mm), mm, mm ** 2, nn, nn * mm, nn ** 2]).T
        expy = image_norm[~image_norm.mask]
        y = np.log(expy)

        # Weight the values assuming equal error in the image pixels (we transform by log(y))
        w = expy**2  # 1/sigma**2
        wy = y*w
        wx = x*w[:, None]

        # Solve the linear least squares problem with QR factorization
        q, r = np.linalg.qr((wx.T * wy).T)
        x = np.linalg.solve(r, q.T @ wy ** 2)

        # Return the fit
        return SuperGaussianResult(
            mu=np.array([
                (-1 * x[3] * x[4] + 2 * x[1] * x[5]) / (x[4] ** 2 - 4 * x[2] * x[5]),
                (-2 * x[2] * x[3] + x[1] * x[4]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5])
            ]),
            sigma=-np.linalg.inv(np.array([[x[2], x[4] / 2], [x[4] / 2, x[5]]]))/2,
            a=hi-lo,
            o=lo,
        )

    def get_name(self):
        return 'GaussianLinearLeastSquares'


def fit_gaussian_linear_least_squares(image, sigma_threshold=2, plot=False):  # Backwards compatibility
    return GaussianLinearLeastSquares(sigma_threshold=sigma_threshold).fit(image).h
