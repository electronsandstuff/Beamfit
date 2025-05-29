import unittest
import numpy as np
import beamfit


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


class TestSigmaParameterization(unittest.TestCase):
    def setUp(self):
        self.matrices = [
            np.identity(2),
            np.array([[1, 2], [2, 5]]),
            np.array([[100, 1], [1, 0.1]]),
        ]
        self.m = np.array([[1, 1], [1, 5]])  # Example from paper I am following
        self.ps = [
            beamfit.Cholesky(),
            beamfit.LogCholesky(),
            beamfit.Spherical(),
            beamfit.MatrixLogarithm(),
            beamfit.Givens(),
        ]

    def test_inverses(self):
        """Make sure all parameterizations have a valid inverse"""
        for p in self.ps:
            for m in self.matrices:
                try:
                    np.testing.assert_allclose(p.reverse(p.forward(m)), m, atol=1e-9)
                except:
                    print(f"Failed at {p}")
                    raise

    def test_cholesky(self):
        np.testing.assert_allclose(
            beamfit.Cholesky().forward(self.m), np.array([1, 1, 2]), atol=1e-9
        )

    def test_log_cholesky(self):
        np.testing.assert_allclose(
            beamfit.LogCholesky().forward(self.m),
            np.array([0, 1, np.log(2)]),
            atol=1e-9,
        )

    def test_spherical(self):
        np.testing.assert_allclose(
            beamfit.Spherical().forward(self.m),
            np.array([0, np.log(5) / 2, -0.608]),
            rtol=1e-3,
        )

    def test_grads_numerical(self):
        x0 = np.array([2, 3, 5])
        hs = np.ones(100) * 1e-4
        hs[0] = 1.0  # Manually set h values for specific parameterizations

        for p, h in zip(self.ps, hs):
            try:
                j, err = calc_gradient_central_difference(p.reverse, x0=x0, h=h)
                # Next line gives estimate of relative truncation error, but doesn't include roundoff error. Try to
                # select the largest h such that the error is like 1e-9.
                # print(err)
                self.assertTrue(
                    (err[np.isfinite(err)] < 1e-6).all()
                )  # If it fails here, need to make h smaller
                j_actual = p.reverse_grad(x0)
                np.testing.assert_allclose(j_actual, j, atol=1e-8, rtol=1e-5)
            except:
                print(f"Failed at {p}")
                raise

    def test_eigen2d_grad_numerical(self):
        x0 = np.array([2, 3, 5])

        def rev(s):
            a = beamfit.eigen2d(s)
            return np.array([[a[0], a[1]], [a[1], a[2]]])

        j, err = calc_gradient_central_difference(rev, x0=x0, h=1e-4)
        j_actual = beamfit.eigen2d_grad(x0)
        np.testing.assert_allclose(j_actual, j, atol=1e-8, rtol=1e-5)
