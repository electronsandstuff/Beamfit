import pytest
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


@pytest.fixture
def example_matrix():
    """Fixture providing the example matrix from paper."""
    return np.array([[1, 1], [1, 5]])


@pytest.fixture
def test_point():
    """Fixture providing test point for gradient tests."""
    return np.array([2, 3, 5])


# Parameterization data for gradient tests
GRADIENT_TEST_PARAMS = [
    (beamfit.Cholesky(), 1.0),
    (beamfit.LogCholesky(), 1e-4),
    (beamfit.Spherical(), 1e-4),
    (beamfit.MatrixLogarithm(), 1e-4),
    (beamfit.Givens(), 1e-4),
]

# Test matrices for inverse tests
TEST_MATRICES = [
    np.identity(2),
    np.array([[1, 2], [2, 5]]),
    np.array([[100, 1], [1, 0.1]]),
]

# Parameterizations for inverse tests
PARAMETERIZATIONS = [
    beamfit.Cholesky(),
    beamfit.LogCholesky(),
    beamfit.Spherical(),
    beamfit.MatrixLogarithm(),
    beamfit.Givens(),
]


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
@pytest.mark.parametrize("test_matrix", TEST_MATRICES)
def test_inverses(parameterization, test_matrix):
    """Make sure parameterization has a valid inverse for given matrix."""
    np.testing.assert_allclose(
        parameterization.reverse(parameterization.forward(test_matrix)),
        test_matrix,
        atol=1e-9,
    )


def test_cholesky(example_matrix):
    """Test Cholesky parameterization forward transform."""
    result = beamfit.Cholesky().forward(example_matrix)
    expected = np.array([1, 1, 2])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_log_cholesky(example_matrix):
    """Test LogCholesky parameterization forward transform."""
    result = beamfit.LogCholesky().forward(example_matrix)
    expected = np.array([0, 1, np.log(2)])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_spherical(example_matrix):
    """Test Spherical parameterization forward transform."""
    result = beamfit.Spherical().forward(example_matrix)
    expected = np.array([0, np.log(5) / 2, -0.608])
    np.testing.assert_allclose(result, expected, rtol=1e-3)


@pytest.mark.parametrize("parameterization,h_value", GRADIENT_TEST_PARAMS)
def test_grads_numerical(parameterization, h_value, test_point):
    """Test numerical gradients for parameterization."""
    j, err = calc_gradient_central_difference(
        parameterization.reverse, x0=test_point, h=h_value
    )
    # Next line gives estimate of relative truncation error, but doesn't include roundoff error. Try to
    # select the largest h such that the error is like 1e-9.
    # print(err)
    assert (err[np.isfinite(err)] < 1e-6).all(), (
        f"Error too large for {parameterization} - need to make h smaller"
    )
    j_actual = parameterization.reverse_grad(test_point)
    np.testing.assert_allclose(j_actual, j, atol=1e-8, rtol=1e-5)


def test_eigen2d_grad_numerical(test_point):
    """Test numerical gradient for eigen2d function."""

    def rev(s):
        a = beamfit.eigen2d(s)
        return np.array([[a[0], a[1]], [a[1], a[2]]])

    j, err = calc_gradient_central_difference(rev, x0=test_point, h=1e-4)
    j_actual = beamfit.eigen2d_grad(test_point)
    np.testing.assert_allclose(j_actual, j, atol=1e-8, rtol=1e-5)
