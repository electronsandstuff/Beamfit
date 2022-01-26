from gaussufunc import supergaussian_internal, supergaussian_grad_internal
import numpy as np


def supergaussian(x, y, mux, muy, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    return supergaussian_internal(x, y, mux, muy, sigma_xx, sigma_xy, sigma_yy, n, a, o)


def supergaussian_grad(x, y, mux, muy, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    return np.array(supergaussian_grad_internal(x, y, mux, muy, sigma_xx, sigma_xy, sigma_yy, n, a, o)).T
