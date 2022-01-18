import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndimage

from gaussufunc import supergaussian, supergaussian_grad
from .utils import chunk_it
from .gaussian_linear_least_squares import fit_gaussian_linear_least_squares
from .gaussian_fit_1d import fit_gaussian_1d


def fit_func(xdata, mux, muy, vxx, vxy, vyy, n, a, o):
    return supergaussian(xdata[0], xdata[1], mux, muy, vxx, vxy, vyy, n, a, o)


def fit_func_jac(xdata, mux, muy, vxx, vxy, vyy, n, a, o):
    return supergaussian_grad(xdata[0], xdata[1], mux, muy, vxx, vxy, vyy, n, a, o).T


# Functions to convert from bounded parameters to parameters for unbounded optimization
# TODO: add different options of this conversion: log
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


def fit_scipy_curvefit(x, y, w, h0, maxfev):
    # Call scipy's curve_fit
    h, C = opt.curve_fit(fitfun_bounded, x, y, h_to_hb(h0), 1 / np.sqrt(w), absolute_sigma=True, jac=fitjac_bounded,
                         maxfev=maxfev)

    return hb_to_h(h), C


def fit_stochastic_lma(x, y, w, h0, LMA_lambda=1, nbatch=8, epochs=4):
    # Convert to the bounded internal parameters
    hb = h_to_hb(h0)

    for _ in range(epochs):
        # Get a batch
        idx = np.arange(x.shape[1])
        np.random.shuffle(idx)
        batches = chunk_it(idx, nbatch)

        # Minimize once for each batch
        for batch in batches:
            # Calculate the gradient and value of the function
            model_grad = np.sqrt(w[batch]) * h_to_hb_grad(hb, fit_func_jac(x[:, batch], *hb_to_h(hb)).T)
            model = hb[-2] * model_grad[-2, :] + np.sqrt(w[batch]) * hb[-1]

            # Turn this into the gradient of the loss function and the approximation of the hessian
            loss_grad = 2 * model_grad @ (np.sqrt(w[batch]) * y[batch] - model).T
            hess = model_grad @ model_grad.T

            # Add the LMA damping term
            diag = np.diag_indices(8)
            hess[diag] = hess[diag] * (1 + LMA_lambda)

            # Calculate the update
            delta = np.linalg.solve(hess, loss_grad)

            # Update the parameters
            hb = hb + delta

    # Calculate the Hessian of the loss function for error estimation
    model_grad = np.sqrt(w[batch]) * h_to_hb_grad(hb, fit_func_jac(x[:, batch], *hb_to_h(hb)).T)
    hess = model_grad @ model_grad.T

    # Invert to get the variance-covariance matrix
    C = np.linalg.inv(hess)

    # Return the parameters and the variance covariance matrix
    return hb_to_h(hb), C


def fit_supergaussian(image, image_weights=None, prediction_func="2D_linear_Gaussian", sigma_threshold=3,
                      sigma_threshold_guess=1, smoothing=5, maxfev=100):
    # Double check the input
    if not isinstance(image, (list, np.ndarray)):
        raise ValueError(f"Image provided to supergaussian fit must be numpy compatible not the received type:"
                         f" \"{type(image)}\"")
    if isinstance(image, list):
        image = np.array(image)
    if len(image.shape) != 2:
        raise ValueError(f"Image array provided to superagussian fit must have dimension 2, not {len(image.shape)}")
    if image.size == 0:
        raise ValueError(f"Image array provided to superagussian fit must contain more than zero pixels")
    # Calculate the threshold
    threshold = np.exp(-1 * sigma_threshold ** 2 / 2)

    # If there isn't a mask, make one
    if not np.ma.isMaskedArray(image):
        image = np.ma.array(image)

    if image_weights is None:
        image_weights = np.ones_like(image)

    # Get a median filtered image for thresholding
    image_filtered = ndimage.median_filter(image, size=smoothing)

    # Make a good initial guess
    if prediction_func == "2D_linear_Gaussian":
        h0 = fit_gaussian_linear_least_squares(image_filtered, sigma_threshold=sigma_threshold_guess)

    elif prediction_func == "1D_Gaussian":
        h0 = fit_gaussian_1d(image_filtered)

    else:
        raise ValueError("Unrecognized prediction method \"{:s}\"".format(prediction_func))

    # Get the Y data
    #image_unwrapped = ndimage.median_filter(image, size=2).ravel()
    image_unwrapped = image.ravel()

    # Create a mask based on a threshold and a the actual mask
    mask_from_image = (image.mask == False)

    # Find the peak and bottom
    masked_image = np.array(image)[mask_from_image].ravel()
    ind = np.argsort(masked_image)
    low = np.mean(masked_image[ind][2:12])
    high = np.mean(masked_image[ind][-12:-2])

    # Get the X data
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    MN = np.vstack((M.ravel(), N.ravel()))

    # Make the threshold mask
    thresh_image = supergaussian(M, N, h0[0], h0[1], h0[2], h0[3], h0[4], 1, 1, 0)
    mask_from_threshold = thresh_image > threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold).ravel()

    # Mask the array
    y = np.array(image_unwrapped[mask_combined])
    w = np.array(image_weights.ravel()[mask_combined])

    # Mask it
    x = MN[:, mask_combined]

    # Fit it
    h, C = fit_scipy_curvefit(x, y, w, h0, maxfev=maxfev)

    # Return the fit and the covariance variance matrix
    return h, C
