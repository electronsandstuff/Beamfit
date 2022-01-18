import numpy as np


def fit_gaussian_linear_least_squares(image, sigma_threshold=2, plot=False):
    """
    Fits an image to a normal gaussian using linear least squares.  This
    method is intended to be used as a stable starting point to find a guess
    for the non-linear fitting routine
    """
    # Calculate the threshold
    threshold = np.exp(-1 * sigma_threshold)

    # If there isn't a mask, make one
    if not np.ma.isMaskedArray(image):
        image = np.ma.array(image)

    # Force the images to be non-zero
    non_zero_image = image - image.min() + np.exp(-10)

    # Get the y values and determine the mask we will use
    image_unwrapped = non_zero_image.ravel()

    # Get a median filtered image for thresholding
    image_filtered = image  # ndimage.median_filter(image, size=10)

    # Create a mask based on a threshold and a the actual mask
    mask_from_image = (image_unwrapped.mask == False)
    mask_from_threshold = image_unwrapped > image_filtered.ravel().max() * threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold)

    # Mask the array
    image_masked = np.array(image_unwrapped[mask_combined])

    # Get the y values in the fit from the image
    y = np.log(image_masked)

    # Create the a grid to evaluate the loss function over
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    mm = M.ravel()[mask_combined]
    nn = N.ravel()[mask_combined]

    # Find the A matrix for the fit
    A = np.array([np.ones_like(mm), mm, mm ** 2, nn, nn * mm, nn ** 2]).T

    # Solve the linear least squares problem with QR factorization
    Q, R = np.linalg.qr((A.T * y).T)
    x = np.linalg.solve(R, Q.T @ y ** 2)

    # Create the full matrix
    M_full, N_full = np.mgrid[:image.shape[0], :image.shape[1]]
    mm_full = M_full.ravel()
    nn_full = N_full.ravel()
    A_full = np.array([np.ones_like(mm_full), mm_full, mm_full ** 2, nn_full,
                       nn_full * mm_full, nn_full ** 2]).T

    # Get the fit image and residuals
    fit_image = np.exp(np.reshape(A_full @ x, image.shape))
    residual = fit_image - image

    # Find the centroid from the fit
    mu = np.array([
        (-1 * x[3] * x[4] + 2 * x[1] * x[5]) / (x[4] ** 2 - 4 * x[2] * x[5]),
        (-2 * x[2] * x[3] + x[1] * x[4]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5])])

    # Get the sigma matrix too
    sigma_inv = np.array([[x[2], x[4] / 2], [x[4] / 2, x[5]]])
    sigma = -2 * np.linalg.inv(sigma_inv)

    # Calculate the twiss parameters from the gaussian fit
    alpha = x[4] / np.sqrt(4 * x[2] * x[5] - x[4] ** 2)
    root_beta = np.sqrt(-2 * x[5] / np.sqrt(4 * x[2] * x[5] - x[4] ** 2))
    root_epsilon = np.sqrt(2 / np.sqrt(4 * x[2] * x[5] - x[4] ** 2))

    # Find the peak and bottom
    masked_image = np.array(image)[mask_from_image].ravel()
    ind = np.argsort(masked_image)
    low = np.mean(masked_image[ind][2:12])
    high = np.mean(masked_image[ind][-12:-2])

    # Convert to the new guess
    h0 = np.array([
        mu[0],
        mu[1],
        root_beta ** 2 * root_epsilon ** 2,
        -alpha * root_epsilon ** 2,
        (1 + alpha ** 2) / root_beta ** 2 * root_epsilon ** 2,
        1,
        high - low,
        low
    ])

    # Return the fit
    return h0
