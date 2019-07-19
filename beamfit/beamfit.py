#!/usr/bin/env python

################################################################################
# File: beamfit.py
# Author: Christopher M. Pierce (cmp285@cornell.edu)
################################################################################

################################################################################
# Imports
################################################################################
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import scipy.ndimage as ndimage

################################################################################
# Functions
################################################################################
def get_super_gaussian(x, y, A, alpha, root_beta, root_epsilon, mux, muy, n, offset):
    '''Returns the value of a supergaussian function evaluated at the position
    (x, y).  The remaining parameters determine the shape of the function.
    Alpha, root_beta and root_epsilon refer to the Twiss parameterization of
    the ellipse.  mux and muy are the mean of the distribution in x and y.  n
    is the supergaussian parameter and offset is a constant added to the
    function to represent a noise floor.'''
    beta = root_beta**2
    epsilon = root_epsilon**2
    return A*np.exp(-1/2*(2*mux*muy*alpha/epsilon - 2*muy*x*alpha/epsilon -
        2*mux*y*alpha/epsilon + 2*x*y*alpha/epsilon + mux**2/beta/epsilon -
        2*mux*x/beta/epsilon + x**2/beta/epsilon + mux**2*alpha**2/beta/epsilon
        - 2*mux*x*alpha**2/beta/epsilon + x**2*alpha**2/beta/epsilon +
        muy**2*beta/epsilon - 2*muy*y*beta/epsilon + y**2*beta/epsilon)**n) + offset

def get_super_gaussian_grad(X, Y, A, Alpha, RootBeta, RootEpsilon, MuX, MuY, N, Offset):
    '''Returns the gradient of the above supergaussian function WRT the
    parameters.  Warning: dragons be here!'''
    return np.array([
        np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
            MuX**2/(RootBeta**2*RootEpsilon**2) +
            (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
            (MuY**2*RootBeta**2)/RootEpsilon**2 -
            (2*Alpha*MuY*X)/RootEpsilon**2 -
            (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
            X**2/(RootBeta**2*RootEpsilon**2) +
            (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha*MuX*Y)/RootEpsilon**2 -
            (2*MuY*RootBeta**2*Y)/RootEpsilon**2 + (2*Alpha*X*Y)/RootEpsilon**2
            + (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.),
        -(A*N*((2*MuX*MuY)/RootEpsilon**2 +
            (2*Alpha*MuX**2)/(RootBeta**2*RootEpsilon**2) -
            (2*MuY*X)/RootEpsilon**2 -
            (4*Alpha*MuX*X)/(RootBeta**2*RootEpsilon**2) +
            (2*Alpha*X**2)/(RootBeta**2*RootEpsilon**2) -
            (2*MuX*Y)/RootEpsilon**2 +
            (2*X*Y)/RootEpsilon**2)*((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**(-1 +
                    N)*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                        MuX**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                        (MuY**2*RootBeta**2)/RootEpsilon**2 -
                        (2*Alpha*MuY*X)/RootEpsilon**2 -
                        (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                        X**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha*MuX*Y)/RootEpsilon**2 -
                        (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                        (2*Alpha*X*Y)/RootEpsilon**2 +
                        (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.))/2.,
        -(A*N*((-2*MuX**2)/(RootBeta**3*RootEpsilon**2) -
            (2*Alpha**2*MuX**2)/(RootBeta**3*RootEpsilon**2) +
            (2*MuY**2*RootBeta)/RootEpsilon**2 +
            (4*MuX*X)/(RootBeta**3*RootEpsilon**2) +
            (4*Alpha**2*MuX*X)/(RootBeta**3*RootEpsilon**2) -
            (2*X**2)/(RootBeta**3*RootEpsilon**2) -
            (2*Alpha**2*X**2)/(RootBeta**3*RootEpsilon**2) -
            (4*MuY*RootBeta*Y)/RootEpsilon**2 +
            (2*RootBeta*Y**2)/RootEpsilon**2)*((2*Alpha*MuX*MuY)/RootEpsilon**2
                + MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**(-1 +
                    N)*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                        MuX**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                        (MuY**2*RootBeta**2)/RootEpsilon**2 -
                        (2*Alpha*MuY*X)/RootEpsilon**2 -
                        (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                        X**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha*MuX*Y)/RootEpsilon**2 -
                        (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                        (2*Alpha*X*Y)/RootEpsilon**2 +
                        (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.))/2.,
        -(A*N*((-4*Alpha*MuX*MuY)/RootEpsilon**3 -
            (2*MuX**2)/(RootBeta**2*RootEpsilon**3) -
            (2*Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**3) -
            (2*MuY**2*RootBeta**2)/RootEpsilon**3 +
            (4*Alpha*MuY*X)/RootEpsilon**3 +
            (4*MuX*X)/(RootBeta**2*RootEpsilon**3) +
            (4*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**3) -
            (2*X**2)/(RootBeta**2*RootEpsilon**3) -
            (2*Alpha**2*X**2)/(RootBeta**2*RootEpsilon**3) +
            (4*Alpha*MuX*Y)/RootEpsilon**3 +
            (4*MuY*RootBeta**2*Y)/RootEpsilon**3 - (4*Alpha*X*Y)/RootEpsilon**3
            -
            (2*RootBeta**2*Y**2)/RootEpsilon**3)*((2*Alpha*MuX*MuY)/RootEpsilon**2
                + MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**(-1 +
                    N)*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                        MuX**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                        (MuY**2*RootBeta**2)/RootEpsilon**2 -
                        (2*Alpha*MuY*X)/RootEpsilon**2 -
                        (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                        X**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha*MuX*Y)/RootEpsilon**2 -
                        (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                        (2*Alpha*X*Y)/RootEpsilon**2 +
                        (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.))/2.,
        -(A*N*((2*Alpha*MuY)/RootEpsilon**2 +
            (2*MuX)/(RootBeta**2*RootEpsilon**2) +
            (2*Alpha**2*MuX)/(RootBeta**2*RootEpsilon**2) -
            (2*X)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha**2*X)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha*Y)/RootEpsilon**2)*((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**(-1 +
                    N)*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                        MuX**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                        (MuY**2*RootBeta**2)/RootEpsilon**2 -
                        (2*Alpha*MuY*X)/RootEpsilon**2 -
                        (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                        X**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha*MuX*Y)/RootEpsilon**2 -
                        (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                        (2*Alpha*X*Y)/RootEpsilon**2 +
                        (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.))/2.,
        -(A*N*((2*Alpha*MuX)/RootEpsilon**2 +
            (2*MuY*RootBeta**2)/RootEpsilon**2 - (2*Alpha*X)/RootEpsilon**2 -
            (2*RootBeta**2*Y)/RootEpsilon**2)*((2*Alpha*MuX*MuY)/RootEpsilon**2
                + MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**(-1 +
                    N)*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2 +
                        MuX**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                        (MuY**2*RootBeta**2)/RootEpsilon**2 -
                        (2*Alpha*MuY*X)/RootEpsilon**2 -
                        (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                        X**2/(RootBeta**2*RootEpsilon**2) +
                        (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                        (2*Alpha*MuX*Y)/RootEpsilon**2 -
                        (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                        (2*Alpha*X*Y)/RootEpsilon**2 +
                        (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.))/2.,
        -(A*((2*Alpha*MuX*MuY)/RootEpsilon**2 +
            MuX**2/(RootBeta**2*RootEpsilon**2) +
            (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
            (MuY**2*RootBeta**2)/RootEpsilon**2 -
            (2*Alpha*MuY*X)/RootEpsilon**2 -
            (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
            X**2/(RootBeta**2*RootEpsilon**2) +
            (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
            (2*Alpha*MuX*Y)/RootEpsilon**2 -
            (2*MuY*RootBeta**2*Y)/RootEpsilon**2 + (2*Alpha*X*Y)/RootEpsilon**2
            +
            (RootBeta**2*Y**2)/RootEpsilon**2)**N*np.exp(-((2*Alpha*MuX*MuY)/RootEpsilon**2
                + MuX**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                (MuY**2*RootBeta**2)/RootEpsilon**2 -
                (2*Alpha*MuY*X)/RootEpsilon**2 -
                (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                X**2/(RootBeta**2*RootEpsilon**2) +
                (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                (2*Alpha*MuX*Y)/RootEpsilon**2 -
                (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                (2*Alpha*X*Y)/RootEpsilon**2 +
                (RootBeta**2*Y**2)/RootEpsilon**2)**N/2.)*np.log((2*Alpha*MuX*MuY)/RootEpsilon**2
                    + MuX**2/(RootBeta**2*RootEpsilon**2) +
                    (Alpha**2*MuX**2)/(RootBeta**2*RootEpsilon**2) +
                    (MuY**2*RootBeta**2)/RootEpsilon**2 -
                    (2*Alpha*MuY*X)/RootEpsilon**2 -
                    (2*MuX*X)/(RootBeta**2*RootEpsilon**2) -
                    (2*Alpha**2*MuX*X)/(RootBeta**2*RootEpsilon**2) +
                    X**2/(RootBeta**2*RootEpsilon**2) +
                    (Alpha**2*X**2)/(RootBeta**2*RootEpsilon**2) -
                    (2*Alpha*MuX*Y)/RootEpsilon**2 -
                    (2*MuY*RootBeta**2*Y)/RootEpsilon**2 +
                    (2*Alpha*X*Y)/RootEpsilon**2 +
                    (RootBeta**2*Y**2)/RootEpsilon**2))/2.,
        np.ones_like(X)])

def fit_gaussian_linear_least_squares(image, sigma_threshold=0.5, plot=False):
    '''Fits an image to a normal gaussian using linear least squares.  This
    method is intended to be used as a stable starting point to find a guess
    for the non-linear fitting routine'''
    # Calculate the threshold
    threshold = np.exp(-1*sigma_threshold)

    # If there isn't a mask, make one
    if(not np.ma.isMaskedArray(image)):
        image = np.ma.array(image)

    # Force the images to be non-zero
    non_zero_image = image - image.min() + np.exp(-10)

    # Get the y values and determine the mask we will use
    image_unwrapped = non_zero_image.ravel()

    # Get a median filtered image for thresholding
    image_filtered = ndimage.median_filter(image, size=10)

    # Create a mask based on a threshold and a the actual mask
    mask_from_image = (image_unwrapped.mask == False)
    mask_from_threshold = image_unwrapped > image_filtered.ravel().max()*threshold
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
    A = np.array([np.ones_like(mm), mm, mm**2, nn, nn*mm, nn**2]).T

    # Solve the linear least squares problem with QR factorization
    Q,R = np.linalg.qr((A.T*y).T)
    x = np.linalg.solve(R, Q.T@y**2)

    # Create the full matrix
    M_full, N_full = np.mgrid[:image.shape[0], :image.shape[1]]
    mm_full = M_full.ravel()
    nn_full = N_full.ravel()
    A_full = np.array([np.ones_like(mm_full), mm_full, mm_full**2, nn_full,
        nn_full*mm_full, nn_full**2]).T

    # Get the fit image and residuals
    fit_image = np.exp(np.reshape(A_full@x, image.shape))
    residual = fit_image - image

    # If we are plotting it
    if(plot):
        # Set the plots to be larger
        plt.rcParams["figure.figsize"] = 12,3.5

        # Show the fit
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 12}
        plt.rc('font', **font)

        # Plot the image and the fit and the residual
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        # Show them
        ax1.imshow(image)
        ax2.imshow(fit_image)
        ax3.imshow(residual)

    # Find the centroid from the fit
    mu = np.array([
        (-1*x[3]*x[4] + 2*x[1]*x[5])/(x[4]**2 - 4*x[2]*x[5]),
        (-2*x[2]*x[3] + x[1]*x[4])/(-1*x[4]**2 + 4*x[2]*x[5])])

    # Get the sigma matrix too
    sigma_inv = np.array([[x[2], x[4]/2], [x[4]/2, x[5]]])
    sigma = -2*np.linalg.inv(sigma_inv)

    # Calculate the twiss parameters from the gaussian fit
    alpha = x[4]/np.sqrt(4*x[2]*x[5] - x[4]**2)
    root_beta = np.sqrt(-2*x[5]/np.sqrt(4*x[2]*x[5] - x[4]**2))
    root_epsilon = np.sqrt(2/np.sqrt(4*x[2]*x[5] - x[4]**2))

    # Calculate the amplitude
    A = np.exp(x[0] - 2*mu[0]*mu[1]*alpha/root_epsilon**2 +
            mu[0]**2/root_beta**2/root_epsilon**2 +
            mu[0]*alpha**2/root_beta**2/root_epsilon**2 +
            mu[1]**2*root_beta**2/root_epsilon**2)

    # Return the fit
    return np.array([A, alpha, root_beta, root_epsilon, mu[0], mu[1], 1, 0.0]), residual

def fit_supergaussian(image, sigma_integrate = 6, sigma_threshold = 4,
        sigma_threshold_guess = 1, plot=False):
    '''This method fits a supergaussian to the provided image and returns the
    first and second moments.  sigma_integrate refers to how much of the tails
    are used in numerically computing the second moments from the function
    after the fit has already been performed.  This is done through numerical
    integration.  sigma_threshold determines how much of the image is used in
    the nonlinear fitting routing based on amplitude.  sigma_threshold guess is
    the same threshold, but used for the linear least squares routine which
    provided a guess at the non-linear fitting parameters.'''
    # Calculate the threshold
    threshold = np.exp(-1*sigma_threshold)

    # If there isn't a mask, make one
    if(not np.ma.isMaskedArray(image)):
        image = np.ma.array(image)

    # Make a good initial guess
    guess = fit_gaussian_linear_least_squares(ndimage.median_filter(image, size=10),
                        sigma_threshold=sigma_threshold_guess)[0]

    # Get the Y data
    image_unwrapped = image.ravel()

    # Get a median filtered image for thresholding
    image_filtered = ndimage.median_filter(image, size=10)

    # Create a mask based on a threshold and a the actual mask
    mask_from_image = (image_unwrapped.mask == False)
    mask_from_threshold = image_unwrapped > image_filtered.ravel().max()*threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold)

    # Mask the array
    y = np.array(image_unwrapped[mask_combined])

    # Get the X data
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    MN = np.vstack((M.ravel(), N.ravel()))

    # Mask it
    x = MN[:, mask_combined]

    # Perform the fit
    fit_func = lambda xdata, A, alpha, root_beta, root_epsilon, mux, muy, n, offset: get_super_gaussian(xdata[0], xdata[1], A, alpha, root_beta, root_epsilon, mux, muy, n, offset)
    fit_func_jac = lambda xdata, A, alpha, root_beta, root_epsilon, mux, muy, n, offset: get_super_gaussian_grad(xdata[0], xdata[1], A, alpha, root_beta, root_epsilon, mux, muy, n, offset).T
    popt, pcov = opt.curve_fit(fit_func, x, y, p0=guess, jac=fit_func_jac, ftol=1e-9)

    # Use that to determine beam size
    x_for_size = np.copy(popt)
    x_for_size[4] = 0.0
    x_for_size[5] = 0.0
    x_for_size[7] = 0.0
    f = lambda yy, xx: get_super_gaussian(xx, yy, *x_for_size)

    # Get the widths of integration
    x_width = popt[2]*popt[3]
    y_width = np.sqrt(1+popt[1]**2)/popt[2]*popt[3]
    xmin = -1*x_width*sigma_integrate
    xmax = x_width*sigma_integrate
    ymin = -1*y_width*sigma_integrate
    ymax = y_width*sigma_integrate

    # Integrate to find the covariance matrix
    norm = integrate.dblquad(f, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]
    mom_xx, mom_xx_err = integrate.dblquad(lambda y, x: f(y,x)*x*x, xmin, xmax,
            lambda x: ymin, lambda x: ymax)
    mom_xy, mom_xy_err = integrate.dblquad(lambda y, x: f(y,x)*x*y, xmin, xmax,
            lambda x: ymin, lambda x: ymax)
    mom_yy, mom_yy_err = integrate.dblquad(lambda y, x: f(y,x)*y*y, xmin, xmax,
            lambda x: ymin, lambda x: ymax)

    # Construct the sigma matrix from it
    sigma = np.array([[mom_xx, mom_xy], [mom_xy, mom_yy]])/norm

    # Get the Gaussian
    gaussian = get_super_gaussian(M, N, *popt)

    # Make the residual
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    residual = (get_super_gaussian(M, N, *popt) - image)

    # Find mu
    mu = popt[4:6]

    # If we are plotting it
    if(plot):
        # Set the plots to be larger
        plt.rcParams["figure.figsize"] = 12,3.5

        # Show the fit
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 12}
        plt.rc('font', **font)

        # Plot the image and the fit and the residual
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        # Show them
        ax1.imshow(image)
        ax2.imshow(gaussian)
        ax3.imshow(residual)

    # Return everything
    return mu, sigma, x, gaussian, residual
