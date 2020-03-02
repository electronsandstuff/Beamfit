#!/usr/bin/env python

'''
BeamFit - Robust laser and charged particle beam image analysis
Copyright (C) 2020 Christopher M. Pierce (contact@chris-pierce.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

################################################################################
# Imports
################################################################################
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import scipy.special as special
import matplotlib.pyplot as plt
from . import gaussufunc
from .gaussufunc import *

################################################################################
# Functions for fitting
################################################################################
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
    image_filtered = image # ndimage.median_filter(image, size=10)

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

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# Wrapper functions
fit_func = lambda xdata, mux, muy, vxx, vxy, vyy, n, a, o: supergaussian(xdata[0], xdata[1], mux, muy, vxx, vxy, vyy, n, a, o)
fit_func_jac = lambda xdata, mux, muy, vxx, vxy, vyy, n, a, o: np.array(supergaussian_grad(xdata[0], xdata[1], mux, muy, vxx, vxy, vyy, n, a, o)).T

def h_to_hb(h):
    return np.array([
        h[0],
        h[1],
        np.sqrt(h[2]),
        h[3],
        np.sqrt(h[2]*h[4] - h[3]**2),
        np.sqrt(h[5]),
        h[6],
        h[7]
    ])

def h_to_hb_grad(hb, grad):
    return np.array([
        grad[0,:],
        grad[1,:],
        2*hb[2]*grad[2,:] - ((2*(hb[3]**2 + hb[4]**2))/hb[2]**3)*grad[4,:],
        grad[3,:] + (2*hb[3])/hb[2]**2*grad[4,:],
        (2*hb[4])/hb[2]**2*grad[4,:],
        2*hb[5]*grad[5,:],
        grad[6,:],
        grad[7,:]
    ])

def hb_to_h(hb):
    return np.array([
        hb[0],
        hb[1],
        hb[2]**2,
        hb[3],
        (hb[3]**2 + hb[4]**2)/hb[2]**2,
        hb[5]**2,
        hb[6],
        hb[7]
    ])

def fit_stochastic_LMA(x, y, w, h0, LMA_lambda=1, nbatch=8, epochs=4):
    # Convert to the bounded internal parameters
    hb = h_to_hb(h0)

    for _ in range(epochs):
        # Get a batch
        idx = np.arange(x.shape[1])
        np.random.shuffle(idx)
        batches = chunkIt(idx, nbatch)

        # Minimize once for each batch
        for batch in batches:
            # Calculate the gradient and value of the function
            model_grad = np.sqrt(w[batch])*h_to_hb_grad(hb, fit_func_jac(x[:,batch], *hb_to_h(hb)).T)
            model = hb[-2]*model_grad[-2,:] + np.sqrt(w[batch])*hb[-1]

            # Turn this into the gradient of the loss function and the approximation of the hessian
            loss_grad = 2*model_grad @ (np.sqrt(w[batch])*y[batch] - model).T
            hess = model_grad @ model_grad.T

            # Add the LMA damping term
            diag = np.diag_indices(8)
            hess[diag] = hess[diag]*(1+LMA_lambda)

            # Calculate the update
            delta = np.linalg.solve(hess, loss_grad)

            # Update the parameters
            hb = hb + delta

    # Calculate the Hessian of the loss function for error estimation
    model_grad = np.sqrt(w[batch])*h_to_hb_grad(hb, fit_func_jac(x[:,batch], *hb_to_h(hb)).T)
    hess = model_grad @ model_grad.T

    # Invert to get the variance-covariance matrix
    C = np.linalg.inv(hess)

    # Return the parameters and the variance covariance matrix
    return hb_to_h(hb), C

def fit_supergaussian(image, image_weights, sigma_threshold=4,
                      sigma_threshold_guess=1, nbatch=8, epochs=4,
                      smoothing=5, LMA_lambda=1):
    # Calculate the threshold
    threshold = np.exp(-1*sigma_threshold**2/2)

    # If there isn't a mask, make one
    if(not np.ma.isMaskedArray(image)):
        image = np.ma.array(image)

    # Get a median filtered image for thresholding
    image_filtered = ndimage.median_filter(image, size=smoothing)

    # Make a good initial guess
    guess = fit_gaussian_linear_least_squares(image_filtered,
                        sigma_threshold=sigma_threshold_guess)[0]

    # Get the Y data
    image_unwrapped = image.ravel()

    # Create a mask based on a threshold and a the actual mask
    mask_from_image = (image.mask == False)

    # Find the peak and bottom
    masked_image = np.array(image)[mask_from_image].ravel()
    ind = np.argsort(masked_image)
    low = np.mean(masked_image[ind][2:12])
    high = np.mean(masked_image[ind][-12:-2])

    # Make the threshold mask
    mask_from_threshold = (image - low)/(high - low) > threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold).ravel()

    # Convert to the new guess
    h0 = np.array([
        guess[4],
        guess[5],
        guess[2]**2*guess[3]**2,
        -guess[1]*guess[3]**2,
        (1+guess[1]**2)/guess[2]**2*guess[3]**2,
        guess[6],
        high - low,
        low
    ])

    # Mask the array
    y = np.array(image_unwrapped[mask_combined])
    w = np.array(image_weights.ravel()[mask_combined])

    # Get the X data
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    MN = np.vstack((M.ravel(), N.ravel()))

    # Mask it
    x = MN[:, mask_combined]

    # Fit it
    h, C = fit_stochastic_LMA(x, y, w, h0, nbatch=nbatch, epochs=epochs, LMA_lambda=LMA_lambda)

    # Return the fit and the covariance variance matrix
    return h, C

################################################################################
# Post processing
################################################################################
def get_mu_sigma(h, pixel_size):
    # Pull out the parameters
    mu = np.array([h[0], h[1]])*pixel_size

    # Get sigma
    sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
    n = h[5]
    scaling_factor = np.power(2, 3/n - 2)*special.gamma(1/n + 0.5)/np.sqrt(np.pi)
    sigma = sigma*scaling_factor*pixel_size**2

    # Return them
    return mu, sigma

def get_mu_sigma_std(h, C, pixel_size, pixel_size_std):
    # Pull out the parameters
    mu = np.array([h[0], h[1]])

    # Get sigma
    sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
    n = h[5]
    scaling_factor = np.power(2, 3/n - 2)*special.gamma(1/n + 0.5)/np.sqrt(np.pi)
    sigma = sigma*scaling_factor

    # Calculate mu's variance
    mu_var = np.array([C[0,0], C[1,1]])

    # Calculate Sigma's variance
    sigma_var = np.array([[C[2,2], C[3,3]], [C[3,3], C[4,4]]])
    n = h[5]
    n_var = C[5,5]
    scaling_factor_deriv = scaling_factor*(-1*np.log(8) + special.polygamma(0, 1/n + 0.5))/n**2
    scaling_factor_var = n_var*scaling_factor_deriv**2
    sigma_var = sigma_var*scaling_factor_var + sigma**2*scaling_factor_var + scaling_factor**2*sigma_var

    # Scale by the pixel size and calculate variances
    pixel_size_var = pixel_size_std**2
    mu_scaled_var = mu_var*pixel_size_var + pixel_size_var*mu**2 + mu_var*pixel_size**2
    pixel_size_squared_var = 4*pixel_size**2*pixel_size_var
    sigma_scaled_var = sigma_var*pixel_size_squared_var + pixel_size_squared_var*sigma**2 + sigma_var*pixel_size**4

    # Return them
    return np.sqrt(mu_scaled_var), np.sqrt(sigma_scaled_var)

def pretty_print_loc_and_size(h, C, pixel_size, pixel_size_std):
    # Pull out the components
    mu, sigma = get_mu_sigma(h, pixel_size)
    mu_std, sigma_std = get_mu_sigma_std(h, C, pixel_size, pixel_size_std)

    # Print them
    np.set_printoptions(precision=3)
    print('Position:  ', end='')
    print(mu*1e3, end='')
    print(' mm +/- ', end='')
    print(mu_std*1e6, end='')
    print(' um')
    print('Spot Size: ', end='')
    print(np.sqrt(sigma.diagonal())*1e3, end='')
    print(' mm +/- ', end='')
    size_std = np.abs(0.5/np.sqrt(sigma.diagonal())) * sigma_std.diagonal()
    print(size_std*1e6, end='')
    print(' um')

################################################################################
# Fit plotting
################################################################################
def plot_residuals(image, h):
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    residual = image - supergaussian(M,N,*h)
    plt.imshow(residual, cmap='seismic')

def plot_beam_contours(image, h):
    plt.imshow(image)
    M, N = np.mgrid[:image.shape[0], :image.shape[1]]
    gauss = supergaussian(M,N,*h)
    plt.contour(gauss, colors='r', levels=3)

def plot_threshold(image, sigma_threshold=4):
    # Calculate the threshold
    threshold = np.exp(-1*sigma_threshold**2/2)

    # Get a median filtered image for thresholding
    image_filtered = ndimage.median_filter(image, size=6)

    # Get the mask
    mask_from_image = (image.mask == False)

    # Find the peak and bottom
    masked_image = np.array(image)[mask_from_image].ravel()
    ind = np.argsort(masked_image)
    low = np.mean(masked_image[ind][2:12])
    high = np.mean(masked_image[ind][-12:-2])

    # Make the threshold mask
    mask_from_threshold = (image - low)/(high - low) > threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold)

    # Show it
    plt.imshow(mask_combined)
