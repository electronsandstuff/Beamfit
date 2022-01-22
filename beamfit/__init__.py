from .supergaussian import fit_supergaussian, SuperGaussian
from .fit_param_conversion import get_mu_sigma, get_mu_sigma_std
from .plotting_and_output import pretty_print_loc_and_size, plot_threshold, plot_residuals, plot_beam_contours
from .utils import get_image_and_weight
from gaussufunc import supergaussian
from .gaussian_fit_1d import fit_gaussian_1d, GaussianProfile1D
from .gaussian_linear_least_squares import GaussianLinearLeastSquares

from .factory import register, create, unregister, get_names
for o in [GaussianProfile1D, GaussianLinearLeastSquares, SuperGaussian]:  # Register all analysis methods to the factory
    register(o().get_name(), o)
