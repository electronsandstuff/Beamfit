from .factory import register, create, unregister, get_names
from .fit_param_conversion import get_mu_sigma, get_mu_sigma_std
from .gaussian_fit_1d import fit_gaussian_1d, GaussianProfile1D
from .gaussian_linear_least_squares import GaussianLinearLeastSquares
from .plotting_and_output import pretty_print_loc_and_size, plot_threshold, plot_residuals, plot_beam_contours
from .supergaussian import fit_supergaussian, SuperGaussian
from .utils import get_image_and_weight, get_config_dict_analysis_method, create_analysis_method_from_dict
from .sigma_transformations import Cholesky, LogCholesky, Spherical, MatrixLogarithm, Givens, eigen2d_grad, eigen2d
from .rms_integration import RMSIntegration
from gaussufunc import supergaussian

for o in [GaussianProfile1D, GaussianLinearLeastSquares, SuperGaussian]:  # Register all analysis methods to the factory
    register('analysis', o.__name__, o)

for o in [Cholesky, LogCholesky, Spherical, MatrixLogarithm, Givens]:  # Register the sigma parameterizations
    register('sig_param', o.__name__, o)
