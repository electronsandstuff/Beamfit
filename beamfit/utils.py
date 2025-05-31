from __future__ import annotations
import numpy as np
import scipy.special as special
import scipy.signal as signal
from . import factory
from dataclasses import dataclass
from typing import List, Dict, Union, Any


@dataclass
class Setting:
    name: str
    default: str
    stype: str = "str"  # can be 'str', 'list', 'settings_list'
    list_values: List[str] = None  # Options used with 'list' type
    list_settings: Dict[str, List[Setting]] = (
        None  # Map from names in list_values to lists of settings for that option
    )


def get_image_and_weight(raw_images, dark_fields, mask):
    image = np.ma.masked_array(
        data=np.mean(raw_images, axis=0) - np.mean(dark_fields, axis=0), mask=mask
    )
    std_image = np.ma.masked_array(
        data=np.sqrt(
            np.std(raw_images, axis=0) ** 2 + np.std(dark_fields, axis=0) ** 2
        ),
        mask=mask,
    )
    image_weight = len(raw_images) / std_image**2
    return image, image_weight


class AnalysisMethod:
    def __init__(self, sigma_threshold=None, median_filter_size=None):
        self.sigma_threshold = sigma_threshold
        self.median_filter_size = median_filter_size

    def fit(self, image, image_sigmas=None):
        """
        Measure the RMS size and centroid of the supplied image. If the image is passed as a masked array then
        (depending on support by the fitting method) only unmasked pixels are considered. This can be useful for
        selecting oddly shaped regions of interest.

        :param image: 2D np.ndarray or np.ma.array, the image as a grayscale array or masked array of pixels
        :param image_sigmas: (optional) the uncertainty in each pixel intensity
        :return: AnalysisResult object (depends on analysis method)
        """
        if not np.ma.isMaskedArray(image):  # Make a mask if there isn't one
            image = np.ma.array(image)
        if self.median_filter_size is not None:  # Median filter the image if required
            image = np.ma.array(
                signal.medfilt2d(image, kernel_size=self.median_filter_size),
                mask=image.mask,
            )
        if self.sigma_threshold is not None:  # Apply threshold if provided
            image.mask = np.bitwise_or(
                image.mask, image < (image.max() * np.exp(-(self.sigma_threshold**2)))
            )
        return self.__fit__(image, image_sigmas)

    def __fit__(self, image, image_sigmas=None):
        raise NotImplementedError

    def get_config_dict(self):
        ret = {
            "sigma_threshold": self.sigma_threshold,
            "median_filter_size": self.median_filter_size,
        }
        ret.update(self.__get_config_dict__())
        return ret

    def __get_config_dict__(self):
        return {}

    def get_settings(self) -> List[Setting]:
        arr = [
            Setting("Sigma Threshold", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Sigma Threshold Size", "3.0"),
            Setting("Median Filter", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Median Filter Size", "3"),
        ]
        return arr + self.__get_settings__()

    def __get_settings__(self) -> List[Setting]:
        raise NotImplementedError()

    def set_from_settings(self, settings: Dict[str, str]):
        if settings["Sigma Threshold"] == "On":
            sigma_threshold = float(settings["Sigma Threshold Size"])
            if sigma_threshold <= 0.0:
                raise ValueError(
                    f"Sigma threshold must be greater than zero, got {sigma_threshold}"
                )
            self.sigma_threshold = sigma_threshold
        elif settings["Sigma Threshold"] == "Off":
            self.sigma_threshold = None
        else:
            raise ValueError(
                f'Unrecognized value for "Sigma Threshold": "{settings["Sigma Threshold"]}"'
            )
        if settings["Median Filter"] == "On":
            median_filter_size = int(settings["Median Filter Size"])
            if median_filter_size < 3:
                raise ValueError(
                    f"Median filter size must be at least 3, got {median_filter_size}"
                )
            if median_filter_size % 2 == 0:
                raise ValueError(
                    f"Median filter size must be odd integer, not {median_filter_size}"
                )
            self.median_filter_size = median_filter_size
        elif settings["Median Filter"] == "Off":
            self.median_filter_size = None
        else:
            raise ValueError(
                f'Unrecognized value for "Median Filter": "{settings["Median Filter"]}"'
            )
        self.__set_from_settings__(settings)

    def __set_from_settings__(self, settings: Dict[str, Union[str, Dict[str, Any]]]):
        raise NotImplementedError()


def get_config_dict_analysis_method(m: AnalysisMethod):
    return {"type": type(m).__name__, "config": m.get_config_dict()}


def create_analysis_method_from_dict(d):
    return factory.create("analysis", d["type"], **d["config"])


class AnalysisResult:
    def get_mean(self):
        raise NotImplementedError

    def get_covariance_matrix(self):
        raise NotImplementedError

    def get_mean_std(self):
        return None

    def get_covariance_matrix_std(self):
        return None


def super_gaussian_scaling_factor(n):
    return special.gamma((2 + n) / n) / 2 / special.gamma(1 + 1 / n)


def super_gaussian_scaling_factor_grad(n):
    n = n
    scaling_factor_deriv = (
        special.gamma((2 + n) / n) / 2 / n**2 * special.polygamma(0, 1 + 1 / n)
    )
    scaling_factor_deriv += (
        (1 / n - (2 + n) / n**2)
        * special.gamma((2 + n) / n)
        * special.polygamma(0, (2 + n) / n)
        / 2
    )
    scaling_factor_deriv /= special.gamma(1 + 1 / n)
    return scaling_factor_deriv


class SuperGaussianResult(AnalysisResult):
    def __init__(
        self, mu=np.zeros(2), sigma=np.identity(2), a=1.0, o=0.0, n=1.0, c=None, h=None
    ):
        if h is None:
            self.mu = mu  # Centroid
            self.sigma = sigma  # Variance-covariance matrix
            self.a = a  # Amplitude
            self.o = o  # Background offset
            self.n = n  # Supergaussian parameter
        else:
            self.h = h
        self.c = c  # covariance matrix of h

    @property
    def h(self):
        return np.array(
            [
                self.mu[0],
                self.mu[1],
                self.sigma[0, 0],
                self.sigma[0, 1],
                self.sigma[1, 1],
                self.n,
                self.a,
                self.o,
            ]
        )

    @h.setter
    def h(self, h):
        self.mu = np.array([h[0], h[1]])
        self.sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
        self.n = h[5]
        self.a = h[6]
        self.o = h[7]

    def get_mean(self):
        return self.mu

    def get_covariance_matrix(self):
        return self.sigma * super_gaussian_scaling_factor(self.n)

    def get_mean_std(self):
        return np.sqrt(np.array([self.c[0, 0], self.c[1, 1]]))

    def get_covariance_matrix_std(self):
        # Find the Jacobian of the scaling transformation
        scaling_j = np.identity(4)
        scaling_j[:3, :3] *= super_gaussian_scaling_factor(self.n)
        scaling_j[:3, 3] = self.h[2:5] * super_gaussian_scaling_factor_grad(self.n)

        # Get the covariance matrix of our variables
        sigma_n_cov = self.c[2:6, 2:6]

        # Transform it
        sigma_n_cov_scaled = scaling_j @ sigma_n_cov @ scaling_j.T
        return np.sqrt(
            np.array(
                [
                    [sigma_n_cov_scaled[0, 0], sigma_n_cov_scaled[1, 1]],
                    [sigma_n_cov_scaled[1, 1], sigma_n_cov_scaled[2, 2]],
                ]
            )
        )
