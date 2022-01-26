import numpy as np
import scipy.special as special


def get_image_and_weight(raw_images, dark_fields, mask):
    image = np.ma.masked_array(data=np.mean(raw_images, axis=0) - np.mean(dark_fields, axis=0), mask=mask)
    std_image = np.ma.masked_array(data=np.sqrt(np.std(raw_images, axis=0) ** 2 + np.std(dark_fields, axis=0) ** 2),
                                   mask=mask)
    image_weight = len(raw_images) / std_image ** 2
    return image, image_weight


class AnalysisMethod:
    def __init__(self, sigma_threshold=None):
        self.sigma_threshold = sigma_threshold

    def fit(self, image):
        if not np.ma.isMaskedArray(image):  # Make a mask if there isn't one
            image = np.ma.array(image)
        if self.sigma_threshold is not None:
            image.mask = np.bitwise_and(image.mask, image < image.max() * np.exp(-self.sigma_threshold))
        return self.__fit__(image)

    def __fit__(self, image):
        raise NotImplementedError

    def get_config_dict(self):
        ret = {'sigma_threshold': self.sigma_threshold}
        ret.update(self.__get_config_dict__())
        return ret

    def __get_config_dict__(self):
        return {}


class AnalysisResult:
    def get_mean(self):
        raise NotImplementedError

    def get_covariance_matrix(self):
        raise NotImplementedError


class SuperGaussianResult:
    def __init__(self, mu=np.zeros(2), sigma=np.identity(2), a=1.0, o=0.0, n=1.0):
        self.mu = mu  # Centroid
        self.sigma = sigma  # Variance-covariance matrix
        self.a = a  # Amplitude
        self.o = o  # Background offset
        self.n = n  # Supergaussian parameter

    @property
    def h(self):
        return np.array([
            self.mu[0], self.mu[1],
            self.sigma[0, 0], self.sigma[0, 1], self.sigma[1, 1],
            self.n,
            self.a,
            self.o
        ])

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
        scaling_factor = special.gamma((2 + self.n) / self.n) / 2 / special.gamma(1 + 1 / self.n)
        return self.sigma * scaling_factor
