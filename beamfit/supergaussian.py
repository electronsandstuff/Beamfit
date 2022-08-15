import numpy as np
from typing import List, Dict, Union, Any
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow informational output
import tensorflow as tf

from . import factory
from .utils import AnalysisMethod, SuperGaussianResult, Setting
from .sigma_transformations import LogCholesky


class SupergaussianLayer(tf.keras.layers.Layer):
    def __init__(self, h=None, sig_param=LogCholesky()):
        super(SupergaussianLayer, self).__init__()
        self.a = tf.Variable(1.0, dtype="float32", trainable=True, )
        self.nl = tf.Variable(1.0, dtype="float32", trainable=True, )  # Log of supergaussian parameter
        self.o = tf.Variable(0.0, dtype="float32", trainable=True, )
        self.mu = tf.Variable(initial_value=[32., 32.], dtype="float32", trainable=True)
        self.theta = tf.Variable([3., 0., 3.], dtype="float32", trainable=True)  # Unconstrained parameters of sigma mat
        self.sp = sig_param
        if h is not None:
            self.set_h(h)

    def set_h(self, h):
        self.mu.assign([h[0], h[1]])
        self.a.assign(h[6])
        self.nl.assign(np.log(h[5]))
        self.o.assign(h[7])
        self.theta.assign(self.sp.forward(np.array([[h[2], h[3]], [h[3], h[4]]])))

    def get_h(self):
        return np.concatenate([
            self.mu.numpy(),
            self.sp.reverse(self.theta).numpy()[np.triu_indices(2)],
            [np.exp(self.nl.numpy())],
            [self.a.numpy()],
            [self.o.numpy()]
        ])

    def call(self, inputs):
        z = tf.linalg.triangular_solve(self.sp.reverse_l(self.theta), tf.transpose(inputs - self.mu))
        return self.a * tf.exp(-tf.math.pow(tf.reduce_sum(z * z, 0) / 2., tf.exp(self.nl))) + self.o


class SuperGaussian(AnalysisMethod):
    def __init__(self, predfun="GaussianProfile1D", predfun_args=None, sig_param='LogCholesky', sig_param_args=None,
                 maxfev=100, **kwargs):
        super().__init__(**kwargs)
        if sig_param_args is None:
            sig_param_args = {}
        if predfun_args is None:
            predfun_args = {}
        self.predfun = factory.create('analysis', predfun, **predfun_args)
        self.predfun_args = predfun_args
        self.maxfev = maxfev
        self.sig_param = factory.create('sig_param', sig_param, **sig_param_args)
        self.sig_param_args = sig_param_args

    def __fit__(self, image, image_sigmas=None):
        lo, hi = image.min(), image.max()  # Normalize image
        image = (image - lo)/(hi - lo)

        # Get the x and y data for the fit
        m, n = np.mgrid[:image.shape[0], :image.shape[1]]
        x = np.vstack((m[~image.mask], n[~image.mask])).T
        y = np.array(image[~image.mask])

        # Construct and fit the supergaussian model
        model = tf.keras.Sequential([SupergaussianLayer(self.predfun.fit(image).h, sig_param=self.sig_param)])
        model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
        model.fit(x.astype(np.float32), y.astype(np.float32), epochs=16, batch_size=10000, verbose=0)

        # Return the fit and the covariance variance matrix
        return SuperGaussianResult(h=model.get_layer(index=0).get_h())

    def __get_config_dict__(self):
        return {'predfun': type(self.predfun).__name__, 'predfun_args': self.predfun_args,
                'sig_param': type(self.sig_param).__name__, 'sig_param_args': self.sig_param_args,
                'maxfev': self.maxfev}

    def __get_settings__(self) -> List[Setting]:
        pred_funs = [x for x in factory.get_names('analysis') if x != 'SuperGaussian']
        pred_fun_settings = {x: factory.create('analysis', x).get_settings() for x in pred_funs}
        return [
            Setting(
                'Intial Prediction Method', 'GaussianProfile1D', stype='settings_list', list_values=pred_funs,
                list_settings=pred_fun_settings
            ),
            Setting(
                'Covariance Matrix Parameterization', 'LogCholesky', stype='list',
                list_values=factory.get_names('sig_param')
            ),
            Setting('Max Function Evaluation', '100')
        ]

    def __set_from_settings__(self, settings: Dict[str, Union[str, Dict[str, Any]]]):
        self.predfun = factory.create('analysis', settings['Intial Prediction Method']['name'])
        self.predfun.set_from_settings(settings['Intial Prediction Method']['settings'])
        self.sig_param = factory.create('sig_param', settings['Covariance Matrix Parameterization'])
        maxfev = int(settings['Max Function Evaluation'])
        if maxfev < 1:
            raise ValueError(f'maxfev must be greater than zero, got {maxfev}')
        self.maxfev = maxfev


def fit_supergaussian(image, image_weights=None, prediction_func="2D_linear_Gaussian", sigma_threshold=3,
                      sigma_threshold_guess=1, smoothing=5, maxfev=100):  # Backwards compatibility
    predfun = {'2D_linear_Gaussian': 'GaussianLinearLeastSquares', '1D_Gaussian': 'GaussianProfile1D'}[prediction_func]
    ret = SuperGaussian(predfun=predfun, predfun_args={'sigma_threshold': sigma_threshold_guess,
                                                       'median_filter_size': smoothing},
                        sigma_threshold=sigma_threshold, maxfev=maxfev).fit(image, image_weights)
    return ret.h, ret.c
