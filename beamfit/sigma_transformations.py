"""
Many transformations in this section are from the paper Pinheiro, J. C., & Bates, D. M. (1996). Unconstrained
Parameterizations for Variance-Covariance Matrices. Statistics and Computing, 6, 289–296.
"""
import numpy as np

class SigmaParameterization:
    def forward(self, s):
        """
        Transforms the 2x2 matrix s into an unconstrained parameterization
        """
        raise NotImplementedError

    def reverse(self, st):
        """
        Transforms the unconstrained parameterization into the 2x2 sigma matrix
        """
        raise NotImplementedError

    def forward_grad(self, s, grad):
        """
        Transforms the gradient of a function WRT the sigma matrix into the gradient WRT to the unconstrained
        parameterization
        """
        raise NotImplementedError


class Cholesky(SigmaParameterization):
    def forward(self, s):
        a = np.sqrt(s[0, 0])
        b = s[0, 1] / a
        c = np.sqrt(s[1, 1] - b**2)
        return np.array([a, b, c])

    def reverse(self, st):
        return np.array([[st[0]**2, st[0]*st[1]], [st[0]*st[1], st[1]**2 + st[2]**2]])


class LogCholesky(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        return np.array([np.log(st[0]), st[1], np.log(st[2])])

    def reverse(self, st):
        return self.ch.reverse(np.array([np.exp(st[0]), st[1], np.exp(st[2])]))


class Spherical(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        theta = np.arctan(st[2]/st[1])
        return np.array([np.log(st[0]), np.log(st[1]**2 + st[2]**2)/2, np.log(theta/(np.pi - theta))])

    def reverse(self, st):
        a = np.exp(st[2])
        theta = np.pi*a/(1+a)
        return self.ch.reverse(np.array([np.exp(st[0]), np.cos(theta)*np.exp(st[1]), np.sin(theta)*np.exp(st[1])]))
