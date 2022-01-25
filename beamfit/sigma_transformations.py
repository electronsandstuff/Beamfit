"""
Many transformations in this section are from the paper Pinheiro, J. C., & Bates, D. M. (1996). Unconstrained
Parameterizations for Variance-Covariance Matrices. Statistics and Computing, 6, 289–296.
"""
import numpy as np


def eigen2d(s):
    if np.isclose(s[0, 1], 0.0):
        return np.identity(2), np.diag(s)
    delta = np.sqrt(4 * s[0, 1] ** 2 + (s[0, 0] - s[1, 1]) ** 2)
    lmbda = np.array([(s[0, 0] + s[1, 1] - delta) / 2, (s[0, 0] + s[1, 1] + delta) / 2])
    u = np.array([[(lmbda[0] - s[1, 1]) / s[0, 1], (lmbda[1] - s[1, 1]) / s[0, 1]], [1, 1]])
    u[:, 0] /= np.sqrt(u[0, 0] ** 2 + u[1, 0] ** 2)
    u[:, 1] /= np.sqrt(u[0, 1] ** 2 + u[1, 1] ** 2)
    return u, lmbda


def a_to_theta(a):
    b = np.exp(a)
    return np.pi * b / (1 + b)


def a_to_theta_grad(a):
    b = np.exp(a)
    return b*np.pi/(1+b) - b**2*np.pi/(1+b)**2


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

    def reverse_grad(self, st):
        raise NotImplementedError


class Cholesky(SigmaParameterization):
    def forward(self, s):
        a = np.sqrt(s[0, 0])
        b = s[0, 1] / a
        c = np.sqrt(s[1, 1] - b**2)
        return np.array([a, b, c])

    def reverse(self, st):
        return np.array([[st[0]**2, st[0]*st[1]], [st[0]*st[1], st[1]**2 + st[2]**2]])

    def reverse_grad(self, st):
        return np.array([[2*st[0], 0, 0], [st[1], st[0], 0], [0, 2*st[1], 2*st[2]]])


class LogCholesky(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        return np.array([np.log(st[0]), st[1], np.log(st[2])])

    def reverse(self, st):
        return self.ch.reverse(np.array([np.exp(st[0]), st[1], np.exp(st[2])]))

    def reverse_grad(self, st):
        jf = np.array([[np.exp(st[0]), 0, 0], [0, 1, 0], [0, 0, np.exp(st[2])]])
        return self.ch.reverse_grad(np.array([np.exp(st[0]), st[1], np.exp(st[2])])) @ jf


class Spherical(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        theta = np.arctan(st[2]/st[1])
        return np.array([np.log(st[0]), np.log(st[1]**2 + st[2]**2)/2, np.log(theta/(np.pi - theta))])

    def reverse(self, st):
        theta = a_to_theta(st[2])
        return self.ch.reverse(np.array([np.exp(st[0]), np.cos(theta)*np.exp(st[1]), np.sin(theta)*np.exp(st[1])]))

    def reverse_grad(self, st):
        theta = a_to_theta(st[2])
        x = a_to_theta_grad(st[2])
        jf = np.array([[np.exp(st[0]), 0, 0], [0, np.cos(theta)*np.exp(st[1]), -np.sin(theta)*x*np.exp(st[1])],
                       [0, np.sin(theta)*np.exp(st[1]), np.cos(theta)*x*np.exp(st[1])]])
        return self.ch.reverse_grad(
            np.array([np.exp(st[0]), np.cos(theta)*np.exp(st[1]), np.sin(theta)*np.exp(st[1])])) @ jf


class MatrixLogarithm(SigmaParameterization):
    def forward(self, s):
        u, v = eigen2d(s)
        return (u @ np.diag(np.log(v)) @ u.T)[np.triu_indices(2)]

    def reverse(self, st):
        u, v = eigen2d(np.array([[st[0], st[1]], [st[1], st[2]]]))
        return u @ np.diag(np.exp(v)) @ u.T


class Givens(SigmaParameterization):
    def forward(self, s):
        u, v = eigen2d(s)
        theta = np.arccos(u[0, 0])
        return np.array([np.log(v[0]), np.log(v[1] - v[0]), np.log(theta/(np.pi - theta))])

    def reverse(self, st):
        theta = a_to_theta(st[2])
        u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        v = [np.exp(st[0]), np.exp(st[0]) + np.exp(st[1])]
        return u @ np.diag(v) @ u.T
