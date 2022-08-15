"""
Many transformations in this section are from the paper Pinheiro, J. C., & Bates, D. M. (1996). Unconstrained
Parameterizations for Variance-Covariance Matrices. Statistics and Computing, 6, 289–296.
"""
import numpy as np
import tensorflow as tf


def is_tf(x):
    """
    Detects if x is a tensorflow object
    """
    return isinstance(x, tf.Variable) or isinstance(x, tf.Tensor)


def eigen2d(s):
    sqrt, atan2 = (tf.sqrt, tf.atan2) if is_tf(s) else (np.sqrt, np.arctan2)
    delta = sqrt(4 * s[1] ** 2 + (s[0] - s[2]) ** 2)
    lmbda1 = (s[0] + s[2] - delta)/2
    lmbda2 = (s[0] + s[2] + delta)/2
    theta = atan2(s[1], lmbda1 - s[2])
    return tf.stack([theta, lmbda1, lmbda2]) if is_tf(s) else np.array([theta, lmbda1, lmbda2])


def eigen2d_grad(s, mindelta=1e-100):
    delta = max(np.sqrt(4 * s[1] ** 2 + (s[0] - s[2]) ** 2), mindelta)

    lmbda1 = (s[0] + s[2] - delta)/2
    dlmbda1 = [(1 - (s[0] - s[2])/delta)/2, -2*s[1]/delta, (1 + (s[0] - s[2])/delta)/2]
    dlmbda2 = [(1 + (s[0] - s[2])/delta)/2, 2*s[1]/delta, (1 - (s[0] - s[2])/delta)/2]

    if np.isclose(s[1], 0):
        dtheta = [0, 0, 0]
    else:
        da = np.array([
            dlmbda1[0] / s[1],
            (s[1] * dlmbda1[1] - lmbda1) / s[1] ** 2 + s[2]/s[1]**2,
            (dlmbda1[2] - 1) / s[1]
        ])
        a = (lmbda1 - s[2]) / s[1]
        x = a / np.sqrt(1 + a ** 2)
        dx = (da*np.sqrt(1 + a ** 2) - a/np.sqrt(1 + a ** 2)*a*da)/(1 + a**2)
        dtheta = -dx/np.sqrt(1 - x**2)

    return np.array([dtheta, dlmbda1, dlmbda2])


def rot_mat_2d(t):
    c, s = (tf.cos(t), tf.sin(t)) if is_tf(t) else (np.cos(t), np.sin(t))
    return tf.reshape(tf.stack([c, s, -s, c]), 2*[2]) if is_tf(t) else np.array([[c, s], [-s, c]])


def a_to_theta(a):
    b = tf.exp(a) if is_tf(a) else np.exp(a)
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
        l = self.reverse_l(st)  # If not overriden, use the lower cholesky factor
        return l @ (tf.transpose if isinstance(st, tf.Variable) else np.transpose)(l)

    def reverse_l(self, st):
        """
        Transforms the unconstrained parameterization into the 2x2 lower triangular cholesky factor
        """
        raise NotImplementedError

    def reverse_grad(self, st):
        raise NotImplementedError


class Cholesky(SigmaParameterization):
    def forward(self, s):
        a = np.sqrt(s[0, 0])
        b = s[0, 1] / a
        c = np.sqrt(s[1, 1] - b ** 2)
        return np.array([a, b, c])

    def reverse_l(self, st):
        xp = tf if is_tf(st) else np
        return xp.reshape(xp.stack([st[0], 0.0, st[1], st[2]]), (2, 2))

    def reverse_grad(self, st):
        return np.array([[2*st[0], 0, 0], [st[1], st[0], 0], [0, 2*st[1], 2*st[2]]])


class LogCholesky(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        return np.array([np.log(st[0]), st[1], np.log(st[2])])

    def reverse_l(self, st):
        xp = tf if is_tf(st) else np
        return self.ch.reverse_l(xp.stack([xp.exp(st[0]), st[1], xp.exp(st[2])]))

    def reverse_grad(self, st):
        jf = np.array([[np.exp(st[0]), 0, 0], [0, 1, 0], [0, 0, np.exp(st[2])]])
        return self.ch.reverse_grad(np.array([np.exp(st[0]), st[1], np.exp(st[2])])) @ jf


class Spherical(SigmaParameterization):
    def __init__(self):
        self.ch = Cholesky()

    def forward(self, s):
        st = self.ch.forward(s)
        theta = np.arccos(st[1]/np.sqrt(st[1]**2 + st[2]**2))
        return np.array([np.log(st[0]), np.log(st[1]**2 + st[2]**2)/2, np.log(theta/(np.pi - theta))])

    def reverse_l(self, st):
        th = a_to_theta(st[2])
        xp = tf if is_tf(st) else np
        return xp.reshape(xp.stack([xp.exp(st[0]), 0.0, xp.cos(th)*xp.exp(st[1]), xp.sin(th)*xp.exp(st[1])]), (2, 2))

    def reverse_grad(self, st):
        theta = a_to_theta(st[2])
        dtheta = a_to_theta_grad(st[2])
        jf = np.array([
            [np.exp(st[0]), 0, 0],
            [0, np.cos(theta)*np.exp(st[1]), -np.sin(theta)*dtheta*np.exp(st[1])],
            [0, np.sin(theta)*np.exp(st[1]), np.cos(theta)*dtheta*np.exp(st[1])]
        ])
        return self.ch.reverse_grad(np.array([
            np.exp(st[0]),
            np.cos(theta)*np.exp(st[1]),
            np.sin(theta)*np.exp(st[1])
        ])) @ jf


class MatrixLogarithm(SigmaParameterization):
    def forward(self, s):
        theta, v1, v2 = eigen2d([s[0, 0], s[0, 1], s[1, 1]])
        u = rot_mat_2d(theta)
        return (u.T @ np.diag(np.log([v1, v2])) @ u)[np.triu_indices(2)]

    def reverse_l(self, st):
        eigs = eigen2d(st)
        u = rot_mat_2d(eigs[0])
        diag, xp = (tf.linalg.diag, tf) if is_tf(st) else (np.diag, np)
        return xp.transpose(u) @ diag(xp.sqrt(xp.exp([eigs[1], eigs[2]])))

    def reverse_grad(self, st):
        theta, v1, v2 = eigen2d(st)
        dtheta, dv1, dv2 = eigen2d_grad(st)

        u = rot_mat_2d(theta)
        c, s = np.cos(theta), np.sin(theta)
        du = np.array([[-s, c], [-c, -s]])[None, :, :] * dtheta[:, None, None]

        x = np.diag(np.exp([v1, v2]))
        dx = np.zeros((3, 2, 2))
        dx[:, 0, 0] = np.exp(v1)*dv1
        dx[:, 1, 1] = np.exp(v2)*dv2

        dr = np.einsum('ikj,kl,lm->ijm', du, x, u)
        dr += np.einsum('kj,ikl,lm->ijm', u, dx, u)
        dr += np.einsum('kj,kl,ilm->ijm', u, x, du)

        return np.array([dr[:, 0, 0], dr[:, 0, 1], dr[:, 1, 1]])


class Givens(SigmaParameterization):
    def forward(self, s):
        theta, v1, v2 = eigen2d([s[0, 0], s[0, 1], s[1, 1]])
        return np.array([np.log(v1), np.log(v2 - v1), np.log(theta/(np.pi - theta))])

    def reverse_l(self, st):
        diag, xp = (tf.linalg.diag, tf) if is_tf(st) else (np.diag, np)
        u = rot_mat_2d(a_to_theta(st[2]))
        v = [xp.exp(st[0]), xp.exp(st[0]) + xp.exp(st[1])]
        return xp.transpose(u) @ xp.sqrt(diag(v))

    def reverse_grad(self, st):
        theta = a_to_theta(st[2])
        dtheta = a_to_theta_grad(st[2])

        u = rot_mat_2d(theta)
        c, s = np.cos(theta), np.sin(theta)
        du = np.array([[-s, c], [-c, -s]])[None, :, :] * np.array([0, 0, dtheta])[:, None, None]

        v = [np.exp(st[0]), np.exp(st[0]) + np.exp(st[1])]
        dv1 = [np.exp(st[0]), 0, 0]
        dv2 = [np.exp(st[0]), np.exp(st[1]), 0]

        x = np.diag(v)
        dx = np.zeros((3, 2, 2))
        dx[:, 0, 0] = dv1
        dx[:, 1, 1] = dv2

        dr = np.einsum('ikj,kl,lm->ijm', du, x, u)
        dr += np.einsum('kj,ikl,lm->ijm', u, dx, u)
        dr += np.einsum('kj,kl,ilm->ijm', u, x, du)

        return np.array([dr[:, 0, 0], dr[:, 0, 1], dr[:, 1, 1]])
