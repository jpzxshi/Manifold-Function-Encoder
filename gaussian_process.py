"""
@author: jpzxshi
"""
import numpy as np
from sklearn import gaussian_process as gp
from itertools import product


class Gaussian_process:
    '''Generate Gaussian process.
    '''
    def __init__(self, intervals, mean, std, length_scale, features, e=1e-12):
        self.intervals = intervals # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000
        self.e = e # 1e-12

    def generate(self, num):
        if isinstance(self.intervals[0], list):
            itvs = []
            for interval in self.intervals:
                itvs.append(np.linspace(interval[0], interval[1], num=self.features))
            x = np.array(list(product(*itvs)))
            d = len(self.intervals)
        else:
            x = np.linspace(self.intervals[0], self.intervals[1], num=self.features)[:, None]
            d = 1
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + self.e * np.eye(x.shape[0]))
        res = (L @ np.random.randn(x.shape[0], num)).transpose() * self.std + self.mean # [num, features ** d]
        return res.reshape([num] + [self.features] * d)

class Gaussian_process_period:
    '''Generate period Gaussian process.
    '''
    def __init__(self, intervals, mean, std, length_scale, features, period, e=1e-12):
        self.intervals = intervals # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000
        self.e = e # 1e-12
        self.period = period

    def generate(self, num):
        if isinstance(self.intervals[0], list):
            itvs = []
            for interval in self.intervals:
                itvs.append(np.linspace(interval[0], interval[1], num=self.features))
            x = np.array(list(product(*itvs)))
            d = len(self.intervals)
        else:
            x = np.linspace(self.intervals[0], self.intervals[1], num=self.features)[:, None]
            d = 1
        #A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        A = gp.kernels.ExpSineSquared(length_scale=self.length_scale, periodicity = self.period)(x)
        L = np.linalg.cholesky(A + self.e * np.eye(x.shape[0]))
        res = (L @ np.random.randn(x.shape[0], num)).transpose() * self.std + self.mean # [num, features ** d]
        return res.reshape([num] + [self.features] * d)

