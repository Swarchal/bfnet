"""
Basic transforms using numpy arrays rather than PIL images

NOTE:
    Have to return copies or pytorch complains about negative strides.
"""

import random
import numpy as np


class RandomHFlip:
    """Random horizontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            x = x[:, ::-1].copy()
        return x


class RandomVFlip:
    """Random vertical flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            x = x[::-1, :].copy()
        return x


class RandomRotate90:
    """Random 90 degree rotation"""
    def __call__(self, x):
        n_rotations = random.choice([0, 1, 2, 3])
        if n_rotations != 0:
            x = np.rot90(x, n_rotations, axes=(0, 1)).copy()
        return x



###########################################
# transforms which work on pairs of images
###########################################

class RandomHFlipPair:
    """Random horizontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, tup):
        x, y = tup
        if random.random() < self.prob:
            x = x[:, ::-1].copy()
            y = y[:, ::-1].copy()
        return x, y


class RandomVFlipPair:
    """Random vertical flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, tup):
        x, y = tup
        if random.random() < self.prob:
            x = x[::-1, :].copy()
            y = y[::-1, :].copy()
        return x, y


class RandomRotate90Pair:
    """Random 90 degree rotation"""
    def __call__(self, tup):
        x, y = tup
        n_rotations = random.choice([0, 1, 2, 3])
        if n_rotations != 0:
            x = np.rot90(x, n_rotations, axes=(0, 1)).copy()
            y = np.rot90(y, n_rotations, axes=(0, 1)).copy()
        return x, y

