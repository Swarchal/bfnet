"""
Simple Online variance calculation with numpy arrays
"""


class OnlineVariance:
    """Welfords' online variance calculation"""
    def __init__(self, arr=None):
        self.arr = arr
        self.mean = arr if arr else 0
        self.count = 1 if arr else 0
        self._M2 = 0

    def __call__(self, x):
        self.update(x)

    @property
    def variance(self):
        return self._M2 / self.count

    @property
    def sample_variance(self):
        return self._M2 / (self.count - 1)

    def update(self, arr):
        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta_prime = arr - self.mean
        self._M2 += delta * delta_prime

