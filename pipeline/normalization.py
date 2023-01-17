from typing import Dict
import numpy as np
from utils import Normalization


class WelfordsZTransform(Normalization):
    """
    Applies a z-transform to the extracted features. Each step, mean and standard deviation
    are updated according to Welford's algorithm to estimate variance.

    Parameters:
        biased_std (bool): if True, use biased standard deviation 1 / n instead of 1 / (n-1)
    """

    def __init__(self, biased_std: bool = False):
        self.count = 0
        self.biased = biased_std
        self.mean = {}
        self.m2 = {}

    def normalize(self, processed: Dict[str, float]):
        """
        Updates mean and M2 according to Welford's algorithm and applies a z-transform to
        the extracted features.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        self.count += 1

        for key, val in processed.items():
            # initialize running stats
            if key not in self.mean:
                self.mean[key] = val
            if key not in self.m2:
                self.m2[key] = 0

            # update running stats according to Welford's algorithm
            delta = val - self.mean[key]
            self.mean[key] += delta / self.count
            delta2 = val - self.mean[key]
            self.m2[key] += delta * delta2

            # normalize current feature
            if self.count < 2:
                val = 0
            else:
                n = self.count if self.biased else self.count - 1
                val = (val - self.mean[key]) / np.sqrt(self.m2[key] / n)
            processed[key] = val
