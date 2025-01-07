import numpy as np


def z_normalize(ts):
    if ts.ndim == 1:
        return (ts - np.mean(ts)) / np.std(ts)

    return (ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)
