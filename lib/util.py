import numpy as np

def normalize_power(d, return_multiplier=False):
    multiplier = np.sqrt(len(d)/np.dot(d, d))
    normalized = multiplier*d
    return (normalized, multiplier) if return_multiplier else normalized