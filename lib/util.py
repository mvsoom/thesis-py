import numpy as np
import dynesty
from dynesty import plotting
import matplotlib.pyplot as plt

def normalize_power(d, return_multiplier=False):
    multiplier = np.sqrt(len(d)/np.dot(d, d))
    normalized = multiplier*d
    return (normalized, multiplier) if return_multiplier else normalized

def dyplot(results, names=None):
    display(results.summary())
    display('Information (bans)', results.information[-1] * np.log10(np.exp(1)))

    fig, axes = dynesty.plotting.traceplot(
        results, show_titles=True,
        labels=names,
        verbose=True
    )
    plt.tight_layout()
    plt.show()

    fg, ax = dynesty.plotting.cornerplot(results, labels=names)
    plt.tight_layout()
    plt.show()

def importance_weights(results):
    weights = np.exp(results.logwt - results.logz[-1])
    return weights

def get_posterior_moments(results):
    mean, cov = dynesty.utils.mean_and_cov(
        results.samples, importance_weights(results)
    )
    return mean, cov

def resample_equal(results, n):
    samples = dynesty.utils.resample_equal(
        results.samples, importance_weights(results)
    )
    i = np.random.choice(len(samples), size=n, replace=False)
    return samples[i,:]