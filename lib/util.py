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