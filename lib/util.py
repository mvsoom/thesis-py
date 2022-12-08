import numpy as np
import scipy.linalg
import dynesty
from dynesty import plotting
import matplotlib.pyplot as plt

def normalize_power(d, return_multiplier=False):
    multiplier = np.sqrt(len(d)/np.dot(d, d))
    normalized = multiplier*d
    return (normalized, multiplier) if return_multiplier else normalized

def nats_to_ban(x):
    return x*np.log10(np.exp(1))

def dyplot(results, names=None, runplot=True, traceplot=True, cornerplot=True):
    display(results.summary())
    display('Information (bans)', nats_to_ban(results.information[-1]))
    
    if runplot:
        fig, axes = dynesty.plotting.runplot(results)
        plt.tight_layout()
        plt.show()

    if traceplot:
        fig, axes = dynesty.plotting.traceplot(
            results, show_titles=True,
            labels=names,
            verbose=True
        )
        plt.tight_layout()
        plt.show()

    if cornerplot:
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

def correlationmatrix(cov):
    """https://en.wikipedia.org/wiki/Correlation#Correlation_matrices"""
    sigma = np.sqrt(np.diag(cov))
    corr = np.diag(1/sigma) @ cov @ np.diag(1/sigma)
    return corr

def kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    m_to, S_to = to
    m_fr, S_fr = fr
    
    d = m_fr - m_to
    
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.