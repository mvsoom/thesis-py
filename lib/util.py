import numpy as np
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

# https://stackoverflow.com/a/55688087/6783015
# I checked this and it is correct
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian 1 to Gaussian 0

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 