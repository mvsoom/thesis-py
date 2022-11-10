import numpy as np
import scipy.stats

def sample(x0, xbar, size=1):
    K = len(xbar)
    X = [x0, *xbar]

    # Calculate scale parameters for the u ~ Exp(beta)
    beta = [(X[k+1] - X[k])/X[k+1] for k in range(K)]

    # Draw the u
    u = scipy.stats.expon.rvs(scale=beta, size=(size,K))
    
    # Transform to x
    x = x0*np.exp(np.cumsum(u, axis=1))
    
    return x # (size, K)

def sample_truncated(x0, xbar, xmax, size=1):
    def get_batch(size):
        x = sample(x0, xbar, size)
        keep = np.all(x <= xmax, axis=1)
        return x[keep,:]

    accept = get_batch(size)
    p = max(accept.shape[0]/size, 1/20)

    while accept.shape[0] < size:
        new = int((size - accept.shape[0])/p)
        batch = get_batch(new)
        accept = np.concatenate((accept, batch), axis=0)

    return accept[:size,:] # (size, K)