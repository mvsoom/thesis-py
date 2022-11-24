import numpy as np
import scipy.stats

def assign_xnullbar(K, xmin, xmax):
    """Heuristic to assign the expected VTR frequency values"""
    xbar = xmin + (xmax - xmin)/(K+1)*np.arange(1, K+1)
    xnullbar = np.array([xmin, *xbar])
    return xnullbar

def sample_x(xnullbar, size=1):
    K = len(xnullbar) - 1

    # Calculate scale parameters for the u ~ Exp(beta)
    beta = [(xnullbar[k+1] - xnullbar[k])/xnullbar[k+1] for k in range(K)]

    # Draw the u
    u = scipy.stats.expon.rvs(scale=beta, size=(size,K))
    
    # Transform to x
    x = xnullbar[0]*np.exp(np.cumsum(u, axis=1))
    
    return x # (size, K)

def sample_x_ppf(q, K, xnullbar):
    assert K == len(q) and len(xnullbar) == K + 1

    # Calculate scale parameters for the u ~ Exp(beta) such
    # that the marginal moments agree with Ex
    beta = [(xnullbar[j+1] - xnullbar[j])/xnullbar[j+1] for j in range(K)]
    
    # Draw the u
    u = np.atleast_1d(scipy.stats.expon.ppf(q, scale=beta))
    
    # Transform to x
    x0 = xnullbar[0]
    x = [x0*np.exp(np.sum(u[0:j+1])) for j in range(K)]
    return np.array(x)

def sample_x_truncated(xnullbar, xmax, size=1):
    def get_batch(size):
        x = sample_x(xnullbar, size)
        keep = np.all(x <= xmax, axis=1)
        return x[keep,:]

    accept = get_batch(size)
    p = max(accept.shape[0]/size, 1/20)

    while accept.shape[0] < size:
        new = int((size - accept.shape[0])/p)
        batch = get_batch(new)
        accept = np.concatenate((accept, batch), axis=0)

    return accept[:size,:] # (size, K)

def sample_jeffreys_ppf(q, bounds):
    J = len(q)
    lower, upper = bounds
    
    assert len(lower) == J and len(upper) == J
    
    a = np.log(lower[:J])
    b = np.log(upper[:J])
    x = a + q*(b-a)
    return np.exp(x)