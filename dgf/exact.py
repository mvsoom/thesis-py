"""Exact implementation of the DGF prior with RBF kernel"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erf as erf

def erfsum(x, y):
    return erf(x) + erf(y)

def kernelmatrix(x, y, scale):
    d = (x - y)/scale
    return jnp.exp(-d**2/2.)

def kernelmatrix_null_integral(x, y, scale, T):
    """Project down to functions integrating to zero on `[0,T]`"""
    s = 1/(jnp.sqrt(2)*scale)

    def erfs(x):
        return erfsum(s*(T-x), s*x)

    numerator = jnp.pi*scale*erfs(x)*erfs(y)
    denominator = 4*scale*jnp.exp(-(T/scale)**2/2) - 2*(2*scale - jnp.sqrt(2*jnp.pi)*T*erf(s*T))
    
    return kernelmatrix(x, y, scale) - numerator/denominator

def kernelmatrix_gfd(x, y, scale, T):
    """Project down to functions going through `(0,0)` and integrating to zero on `[0,T]`"""
    def k(x, y):
        return kernelmatrix_null_integral(x, y, scale, T)
    
    return k(x, y) - k(x, 0.)*k(0., y)/k(0., 0.)

def kernelmatrix_root_gfd(x, y, scale, T, nugget=1e-12):
    K = kernelmatrix_gfd(x, y, scale, T)
    R = jnp.linalg.cholesky(K + nugget*jnp.eye(K.shape[0]))
    return R