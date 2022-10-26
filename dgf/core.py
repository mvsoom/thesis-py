"""Mathemetical core implemented as functional programming to be JAX-friendly"""
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates.jax.math import cholesky_update as jax_cholesky_update
from jax import jit, value_and_grad, grad
from functools import partial

@partial(jit, inline=True)
def phi(t, m, L):
    return jnp.sqrt(2/L)*jnp.sin(jnp.pi*m*t/L)

@partial(jit, inline=True)
def phi_integrated(t, m, L):
    return jnp.sqrt(2*L)*(1. - jnp.cos(jnp.pi*m*t/L))/(m*jnp.pi)

@partial(jit, inline=True)
def phi_singlepole(t, m, T, L, p):
    """Convolve `phi(t, m, L)` on $[0,T]$ with a single $e^{pt}$ pole"""
    π = jnp.pi
    st, ct = jnp.sin(π*m*t/L), jnp.cos(π*m*t/L)
    sT, cT = jnp.sin(π*m*T/L), jnp.cos(π*m*T/L)
    
    a = m*π*(jnp.exp(p*t) - ct)
    b = -p*L*st
    c = m*π*ct - jnp.exp(p*(t - T))*(m*π*cT + p*L*sT)
    
    # TODO: write as `jnp.piecewise()`?
    out = jnp.where(0. < t, a, 0.)               # t ∈ (0,∞]
    out += jnp.where((0. < t) & (t <= T), b, 0.) # t ∈ (0,T]
    out += jnp.where(T < t, c, 0.)               # t ∈ (T,∞)
    
    scale = jnp.sqrt(2*L)/((π*m)**2 + (p*L)**2)
    return out*scale

@partial(jit, static_argnames="M", inline=True)
def phi_matrix(t, M, L):
    """Hilbert basis functions"""
    m = jnp.arange(1, M+1)
    return phi(t[:,None], m[None,:], L) # (len(t), M)

@partial(jit, static_argnames="M")
def phi_integrated_matrix(t, M, L):
    """Integrated Hilbert basis functions"""
    m = jnp.arange(1, M+1)
    return phi_integrated(t[:,None], m[None,:], L) # (len(t), M)

@partial(jit, static_argnames="M")
def phi_integrated_matrix_at(T, M, L):
    m = jnp.arange(1, M+1)
    return phi_integrated(T, m, L) # (M,)

@partial(jit, static_argnames="M")
def phi_poles_matrix(t, M, T, L, poles):
    """Hilbert basis functions convolved with a set of poles"""
    m = jnp.arange(1, M+1)
    phi = phi_singlepole(t[:,None,None], m[None,:,None], T, L, poles[None,None,:])
    return phi # (len(t), M, len(poles))

@partial(jit, static_argnames="M")
def phi_transfer_matrix(t, M, T, L, poles):
    """Hilbert basis functions convolved with an all-pole transfer function"""
    c = pole_coefficients(poles)
    phi = phi_poles_matrix(t, M, T, L, poles)
    phi = jnp.real(phi @ (2*c))
    return phi # (len(t), M)

@partial(jit, static_argnames=("kernel", "M"))
def sqrt_gamma_coefficients(kernel, var, scale, M, L):
    """Calculate $\sqrt{S(-\lambda_m)}$"""
    m = jnp.arange(1, M+1)
    gamma = kernel(var, scale).spectral_density(m*jnp.pi/L)
    return jnp.sqrt(gamma) # (M,)

@partial(jit, static_argnames=("kernel", "M"))
def kernelmatrix_root_hilbert(kernel, var, scale, t, M, L):
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    R = d[None,:] * phi_matrix(t, M, L)
    R = impose_domain(R, t, 0., L)
    return R # (len(t), M)

@partial(jit, static_argnames=("kernel", "M", "c", "impose_null_integral"))
def kernelmatrix_root_gfd(kernel, var, scale, t, M, T, c, impose_null_integral=True):
    L = c*T
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    R = d[None,:] * phi_matrix(t, M, L)
    if impose_null_integral:
        R = impose_null_integral_constraint(d, R, M, T, L) # O(M²)
    R = impose_domain(R, t, 0., T)
    return R # (len(t), M)

@partial(jit, static_argnames=("kernel", "M", "c", "impose_null_integral"))
def kernelmatrix_root_gfd_oq(kernel, var, r, t, M, T, Oq, c, impose_null_integral=True):
    # Manually intervene to limit `Oq <= 1`, since this function
    # will return perfectly sensible results if this is not the case
    # (the waveform gets shifted to the left and the period is stretched)
    Oq = jax.lax.cond(Oq <= 1., lambda: Oq, lambda: jnp.nan)

    GOI = T*(1 - Oq)
    scale = r*T*Oq
    R = kernelmatrix_root_gfd(kernel, var, scale, t - GOI, M, T*Oq, c, impose_null_integral)
    return R # (len(t), M)

@partial(jit, static_argnames=("kernel", "M", "c"))
def kernelmatrix_root_convolved_gfd(kernel, var, scale, t, M, T, c, poles):
    L = T*c
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    R = d[None,:] * phi_transfer_matrix(t, M, T, L, poles)
    R = impose_null_integral_constraint(d, R, M, T, L)
#   R = impose_domain(R, t, 0., jnp.inf) # Already imposed by `phi_transfer_matrix()`
    return R # (len(t), M)

@jit
def impose_domain(R, t, a, b):
    """Make sure basis functions in `R` are nonzero only in `[a,b]`"""
    outside_domain = (t[:,None] < a) | (b < t[:,None])
    return jnp.where(outside_domain, 0., R)

@partial(jit, static_argnames=("M"))
def impose_null_integral_constraint(d, R, M, T, L):
    assert R.shape[1] == M
    q = d * phi_integrated_matrix_at(T, M, L)
    q /= jnp.linalg.norm(q)
    return R @ cholesky_of_projection(q, M)

@partial(jit, static_argnames="M")
def cholesky_of_projection(q, M):
    assert len(q) == M
    nugget = M*jnp.finfo(float).eps
    I = jnp.diag(jnp.repeat(1. + nugget, M))
    return jax_cholesky_update(I, q, -1.) # O(M²)

@partial(jit, inline=True)
def excluded_pole_product(ps):
    diff = ps[None,:] - ps[:,None]
    diff = diff.at[jnp.diag_indices_from(diff)].set(1.)
    denom = jnp.prod(diff, axis=0)
    return 1./denom

@jit
def pole_coefficients(poles):
    gain = jnp.prod(jnp.abs(poles)**2)
    ps = jnp.concatenate([poles, jnp.conj(poles)])
    return gain*excluded_pole_product(ps)[:len(poles)] # (len(poles),)

@jit
def make_poles(bandwidth, frequency):
    poles = (-bandwidth*jnp.pi + (1j)*frequency*2*jnp.pi)/1000.
    return poles

@jit
def loglikelihood_hilbert(R, y, noise_power):
    """Implement *positive* log likelihood from Section 3.2 in Solin & Särkkä (2020)"""
    D = jnp.eye(R.shape[1])
    Z = noise_power * D + R.T @ R

    L, lower = jax.scipy.linalg.cho_factor(Z, lower=True, check_finite=False)

    logabsdet_Z = 2*jnp.sum(jnp.log(jnp.diag(L)))

    N, M = R.shape
    log_Q_term = (N - M)*jnp.log(noise_power) + logabsdet_Z

    # Find the least-squares solution to `R @ x = y`.
    # This is the same `x` that solves the normal equations `Z @ x = R.T @ y`.
    b = R.T @ y
    x = jax.scipy.linalg.cho_solve((L, lower), b, check_finite=False)

    bilinear_term = 1/noise_power*(jnp.dot(y, y) - jnp.dot(b, x))
    
    order_term = N*jnp.log(2*jnp.pi)

    negative_logl = log_Q_term/2 + bilinear_term/2 + order_term/2
    return -negative_logl

@partial(jit, static_argnames=("kernel", "M"))
def loglikelihood_hilbert_grid(kernel, var, scale, M, y, noise_power):
    """Evaluate the log likelihood of a Hilbert kernel on a grid
    
    Note that it is required to have `N > M`.
    
    This function is equivalent to `loglikelihood_hilbert()` evaluated on a grid spanning
    the compact domain `[0, L]`:
    ````
        N = 1000
        M = 128
        
        y = np.random.rand(N)  # Generate some data lying on the grid
        t = np.arange(N)       # Implied grid spanning `[0, L]`
        L = N - 1              # Endpoint of the compact domain `[0, L]`

        kernel = isokernels.Matern32Kernel
        var = 1.3
        scale = 2.5
        noise_power = .1
        
        # Calculate the log likelihood using the general method
        R = kernelmatrix_root_hilbert(kernel, var, scale, t, M, L)
        L_general = loglikelihood_hilbert(R, y, noise_power)
        
        # Now make use of the fact that the grid spans the compact domain `[0, L]`
        L_grid = loglikelihood_hilbert_grid(kernel, var, scale, M, y, noise_power)
        
        assert(np.isclose(L_general, L_grid))
    ````
    As can be seen from the above code, the grid is not given as an argument but implied
    by the data `y`; it is `[0, 1, 2, ..., N - 1]`, where `N = len(y)`. Since there is no
    boundary factor `c`, using this method implies some error at the compact domain endpoints
    `{0, N-1}` as the GP is always zero at these points.
    """
    N = len(y)
    L = N - 1

    S = jnp.square(sqrt_gamma_coefficients(kernel, var, scale, M, L))
    Z_diag = 1. + noise_power/S

    logabsdet_Z = jnp.sum(jnp.log(Z_diag))

    log_Q_term = (N - M)*jnp.log(noise_power) + logabsdet_Z + jnp.sum(jnp.log(S))

    # Find `b = Phi.T @ y` using FFT projection
    a = jnp.fft.rfft(y, 2*L)
    b = -jnp.sqrt(2/L)*a.imag[1:M+1]

    # If not `N > M` the following line will raise an Exception.
    bilinear_term = 1/noise_power*(jnp.dot(y, y) - jnp.dot(b, b/Z_diag))

    order_term = N*jnp.log(2*jnp.pi)

    negative_logl = log_Q_term/2 + bilinear_term/2 + order_term/2
    return -negative_logl