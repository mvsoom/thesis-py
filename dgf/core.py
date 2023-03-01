"""Mathemetical core implemented as functional programming to be JAX-friendly"""
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates.jax.math import cholesky_update as jax_cholesky_update
from jax import jit, value_and_grad, grad

def phi(t, m, L):
    return jnp.sqrt(2/L)*jnp.sin(jnp.pi*m*t/L)

def phi_integrated(t, m, L):
    return jnp.sqrt(2*L)*(1. - jnp.cos(jnp.pi*m*t/L))/(m*jnp.pi)

def phi_singlepole(t, m, T, L, p):
    """Convolve `phi(t, m, L)` on $[0,T]$ with a single $e^{pt}$ pole
    
    Note: we assume `t, T, L` in msec and `p` in Hz.
    """
    p = p/1000. # Convert to kHz, conjugate to time units in msec
    
    π = jnp.pi
    st, ct = jnp.sin(π*m*t/L), jnp.cos(π*m*t/L)
    sT, cT = jnp.sin(π*m*T/L), jnp.cos(π*m*T/L)
    
    a = m*π*(jnp.exp(p*t) - ct)
    b = -p*L*st
    c = m*π*ct - jnp.exp(p*(t - T))*(m*π*cT + p*L*sT)
    
    out = jnp.where(0. < t, a, 0.)               # t ∈ (0,∞]
    out += jnp.where((0. < t) & (t <= T), b, 0.) # t ∈ (0,T]
    out += jnp.where(T < t, c, 0.)               # t ∈ (T,∞)
    
    scale = jnp.sqrt(2*L)/((π*m)**2 + (p*L)**2)
    return out*scale

def phi_matrix(t, M, L):
    """Hilbert basis functions"""
    m = jnp.arange(1, M+1)
    return phi(t[:,None], m[None,:], L) # (len(t), M)

def phi_integrated_matrix(t, M, L):
    """Integrated Hilbert basis functions"""
    m = jnp.arange(1, M+1)
    return phi_integrated(t[:,None], m[None,:], L) # (len(t), M)

def phi_integrated_matrix_at(T, M, L):
    m = jnp.arange(1, M+1)
    return phi_integrated(T, m, L) # (M,)

def phi_integrated_total_flow(M, T, L):
    """
    Analytical calculation of total glottal flow for
    `phi_integrated` basis functions on the interval `[0,T]`.
    Note: the gamma coefficients (determined by the spectral density)
    are not taken into account here.
    """
    m = jnp.arange(1, M+1)
    prefactor = jnp.sqrt(2*L)/((m*jnp.pi)**2)
    total_flow = prefactor*(m*jnp.pi*T - L*jnp.sin(m*jnp.pi*T/L))
    return total_flow # (M,)

def phi_poles_matrix(t, M, T, L, poles):
    """Hilbert basis functions convolved with a set of poles"""
    m = jnp.arange(1, M+1)
    phi = phi_singlepole(t[:,None,None], m[None,:,None], T, L, poles[None,None,:])
    return phi # (len(t), M, len(poles))

def phi_transfer_matrix(t, M, T, L, poles, c):
    """Hilbert basis functions convolved with a rational transfer function"""
    phi = phi_poles_matrix(t, M, T, L, poles)
    phi = jnp.real(phi @ (2*c))
    return phi # (len(t), M)

def sqrt_gamma_coefficients(kernel, var, scale, M, L):
    """Calculate $\sqrt{S(-\lambda_m)}$ which has units [rad/msec] if $L$ has units msec"""
    m = jnp.arange(1, M+1)
    gamma = kernel(var, scale).spectral_density(m*jnp.pi/L)
    return jnp.sqrt(gamma) # (M,)

def kernelmatrix_root_hilbert(kernel, var, scale, t, M, L):
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    phi = phi_matrix(t, M, L)
    R = impose_kernel(phi, d)
    R = impose_domain(R, t, 0., L)
    return R # (len(t), M)

def kernelmatrix_root_gfd(
    kernel, var, scale, t, M, T, bf,
    impose_null_integral=True, integrate=False, regularize_flow=True
):
    L = bf*T
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    phi = phi_integrated_matrix(t, M, L) if integrate else phi_matrix(t, M, L) 
    
    R = impose_kernel(phi, d)
    R = impose_constraints(
        d, R, var, M, T, L, impose_null_integral, regularize_flow
    ) # O(M²)
    R = impose_domain(R, t, 0., T)

    return R # (len(t), M)

def kernelmatrix_root_gfd_oq(
    kernel, var, r, t, M, T, Oq, bf,
    impose_null_integral=True, integrate=False, regularize_flow=True
):
    # Manually intervene to limit `Oq <= 1`, since this function
    # will return perfectly sensible results if this is not the case
    # (the waveform gets shifted to the left and the period is stretched)
    Oq = jax.lax.cond(Oq <= 1., lambda: Oq, lambda: jnp.nan)

    GOI = T*(1 - Oq)
    scale = r*T*Oq
    R = kernelmatrix_root_gfd(
        kernel, var, scale, t - GOI, M, T*Oq, bf,
        impose_null_integral, integrate, regularize_flow
    )

    return R # (len(t), M)

def kernelmatrix_root_convolved_gfd(
    kernel, var, scale, t, M, T, bf, poles, c,
    impose_null_integral=True, regularize_flow=True
):
    L = T*bf
    d = sqrt_gamma_coefficients(kernel, var, scale, M, L)
    phi = phi_transfer_matrix(t, M, T, L, poles, c)
    
    R = impose_kernel(phi, d)
    R = impose_constraints(d, R, var, M, T, L, impose_null_integral, regularize_flow)
#   R = impose_domain(R, t, 0., jnp.inf) # Already imposed by `phi_transfer_matrix()`

    return R # (len(t), M)

def kernelmatrix_root_convolved_gfd_oq(
    kernel, var, r, t, M, T, Oq, bf, poles, c,
    impose_null_integral=True, regularize_flow=True
):
    # Note: we assume `Oq <= 1` here, otherwise this will trigger JAX's
    # NaN debugger for some reason. This is OK anyway since this function
    # is only used in conjunction with p(theta), which guarantees `Oq <= 1`
#   Oq = jax.lax.cond(Oq <= 1., lambda: Oq, lambda: jnp.nan)
    
    GOI = T*(1 - Oq)
    scale = r*T*Oq
    R = kernelmatrix_root_convolved_gfd(
        kernel, var, scale, t - GOI, M, T*Oq, bf, poles, c,
        impose_null_integral, regularize_flow
    )
    return R # (len(t), M)

def impose_kernel(phi, d):
    R = d[None,:] * phi # == phi @ jnp.diag(d)
    return R

def impose_constraints(
    d, R, var, M, T, L, impose_null_integral, regularize_flow
):
    """
    Constraints are imposed by multiplying the basis function matrix `R`
    to the right by Cholesky factors `XXX_tril`encoding those constraints.
    """
    constraints = jnp.eye(M) # Start out with no constraints
    
    if impose_null_integral:
        integral_tril = null_integral_constraint(d, M, T, L)
        constraints = constraints @ integral_tril
    else:
        integral_tril = jnp.eye(M) # No constraint
        
    if regularize_flow:
        flow_tril = expected_flow_constraint(d, integral_tril, var, M, T, L)
        constraints = constraints @ flow_tril

    R = R @ constraints
    return R

def impose_domain(R, t, a, b):
    """Make sure basis functions in `R` are nonzero only in `[a,b]`"""
    outside_domain = (t[:,None] < a) | (b < t[:,None])
    R = jnp.where(outside_domain, 0., R)
    return R

def null_integral_constraint(d, M, T, L):
    q = d * phi_integrated_matrix_at(T, M, L)
    q = q/jnp.linalg.norm(q)
    tril = cholesky_of_projection(q, M) # Project down to the space of functions
    return tril                         # integrating to 0 at O(M²) cost

# This is derived from the analytical RBF model in `dgf/expected_total_flow.nb`
_TOTAL_FLOW_FACTOR = 0.0969358

def expected_flow_constraint(
    d, integral_tril, var, M, T, L, flow_constraint=None
):
    """Calculate the Cholesky constraint such that the expected flow is `flow_constraint`"""
    a = dgf_expected_total_flow(d, integral_tril, M, T, L)
    
    if not flow_constraint:
        flow_constraint = _TOTAL_FLOW_FACTOR*jnp.sqrt(var)*T**2
    
    anorm = jnp.linalg.norm(a)
    gamma = 1. - (flow_constraint/anorm)**2
    
    q = a/anorm
    tril = cholesky_update_to_identity(q, -gamma) # O(M²)
    return tril

def dgf_expected_total_flow(d, integral_tril, M, T, L):
    """
    Calculate the expected total flow for each DGF basisfunction
    taking into account kernel characteristics (through `d`) and null
    integral constraint (through `integral_tril`).
    """
    omega = phi_integrated_total_flow(M, T, L)
    expected_total_flow = (omega * d) @ integral_tril
    return expected_total_flow # (M,)

def cholesky_of_projection(q, M):
    return cholesky_update_to_identity(q, -1.)

def cholesky_update_to_identity(q, multiplier):
    # Calculate chol(I + multiplier*(q @ q.T))
    M = len(q)
    nugget = M*jnp.finfo(float).eps
    I = jnp.diag(jnp.repeat(1. + nugget, M))
    return jax_cholesky_update(I, q, multiplier) # O(M²)

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



####

def kernelmatrix_root_gfd_new(
    kernel, var, scale, t, M, T, bf,
    impose_null_integral=True, integrate=False, regularize_flow=True
):
    left = domain_constraint(t, 0., T)
    phi = phi_integrated_matrix(t, M, L) if integrate else phi_matrix(t, M, L)
    right = cholesky_constraints(
        kernel, var, scale, t, M, T, bf, impose_null_integral, regularize_flow
    )
    
    R = left * jnp.linalg.multi_dot([phi, *right])
    return R

def domain_constraint(t, a, b):
    mask = (a <= t[:,None]) & (t[:,None] <= b)
    return mask # (len(t), 1)