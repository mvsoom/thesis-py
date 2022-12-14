"""A JAX implementation of the Liljencrants-Fant model for the glottal flow waveform"""
import jax
import jax.numpy as jnp
import jaxopt

def _get_initial_bracket(T0, tol, UPPER_FACTOR=1e3):
    lower = (1/T0)*tol
    upper = (1/T0)*UPPER_FACTOR
    return lower, upper

def _bisect_exponential_rate(f, tol, initial_bracket, maxiter, fail_val=jnp.nan, **kwargs):
    lower, upper = initial_bracket
    bisec = jaxopt.Bisection(
        optimality_fun=f,
        lower=lower,
        upper=upper,
        maxiter=maxiter,
        tol=tol,
        check_bracket=False
    )
    result = bisec.run(**kwargs)
    success = result.state.error < tol
    exponential_rate = result.params
    return jax.lax.cond(success, lambda: exponential_rate, lambda: fail_val)

def _bisect_epsilon(p, tol, initial_bracket, maxiter):
    def f(epsilon, p):
        LHS = epsilon*p['Ta']
        RHS = 1 - jnp.exp(-epsilon*(p['T0'] - p['Te']))
        return LHS - RHS

    return _bisect_exponential_rate(f, tol, initial_bracket, maxiter, p=p)

def _bisect_a(p, epsilon, tol, initial_bracket, maxiter):
    def f(a, p, epsilon):
        factor = 1/(a**2 + (jnp.pi/p['Tp'])**2)
        term1 = jnp.exp(-a*p['Te'])*(jnp.pi/p['Tp'])/jnp.sin(jnp.pi*p['Te']/p['Tp'])
        term2 = a
        term3 = jnp.pi/p['Tp']*(1/jnp.tan(jnp.pi*p['Te']/p['Tp'])) # cotg() is 1/tan()
        LHS = factor*(term1 + term2 - term3)

        RHS = (p['T0'] - p['Te'])/(jnp.exp(epsilon*(p['T0'] - p['Te'])) - 1) - 1/epsilon

        return LHS - RHS

    # Assume that the bisection will only fail when `a -> 0`
    return _bisect_exponential_rate(
        f, tol, initial_bracket, maxiter, fail_val=0., p=p, epsilon=epsilon
    )

def _dgf(t, p, epsilon, a, **kwargs):
    """Implement the LF model per Doval et al. (2006) Section A1.4."""
    def rise(t):
        return -jnp.exp(a*(t - p['Te']))*jnp.sin(jnp.pi*t/p['Tp'])/jnp.sin(jnp.pi*p['Te']/p['Tp'])

    def decrease(t):
        return -1/(epsilon*p['Ta'])*(jnp.exp(-epsilon*(t - p['Te'])) - jnp.exp(-epsilon*(p['T0'] - p['Te'])))

    return jnp.piecewise(
        t,
        [t < 0, (0 <= t) & (t < p['Te']), (p['Te'] <= t) & (t <= p['T0'])],
        [0., rise, decrease, 0.]
    )

def _nans_like(a):
    return jnp.full_like(a, jnp.nan)

def dgf(t, p, offset=0., tol=1e-6, initial_bracket=None, maxiter=100):
    """Calculate the glottal flow derivative (DGF) using the LF (Liljencrants-Fant) model
    
    This JAX implementation can be jitted and is differentiable with respect to `p`.
    Jitting this function will gain several a speedup of several orders of magnitude.
    
    The LF model [1,2] is nonzero only for values of `t` in `[offset, offset + T0]`, where
    `T0 = p['T0']`. If the LF parameters in `p` are inconsistent, such that the
    implicit equations cannot be solved, the function's output `u` will be all `nan`s.
    
    Args:
        `t` (array): Time points at which to evaluate the DGF `u(t)`
        `p` (dict): Dictionary with the LF parameters. It must have the following keys:
            `['T0', 'Te', 'Tp', 'Ta']`. `Ee` is always assumed to be 1.
        `offset` (scalar): The DGF waveform starts at this offset and is nonzero in
            `[offset, offset + T0]`.
        `tol`, `initial_bracket`, `maxiter` (scalar): Parameters controlling the bisection
            routine `_bisect_exponential_rate()` to solve the two equations implicit in
            the LF model.

    Returns:
        `u` (DeviceArray): The glottal flow derivative evaluated at times `t` if the LF
            parameters in `p` are consistent, otherwise an array of `nan`s shaped as `t`.
        
    References:
        [1] G. Fant, J. Liljencrants, and Q. Lin, "A four-parameter model of glottal flow",
            STL-QPSR, vol. 4, no. 1985, pp. 1???13, 1985.
        [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of
            glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
    """
    if initial_bracket is None:
        initial_bracket = _get_initial_bracket(p['T0'], tol)

    epsilon = _bisect_epsilon(p, tol=tol, initial_bracket=initial_bracket, maxiter=maxiter)
    a = _bisect_a(p, epsilon=epsilon, tol=tol, initial_bracket=initial_bracket, maxiter=maxiter)
    u = _dgf(t - offset, p=p, epsilon=epsilon, a=a)
    inconsistent_parameters = jnp.any(jnp.isnan(u))
    return jax.lax.cond(inconsistent_parameters, lambda: _nans_like(u), lambda: u)

def consistent_lf_params(p):
    """Check whether the implicit LF equations can be solved given the LF parameters in `p`"""
    return ~jnp.any(jnp.isnan(dgf(0., p)))

def _select_keys(q, *keys):
    return {k: q[k] for k in keys if k in q}

def convert_lf_params(p, s, join=True):
    if s == 'R -> generic':
        # Perrotin et al. (2021) Eq. (A1)
        Ra, Rk, Rg = p['Ra'], p['Rk'], p['Rg']
        Oq = (1 + Rk)/(2*Rg)
        am = 1/(1 + Rk)
        Qa = Ra/(1 - Oq) # Doval (2006) p. 5
        q = dict(Oq=Oq, am=am, Qa=Qa)
    elif s == 'generic -> R':
        Oq, am, Qa = p['Oq'], p['am'], p['Qa']
        Rk = 1/am - 1
        Rg = (1 + Rk)/(2*Oq)
        Ra = Qa*(1 - Oq)
        q = dict(Ra=Ra, Rk=Rk, Rg=Rg)
    elif s == 'T -> R':
        # Fant (1994)
        T0, Te, Tp, Ta = p['T0'], p['Te'], p['Tp'], p['Ta']
        Ra = Ta/T0
        Rk = (Te - Tp)/Tp
        Rg = T0/(2*Tp)
        q = dict(Ra=Ra, Rk=Rk, Rg=Rg)
    elif s == 'R -> T':
        T0, Ra, Rk, Rg = p['T0'], p['Ra'], p['Rk'], p['Rg']
        Ta = T0*Ra
        Tp = T0/(2*Rg)
        Te = Tp*(1 + Rk)
        q = dict(Te=Te, Tp=Tp, Ta=Ta)
    elif s == 'generic -> T':
        q = convert_lf_params(convert_lf_params(p, 'generic -> R'), 'R -> T')
        q = _select_keys(q, 'Te', 'Tp', 'Ta')
    elif s == 'T -> generic':
        q = convert_lf_params(convert_lf_params(p, 'T -> R'), 'R -> generic')
        q = _select_keys(q, 'Oq', 'am', 'Qa')
    elif s == 'Rd -> R':
        # Perrotin et al. (2021) Eq. (A1)
        # See also Fant (1994) for the meaning of these dimensionless parameters
        Rd = p['Rd']
        Ra = (-1 + 4.8*Rd)/100
        Rk = (22.4 + 11.8*Rd)/100
        Rg = Rk*(0.5 + 1.2*Rk)/(0.44*Rd - 4*Ra*(0.5 + 1.2*Rk))
        q = dict(Ra=Ra, Rk=Rk, Rg=Rg)
    elif s == 'Rd -> T':
        q = convert_lf_params(convert_lf_params(p, 'Rd -> R'), 'R -> T')
        q = _select_keys(q, 'Te', 'Tp', 'Ta')
    else:
        raise ValueError(f'Unknown conversion: {s}')
    return {**p, **q} if join else q

def fant_params(s):
    """Generate typical LF model parameters for a male or female vowel
    
    Note that the unit of time is `msec`. Values are taken from [1, p. 121].
    
      [1] Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis."
          Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
    """
    if s == "female vowel":
        return {
            'T0': 5.0,
            'Te': 3.25,
            'Tp': 2.5,
            'Ta': 0.3978873577297384
        }
    elif s == "male vowel":
        return {
            'T0': 8.333333333333334,
            'Te': 4.51388888888889,
            'Tp': 3.4722222222222228,
            'Ta': 0.22736420441699334
        }
    else:
        raise ValueError(s)