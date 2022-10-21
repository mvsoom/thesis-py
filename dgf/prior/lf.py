"""Define the prior over the LF model for the glottal flow derivative"""
from init import __memory__
from dgf.prior import period
from dgf.prior import holmberg
from dgf import constants
from dgf import bijectors
from dgf import isokernels
from lib import lfmodel

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import numpy as np
import warnings

#########################
# Sampling $p(R_k|R_e)$ #
#########################
_UNCONTAINED = np.inf
def _contains(a, x):
    return (a[0] < x) & (x < a[1])

AM_BOUNDS = [constants.MIN_AM, constants.MAX_AM]

def estimate_sigma2_Rk(numsamples=10000, seed=5571):
    """
    Estimate the variance of Rk from the known theoretical range of $\alpha_m$.
    """
    alpha_m = np.random.default_rng(seed).uniform(*AM_BOUNDS, size=numsamples)
    Rk = 1/alpha_m - 1 # Perrotin (2021) Eq. (A1)
    return np.var(Rk)

def estimate_Rk_stddev():
    """
    Estimate Fant (1994) Eq. (2) regression error given the value of
    the Pearson's correlation coefficient `r = 0.93` and a range of plausible
    values of `Rk` from its connection to $\alpha_m$.
    
    For this we only need the conditional variance of the bivariate Normal
    distribution given here: <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case_2>.
    """
    rho = 0.93 # Pearson's correlation coefficient
    sigma2_Rk = estimate_sigma2_Rk()
    Rk_stddev = np.sqrt((1 - rho**2)*sigma2_Rk)
    return Rk_stddev

RK_STDDEV = estimate_Rk_stddev()
RK_BOUNDS = [constants._ZERO, 1.] # Is a percentage

def sample_Rk(Re, rng):
    """
    Use the regression in Fant (1994) Eq. (2) + reverse-engineered noise based on the
    given `r = 0.93` value and the estimated standard deviation of Rk.
    """
    Rk = _UNCONTAINED
    while not _contains(RK_BOUNDS, Rk): Rk = (22.4 + 11.8*Re)/100 + RK_STDDEV*rng.normal()
    return Rk

#########################
# Sampling $p(R_a|R_e)$ #
#########################
OQ_BOUNDS = [constants.MIN_OQ, constants.MAX_OQ]
QA_BOUNDS = [constants.MIN_QA, constants.MAX_QA]

def estimate_sigma2_Ra(numsamples=10000, seed=6236):
    """
    Estimate the variance of Ra from the known theoretical range of OQ and Qa.
    """
    rng = np.random.default_rng(seed)
    Oq = rng.uniform(*OQ_BOUNDS, numsamples) # @Drugman2019, Table 1
    Qa = rng.uniform(*QA_BOUNDS, numsamples) # Doval (2006), p. 5
    Ra = (1 - Oq)*Qa
    return np.var(Ra)

def estimate_Ra_stddev():
    """
    Estimate Fant (1994) Eq. (3) regression error given the value of
    the Pearson's correlation coefficient `r = 0.91` and a range of plausible
    values of `Ra` from its connection to OQ and Qa (the latter one of Doval's
    generic parameters.) Same logic as `estimate_Rk_stddev()`.
    """
    rho = 0.91 # Pearson's correlation coefficient
    sigma2_Ra = estimate_sigma2_Ra()
    Ra_stddev = np.sqrt((1 - rho**2)*sigma2_Ra)
    return Ra_stddev

RA_STDDEV = estimate_Ra_stddev()
RA_BOUNDS = [constants._ZERO, 1.] # Is a percentage

def sample_Ra(Re, rng):
    """
    Use the regression in Fant (1994) Eq. (3) + reverse-engineered noise based on the
    given `r = 0.91` value and the estimated standard deviation of Ra.
    
    Note that `Re` in Fant (1994) is called `Rd` in Fant (1995) and Perrotin (2021).
    """
    Ra = _UNCONTAINED
    while not _contains(RA_BOUNDS, Ra): Ra = (-1 + 4.8*Re)/100 + RA_STDDEV*rng.normal()
    return Ra

###################################
# Sampling $p(R_a, R_k, R_g|R_e)$ #
###################################
RG_BOUNDS = np.array([
    (1 + RK_BOUNDS[0]) / (2*OQ_BOUNDS[1]),
    (1 + RK_BOUNDS[1]) / (2*OQ_BOUNDS[0])
]) # From Rg = (1 + Rk)/(2*Oq)

RE_BOUNDS = [0.3, 2.7] # Main range of variation (Fant 1995)

def sample_R_triple(Re, rng):
    """
    From Perrotin (2021) Eq. (A1). Regress the other `R` parameters given `Re`.

    We use the equations that assume that `Re` is in `[0.3, 2.7]`, the main range
    of variation. The upper range `Re in [2.7, 5]` is "intended for transitions towards
    complete abduction as in prepause voice terminations" (Fant 1995, p. 123).
    

    See Fant (1994) for the meaning of these dimensionless parameters
    * Note that in that paper they are given in percent (%)
    * Note that `Re` in Fant (1994) is called `Rd` in Fant (1995) and Perrotin (2021).
    * Note that `Rg` can be larger than 1, unlike `Ra `and `Rg` (Fant 1994, Fig. 3)
    """
    if not _contains(RE_BOUNDS, Re):
        Ra, Rk, Rg = np.nan, np.nan, np.nan
    else:
        Rg = _UNCONTAINED
        while not _contains(RG_BOUNDS, Rg):
            Ra = sample_Ra(Re, rng)
            Rk = sample_Rk(Re, rng)
            Rg = Rk*(0.5 + 1.2*Rk)/(0.44*Re - 4*Ra*(0.5 + 1.2*Rk)) # Uncertainty from Ra and Rk transfers to Rg
    return Ra, Rk, Rg

#######################################
# Sampling $p(R_a, R_k, R_g, R_e|T0)$ #
#######################################
def calculate_Re(T0, Td):
    """
    We choose the declination time `T0 = U_0/E_e` as the independent variable
    In this way we can induce correlations between T0 and all other variables, as
    empirically observed (Henrich 2005).
    
    Note that `Re` in Fant (1994) is called `Rd` in Fant (1995) and Perrotin (2021)
    and in Freixes (2018).
    """
    F0 = 1/T0 # kHz
    Re = Td * F0 / (0.11) # Fant (1994) Eq. (4)
    return Re

def sample_R_params(T0, Td, rng):
    p = dict()
    p['T0'] = T0
    p['Td'] = Td
    p['Re'] = calculate_Re(T0, Td)
    p['Ra'], p['Rk'], p['Rg'] = sample_R_triple(p['Re'], rng)
    return p

def _collect_list_of_dicts(ld):
    """Convert a list of dicts to a dict of lists"""
    return {k: jnp.array([d[k] for d in ld]) for k in ld[0]}

def _apply_mask(p, mask):
    p = {k: v[mask] for k, v in p.items()}
    return p

@__memory__.cache
def sample_lf_params(fs=constants.FS_KHZ, numsamples=int(1e5), seed=2387):
    """Sample the `R`, `T` and `generic` parameters of the LF model

    This will return less samples than `numsamples`, because we emply rejection
    sampling. The regression equations are valid only for a certain range of `Re`
    (the `RE_BOUNDS`) and we impose bounds on all `R` and `generic` parameters.
    """
    # Manage both Numpy and JAX RNGs
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    rng_seed = int(jax.random.randint(key1, (1,), minval=0, maxval=int(1e4)))
    rng = np.random.default_rng(rng_seed)

    # Sample pitch periods and declination times (both in msec)
    T0 = period.period_marginal_prior().sample(numsamples, seed=key2).squeeze()
    Td = holmberg.declination_time_prior().sample(numsamples, seed=key3).squeeze()
    
    # Sample the `R` parameters based on the pitch periods
    p = _collect_list_of_dicts([sample_R_params(T, Td, rng) for T, Td in zip(T0, Td)])
    
    # Reject samples with out-of-bounds `Re` values
    mask = _contains(RE_BOUNDS, p['Re'])
    p = _apply_mask(p, mask)
    
    # Transform the `R` parameters to the equivalent `T` and generic representations
    p = lfmodel.convert_lf_params(p, 'R -> T')
    p = lfmodel.convert_lf_params(p, 'R -> generic')
    
    # Calculate the power of the DGF waveform on the support `t`
    T0_max = jnp.max(T0)
    t = jnp.linspace(0., T0_max, int(T0_max*fs) + 1)
    
    dgf = jax.jit(lfmodel.dgf)
    
    @jax.jit
    def calc_dgf_power(p):
        u = dgf(t, p)
        u = jnp.where(u, u, jnp.nan) # Ignore points outside of the pitch period
        p['power'] = jnp.nanvar(u)
        return p
    
    p = jax.vmap(calc_dgf_power)(p)
    
    # Reject all samples which have inconsistent LF parameters as signalled by
    # their power being `nan`, as in `lfmodel.consistent_lf_params()` ...
    mask = ~jnp.isnan(p['power'])
    
    # ... and all samples whose *generic* parameters are out of bounds
    for k, bounds in constants.LF_GENERIC_BOUNDS.items():
        lower, upper = bounds
        mask &= (lower < p[k]) & (p[k] < upper)

    p = _apply_mask(p, mask)
    return p

################################################################################
# Finally, define the priors based on the fitted distributions in the z domain #
################################################################################
def generic_params_prior(cacheid=98171):
    """
    Prior for the generic parameters of the LF model. Running this for the first
    time takes O(1) minutes. Equivalent to `generic_params_trajectory_prior(1)`.
    """
    p = sample_lf_params()
    
    # Select the `generic` parameters from the collection of LF parameters in `p`
    samples = jnp.vstack([p[v] for v in constants.LF_GENERIC_PARAMS]).T
    bounds = jnp.array([
        constants.LF_GENERIC_BOUNDS[k] for k in constants.LF_GENERIC_PARAMS
    ])
    
    # Model the empirical distribution of `samples` with a nonlinear
    # coloring prior using maximum likelihood.
    nonlinear_coloring = bijectors.fit_nonlinear_coloring_bijector(
        samples, bounds, cacheid
    )
    
    ndim = len(constants.LF_GENERIC_PARAMS)
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(ndim))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=nonlinear_coloring,
        name="LFGenericParamsPrior"
    )

    return prior

def generic_params_trajectory_prior(
    num_pitch_periods,
    envelope_kernel_name=None,
    envelope_lengthscale=None,
    envelope_noise_sigma=None
):
    # Get the marginal (at a given pitch period) means and correlations in
    # terms of the nonlinear coloring bijector
    nonlinear_coloring_bijector = generic_params_prior().bijector

    # Get the envelope (longitudinal) correlations
    if None in [envelope_kernel_name,
                envelope_lengthscale,
                envelope_noise_sigma]:
        # Default to the latent GP of the period bijector
        period_kernel_name, _, period_results = \
            period.fit_period_trajectory_kernel()
        period_envelope_lengthscale, period_envelope_noise_sigma = \
            period.maximum_likelihood_envelope_params(period_results)
        
        if envelope_kernel_name is None:
            envelope_kernel_name = period_kernel_name
        if envelope_lengthscale is None:
            envelope_lengthscale = period_envelope_lengthscale
        if envelope_noise_sigma is None:
            envelope_noise_sigma = period_envelope_noise_sigma
    
    # Get the corresponding bijector from white noise vector to trajectory matrix
    bijector = bijectors.nonlinear_coloring_trajectory_bijector(
        nonlinear_coloring_bijector,
        num_pitch_periods,
        envelope_kernel_name,
        envelope_lengthscale,
        envelope_noise_sigma
    )
    
    # Get the white noise distribution
    ndim = len(constants.LF_GENERIC_PARAMS)*num_pitch_periods
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(ndim))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=bijector,
        name='LFGenericParamsTrajectoryPrior'
    )
    
    return prior # `prior.sample()` has shape (num_pitch_periods, len(constants.LF_GENERIC_PARAMS))

def generic_params_to_dict(x, squeeze=False):
    x = jnp.atleast_2d(x)
    p = {k: x[:, i] for i, k in enumerate(constants.LF_GENERIC_PARAMS)}
    if squeeze:
        p = {k: jnp.squeeze(v) for k, v in p.items()}
    return p

def sample_and_log_prob_xt(prior, seed):
    """Works with `generic_params_prior` and `generic_params_trajectory_prior`"""
    # Sample and get probability of `generic` LF parameters...
    xg, log_prob_xg = prior.experimental_sample_and_log_prob(
        seed=seed
    )
    
    # ... and convert to `T` parameters.
    def xg_to_xt(xg):
        p = generic_params_to_dict(xg, squeeze=True)
        p = lfmodel.convert_lf_params(p, 'generic -> T')
        return jnp.array([p[k] for k in constants.LF_T_PARAMS]).T, p
    
    def logabsdetjacobian(xg):
        jacobian, _ = jax.jacobian(xg_to_xt, has_aux=True)(xg)
        return jnp.abs(jnp.linalg.slogdet(jacobian)[1])
    
    xt, p = xg_to_xt(xg)
    factors = jax.vmap(logabsdetjacobian)(jnp.atleast_2d(xg))
    log_prob_xt = log_prob_xg - jnp.sum(factors)
    
    return xt, log_prob_xt, p

def sample_dgf(
    num_pitch_periods,
    prior,
    seed,
    fs=constants.FS_KHZ
):
    xg = prior.sample(seed=jax.random.PRNGKey(seed))
    
    p = generic_params_to_dict(xg)
    p = lfmodel.convert_lf_params(p, 'generic -> T')

    GOI = jnp.cumsum(p['T0']) - p['T0'][0]
    end = GOI[-1] + p['T0'][-1]
    t = np.linspace(0, end, int(end*fs))
    
    def warn_if_nans(u):
        mask = jnp.any(jnp.isnan(u), axis=1)
        if jnp.sum(mask) > 0:
            pitch_period_indices = jnp.where(mask)[0]
            warnings.warn(
                'Inconsistent LF parameters detected at the following pitch '
                f'period indices: {pitch_period_indices}'
            )

    u = jax.vmap(jax.jit(lfmodel.dgf), in_axes=[None, 0, 0])(t, p, GOI)
    warn_if_nans(u)
    u = jnp.nansum(u, axis=0)

    return t, u

def sample_and_logprob_dgf(
    num_pitch_periods,
    prior,
    seed,
    fs=constants.FS_KHZ
):
    return 1