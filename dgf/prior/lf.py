"""Define the prior over the LF model for the glottal flow derivative"""
from init import __memory__
from dgf.prior import lf
from dgf.prior import period
from dgf.prior import holmberg
from dgf import constants
from lib import lfmodel

import jax
import jax.numpy as jnp
import numpy as np

#########################
# Sampling $p(R_k|R_e)$ #
#########################
_UNCONTAINED = np.inf
def _contains(a, x):
    return (a[0] < x) and (x < a[1])

ALPHA_M_RANGE = np.array([0.55, 0.8]) # Common range; Drugman (2019), Table 1

def estimate_sigma2_Rk(numsamples=10000, seed=5571):
    """
    Estimate the variance of Rk from the known theoretical range of $\alpha_m$.
    """
    alpha_m = np.random.default_rng(seed).uniform(*ALPHA_M_RANGE, size=numsamples)
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
RK_RANGE = np.array([0., 1.]) # Is a percentage

def sample_Rk(Re, rng):
    """
    Use the regression in Fant (1994) Eq. (2) + reverse-engineered noise based on the
    given `r = 0.93` value and the estimated standard deviation of Rk.
    """
    Rk = _UNCONTAINED
    while not _contains(RK_RANGE, Rk): Rk = (22.4 + 11.8*Re)/100 + RK_STDDEV*rng.normal()
    return Rk

#########################
# Sampling $p(R_a|R_e)$ #
#########################
OQ_RANGE = np.array([0.3, 0.9]) # Common range; Drugman (2019), Table 1
QA_RANGE = np.array([0., 1.0]) # Theoretical range; Doval (2006), p. 5

def estimate_sigma2_Ra(numsamples=10000, seed=6236):
    """
    Estimate the variance of Ra from the known theoretical range of OQ and Qa.
    """
    rng = np.random.default_rng(seed)
    Oq = rng.uniform(*OQ_RANGE, numsamples) # @Drugman2019, Table 1
    Qa = rng.uniform(*QA_RANGE, numsamples) # Doval (2006), p. 5
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
RA_RANGE = np.array([0., 1.]) # Is a percentage

def sample_Ra(Re, rng):
    """
    Use the regression in Fant (1994) Eq. (3) + reverse-engineered noise based on the
    given `r = 0.91` value and the estimated standard deviation of Ra.
    
    Note that `Re` in Fant (1994) is called `Rd` in Perrotin (2021)!
    """
    Ra = _UNCONTAINED
    while not _contains(RA_RANGE, Ra): Ra = (-1 + 4.8*Re)/100 + RA_STDDEV*rng.normal()
    return Ra

###################################
# Sampling $p(R_a, R_k, R_g|R_e)$ #
###################################
RG_RANGE = np.array([
    (1 + RK_RANGE[0]) / (2*OQ_RANGE[1]),
    (1 + RK_RANGE[1]) / (2*OQ_RANGE[0])
]) # From Rg = (1 + Rk)/(2*Oq)

def sample_R_triple(Re, rng):
    """
    From Perrotin (2021) Eq. (A1). `Re` is typically in `[0.3, 2.7]`.

    See Fant (1994) for the meaning of these dimensionless parameters
    * Note that in that paper they are given in percent (%)
    * Note that `Re` in Fant (1994) is called `Rd` in Perrotin (2021).
    * Note that `Rg` can be larger than 1, unlike `Ra `and `Rg` (Fant 1994, Fig. 3)
    """
    Rg = _UNCONTAINED
    while not _contains(RG_RANGE, Rg):
        Ra = sample_Ra(Re, rng)
        Rk = sample_Rk(Re, rng)
        Rg = Rk*(0.5 + 1.2*Rk)/(0.44*Re - 4*Ra*(0.5 + 1.2*Rk)) # Uncertainty from Ra and Rk transfers to Rg
    return Ra, Rk, Rg

#######################################
# Sampling $p(R_a, R_k, R_g, R_e|T0)$ #
#######################################
def _collect_list_of_dicts(ld):
    """Convert a list of dicts to a dict of lists"""
    return {k: jnp.array([d[k] for d in ld]) for k in ld[0]}

def calculate_Re(T0, Td):
    """
    We choose the declination time `T0 = U_0/E_e` as the independent variable
    In this way we can induce correlations between T0 and all other variables, as
    empirically observed (Henrich 2005).
    
    Note that `Re` in Fant (1994) is called `Rd` in Perrotin (2021).
    """
    F0 = 1/T0 # kHz
    Re = Td * F0 / (0.11) # Fant (1994) Eq. (4)
    return Re

def sample_R_params(T0, Td, rng):
    p = dict()
    p['T0'] = T0
    p['Td'] = Td
    p['Re'] = lf.calculate_Re(T0, Td)
    p['Ra'], p['Rk'], p['Rg'] = lf.sample_R_triple(p['Re'], rng)
    return p

_eps = 1e-3
LF_GENERIC_BOUNDS = {
    'power': [_eps, 10.],
    'T0': [constants.MIN_PERIOD_LENGTH_MSEC, constants.MAX_PERIOD_LENGTH_MSEC],
    'Oq': [constants.MIN_OQ, constants.MAX_OQ],
    'am': [0.5, 1-_eps], # Theoretical (restricted) range; Doval (2006), p. 5
    'Qa': [_eps, 1-_eps], # Theoretical range; Doval (2006), p. 5
}

@__memory__.cache
def sample_lf_params(fs=10., numsamples=50000, seed=2387):
    # Manage both Numpy and JAX RNGs
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    rng_seed = int(jax.random.randint(key1, (1,), minval=0, maxval=int(1e4)))
    rng = np.random.default_rng(rng_seed)
    
    # Sample pitch periods and declination times (both in msec)
    T0 = period.marginal_prior().sample(numsamples, seed=key2)
    Td = holmberg.declination_time_prior().sample(numsamples, seed=key3)
    
    # Sample the `R` parameters based on the pitch periods
    p = _collect_list_of_dicts([sample_R_params(T, Td, rng) for T, Td in zip(T0, Td)])
    
    # Transform the `R` parameters to the equivalent `T` and generic representations
    p['Ee'] = jnp.ones(numsamples)
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
    
    # Remove all samples which have inconsistent LF parameters as signalled by
    # their power being `nan`, as in `lfmodel.consistent_lf_params()`
    mask = ~jnp.isnan(p['power'])
    
    # Remove all samples whose generic parameters are out of bounds
    for k, bounds in LF_GENERIC_BOUNDS.items():
        lower, upper = bounds
        mask &= (lower < p[k]) & (p[k] < upper)
    
    p = {k: v[mask] for k, v in p.items()}
    return p

#########################################################################
# The priors based on the fitted empirical distributions in the z domain #
#########################################################################
def generic_params_prior(num_pitch_periods):
    pass

def dgf_prior(num_pitch_periods):
    p = generic_params_prior(num_pitch_periods)
    pass