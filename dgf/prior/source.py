"""Fitting the parameters of the source"""
from init import __memory__
from dgf.prior import lf
from dgf.prior import period
from dgf import constants
from dgf import bijectors
from dgf import isokernels
from dgf import core
from lib import lfmodel

import jax
import jax.numpy as jnp

import numpy as np
import scipy.stats
import warnings
import dynesty

MODEL_LF_SAMPLE_PARAMS = ('noise_power', *constants.SOURCE_PARAMS)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(3e5)}

KERNEL_MS = (16, 32, 64, 128, 256)
KERNEL_NAMES = ('Matern12Kernel', 'Matern32Kernel', 'Matern52Kernel', 'SqExponentialKernel')

def _determine_GCI_index(t, u, Te, treshold):
    mask = (t > Te) & (np.abs(u) < treshold)
    return np.argmax(mask) # Returns 0 if mask is `False` everywhere

def _swap_dgf(t, u, Te, treshold):
    i = _determine_GCI_index(t, u, Te, treshold)
    return np.concatenate((u[i:], u[:i]))

def _nonzero_dgf_indices(t, T0):
    return np.flatnonzero((0 <= t) & (t <= T0))

def closed_phase_leading(t, u, p, treshold, offset=0.):
    """Take a DGF waveform and put its closed phase before the GOI"""
    i = _nonzero_dgf_indices(t - offset, p['T0'])
    t_nonzero = t[i]
    u_nonzero = u[i]
    u_nonzero_swapped = _swap_dgf(t_nonzero - offset, u_nonzero, p['Te'], treshold)
    return np.concatenate((u[:i[0]], u_nonzero_swapped, u[i[-1]+1:]))

def _sample_and_log_prob_xt(rng):
    prior = lf.generic_params_prior()
    seed = jax.random.PRNGKey(rng.integers(int(1e4)))
    return lf.sample_and_log_prob_xt(prior, seed)

def _t_params_to_dict(xt):
    p = {k: xt[i] for i, k in enumerate(constants.LF_T_PARAMS)}
    return p

def _sample_and_jacobian(normalized_dgf, p):
    u = normalized_dgf(p)
    
    pk = lfmodel._select_keys(p, *constants.LF_T_PARAMS)
    jacobian_dict = jax.jacobian(normalized_dgf)(pk)
    
    # The shape of the Jacobian is `(N, P)` where `N` is the number of data points
    # in `u` and `P == 4` is the number of relevant `T` parameters of the LF model
    jacobian = jnp.vstack([jacobian_dict[k] for k in constants.LF_T_PARAMS]).T # (N, P)

    return u, jacobian

def _log_prob_u(log_prob_xt, sigma, jacobian):
    N, P = jacobian.shape

    term1 = log_prob_xt
    term2 = (-1/2)*(N - P)*jnp.log(2*jnp.pi*sigma**2)
    term3 = (-1/2)*jnp.linalg.slogdet(jacobian.T @ jacobian)[1]
    
    return term1 + term2 + term3

def _add_noise(u, sigma, rng):
    return u + rng.normal(size=len(u))*sigma

def sample_and_logprob_q(fs, T, rng):
    """Sample and get probability of standardized DGF waveforms from `q(u)` where `u = u(t)`
    
    "Standardized" means that the initial DGF waveform `u0` has gone through the following pipe,
    i.e., `u = pipe(u0)`:
      
      1. Power normalization
      2. Noise is added such that the GP inference problem's noise is lower bounded
         at `constants.NOISE_FLOOR_POWER`
      3. Finally, the closed phase is swapped such that it leads the DGF rather than
         trails it (as, e.g., in Doval 2006).
    
    Steps (1) and (2) affect the final probability such that `q(u) != q(u0)`.
    """
    # Sample and get probability of `T` parameters
    xt, log_prob_xt, p = _sample_and_log_prob_xt(rng)
    
    # Define the grid
    if T is None:
        T = p['T0']
    N = int(np.ceil(T*fs) + 1)
    t = np.arange(N)/fs

    # Convert the sampled parameters `p` into a normalized DGF waveform `u`
    # and get the Jacobian `\del(u)/\del(p)`
    tol = 1e-6
    initial_bracket = lfmodel._get_initial_bracket(p['T0'], tol=tol)
    
    def normalized_dgf(p):
        u0 = lfmodel.dgf(t, p, tol=tol, initial_bracket=initial_bracket)
        power = jnp.mean(jnp.sum(u0**2))
        u = u0/jnp.sqrt(power)
        return u
    
    u, jacobian = _sample_and_jacobian(normalized_dgf, p)

    # Install the noise floor
    sigma = np.sqrt(constants.NOISE_FLOOR_POWER)
    u = _add_noise(u, sigma, rng)
    
    # Calculate the probability `q(u)` given the noise floor and power normalization
    log_prob_u = _log_prob_u(log_prob_xt, sigma, jacobian)
    
    # And finally, swap the closed phase to the front (doesn't affect `q(u)`)
    u = closed_phase_leading(t, u, p, treshold=sigma/3)
    return p, t, u, log_prob_u

@__memory__.cache
def get_lf_samples(
    num_samples=50,
    fs=constants.FS_KHZ,
    T=None,
    seed=4879
):
    rng = np.random.default_rng(seed)
    
    def sample():
        p, t, u, log_prob_u = sample_and_logprob_q(fs, T, rng)
        sample = dict(p=p, t=t, u=u, log_prob_u=log_prob_u)
        return sample
    
    samples = [sample() for _ in range(num_samples)]
    return samples

def fit_lf_sample(
    t,
    u,
    kernel_name,
    kernel_M,
    use_oq,
    impose_null_integral,
    cacheid,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS
):  
    kernel = isokernels.resolve(kernel_name)
    
    ndim = len(constants.SOURCE_PARAMS) + 1
    npdim = ndim if use_oq else ndim - 1
    
    # Define the log likelihood function
    @jax.jit
    def loglike(x, c=constants.BOUNDARY_FACTOR):
        noise_power, var, r, T, Oq = x
        R = core.kernelmatrix_root_gfd_oq(
            kernel, var, r, t, kernel_M, T, Oq, c, impose_null_integral
        )
        logl = core.loglikelihood_hilbert(R, u, noise_power)
        return jax.lax.cond(jnp.isnan(logl), lambda: -jnp.inf, lambda: logl)

    def ptform(
        u,
        prior=scipy.stats.lognorm(np.log(10))
    ):
        if not use_oq:
            # Hack: constrain `Oq` to 1 (= the median of the prior)
            u = np.append(u, .5)
        
        x = prior.ppf(u)
        return x
    
    # Run the sampler and cache results based on `cacheid`
    if 'nlive' not in samplerargs:
        samplerargs['nlive'] = ndim*5

    @__memory__.cache
    def run_nested(cacheid, samplerargs, runargs):
        seed = cacheid
        rng = np.random.default_rng(seed)
        sampler = dynesty.NestedSampler(
            loglike, ptform, ndim=ndim, npdim=npdim,
            rstate=rng, **samplerargs
        )
        sampler.run_nested(**runargs)
        return sampler.results
    
    results = run_nested(cacheid, samplerargs, runargs)
    return results

def yield_fitted_lf_samples(
    seed=67011,
    kernel_ms=KERNEL_MS,
    kernel_names=KERNEL_NAMES
):
    lf_samples = get_lf_samples()
    rng = np.random.default_rng(seed)

    for i, sample in enumerate(lf_samples):
        t = sample['t']
        u = sample['u']
        for kernel_M in kernel_ms:
            for kernel_name in kernel_names:
                for use_oq in (False, True):
                    for impose_null_integral in (False, True):
                        config = dict(
                            kernel_name = kernel_name,
                            kernel_M = kernel_M,
                            use_oq = use_oq,
                            impose_null_integral = impose_null_integral,
                            cacheid=rng.integers(int(1e8))
                        )

                        results = fit_lf_sample(t=t, u=u, **config)
                        print(i, config, results['logz'][-1])
                    
                        yield dict(
                            sample=i,
                            config=config,
                            results=results
                        )

def get_fitted_lf_samples():
    return list(yield_fitted_lf_samples())