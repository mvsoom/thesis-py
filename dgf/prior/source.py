"""Fitting the parameters of the source"""
from init import __memory__, __cache__
from dgf.prior import lf
from dgf.prior import period
from lib import constants
from dgf import bijectors
from dgf import isokernels
from dgf import core
from lib import lfmodel
from lib import util

import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import numpy as np
import scipy.stats
import warnings
import dynesty
import hashlib
import json

FIT_LF_SAMPLE_PARAMS = ('noise_power_sigma', *constants.SOURCE_PARAMS)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(3e5)}

KERNEL_MS = (16, 32, 64, 128, 256)
KERNEL_NAMES = ('Matern12Kernel', 'Matern32Kernel', 'Matern52Kernel', 'SqExponentialKernel')

@__memory__.cache
def get_lf_samples(
    num_samples=50,
    fs=constants.FS_KHZ,
    noise_floor_power=constants.NOISE_FLOOR_POWER,
    seed=48790
):
    prior = lf.generic_params_prior()
    
    def sample(key):
        t, u, log_prob_u, context = lf.sample_and_log_prob_dgf(
            prior, key, return_full=True, fs=fs,
            noise_floor_power=noise_floor_power
        )
        p = context['p']
        sample = dict(
            p=p, t=t, u=u,
            log_prob_u=log_prob_u,
            noise_floor_power=noise_floor_power
        )
        return sample
    
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    samples = [sample(key) for key in keys]
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
        # Upacking must match FIT_LF_SAMPLE_PARAMS
        noise_power_sigma, var_sigma, r, T, Oq = x
        
        noise_power = noise_power_sigma**2
        var = var_sigma**2
        
        R = core.kernelmatrix_root_gfd_oq(
            kernel, var, r, t, kernel_M, T, Oq, c, impose_null_integral
        )
        logl = core.loglikelihood_hilbert(R, u, noise_power)
        
        # NaN logl values occur for Oq > 1 and extreme values of x
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
        samplerargs = samplerargs.copy()
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
    kernel_names=KERNEL_NAMES,
    verbose=False
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
                        if verbose: print(i, config, results['logz'][-1])
                    
                        yield dict(
                            i=i,
                            sample=sample,
                            config=config,
                            results=results
                        )

def get_fitted_lf_samples():
    return list(yield_fitted_lf_samples())

def process_fitted_lf_samples():
    def process():
        for fit in yield_fitted_lf_samples():
            i, sample, config, results =\
                fit['i'], fit['sample'], fit['config'], fit['results']
            
            lf_p = sample['p'].copy()
            lf_p['Oq_LF'] = lf_p.pop('Oq') # Rename to avoid name clash
            
            mean, _ = util.get_posterior_moments(results)
            mean_dict = {
                k: mean[i] for i, k in enumerate(FIT_LF_SAMPLE_PARAMS)
            }
            
            log_prob_p = results.logz[-1] # Use analytical estimate
            log_prob_p_sd = results.logzerr[-1] # Use analytical estimate
            
            d = dict(
                **config,
                i=i,
                log_prob_p=log_prob_p,
                log_prob_p_sd=log_prob_p_sd,
                log_prob_q=sample['log_prob_u'],
                **lf_p,
                **mean_dict,
                SNR_q=-10*np.log10(sample['noise_floor_power']),
                SNR_p=-10*np.log10(mean_dict['noise_power_sigma']**2)
            )
            
            yield {k: _maybe_native(v) for k, v in d.items()}
            
    return list(process())

def _maybe_native(v): # https://stackoverflow.com/a/11389998/6783015
    try: return np.array(v).item()
    except: return v

@__memory__.cache
def posterior_of_fitted_lf_values(selection=[], numsamples=100):
    def process():
        for fit in yield_fitted_lf_samples():
            config, results = fit['config'], fit['results']
            
            if not all([_match_config(config, c) for c in selection]):
                continue

            yield util.resample_equal(results, numsamples)

    return np.vstack(list(process()))

def _remove_cacheid(d):
    dc = d.copy()
    dc.pop('cacheid', None)
    return dc
    
def _match_config(a, b):
    return _remove_cacheid(a) == _remove_cacheid(b)

def _yield_all_configs():
    for kernel_M in KERNEL_MS:
        for kernel_name in KERNEL_NAMES:
            for use_oq in (False, True):
                for impose_null_integral in (False, True):
                    config = dict(
                        kernel_name = kernel_name,
                        kernel_M = kernel_M,
                        use_oq = use_oq,
                        impose_null_integral = impose_null_integral
                    )
                    yield config

########################
# Bijectors and priors #
########################
def SOURCE_PARAMS(config):
    p = constants.SOURCE_PARAMS
    return p if config['use_oq'] else p[:-1]

def SOURCE_NDIM(config):
    return len(SOURCE_PARAMS(config))

def SOURCE_BOUNDS(config):
    bounds = np.vstack(
        [constants.SOURCE_BOUNDS[k] for k in constants.SOURCE_PARAMS]
    )
    return bounds if config['use_oq'] else bounds[:-1,:]

def T_INDEX(config):
    return SOURCE_PARAMS(config).index('T') 

def source_posterior(config, numsamples=100):
    full = posterior_of_fitted_lf_values([config], numsamples=numsamples)
    
    # Cut out the noise power parameter and possibly Oq
    samples = full[:,1:]
    return samples if config['use_oq'] else samples[:,:-1]

def _dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    # https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def _config_to_seed(config):
    return 1 + int(_dict_hash(config), 16)

def _prune_out_of_bounds(a, bounds):
    lower, upper = bounds.T
    mask = ((lower < a) & (a < upper)).all(axis=1)
    return a[mask]

def fit_source_bijector(
    config,
    return_fit_results=False
):
    """Return a bijector for the source params with dimension `SOURCE_NDIM(config)`, i.e., depending on`config['use_oq']`"""
    cacheid = _config_to_seed(config)
    posterior = source_posterior(config)
    bounds = SOURCE_BOUNDS(config)
    pruned = _prune_out_of_bounds(posterior, bounds)
    return bijectors.fit_nonlinear_coloring_bijector(
        pruned, bounds, cacheid,
        return_fit_results=return_fit_results
    )

def _fit_all_source_bijectors():
    for config in _yield_all_configs():
        fit_source_bijector(config)

def trajectify_bijector(bstatic, num_pitch_periods):
    """Turn a static bijector `bstatic` into a trajectory bijector with `num_pitch_periods` using the fitted source GP based on ground truth period trajectories in the APLAWD database""" 
    kernel_name, _, results =\
        period.fit_period_trajectory_kernel()

    envelope_lengthscale, envelope_noise_sigma =\
        period.maximum_likelihood_envelope_params(results)

    btraj = bijectors.nonlinear_coloring_trajectory_bijector(
        bstatic,
        num_pitch_periods,
        kernel_name,
        envelope_lengthscale,
        envelope_noise_sigma
    )

    return btraj

def source_trajectory_bijector(
    num_pitch_periods,
    config,
    T_estimate=None,
    noiseless_estimates=False
):
    """
    Get a bijector sending N(0,I_n) to (var_sigma, r, T[, Oq]) samples where
    n, the total dimension is `n = num_pitch_periods*SOURCE_NDIM(config) = 
    num_pitch_periods*(3 or 4`).
    
    Optionally condition on Praat's estimates of the pitch periods
    `T_estimate` shaped `(num_pitch_periods,)`. If not `noiseless_estimates`,
    then condition on the estimates without taking into account Praat's
    estimation error.
    """
    marginal_bijector = fit_source_bijector(config)
    trajectory_bijector = trajectify_bijector(marginal_bijector, num_pitch_periods)
    
    # Condition the bijector on T estimates (if any)
    ndim = SOURCE_NDIM(config)
    observation = np.full((num_pitch_periods, ndim), np.nan)
    noise_mean = np.zeros(ndim)
    noise_cov = np.zeros((ndim, ndim))
    
    if T_estimate is not None:
        j = T_INDEX(config)
        observation[:,j] = T_estimate
        if not noiseless_estimates:
            noise_mean[j], noise_cov[j, j] =\
                period.fit_praat_estimation_mean_and_cov()

    trajectory_bijector = bijectors.condition_nonlinear_coloring_trajectory_bijector(
        trajectory_bijector,
        observation,
        noise_cov,
        noise_mean
    )
    
    return trajectory_bijector

def source_trajectory_prior(
    num_pitch_periods,
    config,
    T_estimate=None,
    noiseless_estimates=False
):
    b = source_trajectory_bijector(
        num_pitch_periods,
        config,
        T_estimate,
        noiseless_estimates
    )
    
    if T_estimate is None:
        name = 'SourceTrajectoryPrior'
    else:
        name = 'ConditionedSourceTrajectoryPrior'
    
    ndim = SOURCE_NDIM(config)
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(num_pitch_periods*ndim))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=b,
        name=name
    )
    return prior # prior.sample(ns) shaped (ns, num_pitch_periods, ndim)

def source_marginal_prior(
    config,
    T_estimate=None,
    noiseless_estimates=False
):
    ndim = SOURCE_NDIM(config)
    squeeze_bijector = tfb.Reshape(event_shape_out=(ndim,), event_shape_in=(1,ndim))
    prior = tfd.TransformedDistribution(
        distribution=source_trajectory_prior(
            1,
            config,
            T_estimate,
            noiseless_estimates
        ),
        bijector=squeeze_bijector,
        name="SourceMarginalPrior"
    )
    return prior # prior.sample(ns) shaped (ns, ndim)

@__cache__
def get_source_envelope_kernel():
    envelope_kernel_name, _, results =\
        period.fit_period_trajectory_kernel()

    envelope_lengthscale, _ =\
        period.maximum_likelihood_envelope_params(results)
    
    envelope_variance = 1.
    envelope_kernel = isokernels.resolve(envelope_kernel_name)(
        envelope_variance, envelope_lengthscale
    )
    
    return envelope_kernel