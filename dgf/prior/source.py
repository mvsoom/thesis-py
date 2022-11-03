"""Fitting the parameters of the source"""
from init import __memory__
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

import numpy as np
import scipy.stats
import warnings
import dynesty

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