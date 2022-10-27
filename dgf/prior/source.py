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
        sample = dict(p=p, t=t, u=u, log_prob_u=log_prob_u)
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
        noise_power_sigma, var, r, T, Oq = x
        R = core.kernelmatrix_root_gfd_oq(
            kernel, var, r, t, kernel_M, T, Oq, c, impose_null_integral
        )
        logl = core.loglikelihood_hilbert(R, u, noise_power_sigma**2)
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
                            i=i,
                            sample=sample,
                            config=config,
                            results=results
                        )

def get_fitted_lf_samples():
    return list(yield_fitted_lf_samples())