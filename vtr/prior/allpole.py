from init import __memory__
from dgf import core
from vtr.prior import bandwidth
from vtr import spectrum
from lib import constants
from vtr.prior import pareto

import numpy as np
import jax

import warnings
import dynesty
import scipy.stats

K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(3e5)}

def transfer_function_power_dB(x, poles):
    """x an array in kHz, poles in rad kHz"""
    def labs(x):
        return np.log10(np.abs(x))

    s = (1j)*x*2*np.pi # rad kHz
    G = np.sum(2*labs(poles))
    
    denom = np.sum(labs(s[:,None] - poles[None,:]) + labs(s[:,None] - np.conjugate(poles[None,:])), axis=1)
    return 20.*(G - denom)

def analytical_tilt(K):
    """Let s -> infty such that the all the poles look like zeros"""
    tilt = -20.*K*np.log10(4) # = (12*K) dB/octave
    return tilt

def number_of_peaks(f, power):
    F, B = spectrum.get_formants_from_spectrum(f, power)
    K = len(F)
    return K

@__memory__.cache
def get_TFB_samples(
    num_samples=50,
    seed=312178
):
    prior = bandwidth.TFB_prior()
    f = constants.spectrum_frequencies()
    
    def sample_reject(key):
        while True:
            T, *FB = prior.sample(seed=key)
            F, B = np.split(np.array(FB), 2)
            poles = core.make_poles(B, F)
            power = transfer_function_power_dB(f/1000, poles)
            
            if number_of_peaks(f, power) == 3:
                break
            else:
                warnings.warn("Rejected TFB sample that had merged peaks in its power spectrum")
                _, key = jax.random.split(key)
        
        sample = dict(
            T = T,
            F = F,
            B = B,
            f = f,
            power = power
        )
        return sample
    
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    samples = [sample_reject(key) for key in keys]
    return samples

def fit_TFB_sample(
    sample,
    K,
    cacheid,
    rho_alpha=spectrum.RHO_ALPHA,
    rho_beta=spectrum.RHO_BETA,
    h_scale=spectrum.H_SCALE_dB,
    xmin=constants.MIN_X_HZ,
    xmax=constants.MAX_X_HZ,
    ymin=constants.MIN_Y_HZ,
    ymax=constants.MAX_Y_HZ,
    sigma_F=constants.SIGMA_F_REFERENCE_HZ,
    sigma_B=constants.SIGMA_B_REFERENCE_HZ,
    tilt_target=constants.FILTER_SPECTRAL_TILT_DB,
    sigma_tilt=constants.SIGMA_TILT_DB,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS
):
    ndim = 2 + 2*K

    xnullbar = pareto.assign_xnullbar(K, xmin, xmax)
    band_bounds = (
        np.array([ymin]*K), np.array([ymax]*K)
    )

    def unpack(params):
        rho, h = params[:2]
        x, y = np.split(params[2:], 2)
        return rho, h, x, y

    def loglike(
        params,
        f = sample['f'],
        F_true = sample['F'],
        B_true = sample['B']
    ):
        rho, h, x, y = unpack(params)

        if np.any(x >= xmax):
            return -np.inf

        # Calculate all-pole transfer function
        poles = core.make_poles(y, x)
        power = transfer_function_power_dB(f/1000, poles)

        # Heuristically measure formants
        try:
            F, B = spectrum.get_formants_from_spectrum(
                f, power, rho, h
            )
        except np.linalg.LinAlgError:
            return -np.inf

        if len(F) != 3:
            return -np.inf
        
        # Heuristically measure spectral tilt
        tilt = spectrum.fit_tilt(f, power)
        
        if np.isnan(tilt) or tilt > 0.: # NaN occurs if badly conditioned, very rare
            return -np.inf

        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2

        return -(F_err + B_err + tilt_err)/2

    def ptform(
        u,
        rho_prior = scipy.stats.beta(rho_alpha, rho_beta),
        h_prior = scipy.stats.expon(h_scale)
    ):
        us = unpack(u)
        rho = rho_prior.ppf(us[0])
        h = h_prior.ppf(us[1])
        x = pareto.sample_x_ppf(us[2], K, xnullbar)
        y = pareto.sample_jeffreys_ppf(us[3], band_bounds)
        return np.concatenate(([rho, h], x, y))
    
    # Run the sampler and cache results based on `cacheid`
    if 'nlive' not in samplerargs:
        samplerargs = samplerargs.copy()
        samplerargs['nlive'] = ndim*5

    @__memory__.cache
    def run_nested(cacheid, samplerargs, runargs):
        seed = cacheid
        rng = np.random.default_rng(seed)
        sampler = dynesty.NestedSampler(
            loglike, ptform, ndim=ndim,
            rstate=rng, **samplerargs
        )
        sampler.run_nested(**runargs)
        return sampler.results
    
    results = run_nested(cacheid, samplerargs, runargs)
    return results

def yield_fitted_TFB_samples(
    seed=666789,
    Ks=K_RANGE,
    verbose=True
):
    TFB_samples = get_TFB_samples()
    rng = np.random.default_rng(seed)
    
    for K in Ks:
        for i, sample in enumerate(TFB_samples):
            cacheid = rng.integers(int(1e8))
            results = fit_TFB_sample(sample, K, cacheid)
            
            if verbose: print(K, i, results['logz'][-1])

            yield dict(
                K=K,
                i=i,
                sample=sample,
                cacheid=cacheid,
                results=results
            )

def get_fitted_TFB_samples():
    return list(yield_fitted_TFB_samples())





####

def crazy_yield_fitted_TFB_samples(
    seed=666789,
    Ks=K_RANGE,
    verbose=True
):
    import random
    random.seed()
    
    TFB_samples = get_TFB_samples()
    rng = np.random.default_rng(seed)
    
    for K in Ks:
        for i, sample in enumerate(TFB_samples):
            cacheid = rng.integers(int(1e8))
            
            if random.random() > 1/(50*len(Ks)):
                continue
            
            print("accepted", K, i)
            results = fit_TFB_sample(sample, K, cacheid)
            
            if verbose: print(K, i, results['logz'][-1])

            yield dict(
                K=K,
                i=i,
                sample=sample,
                cacheid=cacheid,
                results=results
            )

def crazy():
    while True:
        list(crazy_yield_fitted_TFB_samples())
