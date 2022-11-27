from init import __memory__
from vtr.prior import bandwidth
from vtr import spectrum
from lib import constants
from vtr.prior import pareto
from dgf import core

import numpy as np
import jax

import dynesty
import scipy.stats

K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(3e5)}

def transfer_function_power_dB(f, x, y):
    """Calculate the AP power spectrum of the impulse response in dB
    
    `f, x, y` given in Hz. Power spectrum is normalized such that it is
    0 dB at f = 0 Hz.
    """
    def labs(x):
        return np.log10(np.abs(x))

    poles = -np.pi*y + 2*np.pi*(1j)*x
    s = 2*np.pi*(1j)*f
    G = np.sum(2*labs(poles))
    
    denom = np.sum(labs(s[:,None] - poles[None,:]) + labs(s[:,None] - np.conjugate(poles[None,:])), axis=1)
    return 20.*(G - denom)

def impulse_response(t, x, y):
    """t in msec, x and y in Hz"""
    poles = (-np.pi*y + 2*np.pi*(1j)*x)/1000 # kHz
    c = core.pole_coefficients(poles)
    Y = np.real(2.*c[None,:]*np.exp(t[:,None]*poles[None,:]))
    h = np.sum(Y, axis=1)
    return h
    
def analytical_tilt(K):
    """Let s -> infty such that the all the poles look like zeros"""
    tilt = -20.*K*np.log10(4) # = (12*K) dB/octave
    return tilt

def fit_TFB_sample(
    sample,
    K,
    cacheid,
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
    ndim = 2*K

    xnullbar = pareto.assign_xnullbar(K, xmin, xmax)
    band_bounds = (
        np.array([ymin]*K), np.array([ymax]*K)
    )

    def unpack(params):
        x, y = np.split(params, 2)
        return x, y

    def loglike(
        params,
        f = sample['f'][::2],
        F_true = sample['F'],
        B_true = sample['B']
    ):
        x, y = unpack(params)

        if np.any(x >= xmax):
            return -np.inf

        # Calculate all-pole transfer function
        power = transfer_function_power_dB(f, x, y)

        # Heuristically measure formants
        try:
            F, B = spectrum.get_formants_from_spectrum(f, power)
        except np.linalg.LinAlgError:
            return -np.inf

        if len(F) != 3:
            return -np.inf
        
        # Heuristically measure spectral tilt
        tilt = spectrum.fit_tilt(f, power)
        
        if np.isnan(tilt): # NaN occurs if badly conditioned, very rare
            return -np.inf

        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2

        return -(F_err + B_err + tilt_err)/2

    def ptform(u):
        ux, uy = unpack(u)
        x = pareto.sample_x_ppf(ux, K, xnullbar)
        y = pareto.sample_jeffreys_ppf(uy, band_bounds)
        return np.concatenate((x, y))
    
    # Run the sampler and cache results based on `cacheid`
    if 'nlive' not in samplerargs:
        samplerargs = samplerargs.copy()
        samplerargs['nlive'] = ndim*3

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

def get_fitted_TFB_samples(n_jobs=1):
    return bandwidth.get_fitted_TFB_samples(
        n_jobs,
        fit_func=fit_TFB_sample,
        seed=6667890,
        Ks=K_RANGE
    )