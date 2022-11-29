from init import __memory__
from vtr.prior import polezero
from vtr.prior import bandwidth
from vtr import spectrum
from lib import constants
from vtr.prior import pareto
from dgf import core

import numpy as np

import dynesty
import scipy.stats
import scipy.special

K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(1e7)}

def transfer_function_power_dB(f, x, y, g):
    """Calculate the AP power spectrum of the impulse response in dB where `f, x, y` given in Hz, `g` in msec."""
    def labs(x):
        return np.log10(np.abs(x))

    poles = polezero.poles(x, y)
    s = 2*np.pi*(1j)*f
    logscale = np.sum(2*labs(poles)) + labs(g/1000.) # normalization + rescaling
    
    logdenom = np.sum(labs(s[:,None] - poles[None,:]) + labs(s[:,None] - np.conjugate(poles[None,:])), axis=1)
    return 20.*(logscale - logdenom)
    
def analytical_tilt(K):
    """Let s -> infty such that the all the poles look like zeros"""
    tilt = -20.*K*np.log10(4) # = (12*K) dB/octave
    return tilt

def g_prior_ppf(q, x, y, mu2_msec=1.):
    """p(g|x,y) = N(0, sigma² = mu2_msec/unscaled_energy)"""
    w = scipy.special.ndtri(q)
    unscaled_energy = unscaled_impulse_response_energy(x, y) # kHz
    sigma = np.sqrt(mu2_msec/unscaled_energy) # msec
    g = w*sigma
    return g # msec

def _g_prior_ppf_exp(q, x, y, mu2_msec=1.):
    """p(g²|x,y) = Exp(expval = mu2_msec/unscaled_energy)"""
    unscaled_energy = unscaled_impulse_response_energy(x, y) # kHz
    expval = mu2_msec/unscaled_energy # msec²
    return np.sqrt(-np.log(q)*expval) # msec

def excluded_pole_product(p):
    ps = np.concatenate([p, np.conj(p)])
    diff = ps[None,:] - ps[:,None]
    diff[np.diag_indices_from(diff)] = 1.
    denom = np.prod(diff, axis=0)
    return (1./denom)[:len(p)]

def unscaled_pole_coefficients(x, y):
    p = polezero.poles(x, y)
    normalization = np.prod(np.abs(p)**2)
    return normalization*excluded_pole_product(p) # Hz if x, y in Hz

def pole_coefficients(x, y, g):
    """(x, y) Hz, g in msec"""
    return (g/1000)*unscaled_pole_coefficients(x, y) # dimensionless

def impulse_response(t, x, y, g):
    """t is in msec, (x, y) in Hz, g in msec"""
    c = pole_coefficients(x, y, g) # dimensionless
    p = polezero.poles(x, y)/1000 # kHz
    h = polezero.impulse_response_cp(t, c, p)
    return h

# The next version is identical to impulse_response() but uses core methods
def _impulse_response2(t, x, y, g):
    """t in msec, x and y in Hz, g in msec"""
    poles = (-np.pi*y + 2*np.pi*(1j)*x)/1000 # kHz
    c = core.pole_coefficients(poles) # kHz
    Y = np.real(2.*c[None,:]*np.exp(t[:,None]*poles[None,:]))
    h = np.sum(Y, axis=1) # kHz
    return g*h

def unscaled_impulse_response_energy(x, y):
    c = unscaled_pole_coefficients(x, y) # Hz
    ab = polezero.amplitudes(c) # Hz
    S = polezero.overlap_matrix(x, y) # sec
    energy = (ab.T @ S @ ab)/1000
    return energy # kHz
    
def impulse_response_energy(x, y, g):
    return (g**2)*unscaled_impulse_response_energy(x, y) # msec

def fit_TFB_sample(
    sample,
    K,
    cacheid,
    xmin=constants.MIN_X_HZ,
    xmax=constants.MAX_X_HZ,
    ymin=constants.MIN_Y_HZ,
    ymax=constants.MAX_Y_HZ,
    sigma_F=constants.SIGMA_FB_REFERENCE_HZ,
    sigma_B=constants.SIGMA_FB_REFERENCE_HZ,
    tilt_target=constants.FILTER_SPECTRAL_TILT_DB,
    sigma_tilt=constants.SIGMA_TILT_DB,
    energy_target=constants.IMPULSE_RESPONSE_ENERGY_MSEC,
    sigma_energy=constants.SIGMA_IMPULSE_RESPONSE_ENERGY_MSEC,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS
):
    ndim = 2*K + 1

    xnullbar = pareto.assign_xnullbar(K, xmin, xmax)
    band_bounds = (
        np.array([ymin]*K), np.array([ymax]*K)
    )

    def unpack(params):
        x, y = np.split(params[:-1], 2)
        g = params[-1]
        return x, y, float(g)

    def loglike(
        params,
        f = sample['f'],
        F_true = sample['F'],
        B_true = sample['B']
    ):
        x, y, g = unpack(params)

        if np.any(x >= xmax):
            return -np.inf

        # Calculate all-pole transfer function
        power = transfer_function_power_dB(f, x, y, g)

        # Heuristically measure formants
        try:
            F, B = spectrum.get_formants_from_spectrum(f, power)
        except np.linalg.LinAlgError:
            return -np.inf

        if len(F) != 3:
            return -np.inf
        
        # Heuristically measure spectral tilt
        tilt = spectrum.fit_tilt(f, power, cutoff=F_true[-1])
        
        if np.isnan(tilt): # NaN occurs if badly conditioned, very rare
            return -np.inf

        # Calculate impulse response energy (in msec)
        energy = impulse_response_energy(x, y, g)

        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2
        energy_err = ((energy - energy_target)/sigma_energy)**2

        return -(F_err + B_err + tilt_err + energy_err)/2

    def ptform(u):
        ux, uy, ug = unpack(u)
        x = pareto.sample_x_ppf(ux, K, xnullbar)
        y = pareto.sample_jeffreys_ppf(uy, band_bounds)
        g = g_prior_ppf(ug, x, y)
        return np.concatenate((x, y, [g]))
    
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