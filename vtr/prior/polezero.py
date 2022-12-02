from init import __memory__
from vtr.prior import bandwidth
from vtr import spectrum
from lib import constants
from vtr.prior import pareto

import numpy as np

import warnings
import dynesty
import scipy.stats
import scipy.linalg
import scipy.special

K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)

SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(1e7)}

def transfer_function_power_dB(f, x, y, ab, normalize_gain=False):
    """Calculate the PZ power spectrum of the impulse response in dB
    
    `f, x, y` given in Hz.
    
    If `normalize_gain` normalize s.t. the power spectrum is zero at
    zero frequency. (But this type of normalization is not dimensionally
    consistent and should be avoided.)
    """
    AB = ab[:,None]
    X = x[:,None]
    Y = y[:,None]
    S = (2.*np.pi*1j)*f[None,:]
    
    A, B = np.split(AB, 2)
    
    P = -np.pi*Y + 2.*np.pi*(1j)*X
    C = (A - (1j)*B)/2
    terms = C/(S - P) + np.conj(C)/(S - np.conj(P))
    H = np.sum(terms, axis=0)
    
    if normalize_gain:
        dc = -2*np.sum(np.real(C*np.conj(P))/np.abs(P)**2, axis=0)
        G = 1/dc
        H *= G
    
    power = 20.*np.log10(np.abs(H))
    return power

def analytical_tilt():
    tilt = -10.*np.log10(4) # = -6 dB/octave
    return tilt

def cos_overlap_matrix(x, y):
    x1, x2 = x[:,None], x[None,:]
    y1, y2 = y[:,None], y[None,:]
    
    num = (y1 + y2)*(x1**2 + x2**2 + (y1 + y2)**2)
    den = ((x1 - x2)**2 + (y1 + y2)**2)*((x1 + x2)**2 + (y1 + y2)**2)
    return num/den

def sin_overlap_matrix(x, y):
    x1, x2 = x[:,None], x[None,:]
    y1, y2 = y[:,None], y[None,:]
    
    num = 2*x1*x2*(y1 + y2)
    den = ((x1 - x2)**2 + (y1 + y2)**2)*((x1 + x2)**2 + (y1 + y2)**2)
    return num/den

def cos_sin_overlap_matrix(x, y):
    x1, x2 = x[:,None], x[None,:]
    y1, y2 = y[:,None], y[None,:]
    
    num = x2 *(-x1**2 + x2**2 + (y1 + y2)**2)
    den = (x1**2 - x2**2)**2 + 2*(x1**2 + x2**2)*(y1 + y2)**2 + (y1 + y2)**4
    return num/den
    
def overlap_matrix(x, y):
    X = 2*np.pi*x
    Y = np.pi*y
    
    c = cos_overlap_matrix(X, Y)
    s = sin_overlap_matrix(X, Y)
    cs = cos_sin_overlap_matrix(X, Y)
    
    S = np.block([
        [c,    cs],
        [cs.T, s ]
    ])
    return S

def _nan_like(a):
    return np.nan*a

def mvn_precision_ppf(q, P):
    """Sample from a zero-mean MVN parametrized by its precision matrix `P`
    See http://www.statsathome.com/2018/10/19/sampling-from-multivariate-normal-precision-and-covariance-parameterizations/"""
    w = scipy.special.ndtri(q)
    try:
        L = np.linalg.cholesky(P)
        z = scipy.linalg.solve_triangular(L, w, lower=True)
        return z
    except np.linalg.LinAlgError:
        return _nan_like(q)
    
def amplitudes_prior_ppf(q, x, y, E_msec=1.):
    """The VTRs `x, y` are in Hz, so `E` must be in sec and defaults to 1 msec"""
    K = len(x)
    E = E_msec/1000 # sec
    S = overlap_matrix(x, y)
    precision_matrix = (2*K)/E*S
    ab = mvn_precision_ppf(q, precision_matrix)
    return ab

def pole_coefficients(ab):
    a, b = np.split(ab, 2)
    c = (a - (1j)*b)/2
    return c

def amplitudes(c):
    """Calculate the a (cos) and b (sin) amplitudes corresponding with the pole coefficients c. This is the inverse of pole_coefficients()"""
    a = np.real(c + np.conj(c))
    b = np.real((1j)*(c - np.conj(c)))
    ab = np.concatenate((a, b))
    return ab

def poles(x, y):
    return -np.pi*y + 2*np.pi*(1j)*x

def impulse_response(t, x, y, ab):
    """t is in msec, (x, y) in Hz"""
    c = pole_coefficients(ab)
    p = poles(x, y)/1000. # kHz
    h = impulse_response_cp(t, c, p)
    return h

def impulse_response_cp(t, c, p):
    return np.real(2.*c[None,:]*np.exp(t[:,None]*p[None,:])).sum(axis=1)

def impulse_response_energy(x, y, ab):
    """Return the analytical impulse response energy in msec"""
    S = overlap_matrix(x, y) # sec
    energy = (ab.T @ S @ ab)*1000. # msec
    return energy

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
    ndim = 4*K

    xnullbar = pareto.assign_xnullbar(K, xmin, xmax)
    band_bounds = (
        np.array([ymin]*K), np.array([ymax]*K)
    )

    def unpack(params):
        xy, ab = np.split(params, 2)
        x, y = np.split(xy, 2)
        return x, y, ab

    def loglike(
        params,
        f = sample['f'],
        F_true = sample['F'],
        B_true = sample['B']
    ):
        x, y, ab = unpack(params)
        
        if np.any(np.isnan(ab)):
            return -np.inf # Semi-definite overlap matrix S in p(a,b|x,y)

        if np.any(x >= xmax):
            return -np.inf

        # Calculate pole-zero transfer function
        power = transfer_function_power_dB(f, x, y, ab)

        # Heuristically measure formants
        try:
            F, B = spectrum.get_formants_from_spectrum(f, power)
        except np.linalg.LinAlgError:
            return -np.inf

        if len(F) != 3:
            return -np.inf
        
        # Heuristically measure spectral tilt starting from F3(true)
        tilt = spectrum.fit_tilt(f, power, cutoff=F_true[-1])
        
        if np.isnan(tilt): # NaN occurs if badly conditioned, very rare
            return -np.inf
        
        # Calculate impulse response energy (in msec)
        energy = impulse_response_energy(x, y, ab)

        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2
        energy_err = ((energy - energy_target)/sigma_energy)**2

        return -(F_err + B_err + tilt_err + energy_err)/2

    def ptform(u):
        ux, uy, uab = unpack(u)
        x = pareto.sample_x_ppf(ux, K, xnullbar)
        y = pareto.sample_jeffreys_ppf(uy, band_bounds)
        ab = amplitudes_prior_ppf(uab, x, y)
        return np.concatenate((x, y, ab))
    
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
        seed=842329,
        Ks=K_RANGE
    )