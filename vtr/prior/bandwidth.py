"""Fit the p(T, F^R, B^R) prior for T and reference formant frequency and bandwidh"""
from init import __memory__
from vtr import spectrum
from vtr.prior import formant
from vtr.prior import allpole
from lib import constants
from dgf import bijectors
from vtr.prior import pareto

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import numpy as np
import scipy.stats
import warnings
import joblib

import dynesty
import scipy.stats
import scipy.linalg
import scipy.special

def _hawks_bandwidth(T, F):
    """Implement Hawks+ (1995) Eq. (1)"""
    if F < 500.:
        k  = 165.327516
        x1 = -6.73636734e-1
        x2 = 1.80874446e-03
        x3 = -4.52201682e-06
        x4 = 7.49514000e-09
        x5 = -4.70219241e-12
    else:
        k  = 15.8146139
        x1 = 8.10159009e-02
        x2 = -9.79728215e-05
        x3 = 5.28725064e-08
        x4 = -1.07099364e-11
        x5 = 7.91528509e-16
    F0 = 1000./T # Hz
    S = 1. + 0.25*(F0 - 132)/88
    B = S*(k + x1*F + x2*F**2 + x3*F**3 + x4*F**4 + x5*F**5)
    return B

hawks_bandwidth = np.vectorize(_hawks_bandwidth)

############################
# TF prior p(\hat{T}, F^R) #
############################
TF_NDIM = 4
TF_NAMES = ("T_hat", "F1", "F2", "F3")
TF_BOUNDS = np.array([
    (constants.MIN_PERIOD_LENGTH_MSEC, constants.MAX_PERIOD_LENGTH_MSEC),
    (constants.MIN_FORMANT_FREQ_HZ, constants.MAX_FORMANT_FREQ_HZ),
    (constants.MIN_FORMANT_FREQ_HZ, constants.MAX_FORMANT_FREQ_HZ),
    (constants.MIN_FORMANT_FREQ_HZ, constants.MAX_FORMANT_FREQ_HZ)
])

def TF_prior(
    cacheid=87846
):
    """Fit p(^T, F^R) from TIMIT and VTRFormants training data"""
    data = formant.get_vtrformants_training_data()
    TF = [np.column_stack(pair) for pair in zip(data['praat_T'], data['true_F_trajectories'])]
    samples = np.vstack(TF)
    
    assert samples.shape[1] == TF_NDIM

    # Fit the bijector and construct prior
    bijector = bijectors.fit_nonlinear_coloring_bijector(
        samples, TF_BOUNDS, cacheid
    )
    
    standardnormal = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(TF_NDIM))
    prior = tfd.TransformedDistribution(
        standardnormal,
        bijector
    )
    return prior # prior.sample(ns) returns (ns, 4)

#################################
# TFB prior p(\hat T, F^R, B^R) #
#################################
HAWKS_ERRORS_DB = (1.15, 1.9, 3.2) # Errors for F1, F2, F3 in dB

TFB_NDIM = 7
TFB_NAMES = ("T_hat", "F1", "F2", "F3", "B1", "B2", "B3")

B_BOUNDS = np.array([
    (constants.MIN_FORMANT_BAND_HZ, constants.MAX_FORMANT_BAND_HZ),
    (constants.MIN_FORMANT_BAND_HZ, constants.MAX_FORMANT_BAND_HZ),
    (constants.MIN_FORMANT_BAND_HZ, constants.MAX_FORMANT_BAND_HZ)
])

TFB_BOUNDS = np.array([
    *TF_BOUNDS,
    *B_BOUNDS
])

def sample_TFB(
    numsamples=int(1e5), # Not necessarily the number of samples returned
    seed=2387            # due to rejection sampling
):
    # Sample (T, F)
    TF = TF_prior().sample(numsamples, seed=jax.random.PRNGKey(seed))
    T, F = np.hsplit(TF, (1,))
    
    # Regress B given T, F
    B_hawks = hawks_bandwidth(T, F)
    errors = scipy.stats.laplace(scale=HAWKS_ERRORS_DB).rvs(size=(numsamples,3))
    B_true = B_hawks * 10**(errors/20) # Regress with simulated errors
    TFB = np.column_stack((T, F, B_true))
    
    assert TFB.shape[1] == TFB_NDIM
    
    # Make sure everything is within bounds
    def tighten_bounds(b):
        return np.column_stack(
            (b[:,0] + constants._ZERO, b[:,1] - constants._ZERO)
        )
    
    bounds = tighten_bounds(TFB_BOUNDS)
    mn, mx = bounds.T
    inbounds = ((mn < TFB) & (TFB < mx)).all(axis=1)
    
    return TFB[inbounds,:] # (~numsamples, TFB_NDIM)

def TFB_prior(cacheid=3325495):
    """Fit TFB prior as usual on the regressed TFB samples"""
    TFB = sample_TFB()
    assert TFB.shape[1] == TFB_NDIM
    
    bijector = bijectors.fit_nonlinear_coloring_bijector(
        TFB, TFB_BOUNDS, cacheid
    )

    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(TFB_NDIM))
    prior = tfd.TransformedDistribution(
        standardnormals,
        bijector,
        name='TFBPrior'
    )
    return prior # prior.sample(ns) shaped (ns, TFB_NDIM)

##########################################
# TFB samples and fitting them with VTRs #
##########################################
SAMPLERARGS = {'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False, 'maxcall': int(1e7)}
NUM_LOGL_SAMPLES = 1

@__memory__.cache
def get_TFB_samples(
    num_samples=200,
    seed=312178
):
    """Get samples to fit p(x,y) priors to"""
    prior = TFB_prior()
    
    def sample(key):
        T, *FB = prior.sample(seed=key)
        F, B = np.split(np.array(FB), 2)
        f = constants.spectrum_frequencies(constants.TIMIT_FS_HZ)[::2]
        sample = dict(
            T = float(T),
            F = F,
            B = B,
            f = f
        )
        return sample
    
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    samples = [sample(key) for key in keys]
    return samples

def fit_TFB_sample(
    sample,
    vtfilter,
    cacheid,
    return_full=False,
    xmin=constants.MIN_X_HZ,
    xmax=constants.MAX_X_HZ,
    ymin=constants.MIN_Y_HZ,
    ymax=constants.MAX_Y_HZ,
    sigma_F=constants.SIGMA_FB_REFERENCE_HZ,
    sigma_B=constants.SIGMA_FB_REFERENCE_HZ,
    tilt_target=constants.FILTER_SPECTRAL_TILT_DB,
    sigma_tilt=constants.SIGMA_TILT_DB,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS,
    nloglsamples=NUM_LOGL_SAMPLES
):
    K = vtfilter.K
    ndim = 2*K
    dimlabels = [f"${c}_{{{i+1}}}$" for c in ('x', 'y') for i in range(K)]

    xnullbar = pareto.assign_xnullbar(K, xmin, xmax)
    band_bounds = (
        np.array([ymin]*K), np.array([ymax]*K)
    )
    
    f = sample['f']
    F_true = sample['F']
    B_true = sample['B']
    
    def squared_error(F, B, tilt):
        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2
        return F_err + B_err + tilt_err

    def unpack(params):
        x, y = np.split(params, 2)
        return x, y

    def loglike(power):
        # Calculate pole-zero transfer function
        if np.isnan(power[0]):
            return -np.inf # Semi-definite overlap matrix S in p(a,b|x,y)

        # Heuristically measure formants
        try:
            F, B = spectrum.get_formants_from_spectrum(f, power)
        except np.linalg.LinAlgError:
            return -np.inf # If fitting a linear curve fails (very rare)

        # Impose strictly three formants in the spectrum
        if len(F) != 3:
            return -np.inf
        
        # Heuristically measure spectral tilt starting from F3(true)
        tilt = spectrum.fit_tilt(f, power, cutoff=F_true[-1])
        
        if np.isnan(tilt): # NaN occurs if badly conditioned (very rare)
            return -np.inf
        
        # Calculate log likelihood as square loss
        F_err = np.sum(((F - F_true)/sigma_F)**2)
        B_err = np.sum(((B - B_true)/sigma_B)**2)
        tilt_err = ((tilt - tilt_target)/sigma_tilt)**2
        logl = -(F_err + B_err + tilt_err)/2
        return logl
    
    def loglike_montecarlo(params, ws):
        x, y = unpack(params)
        if np.any(x >= xmax):
            return -np.inf
        
        # Calculate cheap MC estimate of the integral over w ~ N(0,I) amplitudes
        powers = vtfilter.transfer_function_power_dB(f, x, y, ws)
        logls = [loglike(power) for power in powers]
        logl = scipy.special.logsumexp(logls) - np.log(len(ws))
        return logl

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
        ws = vtfilter.randws(nloglsamples, rng=rng)
        sampler = dynesty.NestedSampler(
            loglike_montecarlo, ptform, ndim=ndim,
            rstate=rng, logl_args=[ws], **samplerargs
        )
        sampler.run_nested(**runargs)
        return sampler.results, ws
    
    results, ws = run_nested(cacheid, samplerargs, runargs)
    return (results, locals()) if return_full else results

def yield_fitted_TFB_samples(
    vtfilter,
    seed,
    Ks,
    fit_kwargs={},
    parallel=(1,0)
):
    TFB_samples = get_TFB_samples()
    rng = np.random.default_rng(seed)
    
    n = 0
    for K in Ks:
        for i, sample in enumerate(TFB_samples):
            cacheid = rng.integers(int(1e8))
            
            n += 1
            if (n % parallel[0]) != parallel[1]:
                continue
            
            results = fit_TFB_sample(sample, vtfilter(K), cacheid, **fit_kwargs)

            yield dict(
                K=K,
                i=i,
                sample=sample,
                cacheid=cacheid,
                results=results
            )

def _flatten(l):
    # https://stackoverflow.com/a/952952/6783015
    return [item for sublist in l for item in sublist]
    
def get_fitted_TFB_samples(n_jobs=1, **kwargs):
    def job(div):
        return list(yield_fitted_TFB_samples(parallel=(n_jobs, div), **kwargs))
    
    if n_jobs == 1:
        return job(0)
    else:
        return _flatten(
            joblib.Parallel(n_jobs=n_jobs, batch_size=1, verbose=100)(
                joblib.delayed(job)(div) for div in range(n_jobs)
            )
        )