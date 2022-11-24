"""Fit the p(T, F^R, B^R) prior for T and reference formant frequency and bandwidh"""
from init import __memory__
from vtr import spectrum
from vtr.prior import formant
from vtr.prior import allpole
from lib import constants
from dgf import bijectors

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import numpy as np
import scipy.stats
import warnings
import joblib

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

@__memory__.cache
def get_TFB_samples(
    num_samples=50,
    seed=312178
):
    """Get samples to fit p(x,y) priors to"""
    prior = TFB_prior()
    f = constants.spectrum_frequencies(constants.TIMIT_FS_HZ)
    
    def sample_reject(key):
        while True:
            T, *FB = prior.sample(seed=key)
            F, B = np.split(np.array(FB), 2)
            power = allpole.transfer_function_power_dB(f, F, B)
            
            if spectrum.number_of_peaks(f, power) == 3:
                break
            else:
                warnings.warn("Rejected TFB sample that had merged peaks in its power spectrum")
                _, key = jax.random.split(key)
        
        sample = dict(
            T = float(T),
            F = F,
            B = B,
            f = f,
            power = power
        )
        return sample
    
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    samples = [sample_reject(key) for key in keys]
    return samples

def yield_fitted_TFB_samples(
    fit_func,
    seed,
    Ks,
    fit_func_kwargs={},
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
            
            results = fit_func(sample, K, cacheid, **fit_func_kwargs)

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