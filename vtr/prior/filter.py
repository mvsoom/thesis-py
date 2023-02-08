from init import __memory__, __cache__
from vtr.prior import polezero, allpole
from vtr.prior import formant
from vtr.prior import bandwidth
from dgf.prior import period
from lib import util
from dgf import bijectors
from dgf import isokernels
from lib import constants

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import sys

PZ = polezero.PoleZeroFilter
AP = allpole.AllPoleFilter

def get_module(vtfilter):
    return sys.modules[vtfilter.__module__]

def get_TFB_row(fit):
    row = np.concatenate((
        [fit['sample']['T']],
        fit['sample']['F'],
        fit['sample']['B']
    ))
    return row

@__memory__.cache
def posterior_of_fitted_TFB_values(vtfilterclass, numsamples=50):
    d = dict()
    
    def process(fit):
        row = get_TFB_row(fit)
        rows = np.repeat(row[None,:], numsamples, axis=0)
        samples = util.resample_equal(fit['results'], numsamples)
        return np.column_stack((rows, samples))
                
    for fit in get_module(vtfilterclass).get_fitted_TFB_samples():
        K = fit['K']
        if K not in d:
            d[K] = []
        d[K].append(process(fit))
        
    posterior = {K: np.vstack(v) for K, v in d.items()}
    seed = id(posterior)
    return posterior, seed

def TFBXY_BOUNDS(K):
    tfb = bandwidth.TFB_BOUNDS
    xy = np.array([
        [(constants.MIN_X_HZ, constants.MAX_X_HZ)]*K +\
        [(constants.MIN_Y_HZ, constants.MAX_Y_HZ)]*K,
    ]).squeeze()
    bounds = np.vstack((tfb, xy))
    return bounds

TFB_ndim = 7

def TFBXY_NDIM(K):
    return TFBXY_BOUNDS(K).shape[0]

def fit_TFB_filter_bijector(
    vtfilter,
    return_fit_results=False
):  
    def fit(K, samples, cacheid):
        return bijectors.fit_nonlinear_coloring_bijector(
            samples, TFBXY_BOUNDS(K), cacheid,
            return_fit_results=return_fit_results
        )
    
    posterior, seed = posterior_of_fitted_TFB_values(vtfilter.__class__)
    rng = np.random.default_rng(seed)

    for K, samples in sorted(posterior.items()):
        cacheid = rng.integers(int(1e8))
        if K == vtfilter.K:
            return fit(K, samples, cacheid)

    raise ValueError(f"No posterior samples found for {vtfilter} with K = {vtfilter.K}")

def _fit_all_TFB_filter_bijectors():
    for vtfilterclass in (allpole.AllPoleFilter, polezero.PoleZeroFilter):
        for K in vtfilterclass.K_RANGE:
            fit_TFB_filter_bijector(vtfilterclass(K))

def trajectify_bijector(bstatic, num_pitch_periods):
    """Turn a static bijector `bstatic` into a trajectory bijector with `num_pitch_periods` using the fitted filter GP based on ground truth F1-F3 trajectories in the VTRFormants database""" 
    kernel_name, _, results =\
        formant.fit_formants_trajectory_kernel()

    envelope_lengthscale, envelope_noise_sigma =\
        formant.maximum_likelihood_envelope_params(results)

    btraj = bijectors.nonlinear_coloring_trajectory_bijector(
        bstatic,
        num_pitch_periods,
        kernel_name,
        envelope_lengthscale,
        envelope_noise_sigma
    )

    return btraj

def TFB_filter_trajectory_bijector(
    num_pitch_periods,
    vtfilter,
    T_estimate=None,
    F_estimate=None,
    noiseless_estimates=False
):
    """
    Get a bijector sending N(0,I_n) to (T,F^R,B^R,x,y) samples where the
    reference formants |F^R| = |B^R| = 3 per pitch period and the VTRs
    |x| = |y| = K per pitch period. The total dimension is 
    `n = num_pitch_periods*TFBXY_NDIM(K) = num_pitch_periods*(7 + 2*K)`.
    Optionally condition on Praat's estimates of the pitch periods
    `T_estimate` shaped `(num_pitch_periods,)` and Praat's estimates of
    `F^R = (F1, F2, F3)` shaped `(num_pitch_periods,3)`. If not
    `noiseless_estimates`, then condition on the estimates without taking
    into account Praat's estimation error.
    """
    marginal_bijector = fit_TFB_filter_bijector(vtfilter)
    trajectory_bijector = trajectify_bijector(marginal_bijector, num_pitch_periods)
    
    # Condition the bijector on F and T estimates (if any)
    ndim = TFBXY_NDIM(vtfilter.K)
    observation = np.full((num_pitch_periods, ndim), np.nan)
    noise_mean = np.zeros(ndim)
    noise_cov = np.zeros((ndim, ndim))
    
    if T_estimate is not None:
        observation[:,0] = T_estimate
        if not noiseless_estimates:
            noise_mean[0], noise_cov[0, 0] =\
                period.fit_praat_estimation_mean_and_cov()
    if F_estimate is not None:
        observation[:,1:4] = F_estimate
        if not noiseless_estimates:
            noise_mean[1:4], noise_cov[1:4, 1:4] =\
                formant.fit_praat_estimation_mean_and_cov()

    trajectory_bijector = bijectors.condition_nonlinear_coloring_trajectory_bijector(
        trajectory_bijector,
        observation,
        noise_cov,
        noise_mean
    )
    
    return trajectory_bijector

def filter_trajectory_bijector(
    num_pitch_periods,
    vtfilter,
    T_estimate=None,
    F_estimate=None,
    noiseless_estimates=False
):
    b = TFB_filter_trajectory_bijector(
        num_pitch_periods,
        vtfilter,
        T_estimate,
        F_estimate
    )
    
    # Drop first (TFB) dimensions to retain only (xy)
    TFB_dims = range(bandwidth.TFB_NDIM)
    return bijectors.drop_dimensions(b, TFB_dims)

def filter_trajectory_prior(
    num_pitch_periods,
    vtfilter,
    T_estimate=None,
    F_estimate=None,
    noiseless_estimates=False
):
    """p(x, y) xor p(x, y|\hat T or \hat F^R)"""
    b = filter_trajectory_bijector(
        num_pitch_periods,
        vtfilter,
        T_estimate,
        F_estimate
    )
    
    if (T_estimate is None) and (F_estimate is None):
        name = 'FilterTrajectoryPrior'
    else:
        name = 'ConditionedFilterTrajectoryPrior'
    
    ndim = 2*vtfilter.K
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(num_pitch_periods*ndim))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=b,
        name=name
    )
    return prior # prior.sample(ns) shaped (ns, num_pitch_periods, 2K)

def filter_marginal_prior(
    vtfilter,
    T_estimate=None,
    F_estimate=None,
    noiseless_estimates=False
):
    ndim = 2*vtfilter.K
    squeeze_bijector = tfb.Reshape(event_shape_out=(ndim,), event_shape_in=(1,ndim))
    prior = tfd.TransformedDistribution(
        distribution=filter_trajectory_prior(1, vtfilter, T_estimate, F_estimate),
        bijector=squeeze_bijector,
        name="FilterMarginalPrior"
    )
    return prior # prior.sample(ns) shaped (ns, 2K)

#@__memory__.cache
def get_KL_weights():
    pass