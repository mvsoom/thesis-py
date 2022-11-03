from init import __datadir__, __memory__
from lib import praat
from lib import aplawd
from dgf import bijectors
from dgf import isokernels
from dgf import core
from lib import constants

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import jax
import jax.numpy as jnp
import numpy as np

import parselmouth
import itertools
import scipy.stats
import time
import random
import warnings

def load_recording_and_markers(recordings, markings, key):
    k = recordings.load_shifted(key)
    try:
        m = markings.load(key)
    except FileNotFoundError:
        m = None
    return k, m

def split_markings_into_voiced_groups(m, fs, max_period_length_msec, min_num_periods):
    """Split the markings into groups where the waveform is voiced"""
    periods = np.diff(m) / fs * 1000 # msec
    split_at = np.where(periods > max_period_length_msec)[0] + 1
    for group in np.split(m, split_at):
        if len(group) <= min_num_periods + 1:
            continue
        else:
            yield group

def align_and_intersect(a, b):
    """Align two arrays containing sample indices as closely as possible"""
    a, b = a.copy(), b.copy()
    dist = np.abs(a[:,None] - b[None,:])
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    d = j - i
    if d >= 0:
        intersect = min(len(a), len(b) - d)
        a = a[0 : intersect]
        b = b[d : d + intersect]
    elif d < 0:
        d = np.abs(d)
        intersect = min(len(a) - d, len(b))
        a = a[d : d + intersect]
        b = b[0 : intersect]
    return a, b

def yield_training_pairs(
    recordings,
    markings,
    min_period_length_msec=constants.MIN_PERIOD_LENGTH_MSEC,
    max_period_length_msec=constants.MAX_PERIOD_LENGTH_MSEC,
    min_num_periods=constants.MIN_NUM_PERIODS
):
    """Yield all training pairs consisting of the true and Praat-estimated pitch periods in msec"""
    for key in recordings.keys():
        k, m = load_recording_and_markers(recordings, markings, key)
        if m is None:
            # Only a few dozen
            warnings.warn(
                f'{k.name}: Discarded entire recording because of ground truth markings are missing'
            )
            continue
        
        try:
            praat_pulses = praat.get_pulses(k.s, k.fs)
        except parselmouth.PraatError:
            # Occurs when the recording is too short
            warnings.warn(
                f'{k.name}: Discarded entire recording because of PraatError in `get_pulses()`'
            )
            continue

        voiced_groups = split_markings_into_voiced_groups(
            m, k.fs, max_period_length_msec, min_num_periods
        )

        # We call the APLAWD markings the 'true' group markings
        for true_group in voiced_groups:
            if len(true_group) <= min_num_periods + 1:
                continue # Discard voiced groups which are a priori too short

            # Intersect the current ground truth group as well as possible with Praat estimates
            true_group, praat_group = align_and_intersect(true_group, praat_pulses)
            assert len(true_group) == len(praat_group)
            if len(true_group) <= min_num_periods + 1:
                continue

            true_periods = np.diff(true_group) / k.fs * 1000 # msec
            praat_periods = np.diff(praat_group) / k.fs * 1000 # msec
            
            if np.any(true_periods > max_period_length_msec) or \
               np.any(true_periods < min_period_length_msec):
                # This should be very rare
                warnings.warn(
                    f'{k.name}: Discarded voiced group of GCIs because one of the ground '
                     'truth markings is not within `{min|max}_period_length_msec`'
                )
                continue

            if np.any(praat_periods > max_period_length_msec) or \
               np.any(praat_periods < min_period_length_msec):
                # Discard this and continue; we assume user will never accept
                # such Praat estimates so we don't want to model this case.
                warnings.warn(
                    f'{k.name}: Discarded voiced group of GCIs because one of the synced '
                     'Praat periods is not within `{min|max}_period_length_msec`'
                )
                continue
            
            yield true_periods, praat_periods

@__memory__.cache
def get_aplawd_training_pairs(cacheid=18796):
    """
    Get the training pairs from the APLAWD database.
    
    A training pair consists of (1) the ground truth pitch periods derived from the
    manually verified GCI markings and (2) the pitch periods as estimated
    from Praat's pulses. The latter (2) is aligned as closely as possible to (1).
    """
    # Get the recordings and the GCI markings
    recordings = aplawd.APLAWD(__datadir__('APLAWDW/dataset'))
    markings = aplawd.APLAWD_Markings(__datadir__('APLAWDW/markings/aplawd_gci'))
    
    # Get pairs of 'true' pitch periods and the ones estimated by Praat based on the recordings
    training_pairs = list(yield_training_pairs(
        recordings,
        markings,
        constants.MIN_PERIOD_LENGTH_MSEC + constants._ZERO,
        constants.MAX_PERIOD_LENGTH_MSEC - constants._ZERO,
        constants.MIN_NUM_PERIODS
    ))

    return training_pairs

def get_aplawd_training_pairs_subset(
    subset_size=5000,
    max_num_periods=100,
    seed=411489
):
    """Select a subset of the training pairs with a max number of pitch periods"""
    training_pairs = get_aplawd_training_pairs()

    random.seed(seed)
    subset = random.choices(
        list(
            filter(lambda s: len(s[0]) <= max_num_periods, training_pairs)
        ), k=subset_size
    )

    return subset

# Cannot be cached due to complex return value
def _fit_period_trajectory_kernel():
    """Fit Matern kernels to the APLAWD database and return the MAP one"""
    subset = get_aplawd_training_pairs_subset()
    samples = [d[0][:,None] for d in subset]

    bounds = jnp.array([
        constants.MIN_PERIOD_LENGTH_MSEC,
        constants.MAX_PERIOD_LENGTH_MSEC
    ])[None,:]
    
    def fit(kernel_name, cacheid):
        bijector, results = bijectors.fit_nonlinear_coloring_trajectory_bijector(
            samples, bounds, kernel_name, cacheid, return_fit_results=True
        )
        return kernel_name, bijector, results
    
    kernel_fits = [
        fit("Matern12Kernel", 19863),
        fit("Matern32Kernel", 11279), # Spoiler: this has highest evidence
        fit("Matern52Kernel", 54697),
        fit("SqExponentialKernel", 79543)
    ]
    
    def logz(fit_result):
        kernel_name, bijector, results = fit_result
        return results.logz[-1]
    
    best_fit = max(*kernel_fits, key=lambda fit_result: logz(fit_result))
    return best_fit # == (kernel_name, bijector, results)

def fit_period_trajectory_kernel(
    _best_fit = _fit_period_trajectory_kernel()
):
    """Cache `_fit_period_trajectory_kernel()`"""
    return _best_fit

def fit_period_trajectory_bijector(
    num_pitch_periods=None
):
    kernel_name, bijector, results = fit_period_trajectory_kernel()
    del kernel_name, results
    
    if num_pitch_periods is not None:
        bijector = bijector(num_pitch_periods)

    return bijector

def _fit_praat_estimation_cov():
    subset = get_aplawd_training_pairs_subset()
    true_samples  = [d[0][:,None] for d in subset]
    praat_samples = [d[1][:,None] for d in subset]
    
    b = fit_period_trajectory_bijector(1)
    cov = bijectors.estimate_observation_noise_cov(b, true_samples, praat_samples)
    return cov

def _fit_praat_estimation_mean():
    subset = get_aplawd_training_pairs_subset()
    true_samples  = [d[0][:,None] for d in subset]
    praat_samples = [d[1][:,None] for d in subset]
    
    b = fit_period_trajectory_bijector(1)
    mean = bijectors.estimate_observation_noise_mean(b, true_samples, praat_samples)
    return mean

def fit_praat_estimation_cov(
    _cov = _fit_praat_estimation_cov()
):
    """Cache `_fit_praat_estimation_cov()`"""
    return _cov

def fit_praat_estimation_sigma():
    """Maximum likelihood fit of Praat's observation error's sigma"""
    return jnp.sqrt(fit_praat_estimation_cov()).squeeze()

def maximum_likelihood_envelope_params(results):
    s, envelope_lengthscale, envelope_noise_sigma = results.samples[-1]
    del s
    return envelope_lengthscale, envelope_noise_sigma

def period_trajectory_prior(
    num_pitch_periods,
    praat_estimate=None
):
    """
    A GP prior based on the APLAWD database for pitch period trajectories
    of length `num_pitch_periods`, possibly conditioned on `praat_estimate`,
    which must be an array shaped `(num_pitch_periods,)`. The prior returns
    samples shaped `(num_pitch_periods,)`.
    """
    bijector = fit_period_trajectory_bijector(num_pitch_periods)
    
    if praat_estimate is None:
        name = 'PeriodTrajectoryPrior'
    else:
        name = 'ConditionedPeriodTrajectoryPrior'
        bijector = bijectors.condition_nonlinear_coloring_trajectory_bijector(
            bijector, praat_estimate[:,None], fit_praat_estimation_cov()
        )
    
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(num_pitch_periods))
    
    # Squeeze out the last dimension in `(m,n)` since `n == 1`
    squeeze_bijector = tfb.Chain([
        tfb.Reshape(event_shape_out=(-1,), event_shape_in=(-1, 1)),
        bijector
    ])
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=squeeze_bijector,
        name=name
    )
    return prior # prior.sample(ns) shaped (ns, num_pitch_periods) shaped

def period_marginal_prior():
    squeeze_bijector = tfb.Reshape(event_shape_out=(), event_shape_in=(1,))
    prior = tfd.TransformedDistribution(
        distribution=period_trajectory_prior(1),
        bijector=squeeze_bijector,
        name="PeriodMarginalPrior"
    )
    return prior # prior.sample(ns) shaped (ns,)