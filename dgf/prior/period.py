from init import __datadir__, __memory__
from lib import praat
from lib import aplawd
from dgf import bijectors
from dgf import isokernels
from dgf import core
from dgf import constants

import tensorflow_probability.substrates.jax.distributions as tfd

import numpy as np
import dynesty
import parselmouth
import warnings
import itertools
import scipy.stats
import time

MIN_NUM_PERIODS = 3
MAP_KERNEL = 'Matern32Kernel' # Kernel with highest evidence for APLAWD
HILBERT_EXPANSION_ORDER = 32

SAMPLERARGS = {'nlive': 100, 'bound': 'multi', 'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False}

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
    min_period_length_msec,
    max_period_length_msec,
    min_num_periods, # Reject a voiced group if it has less than `min_num_periods` pitch periods
    rng=None
):
    """Yield all training pairs consisting of the true and Praat-estimated pitch periods in msec"""
    keys = recordings.keys()
    if rng is not None:
        rng.shuffle(keys)
    
    for key in keys:
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
def get_aplawd_training_pairs():
    """
    Get the training pairs from the APLAWD database; both constrained and transformed to the
    unconstrained z-domain. A training pairs consists of (1) the ground truth pitch periods
    derived from the manually verified GCI markings and (2) the pitch periods as estimated
    from Praat's pulses. The latter (2) is aligned as closely as possible to (1).
    """
    # Get the recordings and the GCI markings
    recordings = aplawd.APLAWD(__datadir__('APLAWDW/dataset'))
    markings = aplawd.APLAWD_Markings(__datadir__('APLAWDW/markings/aplawd_gci'))
    
    # Get pairs of 'true' pitch periods and the ones estimated by Praat based on the recordings
    training_pairs = list(yield_training_pairs(
        recordings,
        markings,
        constants.MIN_PERIOD_LENGTH_MSEC,
        constants.MAX_PERIOD_LENGTH_MSEC,
        MIN_NUM_PERIODS
    ))

    # Transform the pitch periods to the z-domain
    def inverse_period_bijector(x):
        return np.array(bijectors.period_bijector().inverse(x))

    training_pairs_z = [
        (inverse_period_bijector(true_periods), inverse_period_bijector(praat_periods))
        for true_periods, praat_periods in training_pairs
    ]

    return training_pairs, training_pairs_z
    
@__memory__.cache
def model_true_pitch_periods(
    kernel_name,
    kernel_M=HILBERT_EXPANSION_ORDER,
    seed=7498,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS
):
    """
    Model the 'true' APLAWD pitch periods transformd to the z domain by a standard Hilbert GP
    parametrized by a `kernel_name` kernel with `kernel_M` basis functions, a constant mean,
    variance, lengthscale and noise (nugget) variance.
    """
    rng = np.random.default_rng(seed)
    
    _, training_pairs_z = get_aplawd_training_pairs()
    kernel = isokernels.resolve(kernel_name)

    def loglike(x):
        return np.sum([loglike_true_model(x, pair) for pair in training_pairs_z])

    def loglike_true_model(x, pair):
        mean, sigma, scale, noise_sigma = x
        var = sigma**2
        noise_power = noise_sigma**2

        true_z, _ = pair
        z = true_z - mean

        if len(z) > kernel_M:
            L = core.loglikelihood_hilbert_grid(
                kernel, var, scale, kernel_M, z, noise_power
            )
        else:
            t = np.arange(len(z))
            R = core.kernelmatrix_root_hilbert(kernel, var, scale, t, kernel_M, t[-1])
            L = core.loglikelihood_hilbert(R, z, noise_power)

        # Can return nan if lengthscale is too large
        return -np.inf if np.isnan(L) else L

    def ptform(u):
        # All parameters have LogNormal(0, 1) priors, i.e., they are all positive and O(1)
        z = scipy.stats.norm.ppf(u)
        return np.exp(z)

    sampler = dynesty.NestedSampler(
        loglike, ptform, ndim=4, rstate=rng, **samplerargs
    )
    sampler.run_nested(**runargs)
    return sampler.results

@__memory__.cache
def model_praat_pitch_periods(
    seed=3176,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS
):
    """
    Model the pitch periods as estimated by Praat as the 'true' APLAWD pitch periods
    transformed to the z domain plus a constant error term.
    """
    rng = np.random.default_rng(seed)
    
    _, training_pairs_z = get_aplawd_training_pairs()
    
    def loglike(x):
        return np.sum([loglike_praat_model(x, pair) for pair in training_pairs_z])

    def loglike_praat_model(praat_sigma, pair):
        # Praat observation model is `L = N(praat_z | true_z, praat_sigma²*I)`
        true_z, praat_z = pair

        d = praat_z - true_z
        bilinear_term = np.dot(d, d)/(praat_sigma**2)

        N = len(praat_z)
        order_term = N*np.log(2*np.pi*praat_sigma**2)

        L = -order_term/2 - bilinear_term/2
        return L

    def ptform(u):
        # The prior for `praat_sigma` is a LogNormal(0, 1) priors
        z = scipy.stats.norm.ppf(u)
        praat_sigma = np.exp(z)
        return praat_sigma
    
    sampler = dynesty.NestedSampler(
        loglike, ptform, ndim=1, rstate=rng, **samplerargs
    )
    sampler.run_nested(**runargs)
    return sampler.results

def posterior_mean_point_estimate(results):
    weights = np.exp(results.logwt - results.logz[-1])
    mu, cov = dynesty.utils.mean_and_cov(results.samples, weights)
    del cov
    return np.squeeze(mu)

def fit_aplawd_z():
    """Return the fit with highest evidence of APLAWD's period data transfored to the z-domain"""
    # Get posterior mean estimates of the 'true' pitch periods model with highest evidence
    true_results = model_true_pitch_periods(MAP_KERNEL)
    mean, sigma, scale, noise_sigma = posterior_mean_point_estimate(true_results)
    
    # Get posterior mean estimate of Praat observation error
    praat_results = model_praat_pitch_periods()
    praat_sigma = posterior_mean_point_estimate(praat_results)

    return {
        'mean': mean,
        'sigma': sigma,
        'scale': scale,
        'noise_sigma': noise_sigma,
        'praat_sigma': praat_sigma
    }

def trajectory_prior(num_pitch_periods=None, praat_estimate=None):
    """A GP prior for pitch period trajectories based on the APLAWD database"""
    fit_z = fit_aplawd_z()
    bijector = bijectors.period_bijector()
    
    if praat_estimate is not None:
        if num_pitch_periods is not None:
            assert num_pitch_periods == len(praat_estimate)
        else:
            num_pitch_periods = len(praat_estimate)
        index_points = np.arange(num_pitch_periods).astype(float)[:,None]
        observation_index_points = index_points
        observations = bijector.inverse(praat_estimate)
        observation_noise_variance = fit_z['noise_sigma']**2 + fit_z['praat_sigma']**2
        predictive_noise_variance = 0.
    else:
        if num_pitch_periods is None:
            num_pitch_periods = 1
        index_points = np.arange(num_pitch_periods).astype(float)[:,None]
        observation_index_points = None
        observations = None
        observation_noise_variance = fit_z['noise_sigma']**2
        predictive_noise_variance = 0.

    map_kernel = isokernels.resolve(MAP_KERNEL)(
        fit_z['sigma']**2, fit_z['scale']
    )
    mean_fn = lambda _: np.array([fit_z['mean']])

    gp = tfd.GaussianProcessRegressionModel(
        map_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        predictive_noise_variance=predictive_noise_variance,
        mean_fn=mean_fn
    )

    prior = tfd.TransformedDistribution(
        distribution=gp,
        bijector=bijector,
        name='PeriodTrajectoryPrior'
    )
    
    return prior

def marginal_prior():
    return trajectory_prior(num_pitch_periods=1)
