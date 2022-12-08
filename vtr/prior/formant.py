"""Prior for the reference formant values F1, F2, F3"""
from init import __memory__, __cache__
from lib import timit
from lib import util
from lib import praat
from dgf import bijectors
from lib import constants

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import jax
import jax.numpy as jnp
import numpy as np

import datatable
import functools
import operator
import warnings
import parselmouth

def select_F_true(fb_file, fs):
    """Return manually verified formant tracks F1-3 in Hz at 10 msec frames"""
    FB = timit.read_fb_file(fb_file)
    F_true = FB[:,:3]*1000. # Hz
    frame_midpoint_idx = ((np.arange(len(F_true)) + 0.5) * timit.FRAME_LENGTH_SEC)*fs
    return frame_midpoint_idx, F_true

def get_F_true_tracks(frame_midpoint_idx, F_true, indices):
    def interpolate_column(j):
        return np.interp(indices, frame_midpoint_idx, F_true[:,j])
    
    return np.column_stack(
        [interpolate_column(j) for j in range(F_true.shape[1])]
    )

def select_vowel_segments(phn_file):
    df = timit.read_phn_file(phn_file)
    
    # https://stackoverflow.com/a/67784024/6783015
    filter = functools.reduce(
        operator.or_,
        (datatable.f.PHONETIC_LABEL == v for v in timit.VOWEL_LABELS)
    )
    return df[filter,:]

def read_wav_file_and_normalize(wav_file):
    d, fs = timit.read_wav_file(wav_file)
    return util.normalize_power(d), fs

def _robustnanmean(a, axis):
    """Identical to `np.nanmean` but issues no warning when input is all NaNs"""
    if not np.all(np.isnan(a)):
        return np.nanmean(a, axis=axis)
    else:
        shape = list(a.shape)
        del shape[axis]
        return np.full(shape, np.nan)

def average_F_over_periods(pulse_idx, F):
    pairwise = zip(pulse_idx, pulse_idx[1:])
    return np.vstack([
        _robustnanmean(F[start:end], axis=0) for start, end in pairwise
    ])

def yield_vowel_segments(d, phn):
    for start, end, label in phn.to_tuples():
        yield label, d[start:end]

def yield_training_data(
    fb_file, phn_file, wav_file, return_full=False
):
    """
    Yield all training tuples consisting of Praat's estimated pitch period (msec),
    and the true and Praat-estimated formant frequencies F1-F3 averaged over
    (i.e., within) the pitch periods estimated by Praat, in Hz
    """
    min_period_length_msec = constants.MIN_PERIOD_LENGTH_MSEC + constants._ZERO
    max_period_length_msec = constants.MAX_PERIOD_LENGTH_MSEC - constants._ZERO
    min_num_periods = constants.MIN_NUM_PERIODS
    
    d, fs = read_wav_file_and_normalize(wav_file)
    phn = select_vowel_segments(phn_file)
    frame_midpoint_idx, F_true_frames = select_F_true(fb_file, fs)
    
    for start, end, vowel in phn.to_tuples():
        segment = d[start:end]
        
        def warn(s):
            warnings.warn(
                f'{wav_file}: Discarding vowel `{vowel}` segment at sample indices {start}:{end}: {s}'
            )
        
        #####################################
        # Estimate the pitch period indices #
        #####################################
        try:
            pulse_idx = praat.get_pulses(segment, fs)
        except parselmouth.PraatError as e:
            warn(
                f"segment too short for Praat's pulse estimation algorithm: {str(e)}"
            )
            continue
        
        num_pulses = len(pulse_idx)
        if num_pulses < min_num_periods + 1:
            warn(f'Praat only gave {num_pulses} < {min_num_periods + 1} pulses')
            continue
        
        T_praat = np.diff(pulse_idx)/fs*1000. # msec
        if np.any(T_praat > max_period_length_msec) or \
           np.any(T_praat < min_period_length_msec):
            # Discard this and continue; we assume user will never accept
            # such Praat estimates so we don't want to model this case.
            warn(f'Estimated Praat periods not within `{{min|max}}_period_length_msec`: {T_praat}')
            continue

        ##############################################
        # Estimate F1-F3 formant tracks in `segment` #
        ##############################################
        F_praat_tracks = praat.get_formant_tracks(segment, fs, num_tracks=3)
        
        ##############################################
        # Get true F1-F3 formant tracks at `segment` #
        ##############################################
        F_true_tracks = get_F_true_tracks(
            frame_midpoint_idx, F_true_frames, np.arange(start, end)
        )
        
        ############################################
        # Average over the estimated pitch periods #
        ############################################
        F_praat_periods = average_F_over_periods(pulse_idx, F_praat_tracks)
        F_true_periods = average_F_over_periods(pulse_idx, F_true_tracks)
        
        #########################################################
        # Intersect the true and estimated formant trajectories #
        #########################################################
        
        # Discard pitch periods where Praat gave empty formant tracks
        # (NaN values). This happens at the boundaries of the current
        # segment, because the pulse values are also typically missing there.
        empty = np.any(np.isnan(F_praat_periods), axis=1)
        if np.sum(np.diff(empty)) > 2:
            # NaN values have occured away from boundary
            warn('Praat gave empty formant tracks away from segment boundaries')
            continue
        
        F_praat_periods = F_praat_periods[~empty, :]
        F_true_periods = F_true_periods[~empty, :]
        assert not np.any(np.isnan(F_praat_periods))
        assert not np.any(np.isnan(F_true_periods))
        
        T_praat = T_praat[~empty]
        
        # Finally, get rid of very short samples
        num_pitch_periods = F_praat_periods.shape[0]
        assert F_true_periods.shape[0] == num_pitch_periods
        if num_pitch_periods < min_num_periods:
            warn(f'Praat gave only {num_pitch_periods} < {min_num_periods} valid formants')
            continue
        
        if return_full:
            yield T_praat, F_true_periods, F_praat_periods, locals()
        else:
            yield T_praat, F_true_periods, F_praat_periods

@__memory__.cache
def get_vtrformants_training_data(cacheid=452369):
    """
    Get a list of training data from the VTRFormants database (TRAINING set).
    
    One training tuple consists of an array of Praat's pitch period estimate
    during the vowel segment shaped `(num_pitch_periods,)`  and a pair of two
    identically shaped `(num_pitch_periods, 3)` matrices of formant F1-F3 values
    in Hz. The first matrix in the pair is the ground truth and second one is
    Praat's estimate.
    
    Note that these estimates use Praat's estimates of the pitch periods
    which also introduces extra noise.
    """
    vtr_root = timit.training_set(timit.VTRFORMANTS)
    timit_root = timit.training_set(timit.TIMIT)
    
    all_training_data = []
    for triple in timit.yield_file_triples(vtr_root, timit_root):
        fb_file, phn_file, wav_file = triple
        
        training_data = list(
            yield_training_data(
                fb_file, phn_file, wav_file
            )
        )
        
        all_training_data.extend(training_data)

    # Organize into a dict
    praat_T, true_F_trajectories, praat_F_trajectories\
        = list(zip(*all_training_data)) # Unzip

    return {
        'praat_T': praat_T,
        'true_F_trajectories': true_F_trajectories,
        'praat_F_trajectories': praat_F_trajectories
    }

@__cache__
def fit_formants_trajectory_kernel():
    """Fit Matern kernels to TIMIT/VTRFormants and return the MAP one"""
    true_F_trajectories = get_vtrformants_training_data()['true_F_trajectories']
    
    a, b = constants.MIN_FORMANT_FREQ_HZ, constants.MAX_FORMANT_FREQ_HZ
    bounds = jnp.array([
        (a, b),
        (a, b),
        (a, b)
    ])
    
    def fit(kernel_name, cacheid):
        bijector, results = bijectors.fit_nonlinear_coloring_trajectory_bijector(
            true_F_trajectories, bounds, kernel_name, cacheid, return_fit_results=True
        )
        return kernel_name, bijector, results
    
    kernel_fits = [
        fit("Matern12Kernel", 23654862),
        fit("Matern32Kernel", 899785662), # Spoiler: this has highest evidence
        fit("Matern52Kernel", 17893652),
        fit("SqExponentialKernel", 9856723)
    ]
    
    def logz(fit_result):
        kernel_name, bijector, results = fit_result
        return results.logz[-1]
    
    best_fit = max(*kernel_fits, key=lambda fit_result: logz(fit_result))
    return best_fit # == (kernel_name, bijector, results)

def fit_formants_trajectory_bijector(
    num_pitch_periods=None
):
    kernel_name, bijector, results = fit_formants_trajectory_kernel()
    del kernel_name, results
    
    if num_pitch_periods is not None:
        bijector = bijector(num_pitch_periods)

    return bijector

@__cache__
def fit_praat_estimation_mean_and_cov():
    training_data = get_vtrformants_training_data()
    true_F_trajectories = training_data['true_F_trajectories']
    praat_F_trajectories = training_data['praat_F_trajectories']
    
    b = fit_formants_trajectory_bijector(1)
    mean, cov = bijectors.estimate_observation_noise_cov(
        b, true_F_trajectories, praat_F_trajectories, return_mean=True
    )
    return mean, cov

def maximum_likelihood_envelope_params(results):
    *s, envelope_lengthscale, envelope_noise_sigma = results.samples[-1]
    del s
    return envelope_lengthscale, envelope_noise_sigma

def formants_trajectory_prior(
    num_pitch_periods,
    praat_estimate=None
):
    """
    A GP prior based on the APLAWD database for pitch period trajectories
    of length `num_pitch_periods`, possibly conditioned on `praat_estimate`,
    which must be an array shaped `(num_pitch_periods,)`. The prior returns
    samples shaped `(num_pitch_periods, 3)`.
    """
    bijector = fit_formants_trajectory_bijector(num_pitch_periods)
    
    if praat_estimate is None:
        name = 'ReferenceFormantsTrajectoryPrior'
    else:
        name = 'ConditionedReferenceFormantsTrajectoryPrior'
        mean, cov = fit_praat_estimation_mean_and_cov()
        bijector = bijectors.condition_nonlinear_coloring_trajectory_bijector(
            bijector, praat_estimate, cov, mean
        )
    
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(3*num_pitch_periods))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=bijector,
        name=name
    )
    return prior # prior.sample(ns) shaped (ns, num_pitch_periods, 3)

def formants_marginal_prior():
    squeeze_bijector = tfb.Reshape(event_shape_out=(-1,), event_shape_in=(1, 3))
    prior = tfd.TransformedDistribution(
        distribution=formants_trajectory_prior(1),
        bijector=squeeze_bijector,
        name="ReferenceFormantsMarginalPrior"
    )
    return prior # prior.sample(ns) shaped (ns, 3)