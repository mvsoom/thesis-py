from init import __memory__
from lib import timit
from lib import util
from lib import praat
from dgf import bijectors
from dgf import constants

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

def yield_training_pairs(
    fb_file, phn_file, wav_file,
    return_full=False,
    min_period_length_msec=constants.MIN_PERIOD_LENGTH_MSEC,
    max_period_length_msec=constants.MAX_PERIOD_LENGTH_MSEC,
    min_num_periods=constants.MIN_NUM_PERIODS
):
    """
    Yield all training pairs consisting of the true and Praat-estimated
    formant frequencies F1-F3 averaged over (i.e., within) the pitch periods
    estimated by Praat
    """
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
        
        T = np.diff(pulse_idx)/fs*1000. # msec
        if np.any(T > max_period_length_msec) or \
           np.any(T < min_period_length_msec):
            # Discard this and continue; we assume user will never accept
            # such Praat estimates so we don't want to model this case.
            warn(f'Estimated Praat periods not within `{{min|max}}_period_length_msec`: {T}')
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
        
        # Finally, get rid of very short samples
        num_pitch_periods = F_praat_periods.shape[0]
        assert F_true_periods.shape[0] == num_pitch_periods
        if num_pitch_periods < min_num_periods:
            warn(f'Praat gave only {num_pitch_periods} < {min_num_periods} valid formants')
            continue
        
        if return_full:
            yield F_true_periods, F_praat_periods, locals()
        else:
            yield F_true_periods, F_praat_periods

@__memory__.cache
def get_vtrformants_training_pairs(cacheid=442369):
    """
    Get a list of training pairs from the VTRFormants database (TRAINING set).
    
    A training pair consists of two identically matrices shaped
    `(num_pitch_periods, 3)` of formant F1-F3 values in Hz. The
    first matrix in the pair is the ground truth and second one is Praat's estimate.
    
    Note that these estimates use Praat's estimates of the pitch periods
    which also introduces extra noise.
    """
    vtr_root = timit.training_set(timit.VTRFORMANTS)
    timit_root = timit.training_set(timit.TIMIT)
    
    all_training_pairs = []
    for triple in timit.yield_file_triples(vtr_root, timit_root):
        fb_file, phn_file, wav_file = triple
        
        training_pairs = list(
            yield_training_pairs(
                fb_file, phn_file, wav_file,
                min_period_length_msec=constants.MIN_PERIOD_LENGTH_MSEC,
                max_period_length_msec=constants.MAX_PERIOD_LENGTH_MSEC,
                min_num_periods=constants.MIN_NUM_PERIODS
            )
        )
        
        all_training_pairs.extend(training_pairs)

    return all_training_pairs

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