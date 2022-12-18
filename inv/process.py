from init import __memory__
from vtr.prior import formant
from dgf.prior import source
from vtr.prior import filter
from lib import util
from lib import constants
from lib import praat

import numpy as np

import scipy.signal
import parselmouth
import warnings

def resample_data(data, fs, fs_new):
    N = len(data)
    N_new = int(N*fs_new/fs)
    new_data = scipy.signal.resample(data, N_new)
    return new_data

def standardize_data(d, fs, standard_fs=constants.FS_HZ):
    d = resample_data(d, fs, standard_fs)
    d = util.normalize_power(d)
    return d, standard_fs

def get_pulse_estimate(d, fs):
    try:
        return praat.get_pulses(d, fs)
    except parselmouth.PraatError as e:
        raise ValueError("Data too short for Praat pulse estimation") from e

def _get_middle_n_elements(a, n):
    start = (len(a) // 2) - (n // 2)
    end = (len(a) // 2) + (n // 2) + (n % 2)
    return a[start:end]

def _num_pitch_periods(pulse_estimate):
    return len(pulse_estimate) - 1

def process_data(
    fulldata,
    fs,
    pulse_estimate=None,
    F_estimate=None,
    anchor=None,
    prepend=constants.PREPEND_PITCH_PERIODS,
    max_NP=np.inf,
    return_full=False
):
    fulldata, fs = standardize_data(fulldata, fs)
    
    ##########
    # Pulses #
    ##########
    # Get estimate from Praat if not supplied
    if pulse_estimate is None:
        pulse_estimate = get_pulse_estimate(fulldata, fs)
    
    # Possibly limit the number of pitch periods
    NP = prepend + _num_pitch_periods(pulse_estimate)
    if NP > max_NP:
        assert max_NP > prepend
        pulse_estimate = _get_middle_n_elements(pulse_estimate, max_NP-prepend+1)
        NP = max_NP
    
    # Discard all data outside first and last pulse
    first, last = pulse_estimate[0], pulse_estimate[-1]
    d = fulldata[first:last]
    d = util.normalize_power(d)
    
    # Get the period estimate per pitch period
    def to_msec(idx):
        return idx*(1000./fs)
    
    Ts = to_msec(np.diff(pulse_estimate))
    T_estimate = np.concatenate(([np.nan]*prepend, Ts))
    assert T_estimate.shape == (NP,)
    
    # Define the time and the time origin
    fullt = to_msec(np.arange(len(fulldata)))
    t = fullt[first:last]
    
    # Define the anchor, i.e., "time origin"
    if anchor is None:
        anchor = 0
    anchort = to_msec(pulse_estimate[anchor])
    
    ############################
    # Reference formant tracks #
    ############################
    if F_estimate is None:
        reference_tracks = praat.get_formant_tracks(
            fulldata, fs, num_tracks=3
        )
        F_estimate = formant.average_F_over_periods(
            pulse_estimate, reference_tracks
        )
        F_estimate = np.vstack([
            np.full((prepend, 3), np.nan),
            F_estimate,
        ])
    
    data = dict(
        fulldata=fulldata,
        fullt=fullt,
        fs=fs,
        d=d,
        t=t,
        prepend=prepend,
        anchor=anchor,
        anchort=anchort,
        NP=NP,
        T_estimate=T_estimate,
        F_estimate=F_estimate
    )
    
    if return_full:
        return data, locals()
    else:
        return data