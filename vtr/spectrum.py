"""Empirical and heuristic spectrum measures"""
from lib import constants

import numpy as np
import scipy.signal

BUTTERWORTH_FILTER_ORDER = 6
PEAK_DISTANCE = 100. # Hz

# Default minimum peak prominence
H_DEFAULT_dB = 3.

# Hyperparameters for the cutoff frequency and peak prominence
# (Currently not used)
SPECTRUM_RIPPLE_PERIOD = 3.6 # Hz
RHO_BETA = 1.
RHO_ALPHA = 2.
H_SCALE_dB = 1.

# Hyperparameters for spectral tilt estimation
GLOTTAL_FORMANT_HZ = 200. # Fulop & Disner (2011)

def get_bandwidths_at_FWHM(envelope, peaks):
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(
        envelope, peaks
    )
    
    # To get the FWHM, we can use Scipy's peak_widths() if we trick it into
    # believing the peaks have prominence of 3 dB, and then measure peak width at
    # 100% of this prominence.
    prominences_hack = np.repeat(3., len(peaks))
    prominence_data = (prominences_hack, left_bases, right_bases)

    widths, width_heights, left_lps, right_lps = scipy.signal.peak_widths(
        envelope, peaks, rel_height=1., prominence_data=prominence_data
    )
    
    return widths

def smooth_spectrum(
    spectrum,
    rho,
    butterworth_filter_order=BUTTERWORTH_FILTER_ORDER
):
    """rho in [0, 1] is cutoff frequency as fraction of fs/2"""
    sos = scipy.signal.butter(
        butterworth_filter_order, rho, 'lowpass', analog=False, output='sos'
    )
    smoothed = scipy.signal.sosfiltfilt(sos, spectrum)
    return smoothed

def get_formants_from_spectrum(
    f,
    power,
    rho=None,
    h=H_DEFAULT_dB,
    return_full=False,
    butterworth_filter_order=BUTTERWORTH_FILTER_ORDER,
    peak_distance=PEAK_DISTANCE
):
    """Estimate formants by peak picking off a low-pass smoothed power spectrum"""
    # Smooth spectrum based on rho
    if rho is not None:
        smoothed = smooth_spectrum(power, rho, butterworth_filter_order)
    else:
        smoothed = power
    
    # Find peaks in the smoothed spectrum based on h
    df = f[1] - f[0]
    peaks, _ = scipy.signal.find_peaks(
        smoothed, distance=max(peak_distance // df, 1), prominence=h
    )

    # Get formant center frequencies and bandwidths at FWHM
    F = f[peaks]
    B = get_bandwidths_at_FWHM(smoothed, peaks)

    if return_full:
        return F, B, locals()
    else:
        return F, B

def number_of_peaks(f, power, **kwargs):
    F, B = get_formants_from_spectrum(f, power, **kwargs)
    K = len(F)
    return K

def fit_tilt(
    f, power_spectrum, cutoff=GLOTTAL_FORMANT_HZ, return_interp=False
):
    """Estimate tilt in dB/oct by linear regression in log domain -- see tilt.ipynb
    
    May return NaN in exceptional cases if very badly conditioned (such
    as an upwardly curving power spectrum). This is very rare.
    """
    accept = f >= cutoff
    
    f2 = np.log2(f[accept])
    a, b = np.polyfit(f2, power_spectrum[accept], 1)
    
    if return_interp:
        interp_log = np.poly1d([a, b])
        def interp(f): return interp_log(np.log2(f))
        return a, interp
    else:
        return a
    
def get_power_spectrum(d, fs):
    dt = 1/fs
    D = np.fft.rfft(d)*dt
    spectrum = 20*np.log10(np.abs(D))
    f = np.fft.rfftfreq(len(d), dt)
    return f, spectrum