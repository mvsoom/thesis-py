import numpy as np
import scipy.signal
from warnings import warn

BUTTERWORTH_FILTER_ORDER = 6
PEAK_DISTANCE = 100. # Hz

MEAN_RHO = 0.2
MEAN_H = 3. # dB

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

def get_formants_from_spectrum(
    f,
    power,
    rho=MEAN_RHO,
    h=MEAN_H,
    return_full=False,
    butterworth_filter_order=BUTTERWORTH_FILTER_ORDER,
    peak_distance=PEAK_DISTANCE
):
    """Estimate formants by peak picking off a low-pass smoothed power spectrum"""
    # Smooth spectrum based on rho
    sos = scipy.signal.butter(
        butterworth_filter_order, rho, 'lowpass', analog=False, output='sos'
    )
    smoothed = scipy.signal.sosfiltfilt(sos, power)
    
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