import numpy as np
from scipy import signal
from warnings import warn

def get_polyorder(width, df):
    n = int(width // df)
    return n if n % 2 else n + 1

def get_bandwidths_at_FWHM(envelope, peaks):
    prominences, left_bases, right_bases = signal.peak_prominences(envelope, peaks)
    
    # To get the FWHM, we can use Scipy's peak_widths() if we trick it into
    # believing the peaks have prominence of 3 dB, and then measure peak width at
    # 100% of this prominence.
    prominences_hack = np.repeat(3., len(peaks))
    prominence_data = (prominences_hack, left_bases, right_bases)

    widths, width_heights, left_lps, right_lps = signal.peak_widths(
        envelope, peaks, rel_height=1., prominence_data=prominence_data
    )
    
    return widths

def stack_formant_samples(samples_list):
    lengths, counts = np.unique([len(s) for s in samples_list], return_counts=True)
    if len(counts) == 1:
        return np.vstack(samples_list)
    else:
        n = len(samples_list)
        warn(
            f'The number of formants Q is not the same across all {n} samples: '
            f'filling out missing values with nans\n'
            f'Histogram of Q: {lengths // 2, counts}'
        )
        
        max_Q = max(lengths) // 2
        
        def nanresize(a):
            return np.pad(a, (0, max_Q - len(a)), constant_values=np.nan)
        
        def align(s):
            bandwidth, center = np.split(s, 2)
            return np.hstack([nanresize(bandwidth), nanresize(center)])
        
        aligned_samples_list = [align(s) for s in samples_list]
        return np.vstack(aligned_samples_list)

def estimate_formants(
    freq,
    spectrum,
    n_samples = 200,
    df_upsample = 1., # Hz
    freq_bounds = (100., 5000.), # Hz
    filter_window_length = 250., # Hz
    peak_prominence = 3., # dB
    peak_distance = 200. # Hz
):
    """Estimate formants from heuristic spectral smoothing. See test_peak.ipynb for details."""
    def get_formants_from_spectrum(sample):
        # Obtain spectral envelope by second-order Savitzky-Golay filter
        df = freq[1] - freq[0]
        envelope = signal.savgol_filter(sample, get_polyorder(filter_window_length, df), 2)
        
        # Upsample envelope to have a greater precision on the frequency axis
        n_upsample = int(len(envelope)*df/df_upsample)
        envelope_up, freq_up = signal.resample(envelope, n_upsample, freq, window='hamming')
       
        # Discard upsampling artifacts at low and high frequencies
        keep = (freq_bounds[0] < freq_up) & (freq_up < freq_bounds[1])
        freq_up = freq_up[keep]
        envelope_up = envelope_up[keep]
        
        # Find peaks indices
        peaks, _ = signal.find_peaks(
            envelope_up, distance=peak_distance // df, prominence=peak_prominence
        )
        
        # Get formant center frequencies and bandwidths (defined by FWHM)
        centers = freq_up[peaks]
        bandwidths = get_bandwidths_at_FWHM(envelope_up, peaks)
        
        return np.hstack([bandwidths, centers])
    
    spectra = sample_gvar(spectrum, n_samples)
    
    formant_samples_list = [get_formants_from_spectrum(s) for s in spectra]
    formant_samples = stack_formant_samples(formant_samples_list)

    mean = np.nanmean(formant_samples, axis=0)
    std = np.nanstd(formant_samples, axis=0)
    formant_estimates = np.array([gvar.gvar(m, s) for m, s in zip(mean, std)])
    return formant_estimates