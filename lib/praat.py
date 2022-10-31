import parselmouth
import numpy as np

DEFAULT_TRACK_OPTIONS = dict(
    formant_references_hz = (550, 1650, 2750, 3850, 4950),
    costs = (1., 1., 1.)
)

def get_pulses(d, fs):
    """Get sample indices of pulses according to Praat heuristics"""
    s = parselmouth.Sound(d, sampling_frequency=fs) # `fs` must be in Hz

    pitch = s.to_pitch()
    pp = parselmouth.praat.call([s, pitch], "To PointProcess (cc)")
    pulses = parselmouth.praat.call(pp, "To Matrix").values[0,:] # Always take pulses from left channel if stereo
    return np.asarray(pulses*fs, dtype=int) # Convert to sample indices

def get_formant_track_interpolator(
    d, fs, num_tracks, burg_options={}, track_options=DEFAULT_TRACK_OPTIONS
):
    s = parselmouth.Sound(d, sampling_frequency=fs)
    
    # Raw formants
    Formant_object_1 = s.to_formant_burg(**burg_options)
    
    # Smooth tracks and select only formants 1:num_tracks
    Formant_object_2 = parselmouth.praat.call(
        Formant_object_1, "Track...", num_tracks,
        *track_options['formant_references_hz'], *track_options['costs']
    )
    
    def track_interpolator(formant_number, indices):
        times = np.array(indices) / fs # sec
        return np.fromiter(
            (Formant_object_2.get_value_at_time(formant_number, t) for t in times),
            dtype=float
        )
    return track_interpolator

def get_formant_tracks(
    d, fs, indices=None, num_tracks=3,
    burg_options={}, track_options=DEFAULT_TRACK_OPTIONS
):
    """
    Get the first `num_tracks` formant frequency tracks from Praat, estimated
    on speech waveform `d` at sample `indices` and sampled at `fs` (Hz).
    Formant frequency values are in Hz and shaped `(len(indices), num_tracks)`.
    
    **Note**: the formant tracks are interpolated and can be evaluated anywhere
    (`indices` can be a float). However, the interpolated values can be NaN
    for some indices, indicating that formant tracks at these indices are
    undefined. This happens when Praat cannot find reliable pulses in the
    waveforms around these indices and is usually caused by endpoint effects:
    Praat's pulse algorithm will not return pulses close to the beginning
    and end of the waveform because of insufficient data, and the lack of these
    pulses will cause the formant tracks to fail.
    """
    track_interpolator = get_formant_track_interpolator(
        d, fs, num_tracks, burg_options, track_options
    )
    
    if indices is None:
        indices = np.arange(len(d))
    
    formant_numbers = np.arange(num_tracks) + 1
    return np.column_stack(
        [track_interpolator(k, indices) for k in formant_numbers]
    ) # (len(indices), num_tracks), in Hz