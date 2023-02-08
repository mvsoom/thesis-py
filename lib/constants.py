"""Physical constants used in the program"""
import numpy as np

_ZERO = 1e-3

# We work up until 5 kHz
FS_KHZ = 10.
FS_HZ = 1000*FS_KHZ

# Sampling power spectra
DF = 1. # Hz
TIMIT_FS_HZ = 16000
def spectrum_frequencies(fs=FS_HZ):
    return np.arange(0, fs/2, DF)

# Number of pitch periods to posit before the first pulse
PREPEND_PITCH_PERIODS = 1

# The boundary factor `c` (Riutort-Mayol 2020)
BOUNDARY_FACTOR = 2.

# Determined empirically from Serwy (2017)
_MIN_FUNDAMENTAL_FREQUENCY_HZ = 50 # Hz
_MAX_FUNDAMENTAL_FREQUENCY_HZ = 500 # Hz

MIN_PERIOD_LENGTH_MSEC = 1000/_MAX_FUNDAMENTAL_FREQUENCY_HZ
MAX_PERIOD_LENGTH_MSEC = 1000/_MIN_FUNDAMENTAL_FREQUENCY_HZ

MEAN_PERIOD_LENGTH_MSEC = 7.141548 # From APLAWD

# From Hawks+ (1995) Fig. 1
MIN_FORMANT_FREQ_HZ = 100.
MAX_FORMANT_FREQ_HZ = 5000.

# Bounds for VTR frequency
MIN_X_HZ = 100.
MAX_X_HZ = FS_HZ/2

# More precise bounds are in `vtr/band_bounds.py`
MIN_FORMANT_BAND_HZ = 10.
MAX_FORMANT_BAND_HZ = 500.

# Bounds for VTR bandwidth
MIN_Y_HZ = 20.
MAX_Y_HZ = FS_HZ/2

# The reference formant F and B tolerances for loosely based on Pinson (1963)
SIGMA_FB_REFERENCE_HZ = np.array([5., 10., 15.])

# Reference spectral tilt for the filter envelope and sensitivity
FILTER_SPECTRAL_TILT_DB = -2.
SIGMA_TILT_DB = 2. + 6. # Take into account the sloppiness (+6 dB) of our estimator

# Expected value of the VT impulse response energy
IMPULSE_RESPONSE_ENERGY_MSEC = 1.

# In APLAWD and VTRFormants, reject a voiced group or vowel segment
# if it has less than `MIN_NUM_PERIODS` pitch periods
MIN_NUM_PERIODS = 3

# Determined empirically from Holmberg (1988)
MIN_DECLINATION_TIME_MSEC = 0.2
MAX_DECLINATION_TIME_MSEC = 4.5

# Lower bounds for the open quotient are based on Drugman (2019, Table 1) and Henrich (2005)
MIN_OQ = 0.30
MAX_OQ = 1 - _ZERO

# Bounds for the asymmetry coefficient are based on Doval (2006, p. 5)
MIN_AM = 0.5
MAX_AM = 1 - _ZERO

# Bounds for the return phase quotient are based on Doval (2006, p. 5)
MIN_QA = _ZERO
MAX_QA = 1 - _ZERO

# Bounds for the generic LF model parameters assuming `Ee == 1`
LF_GENERIC_PARAMS = ('T0', 'Oq', 'am', 'Qa')
LF_GENERIC_BOUNDS = {
    'T0': [MIN_PERIOD_LENGTH_MSEC, MAX_PERIOD_LENGTH_MSEC],
    'Oq': [MIN_OQ, MAX_OQ],
    'am': [MIN_AM, MAX_AM], 
    'Qa': [MIN_QA, MAX_QA]
}

LF_T_PARAMS = ('T0', 'Te', 'Tp', 'Ta')

# Bounds for the variance of a GP given that the data is power-normalized
MIN_VAR_SIGMA = 1e-3
MAX_VAR_SIGMA = 1e2

# Bounds for the relative scale `r`
MIN_R = 1/(np.pi*FS_KHZ*MAX_PERIOD_LENGTH_MSEC)
MAX_R = 10. # Independent of the period `T`

# Bounds for the source parameters
SOURCE_PARAMS = ('var_sigma', 'r', 'T', 'Oq')

SOURCE_BOUNDS = {
    'var_sigma': [MIN_VAR_SIGMA, MAX_VAR_SIGMA],
    'r': [MIN_R, MAX_R],
    'T': [MIN_PERIOD_LENGTH_MSEC, MAX_PERIOD_LENGTH_MSEC],
    'Oq': [MIN_OQ, MAX_OQ]
}

# The "jitterness" of source and filter amplitude trajectories
SOURCE_F_ENVELOPE_NOISE_SIGMA = 1.
FILTER_G_ENVELOPE_NOISE_SIGMA = 1.

# Noise floor
def db_to_power(x, ref=1.):
    return ref*10**(x/10)

NOISE_FLOOR_DB = -60. # Assuming power normalized data, i.e., the power is unity
NOISE_FLOOR_POWER = db_to_power(NOISE_FLOOR_DB)