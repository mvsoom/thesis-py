"""Physical constants used in the program"""
_ZERO = 1e-3

# Determined empirically from Serwy (2017)
MIN_FUNDAMENTAL_FREQUENCY_HZ = 50 # Hz
MAX_FUNDAMENTAL_FREQUENCY_HZ = 500 # Hz

MIN_PERIOD_LENGTH_MSEC = 1000/MAX_FUNDAMENTAL_FREQUENCY_HZ
MAX_PERIOD_LENGTH_MSEC = 1000/MIN_FUNDAMENTAL_FREQUENCY_HZ

# Determined empirically from Holmberg (1988)
MIN_DECLINATION_TIME_MSEC = 0.2
MAX_DECLINATION_TIME_MSEC = 4.5

# Bounds for the power of the DGF waveform of the LF model, given that `Ee == 1`
# This is not a LF parameter but is included for convenience
MIN_DGF_POWER = _ZERO
MAX_DGF_POWER = 5.

# Bounds for the open quotient are based on Drugman (2019, Table 1) and Henrich (2005)
MIN_OQ = 0.30
MAX_OQ = 0.95

# Bounds for the asymmetry coefficient are based on Doval (2006, p. 5)
MIN_AM = 0.5
MAX_AM = 1.

# Bounds for the return phase quotient are based on Doval (2006, p. 5)
MIN_QA = _ZERO
MAX_QA = 1.

# Bounds for the generic LF model parameters
LF_GENERIC_PARAMS = ("power", "T0", "Oq", "am", "Qa")
LF_GENERIC_BOUNDS = {
    'power': [MIN_DGF_POWER, MAX_DGF_POWER],
    'T0': [MIN_PERIOD_LENGTH_MSEC, MAX_PERIOD_LENGTH_MSEC],
    'Oq': [MIN_OQ, MAX_OQ],
    'am': [MIN_AM, MAX_AM], 
    'Qa': [MIN_QA, MAX_QA]
}