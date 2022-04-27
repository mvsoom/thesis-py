"""Physical constants used in the program"""
_ZERO = 1e-3

# Determined empirically from Serwy (2017)
MIN_FUNDAMENTAL_FREQUENCY_HZ = 50 # Hz
MAX_FUNDAMENTAL_FREQUENCY_HZ = 500 # Hz

MIN_PERIOD_LENGTH_MSEC = 1000/MAX_FUNDAMENTAL_FREQUENCY_HZ
MAX_PERIOD_LENGTH_MSEC = 1000/MIN_FUNDAMENTAL_FREQUENCY_HZ

MEDIAN_PERIOD_LENGTH_MSEC = 7.

# Determined empirically from Holmberg (1988)
MIN_DECLINATION_TIME_MSEC = 0.2
MAX_DECLINATION_TIME_MSEC = 4.5

# Lower bounds for the open quotient are based on Drugman (2019, Table 1) and Henrich (2005)
MIN_OQ = 0.30
MAX_OQ = 1.

MEDIAN_OQ = 0.60

# Bounds for the asymmetry coefficient are based on Doval (2006, p. 5)
MIN_AM = 0.5
MAX_AM = 1.

# Bounds for the return phase quotient are based on Doval (2006, p. 5)
MIN_QA = _ZERO
MAX_QA = 1.

# Bounds for the generic LF model parameters assuming `Ee == 1`
LF_GENERIC_PARAMS = ('T0', 'Oq', 'am', 'Qa')
LF_GENERIC_BOUNDS = {
    'T0': [MIN_PERIOD_LENGTH_MSEC, MAX_PERIOD_LENGTH_MSEC],
    'Oq': [MIN_OQ, MAX_OQ],
    'am': [MIN_AM, MAX_AM], 
    'Qa': [MIN_QA, MAX_QA]
}

LF_T_PARAMS = ('T0', 'Te', 'Tp', 'Ta')

SOURCE_PARAMS = ('var', 'r', 'T', 'Oq', 'noise_power')

# Noise floor
def db_to_power(x, ref=1.):
    return ref*10**(x/10)

NOISE_FLOOR_DB = -60. # Assuming power normalized data, i.e., the power is unity
NOISE_FLOOR_POWER = db_to_power(NOISE_FLOOR_DB)

# The boundary factor `c` (Riutort-Mayol 2020)
BOUNDARY_FACTOR = 2.