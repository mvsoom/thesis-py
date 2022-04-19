"""Physical constants used in the program"""
# Determined empirically from Serwy (2017)
MIN_FUNDAMENTAL_FREQUENCY_HZ = 50 # Hz
MAX_FUNDAMENTAL_FREQUENCY_HZ = 500 # Hz

MIN_PERIOD_LENGTH_MSEC = 1000/MAX_FUNDAMENTAL_FREQUENCY_HZ
MAX_PERIOD_LENGTH_MSEC = 1000/MIN_FUNDAMENTAL_FREQUENCY_HZ

# Determined empirically from Holmberg (1988)
MIN_DECLINATION_TIME_MSEC = 0.2
MAX_DECLINATION_TIME_MSEC = 4.5

# Common range; based on Drugman (2019), Table 1
MIN_OQ = 0.30
MAX_OQ = 0.95