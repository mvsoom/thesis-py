import jax
import jax.numpy as jnp
import pandas as pd
import pathlib
from init import __datadir__

"""
SEE ALSO

https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.timit.html
"""
def read_timit_voiced_fxv(file):
    d = pd.read_csv(file, delim_whitespace=True, header=None, names=("t1", "t2", "type", "fx"))
    d['file'] = file
    return d

def exclude_silent_frames(fxv):
    return fxv[fxv.type != 0]

def get_periods(file):
    fxv = read_timit_voiced_fxv(file)
    #fxv = exclude_silent_frames(fxv) # These are harmless; removal not needed
    fxv["T"] = 1/fxv.fx*1000 # msec
    T = jnp.asarray(fxv["T"])
    return T

def get_transformed_periods(file, b):
    T = get_periods(file)
    return b.inverse(T)

### CODE BELOW IS JUST COPY PASTED NEEDS FIXING
timit = __datadir__('TIMIT')
timit_voiced = __datadir__('TIMIT-voiced')

def corresponding_timit_file(fxv, suffix):
    """Valid `suffix` are in `['.WAV', '.PHN', '.TXT', '.WRD']`"""
    return timit / fxv.relative_to(timit_voiced).with_suffix(suffix)

for fxv_file in timit_voiced.rglob("*.fxv"):
    wav_file = corresponding_timit_file(fxv_file, '.WAV')
    s = parselmouth.Sound(wav_file.as_posix())