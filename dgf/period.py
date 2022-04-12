import jax
import jax.numpy as jnp
import pandas as pd

def read_timit_voiced_fxv(file):
    return pd.read_csv(file, delim_whitespace=True, header=None, names=("t1", "t2", "type", "fx"))

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