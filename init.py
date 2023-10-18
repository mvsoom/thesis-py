import os
import pathlib
import warnings

__all__ = [
    '__projectdir__',
    '__datadir__',
    '__cachedir__',
    '__memory__',
    '__cache__',
    'sns',
    'jax',
    'jnp',
    'jaxkey',
]

# Define runtime working environment variables
if not ('WORKING_ENVIRONMENT_ACTIVATED' in os.environ):
    raise RuntimeError(
        'Working environment not activated. '
        'Please run load .env file first'
    )

def __projectdir__(s=''):
    return pathlib.Path(os.environ['PROJECTDIR']) / s
def __datadir__(s=''):
    return pathlib.Path(os.environ['DATADIR']) / s
def __cachedir__(s=''):
    return pathlib.Path(os.environ['CACHEDIR']) / s

# Configure joblib's caching mechanism
# NOTE: joblib cannot cache arbitrary functions, because it cannot
# hash/pickle all possible input/output values. In particular, it
# isn't able to memoize functions that return `tfb.Bijector`s or
# or `tfd.Distribution`s. For these functions we use @__cache__
# which calculates the return value of the function once when it
# is called the first time and then caches it.
import joblib
__memory__ = joblib.Memory(__cachedir__('joblib'), verbose=2)

def __cache__(func):
    def cached_func():
        if not hasattr(cached_func, "_cache"):
            cached_func._cache = func()
        return cached_func._cache
    return cached_func

# Configure global plotting options
import seaborn as sns

# Import and configure JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if 'XLA_FLAGS' in os.environ:
    XLA_FLAGS = os.environ['XLA_FLAGS']
    warnings.warn(f'External XLA configuration is `XLA_FLAGS={XLA_FLAGS}`')

import random
def jaxkey(k=None):
    if k is None: k = random.randint(0, 10000000)
    return jax.random.PRNGKey(k)