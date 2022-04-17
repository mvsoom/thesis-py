import os
import pathlib
import warnings

__all__ = [
    '__projectdir__',
    '__datadir__',
    '__cachedir__',
    '__memory__',
    'jax',
    'jnp'
]

# Define runtime working environment variables
if not ('WORKING_ENVIRONMENT_ACTIVATED' in os.environ):
    raise RuntimeError(
        'Working environment not activated. '
        'Please run `$ source activate` first'
    )

def __projectdir__(s=''):
    return pathlib.Path(os.environ['PROJECTDIR']) / s
def __datadir__(s=''):
    return pathlib.Path(os.environ['DATADIR']) / s
def __cachedir__(s=''):
    return pathlib.Path(os.environ['CACHEDIR']) / s

# Configure joblib's caching mechanism
import joblib
__memory__ = joblib.Memory(__cachedir__('joblib'), verbose=2)

# Import and configure JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if 'XLA_FLAGS' in os.environ:
    XLA_FLAGS = os.environ['XLA_FLAGS']
    warnings.warn(f'External XLA configuration is `XLA_FLAGS={XLA_FLAGS}`')