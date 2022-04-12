__all__ = [
    '__projectdir__',
    '__datadir__',
    '__cachedir__',
    'jax',
    'jnp'
]

# Define runtime working environment variables
import os
if not ('WORKING_ENVIRONMENT_ACTIVATED' in os.environ):
    raise RuntimeError(
        'Working environment not activated. '
        'Please run `$ source activate` first'
    )

import pathlib
def __projectdir__(s=''):
    return pathlib.Path(os.environ['PROJECTDIR']) / s
def __datadir__(s=''):
    return pathlib.Path(os.environ['DATADIR']) / s
def __cachedir__(s=''):
    return pathlib.Path(os.environ['CACHEDIR']) / s

# Import JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax import jit, value_and_grad, grad
from functools import partial