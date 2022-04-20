import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

from dgf import constants

import warnings

# https://github.com/tensorflow/probability/issues/1523
# Note: if this results in too much overhead, we could implement the `bounded_exp_bijector()`
# from scratch with https://github.com/yonesuke/softclip. The implementation of this GitHub
# repo is unfortunately different from the one of TF probability.
import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()
logger.addFilter(CheckTypesFilter())

def bounded_exp_bijector(low, high, eps = 1e-5, hinge_factor=0.01):
    """
    Transform an unbounded real variable to a positive bounded variable in `[low, high]`.
    To avoid numerical problems when the unbounded variable hits one of the boundaries,
    the boundaries are made slightly more permissive by a factor `eps`.
    
    The hinge softness in the SoftClip is determined automatically as `hinge_factor*low`
    to prevent scaling issues. (See comment below in the function's source.)
    
    Note: this function is quite slow because of a TensorFlow bug. See the head of this
    function's source file (https://github.com/tensorflow/probability/issues/1523).
    """
    low = jnp.float64(low) * (1 - eps)
    high = jnp.float64(high) * (1 + eps)
    
    # The hinge softness must be smaller than O(low); the default value
    # `hinge_softness == 1` implies that the range of the constrained
    # values (i.e., in the forward direction) is O(1). If this is not the
    # case, the default value will cause numerical problems or prevent the
    # entire constrained range to be reachable from [-inf, +inf]. You can
    # check if this is the case by evaluating `b.forward(-inf), b.forward(+inf)`
    # where `b = bounded_exp_bijector(low, high, hinge_factor)`.
    hinge_softness = hinge_factor * jnp.abs(low)
    return tfb.Chain([tfb.SoftClip(low, high, hinge_softness), tfb.Exp()])

def period_bijector():
    return bounded_exp_bijector(
        constants.MIN_PERIOD_LENGTH_MSEC,
        constants.MAX_PERIOD_LENGTH_MSEC
    )

def declination_time_bijector():
    return bounded_exp_bijector(
        constants.MIN_DECLINATION_TIME_MSEC,
        constants.MAX_DECLINATION_TIME_MSEC
    )

def lf_generic_params_bijector(**kwargs):
    bounds = jnp.array([
        constants.LF_GENERIC_BOUNDS[k] for k in constants.LF_GENERIC_PARAMS
    ])
    return bounded_exp_bijector(bounds[:,0], bounds[:,1], **kwargs)

def lf_generic_params_trajectory_bijector(num_pitch_periods):
    # Reshape from 'Kronecker' structure to trajectory structure
    reshape = tfb.Reshape(
        event_shape_out=(len(constants.LF_GENERIC_PARAMS), num_pitch_periods),
        event_shape_in=(-1,)
    )

    # Transpose such that `lf_generic_params_bijector()` can broadcast correctly
    matrix_transpose = tfb.Transpose(rightmost_transposed_ndims=2)
    
    # Finally, convert to LF generic parameters. We are more permissive
    # with respect to the boundaries because for a large values of `num_pitch_period`
    # the log Jacobians of this bijector can become infinite. Sampling is
    # not a problem however.
    bijector = lf_generic_params_bijector(eps=1e-1)
    if num_pitch_periods > 50:
        warnings.warn(
            'The log det Jacobian of `lf_generic_params_bijector()` can become unstable for '
            'the `lf_generic_params_trajectory_prior()` because the generic LF parameters '
            'will at some point hit their bounds.'
        )
    
    return tfb.Chain([bijector, matrix_transpose, reshape])