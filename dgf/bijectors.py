import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

# https://github.com/tensorflow/probability/issues/1523
import logging
logger = logging.getLogger("root")
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

def bounded_exp_bijector(low, high, hinge_factor=0.01):
    """
    Transform an unbounded real variable to a positive bounded variable in `[low, high]`.
    
    The hinge softness in the SoftClip is determined automatically as `hinge_factor*low`
    to prevent scaling issues. (See comment below in the function's source.)
    """
    low = jnp.float64(low)
    high = jnp.float64(high)
    
    # The hinge softness must be smaller than O(low); the default value
    # `hinge_softness == 1` implies that the range of the constrained
    # values (i.e., in the forward direction) is O(1). If this is not the
    # case, the default value will cause numerical problems or prevent the
    # entire constrained range to be reachable from [-inf, +inf]. You can
    # check if this is the case by evaluating `b.forward(-inf), b.forward(+inf)`
    # where `b = bounded_exp_bijector(low, high, hinge_factor)`.
    hinge_softness = hinge_factor * jnp.abs(low)
    return tfb.Chain([tfb.SoftClip(low, high, hinge_softness), tfb.Exp()])

def multi_bounded_exp_bijector(low, high, hinge_factor=0.01):
    pass