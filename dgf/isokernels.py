import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.math import psd_kernels as jax_psd_kernels
from jax.nn import relu as ramp
from jax import jit, value_and_grad, grad

import sys

def resolve(kernel_name):
    """Avoid recalculating cached functions which have kernel arguments"""
    this_module = sys.modules[__name__]
    return getattr(this_module, kernel_name)

class IsoKernel():
    def __init__(self, variance, length_scale):
        amplitude = jnp.sqrt(variance)
        super().__init__(amplitude, length_scale)

    def spectral_density(self, s, inline=True):
        return (self.amplitude**2) * self.length_scale * self._unscaled_spectral_density(self.length_scale*s)

    def _unscaled_spectral_density(self, s):
        raise NotImplementedError

"""Ugly hack into ExponentiatedQuadratic to implement custom kernels based on the Euclidean distance"""
class CustomIsoKernel(IsoKernel, jax_psd_kernels.ExponentiatedQuadratic):
    def _kappa(self, r):
        """`k(r) = self._kappa(r)`, where `r` is the Euclidean distance between the inputs"""
        return NotImplementedError
    
    def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0
    ):
        from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
        from tensorflow_probability.substrates.jax.math.psd_kernels.internal import util
        
        r = jnp.sqrt(pairwise_square_distance) # Derive Euclidian distance
        
        inverse_length_scale = self._inverse_length_scale_parameter()
        if inverse_length_scale is not None:
            inverse_length_scale = util.pad_shape_with_ones(
                inverse_length_scale, example_ndims
            )
            r = r * inverse_length_scale
        
        out = self._kappa(r)

        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self.amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            return (amplitude**2) * out
        else:
            return out

def spectral_density_matern12(s): return 2/(1+s**2)

class Matern12Kernel(IsoKernel, jax_psd_kernels.MaternOneHalf):
    def _unscaled_spectral_density(self, s):
        return spectral_density_matern12(s)

def spectral_density_matern32(s): return 12*jnp.sqrt(3)/(3+s**2)**2

class Matern32Kernel(IsoKernel, jax_psd_kernels.MaternThreeHalves):
    def _unscaled_spectral_density(self, s):
        return spectral_density_matern32(s)

def spectral_density_matern52(s): return 400*jnp.sqrt(5)/(3*(5+s**2)**3)

class Matern52Kernel(IsoKernel, jax_psd_kernels.MaternFiveHalves):
    def _unscaled_spectral_density(self, s):
        return spectral_density_matern52(s)

def spectral_density_rbf(s): return jnp.sqrt(2*jnp.pi)*jnp.exp(-s**2/2)

class SqExponentialKernel(IsoKernel, jax_psd_kernels.ExponentiatedQuadratic):
    def _unscaled_spectral_density(self, s):
        return spectral_density_rbf(s)

def spectral_density_sinc(s): return jnp.where(s <= jnp.pi, 1., 0.)

class CenteredSincKernel(CustomIsoKernel):
    """Tobar (2019) Eq. (11)"""
    def _unscaled_spectral_density(self, s):
        return spectral_density_sinc(s)

    def _kappa(self, r):
        return jnp.sinc(r)

def spectral_density_compactpoly0(s): return jnp.sin(jnp.pi*s)**2/(jnp.pi*s)**2

class CompactPoly0(CustomIsoKernel):
    """Rasmussen & Williams (2006) Eq. (4.21) p. 88"""
    def _unscaled_spectral_density(self, s):
        return spectral_density_compactpoly0(s)

    def _kappa(self, r):
        j = 1.
        return ramp(1. - r)**j

def spectral_density_compactpoly1(s):
    return (6*jnp.pi*s*(2 + jnp.cos(2*jnp.pi*s)) - 9*jnp.sin(2*jnp.pi*s))/(2*jnp.pi**5*s**5)

class CompactPoly1(CustomIsoKernel):
    """Rasmussen & Williams (2006) Eq. (4.21) p. 88"""
    def _unscaled_spectral_density(self, s):
        return spectral_density_compactpoly1(s)

    def _kappa(self, r):
        j = 2.
        return (1 + (1 + j)*r)*ramp(1 - r)**(1 + j)

def spectral_density_compactpoly2(s):
    return 105*(-12 + 8*jnp.pi**2*s**2 - 2*(-6 + jnp.pi**2*s**2)*jnp.cos(2*jnp.pi*s) + 
   9*jnp.pi*s*jnp.sin(2*jnp.pi*s))/(4*jnp.pi**8*s**8)

class CompactPoly2(CustomIsoKernel):
    """Rasmussen & Williams (2006) Eq. (4.21) p. 88"""
    def _unscaled_spectral_density(self, s):
        return spectral_density_compactpoly2(s)

    def _kappa(self, r):
        j = 3.
        return (1/3)*(3 + (6 + 3*j)*r + (3 + 4*j + j**2)*r**2)*ramp(1 - r)**(2 + j)

def spectral_density_compactpoly3(s):
    return (945*(64*jnp.pi*s*(-6 + jnp.pi**2*s**2) + 
   2*jnp.pi*s*(-123 + 4*jnp.pi**2*s**2)*jnp.cos(2*jnp.pi*s) + 
   9*(35 - 8*jnp.pi**2*s**2)*jnp.sin(2*jnp.pi*s)))/(4*jnp.pi**11*s**11)

class CompactPoly3(CustomIsoKernel):
    """Rasmussen & Williams (2006) Eq. (4.21) p. 88"""
    def _unscaled_spectral_density(self, s):
        return spectral_density_compactpoly3(s)

    def _kappa(self, r):
        j = 4.
        return (1/15)*(15 + (45 + 15*j)*r + (45 + 36*j + 6*j**2)*r**2 + (15 + 23*j + 
      9*j**2 + j**3)*r**3)*ramp(1 - r)**(3 + j)