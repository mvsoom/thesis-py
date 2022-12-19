from dgf import bijectors
from dgf.prior import source
from vtr.prior import filter
from lib import constants
from dgf import core

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

def ndim_source(hyper):
    return source.SOURCE_NDIM(hyper['source'])*hyper['data']['NP']

def ndim_filter(hyper):
    return hyper['filter'].ndim()*hyper['data']['NP']

def ndim_f(hyper):
    return hyper['source']['kernel_M']*hyper['data']['NP']

def ndim_g(hyper):
    return hyper['filter'].ndim_g()*hyper['data']['NP']

def ndim(hyper):
    """The total dimension of the parameter space. Note that we don't count the source amplitudes since they are marginalized over"""
    ndim_noise = 1
    return ndim_noise + ndim_source(hyper) + ndim_filter(hyper) + ndim_g(hyper)

def _db_to_amplitude(x):
    return np.sqrt(constants.db_to_power(x))

def noise_sigma_bijector():
    """Send N(0,1) to approximately a LogNormal"""
    bounds_db = np.array([
        [constants.NOISE_FLOOR_DB, 0.]
    ])
    
    bounds = _db_to_amplitude(bounds_db)
    mean = _db_to_amplitude(-20.)
    sigma = 1.

    b = bijectors.nonlinear_coloring_bijector(
        np.log(np.atleast_1d(mean)),
        np.atleast_2d(sigma),
        np.log(bounds),
        np.atleast_1d(sigma)
    )
    
    squeeze = tfb.Reshape(event_shape_out=(-1,1), event_shape_in=())
    return tfb.Chain([tfb.Invert(squeeze), b, squeeze])

def theta_trajectory_bijector(hyper):
    bnoise_sigma = noise_sigma_bijector()
    
    bsource = source.source_trajectory_bijector(
        hyper['data']['NP'],
        hyper['source'],
        hyper['data']['T_estimate'],
        hyper['meta']['noiseless_estimates']
    )
    
    bfilter = filter.filter_trajectory_bijector(
        hyper['data']['NP'],
        hyper['filter'],
        hyper['data']['T_estimate'],
        hyper['data']['F_estimate'],
        hyper['meta']['noiseless_estimates']
    )
    
    bg = tfb.Reshape(
        event_shape_in = (ndim_g(hyper),),
        event_shape_out = (
            hyper['data']['NP'], hyper['filter'].ndim_g()
        ),
    )
    
    ndim_noise_sigma = 1
    split = tfb.Split(
        [ndim_noise_sigma, ndim_source(hyper), ndim_filter(hyper), ndim_g(hyper)]
    )
    
    restructure = tfb.Restructure({
        'noise_sigma': 0,
             'source': 1,
             'filter': 2,
                  'g': 3
    })
    
    transform = tfb.JointMap({
        'noise_sigma': bnoise_sigma,
             'source': bsource,
             'filter': bfilter,
                  'g': bg
    })
    
    return tfb.Chain([
        transform, restructure, split
    ])

def theta_trajectory_prior(hyper):
    b = theta_trajectory_bijector(hyper)
    
    standardnormals = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(ndim(hyper)))
    
    prior = tfd.TransformedDistribution(
        distribution=standardnormals,
        bijector=b,
        name="ThetaTrajectoryPrior"
    )
    return prior

def pole_coefficients(theta, hyper):
    x, y = theta['filter'].split(2)
    g = theta['g']
    poles = hyper['filter'].poles(x, y)/1000 # Convert to kHz
    pole_coeffs = hyper['filter'].pole_coefficients(x, y, g)
    return poles, pole_coeffs

def root_matrix(theta, hyper):
    pass

def model_basis_functions(theta, hyper):
    noise_sigma = theta['noise_sigma'].squeeze()
    poles, pole_coeffs = pole_coefficients(theta, hyper)
    #var_sigma, r, T, Oq = unpack_source(theta['source'], hyper)
    return c