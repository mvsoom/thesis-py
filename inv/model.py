from dgf import bijectors
from dgf.prior import source
from vtr.prior import filter
from lib import constants
from dgf import core
from dgf import isokernels

import jax
import jax.numpy as jnp
import numpy as np
import scipy

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

def randf(hyper, rng=np.random.default_rng()):
    return rng.normal(size=ndim_f(hyper))

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
    """Get the bijector `w ~ N(0,I) => theta ~ p(theta|hyper)`"""
    # Split the long vector w ~ N(0, I) into the different parameters...
    ndim_noise_sigma = 1
    split = tfb.Split(
        [ndim_noise_sigma, ndim_source(hyper), ndim_filter(hyper), ndim_g(hyper)]
    )
    
    # ... give them names ...
    restructure = tfb.Restructure({
        'noise_sigma': 0,
             'source': 1,
             'filter': 2,
                  'g': 3
    })
    
    # ... and transform them using nonlinear coloring bijectors.
    squeeze = tfb.Reshape(event_shape_out=(), event_shape_in=(1,))
    bnoise_sigma = tfb.Chain([
        noise_sigma_bijector(),
        squeeze # Undo singleton axis from `split` bijector
    ])
    
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

def _without(d, key):
    d2 = d.copy()
    val = d2.pop(key)
    return val, d2

def unpack_theta(theta, hyper):
    noise_sigma, rest = _without(theta, 'noise_sigma')
    
    def unpack_rest(theta):
        if hyper['source']['use_oq']:
            var_sigma, r, T, Oq = theta['source']
        else:
            var_sigma, r, T = theta['source']
            Oq = 1.
        theta_source = dict(var_sigma=var_sigma, r=r, T=T, Oq=Oq)
    
        x, y = theta['filter'].split(2)
        g = theta['g']
        theta_filter = dict(x=x, y=y, g=g)
        
        return theta_source, theta_filter

    theta_source, theta_filter = jax.vmap(unpack_rest)(rest)
    
    return noise_sigma, theta_source, theta_filter

def get_offset(theta_source, hyper):
    offset = jnp.cumsum(theta_source['T'])
    offset -= offset[hyper['data']['anchor'] + hyper['data']['prepend']]
    offset += hyper['data']['anchort']
    return offset

def pole_coefficients(theta_filter, hyper):
    poles = hyper['filter'].poles(theta_filter['x'], theta_filter['y'])
    c = hyper['filter'].pole_coefficients(
        theta_filter['x'], theta_filter['y'], theta_filter['g']
    )
    return poles, c

def full_kernelmatrix_root(
    theta_source, theta_filter, hyper,
    convolve=True, integrate=False, correlatef=True
):
    offset = get_offset(theta_source, hyper)
    
    def period_root_matrix(offset, theta_source, theta_filter):
        kernel = isokernels.resolve(hyper['source']['kernel_name'])
        if convolve:
            poles, c = pole_coefficients(theta_filter, hyper)
            R = core.kernelmatrix_root_convolved_gfd_oq(
                kernel,
                theta_source['var_sigma']**2,
                theta_source['r'],
                hyper['data']['t'] - offset,
                hyper['source']['kernel_M'],
                theta_source['T'],
                theta_source['Oq'],
                hyper['meta']['bf'],
                poles,
                c,
                hyper['source']['impose_null_integral']
            )
        else:
            R = core.kernelmatrix_root_gfd_oq(
                kernel,
                theta_source['var_sigma']**2,
                theta_source['r'],
                hyper['data']['t'] - offset,
                hyper['source']['kernel_M'],
                theta_source['T'],
                theta_source['Oq'],
                hyper['meta']['bf'],
                hyper['source']['impose_null_integral'],
                integrate
            )
        
        return R

    Rs = jax.vmap(period_root_matrix)(offset, theta_source, theta_filter) # (N_P, N, M)
    R = jnp.hstack(Rs) # FIXME: should be possible to avoid copying
    
    if correlatef:
        R = R @ hyper['ftril']

    return R # (N, M*N_P)

def full_likelihood(theta, hyper, **kwargs):
    noise_sigma, theta_source, theta_filter = unpack_theta(theta, hyper)
    R = full_kernelmatrix_root(theta_source, theta_filter, hyper, **kwargs)
    logl = core.loglikelihood_hilbert(R, hyper['data']['d'], noise_sigma**2)
    return logl