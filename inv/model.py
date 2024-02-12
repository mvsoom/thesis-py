from dgf import bijectors
from dgf.prior import source
from dgf.prior import period
from vtr.prior import filter
from vtr.prior import formant
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
    if hyper['meta']['constant']:
        return source.SOURCE_NDIM(hyper['source'])
    else:
        return source.SOURCE_NDIM(hyper['source'])*hyper['data']['NP']

def ndim_filter(hyper):
    if hyper['meta']['constant']:
        return hyper['filter'].ndim()
    else:
        return hyper['filter'].ndim()*hyper['data']['NP']

def ndim_f(hyper):
    return hyper['source']['kernel_M']*hyper['data']['NP']

def ndim_g(hyper):
    if hyper['meta']['constant']:
        return hyper['filter'].ndim_g()
    else:
        return hyper['filter'].ndim_g()*hyper['data']['NP']

def ndim(hyper):
    """The total dimension of the parameter space. Note that we don't count the source amplitudes since they are marginalized over"""
    ndim_noise = 1
    ndim_delta = 1
    return ndim_noise + ndim_delta + ndim_source(hyper) + ndim_filter(hyper) + ndim_g(hyper)

def source_labels(hyper):
    return constants.SOURCE_PARAMS if hyper['source']['use_oq'] else constants.SOURCE_PARAMS[:-1]

def filter_labels(hyper):
    return [f"${c}_{{{i+1}}}$" for c in ('x', 'y') for i in range(hyper['filter'].K)]

def g_labels(hyper):
    if isinstance(hyper['filter'], filter.PZ):
        return [f"$g_{{{i+1}}}$" for i in range(2*hyper['filter'].K)]
    elif isinstance(hyper['filter'], filter.AP):
        return ["$g_0$"]
    else:
        raise ValueError(f"Unrecognized {hyper['filter']}")

def labels(hyper):
    return None # Need to implement with NP -- need to know order for this
    return (
        "noise_sigma",
        "delta",
        *source_labels(hyper),
        *filter_labels(hyper),
        *g_labels(hyper)
    )

def rough_frequency_limit(hyper):
    """See dgf/test/test_pulse.ipynb for rationale"""
    M = hyper['source']['kernel_M']
    T = np.nanmean(hyper['data']['T_estimate'])
    c = constants.BOUNDARY_FACTOR
    f = M/(2*c*T)
    return f*1000 # Hz

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

def delta_bijector(hyper):
    rel_mu, rel_sigma = period.fit_praat_relative_gci_error() # Fractional
    
    ### TODO: testing!! Seems to work great
    rel_sigma = .05
    
    # Get a reference value for the pitch period
    a = hyper['data']['anchor'] + hyper['data']['prepend']
    Tref = hyper['data']['T_estimate'][a]
    if np.isnan(Tref):
        Tref = constants.MEAN_PERIOD_LENGTH_MSEC
    
    # Convert to absolute values in msec
    mu = Tref*rel_mu
    sigma = Tref*rel_sigma
    
    return tfb.Chain([
        tfb.Shift(mu), tfb.Scale(sigma)
    ])

def filter_g_bijector(hyper):
    """Get the trajectory bijector for the filter `g` amplitudes"""
    # We can use `nonlinear_coloring_trajectory_bijector()` to endow
    # a bijector with trajectory structure, but this method expects a
    # nonlinear component. We hack into it by setting the nonlinear
    # component to identity, since the amplitudes already live in the
    # linear (unbounded) domain
    ndim = hyper['filter'].ndim_g()
    beye = bijectors.color_bijector(
        jnp.zeros(ndim), jnp.eye(ndim)
    )
    bhack = tfb.Chain([
        tfb.Identity(), beye
    ])
    
    # Get the filter envelope kernel and lengthscale. The noise sigma
    # (jitterness of trajectories) is ignored, since it is defined in
    # the log domain. Instead, since all amplitudes have been rescaled
    # to be N(0,1), we set the envelope kernel noise sigma to O(1).
    envelope_kernel_name, _, results =\
        formant.fit_formants_trajectory_kernel()
    envelope_lengthscale, _ =\
        formant.maximum_likelihood_envelope_params(results)
    envelope_noise_sigma = constants.FILTER_G_ENVELOPE_NOISE_SIGMA
    
    # Hack the N(0,I) bijector and the trajectory kernel together
    bg = bijectors.nonlinear_coloring_trajectory_bijector(
        bhack,
        hyper['data']['NP'],
        envelope_kernel_name,
        envelope_lengthscale,
        envelope_noise_sigma,
        constant=hyper['meta']['constant']
    )
    return bg

def theta_trajectory_bijector(hyper):
    """Get the bijector `w ~ N(0,I) => theta ~ p(theta|hyper)`"""
    # Split the long vector w ~ N(0, I) into the different parameters...
    ndim_noise_sigma = 1
    ndim_delta = 1
    split = tfb.Split(
        [ndim_noise_sigma, ndim_delta, ndim_source(hyper), ndim_filter(hyper), ndim_g(hyper)]
    )
    
    # ... give them names ...
    restructure = tfb.Restructure({
        'noise_sigma': 0,
              'delta': 1,
             'source': 2,
             'filter': 3,
                  'g': 4
    })
    
    # ... and transform them using (non)linear coloring bijectors.
    squeeze = tfb.Reshape(event_shape_out=(), event_shape_in=(1,))
    bnoise_sigma = tfb.Chain([
        noise_sigma_bijector(),
        squeeze # Undo singleton axis from `split` bijector
    ])
    
    bdelta = tfb.Chain([
        delta_bijector(hyper),
        squeeze # Undo singleton axis from `split` bijector
    ])
    
    bsource = source.source_trajectory_bijector(
        hyper['data']['NP'],
        hyper['source'],
        hyper['data']['T_estimate'],
        hyper['meta']['noiseless_estimates'],
        constant=hyper['meta']['constant']
    )
    
    bfilter = filter.filter_trajectory_bijector(
        hyper['data']['NP'],
        hyper['filter'],
        hyper['data']['T_estimate'],
        hyper['data']['F_estimate'],
        hyper['meta']['noiseless_estimates'],
        constant=hyper['meta']['constant']
    )
    
    bg = filter_g_bijector(hyper)
    
    transform = tfb.JointMap({
        'noise_sigma': bnoise_sigma,
              'delta': bdelta,
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

def _without(d, keys):
    d2 = d.copy()
    return [d2.pop(key) for key in keys], d2

def unpack_theta(theta, hyper):
    (noise_sigma, delta), rest = _without(theta, ('noise_sigma', 'delta'))
    
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
    
    return noise_sigma, delta, theta_source, theta_filter

def get_offset(delta, theta_source, hyper):
    # Make `cs = (0, T0, T0 + T1, T0 + T1 + T2, ...)` whose
    # diff is `theta_source['T']`
    cs = jnp.cumsum(theta_source['T'])
    cs = jnp.roll(cs, +1)
    cs = cs.at[0].set(0.)
    
    t0 = hyper['data']['anchort'] - delta
    offset = t0 + cs - cs[hyper['data']['anchor'] + hyper['data']['prepend']]
    return offset

def pole_coefficients(theta_filter, hyper):
    poles = hyper['filter'].poles(theta_filter['x'], theta_filter['y'])
    c = hyper['filter'].pole_coefficients(
        theta_filter['x'], theta_filter['y'], theta_filter['g']
    )
    return poles, c

def full_kernelmatrix_root(
    delta, theta_source, theta_filter, hyper,
    convolve=True, integrate=False,
    correlatef=True, regularize_flow=True
):
    offset = get_offset(delta, theta_source, hyper)
    
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
                hyper['source']['impose_null_integral'],
                regularize_flow
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
                integrate,
                regularize_flow
            )
        
        return R

    Rs = jax.vmap(period_root_matrix)(offset, theta_source, theta_filter) # (N_P, N, M)
    R = jnp.hstack(Rs) # FIXME: should be possible to avoid copying
    
    if correlatef:
        # This mixes up basisfunctions such that they can span multiple pitch periods
        R = R @ hyper['ftril']

    return R # (N, M*N_P)

def full_likelihood(theta, hyper, **kwargs):
    noise_sigma, delta, theta_source, theta_filter = unpack_theta(theta, hyper)
    
    # The basisfunctions in `R` are regularized by (control thru `kwargs`):
    #  1. Smoothness determined by kernel type (always activated)
    #  2. Flow regularization (activated by `regularize_flow=True`)
    #  3. Glottal flow amplitude correlations (activated by `correlatef=True`)
    R = full_kernelmatrix_root(delta, theta_source, theta_filter, hyper, **kwargs)
    
    # `logl` includes the effects of `correlatef` and `regularize_flow` automatically
    logl = core.loglikelihood_hilbert(R, hyper['data']['d'], noise_sigma**2)
    
    return logl