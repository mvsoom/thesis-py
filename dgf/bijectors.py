from init import __memory__

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import scipy.stats
import dynesty

from dgf import constants

SAMPLERARGS = {'bound': 'multi', 'sample': 'rslice', 'bootstrap': 10}
RUNARGS = {'save_bounds': False}

def get_log_stats(samples, bounds):
    """
    Collect Gaussian stats for `samples` in the log domain. If there are
    `n` samples of `d` variables, `samples` must be shaped `(n, d)` and
    `bounds` must have shape `(n, 2)` indicating the bounds of the samples
    in the *original* domain.
    """
    # Transform variables to log domain in-place
    samples = jnp.log(samples)
    bounds = jnp.log(bounds)

    # Get Gaussian stats
    mean = jnp.mean(samples, axis=0)
    cov = jnp.cov(samples.T)
    sigma = jnp.sqrt(jnp.diag(cov))
    corr = jnp.diag(1/sigma) @ cov @ jnp.diag(1/sigma)
    L_corr = jnp.linalg.cholesky(corr)
    
    logstats = dict(
        samples=samples, bounds=bounds, mean=mean,
        cov=cov, sigma=sigma, corr=corr, L_corr=L_corr
    )
    return logstats

def softclipexp_bijector(bounds, sigma):
    """
    Transform an unbounded real variable in `R^n` with scale `sigma` to a
    positive bounded variable by first softclipping it to lie within `bounds`
    and then exponentiating it. `sigma` must have shape `(n,)` to indicate
    the scale for each dimension and `bounds` must have shape `(n, 2)`.
    """
    return tfb.Chain([
        tfb.Exp(), tfb.SoftClip(
            bounds[:,0], bounds[:,1], sigma
        )
    ])

def color_bijector(mean, cov):
    """Transform samples from `N(0, I)` to `N(mean, cov)`"""
    tril = jnp.linalg.cholesky(cov)
    return color_bijector_tril(mean, tril)

def color_bijector_tril(mean, tril):
    shift = tfb.Shift(mean)
    scale = tfb.ScaleMatvecTriL(tril)
    return tfb.Chain([shift, scale])

def fit_nonlinear_coloring_bijector(
    samples, bounds, cacheid, 
    samplerargs=SAMPLERARGS, runargs=RUNARGS, return_fit_results=False
):
    """
    **NOTE:** This function memoizes fit results primarily based on the `cacheid`,
    NOT on the `samples` and `bounds`! Suppying unique `cacheid`s is ESSENTIAL.
    
    Fit a **nonlinear coloring bijector** that takes `z ~ N(0, I)` samples
    first through a linear coloring bijector (to get a multivariate Gaussian)
    and then through a nonlinear bijector (to get bounds and exponentiate)
    to the empirical distribution of `samples` as well as possible by
    maximizing the likelihood `L(s) = p(samples|s)`, where `s` is a vector
    of rescaling coefficients that rescale the empirical covariance matrix
    of `samples` *in the log domain*. In other words, we fit `s` by
    direct maximum likelihood to find a new, rescaled covariance matrix
    that we can use in the log domain to model `samples`.
    
    The probability `p(samples|s)` is given by transforming a MVN with the
    `softclipexp` bijector and the prior `p(s) = Exp(1)` expresses that the
    rescaling coefficients are of order 1, such that the empirical covariance
    matrix of `log(samples)` is deemed to be a good fit.
    """
    # Collect statistics in the log domain to setup the fit
    logstats = get_log_stats(samples, bounds)
    
    # Create the static softclipexp bijector which will not be optimized
    softclipexp = softclipexp_bijector(logstats['bounds'], logstats['sigma'])
    
    # Define the likelihood function `L(s) = p(samples|s)`
    @jax.jit
    def loglike(s):
        prior = getprior(s)
        logprob = jnp.sum(prior.log_prob(samples))
        return logprob
    
    def rescaled_normal_tril(s):
        """
        Get `cholesky(Sigma(s))` where `Sigma(s)` is the rescaled
        covariance matrix by the rescaling factors `s`
        """
        return jnp.diag(s*logstats['sigma']) @ logstats['L_corr']
    
    def getprior(s):
        mvn = tfd.MultivariateNormalTriL(
            loc=logstats['mean'],
            scale_tril=rescaled_normal_tril(s)
        )
        prior = tfd.TransformedDistribution(
            distribution=mvn,
            bijector=softclipexp
        )
        return prior
    
    # Define the prior
    def ptform(
        u,
        rescale_prior=scipy.stats.expon(scale=1.)
    ):
        return rescale_prior.ppf(u)
    
    # Run the sampler
    ndim = samples.shape[1]
    if 'nlive' not in samplerargs:
        samplerargs['nlive'] = ndim*5

    @__memory__.cache
    def run_nested(cacheid, samplerargs, runargs):
        seed = cacheid
        rng = np.random.default_rng(seed)
        sampler = dynesty.NestedSampler(
            loglike, ptform, ndim, rstate=rng, **samplerargs
        )
        sampler.run_nested(**runargs)
        return sampler.results
    
    results = run_nested(cacheid, samplerargs, runargs)
    
    # Get the maximum likelihood fit of the rescaling parameters...
    s_ML = results.samples[-1,:]
    
    # ... and use them to create the best bijector `N(0,I)`to `p(samples)`
    tril_ML = rescaled_normal_tril(s_ML)
    color = color_bijector_tril(logstats['mean'], tril_ML)
    
    # Construct the bijector and expose the parameters in a more handy format
    nlc = tfb.Chain([
        softclipexp, color
    ])
    
    nlc.meta = {
        'softclipexp': {
            'bounds': logstats['bounds'],
            'sigma': logstats['sigma']
        },
        'color': {
            'mean': logstats['mean'],
            'tril': tril_ML, # Cholesky of covariance matrix
        }
    }
    
    return (nlc, results) if return_fit_results else nlc

############

def bounded_exp_bijector(low, high, eps = 1e-5, hinge_factor=0.01):
    """
    Transform an unbounded real variable to a positive bounded variable in `[low, high]`.
    To avoid numerical problems when the unbounded variable hits one of the boundaries,
    the boundaries are made slightly more permissive by a factor `eps`.
    
    The hinge softness in the SoftClip is determined automatically as `hinge_factor*low`
    to prevent scaling issues. (See comment below in the function's source.)
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
        import warnings
        warnings.warn(
            'The log det Jacobian of `lf_generic_params_bijector()` can become unstable for '
            'the `lf_generic_params_trajectory_prior()` because the generic LF parameters '
            'will at some point hit their bounds.'
        )
    
    return tfb.Chain([bijector, matrix_transpose, reshape])