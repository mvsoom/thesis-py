from init import __memory__
from dgf import isokernels
from dgf import constants

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import scipy.stats
import dynesty
import itertools
from functools import partial

SAMPLERARGS = {}
RUNARGS = {'save_bounds': False}

def _stabilize(A, sigma=0., eps=jnp.finfo(float).eps):
    n = A.shape[0]
    return A + jnp.eye(n)*(n*eps + sigma**2)

def get_log_stats(samples, bounds):
    """
    Collect Gaussian stats for `samples` in the log domain. If there are
    `b` samples of `n` variables, `samples` must be shaped `(b, n)` and
    `bounds` must have shape `(n, 2)` indicating the bounds of the samples
    in the *original* domain.
    """
    # Transform variables to log domain in-place
    samples = jnp.log(samples)
    bounds = jnp.log(bounds)

    # Get Gaussian stats
    mean = jnp.mean(samples, axis=0)
    cov = jnp.atleast_2d(jnp.cov(samples.T))
    sigma = jnp.sqrt(jnp.diag(cov))
    corr = jnp.diag(1/sigma) @ cov @ jnp.diag(1/sigma)
    L_corr = jnp.linalg.cholesky(_stabilize(corr))
    
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
    tril = jnp.linalg.cholesky(_stabilize(cov))
    return color_bijector_tril(mean, tril)

def color_bijector_tril(mean, tril):
    shift = tfb.Shift(mean)
    scale = tfb.ScaleMatvecTriL(tril)
    return tfb.Chain([shift, scale])

def nonlinear_coloring_bijector(mean, tril, bounds, sigma):
    color = color_bijector_tril(mean, tril)
    softclipexp = softclipexp_bijector(bounds, sigma)
    return tfb.Chain([
        softclipexp, color
    ])

def fit_nonlinear_coloring_bijector(
    samples, bounds, cacheid, 
    samplerargs=SAMPLERARGS, runargs=RUNARGS, return_fit_results=False
):
    """
    **NOTE:** This function memoizes fit results primarily based on the `cacheid`,
    NOT on the `samples` and `bounds`! Suppying unique `cacheid`s is essential.
    
    Fit a **nonlinear coloring bijector** that takes `z ~ N(0, I)` samples
    first through a linear coloring bijector (to get a multivariate Gaussian)
    and then through a nonlinear bijector (to get bounds and exponentiate)
    to the empirical distribution of `samples` as well as possible by
    maximizing the likelihood `L(s) = p(samples|s)`, where `s` is a vector
    of rescaling coefficients that rescale the empirical covariance matrix
    of `samples` *in the log domain*. In other words, we fit `s` by
    direct maximum likelihood to find a new, rescaled covariance matrix
    that we can use in the log domain to model `samples`.
    
    `samples` must be shaped `(n, d)` and `bounds` must have shape `(d, 2)` 
    indicating the bounds of the `d` variables in the *original* domain.
    
    The probability `p(samples|s)` is given by transforming a MVN with the
    `softclipexp` bijector and the prior `p(s) = Exp(1)` expresses that the
    rescaling coefficients are of order 1, such that the empirical covariance
    matrix of `log(samples)` is deemed to be a good fit.
    """
    ndim = samples.shape[1]
    
    # Collect statistics in the log domain to setup the fit
    logstats = get_log_stats(samples, bounds)
    
    # Define the likelihood function `L(s) = p(samples|s)`
    @jax.jit
    def loglike(s):
        logprob = jnp.sum(get_distribution(s).log_prob(samples))
        return logprob
    
    def rescaled_normal_tril(s):
        """
        Get `cholesky(Sigma(s))` where `Sigma(s)` is the rescaled
        covariance matrix by the rescaling factors `s`
        """
        return jnp.diag(s*logstats['sigma']) @ logstats['L_corr']
    
    def get_bijector(s):
        return nonlinear_coloring_bijector(
            logstats['mean'], rescaled_normal_tril(s),
            logstats['bounds'], logstats['sigma']
        )
    
    def get_distribution(
        s,
        standardnormals=tfd.MultivariateNormalDiag(scale_diag=jnp.ones(ndim))
    ):
        return tfd.TransformedDistribution(
            distribution=standardnormals,
            bijector=get_bijector(s)
        )
    
    # Define the prior
    def ptform(
        u,
        rescale_prior=scipy.stats.expon(scale=1.)
    ):
        return rescale_prior.ppf(u)
    
    # Run the sampler and cache results based on `cacheid`
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
    nlc = get_bijector(s_ML)
    
    return (nlc, results) if return_fit_results else nlc

#@partial(jax.jit, static_argnames=("m", "envelope_kernel_name"))
def nonlinear_coloring_trajectory_bijector(
    nonlinear_coloring_bijector,
    m,
    envelope_kernel_name,
    envelope_lengthscale,
    envelope_noise_sigma=0.
):
    """
    Enhance a nonlinear coloring bijector in `R^n` with **trajectory structure**
    expressed as a GP over `m` integer indexes. This bijector takes
    a white noise **vector** `z` shaped `(n*m)` to a **matrix** `X` shaped `(m,n)`.
    Here `n` is the number of variables, i.e., the number of trajectories.
    
    For example, for `n = 2` and `m = 3`, the bijector takes
    
        z = [a1, a2, a3, b1, b2, b3]  ~  MVN(zeros(6), eye(6))
    
    to
    
            [[A1, B1],
        X =  [A2, B2],  ~  {something positive, correlated and bounded}
             [A3, B3]]
    
    where `A1 = f1(a1, a2, a3, b1, b2, b3)`, and so on.
    """
    # Pick apart the bijector of the nonlinear coloring prior
    softclipexp, color = nonlinear_coloring_bijector.bijectors
    
    # Get the underlying MVN of the marginal (cross-sectional) prior
    shift, scale = color.bijectors
    
    marginal_mean = shift.parameters['shift']
    marginal_tril = scale.parameters['scale_tril']
    
    n = len(marginal_mean)

    # Get the envelope (longitudinal) correlations
    envelope_variance = 1.
    envelope_kernel = isokernels.resolve(envelope_kernel_name)(
        envelope_variance, envelope_lengthscale
    )

    index_points = jnp.arange(m).astype(float)[:,None]
    
    envelope_K = envelope_kernel.matrix(index_points, index_points)
    envelope_tril = jnp.linalg.cholesky(_stabilize(envelope_K, envelope_noise_sigma))

    # Construct the MVN with Kronecker kernel `k((r,i), (s,j)) = k(r,s) k(i,j)`
    # Note that `chol(A x B) = chol(A) x chol(B)`, where `x` is Kronecker product
    kron_mean = jnp.kron(marginal_mean, jnp.ones(m))
    kron_tril = jnp.kron(marginal_tril, envelope_tril)
    
    # Construct the bijector from white noise `z ~ N(0,1)` in `R^ndim`
    # where `ndim == n*envelope_points` to the desired trajectory
    # which is a matrix that lives in `R^(envelope_points,n)`
    color = color_bijector_tril(kron_mean, kron_tril)
    
    reshape = tfb.Reshape(
        event_shape_out=(n,m),  # A matrix
        event_shape_in=(n*m,)   # A vector ~ N(0,I)
    )

    # Transpose such that `softclipexp` can broadcast correctly
    matrix_transpose = tfb.Transpose(rightmost_transposed_ndims=2)
    
    return tfb.Chain([
        softclipexp, matrix_transpose, reshape, color
    ])

def fit_nonlinear_coloring_trajectory_bijector(
    samples, bounds, envelope_kernel_name, cacheid,
    samplerargs=SAMPLERARGS, runargs=RUNARGS, return_fit_results=False
):
    """
    samples: list of (m, n) shaped arrays
    bounds: (n, 2) shaped array
    """
    
    # Collect marginal statistics
    mlogstats = get_log_stats(jnp.vstack(samples), bounds)
    n = len(mlogstats['mean'])
    
    # Organize the `samples` into a list of batches with equal trajectory
    # lengths `m`. See `loglike_batch()` for the shape of the elements of `batch`.
    samples = sorted(samples, key=len)
    batches = [jnp.stack(list(g)) for _, g in itertools.groupby(samples, key=len)]

    #@jax.jit
    def loglike(x):
        logprob = np.sum([loglike_batch(x, batch) for batch in batches])
        return logprob
    
    #@jax.jit
    def loglike_batch(x, batch):
        """
        Calculate the log likelihood of a `batch` shaped `(b, m, n)` where
            b: Batch length (number of samples with trajectory length of `m`)
            m: Trajectory length
            n: Amount of marginal variables (equal to the global `n`)
        """
        b, m, n = batch.shape # del b, n
        
        # jitting fails at jitting log_prob because of a bug in TF probability
        # using Transpose.forward()
        #logprob = jnp.sum(get_distribution(x, m).log_prob(batch))
        
        #####
        d = get_distribution(x, m)
        logprob = jnp.sum(d.log_prob(batch))
        del d
        #print(m, b, x, '\t', logprob)
        ######
        
        return logprob

    #@jax.jit # works
    def rescaled_normal_tril(s):
        """
        Get `cholesky(Sigma(s))` where `Sigma(s)` is the rescaled
        covariance matrix by the rescaling factors `s`
        """
        return jnp.diag(s*mlogstats['sigma']) @ mlogstats['L_corr']

    #@jax.jit # works
    def unpack(x):
        return x[:n], x[n], x[n+1]

    #@partial(jax.jit, static_argnames=("m"))
    def get_bijector(x, m):
        s, envelope_lengthscale, envelope_noise_sigma = unpack(x)
        
        marginal_bijector = nonlinear_coloring_bijector(
            mlogstats['mean'], rescaled_normal_tril(s),
            mlogstats['bounds'], mlogstats['sigma']
        )
        
        return nonlinear_coloring_trajectory_bijector(
            marginal_bijector,
            m,
            envelope_kernel_name,
            envelope_lengthscale,
            envelope_noise_sigma,
        )
    
    #@partial(jax.jit, static_argnames=("m")) # works
    def get_standardnormals(m):
        return tfd.MultivariateNormalDiag(scale_diag=jnp.ones(n*m))
    
    @partial(jax.jit, static_argnames=("m")) # works
    def get_distribution(x, m):
        return tfd.TransformedDistribution(
            distribution=get_standardnormals(m),
            bijector=get_bijector(x, m)
        )
    
    # Define the prior
    def ptform(
        u,
        rescale_prior=scipy.stats.expon(scale=1.),
        envelope_lengthscale_prior=scipy.stats.lognorm(np.log(10)),
        envelope_noise_sigma_prior=scipy.stats.lognorm(np.log(10))
    ):
        us = unpack(u)
        return np.array([
            *rescale_prior.ppf(us[0]),
            envelope_lengthscale_prior.ppf(us[1]),
            envelope_noise_sigma_prior.ppf(us[2])
        ])
    
    # Run the sampler and cache results based on `cacheid`
    ndim = n + 2
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
    return results

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