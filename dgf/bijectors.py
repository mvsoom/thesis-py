from init import __memory__
from dgf import isokernels
from lib import constants

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import scipy.stats
import dynesty
import itertools
from functools import partial
import warnings

SAMPLERARGS = {}
RUNARGS = {"save_bounds": False}


def stabilize(A, sigma=0.0, eps=jnp.finfo(float).eps):
    n = A.shape[0]
    return A + jnp.eye(n) * (n * eps + sigma**2)


def get_log_stats(samples, bounds):
    """
    Collect Gaussian stats for `samples` in the log domain. If there are
    `b` samples of `n` variables, `samples` must be shaped `(b, n)` and
    `bounds` must have shape `(n, 2)` indicating the bounds of the samples
    in the *original* domain.
    """
    # Transform variables to log domain in-place
    samples = np.log(samples)
    bounds = np.log(bounds)

    # Get Gaussian stats
    mean = np.mean(samples, axis=0)
    cov = np.atleast_2d(np.cov(samples.T))
    sigma = np.sqrt(np.diag(cov))
    corr = np.diag(1 / sigma) @ cov @ np.diag(1 / sigma)
    L_corr = np.linalg.cholesky(stabilize(corr))

    logstats = dict(
        samples=samples,
        bounds=bounds,
        mean=mean,
        cov=cov,
        sigma=sigma,
        corr=corr,
        L_corr=L_corr,
    )
    return logstats


def softclipexp_bijector(bounds, sigma):
    """
    Transform an unbounded real variable in `R^n` with scale `sigma` to a
    positive bounded variable by first softclipping it to lie within `bounds`
    and then exponentiating it. `sigma` must have shape `(n,)` to indicate
    the scale for each dimension and `bounds` must have shape `(n, 2)`.
    """
    return tfb.Chain([tfb.Exp(), tfb.SoftClip(bounds[:, 0], bounds[:, 1], sigma)])


def color_bijector(mean, cov):
    """Transform samples from `N(0, I)` to `N(mean, cov)`"""
    tril = jnp.linalg.cholesky(stabilize(cov))
    return color_bijector_tril(mean, tril)


def color_bijector_tril(mean, tril):
    shift = tfb.Shift(mean)
    scale = tfb.ScaleMatvecTriL(tril)
    return tfb.Chain([shift, scale])


def nonlinear_coloring_bijector(mean, tril, bounds, sigma):
    color = color_bijector_tril(mean, tril)
    softclipexp = softclipexp_bijector(bounds, sigma)
    return tfb.Chain([softclipexp, color])


def fit_nonlinear_coloring_bijector(
    samples,
    bounds,
    cacheid,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS,
    return_fit_results=False,
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

    `samples` must be shaped `(b, n)` and `bounds` must have shape `(n, 2)`
    indicating the bounds of the `n` variables in the *original* domain.

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
        logl = jnp.sum(get_distribution(s).log_prob(samples))
        return logl

    def rescaled_normal_tril(s):
        """
        Get `cholesky(Sigma(s))` where `Sigma(s)` is the rescaled
        covariance matrix by the rescaling factors `s`
        """
        return jnp.diag(s * logstats["sigma"]) @ logstats["L_corr"]

    def get_bijector(s):
        return nonlinear_coloring_bijector(
            logstats["mean"],
            rescaled_normal_tril(s),
            logstats["bounds"],
            logstats["sigma"],
        )

    def get_distribution(
        s, standardnormals=tfd.MultivariateNormalDiag(scale_diag=jnp.ones(ndim))
    ):
        return tfd.TransformedDistribution(
            distribution=standardnormals, bijector=get_bijector(s)
        )

    # Define the prior
    def ptform(u, rescale_prior=scipy.stats.expon(scale=1.0)):
        return rescale_prior.ppf(u)

    # Run the sampler and cache results based on `cacheid`
    if "nlive" not in samplerargs:
        samplerargs = samplerargs.copy()
        samplerargs["nlive"] = ndim * 5

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
    s_ML = results.samples[-1, :]

    # ... and use them to create the best bijector `N(0,I)`to `p(samples)`
    nlc = get_bijector(s_ML)

    return (nlc, results) if return_fit_results else nlc


def matrix_to_vec_bijector(m, n):
    """Implement the vec(A) operator where A is shaped (m, n) as described at https://en.wikipedia.org/wiki/Vectorization_(mathematics)"""
    return tfb.Invert(vec_to_matrix_bijector(m, n))


def vec_to_matrix_bijector(m, n, constant=False):
    """Inverse of vec(A) operator -- see `matrix_to_vec_bijector()`"""
    if not constant:
        # Transform vector into row-stacked matrix
        reshape = tfb.Reshape(
            event_shape_out=(n, m), event_shape_in=(n * m,)  # A matrix  # A vector
        )

        # Transform row-stacked matrix into column-stacked
        matrix_transpose = tfb.Transpose(rightmost_transposed_ndims=2)

        b = tfb.Chain([matrix_transpose, reshape])
    else:
        b = _vec_to_matrix_copies_bijector(m)
    
    b.ndim = (m, n)
    return b


def _vec_to_matrix_copies_bijector(m):
    """This sends 1d arrays to row-stacked copies. Example:
    > x = jnp.array([1., 2., 3., 4.])
    > hstack_bijector(3).forward(x)
    array([[1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.]])
    """

    def forward_fn(x):
        return jnp.repeat(jnp.expand_dims(x, axis=0), m, axis=0)

    def inverse_fn(y):
        return y[0]

    bijector = tfb.Inline(
        forward_fn=forward_fn,
        inverse_fn=inverse_fn,
        forward_log_det_jacobian_fn=lambda _: 0.0,
        inverse_log_det_jacobian_fn=lambda _: 0.0,
        forward_min_event_ndims=1,
        inverse_min_event_ndims=2,
        name="vec_to_matrix_copies",
    )

    return bijector


# @partial(jax.jit, static_argnames=("m", "envelope_kernel_name"))
def nonlinear_coloring_trajectory_bijector(
    nonlinear_coloring_bijector,
    m,
    envelope_kernel_name,
    envelope_lengthscale,
    envelope_noise_sigma=0.0,
    constant=False,
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
    if constant:
        # Return constant trajectories, where for each pitch period
        # the same value is repeated for each variable
        b = nonlinear_coloring_trajectory_bijector(
            nonlinear_coloring_bijector,
            1,
            envelope_kernel_name,
            envelope_lengthscale,
            envelope_noise_sigma,
            constant=False
        )

        softclipexp, vec_to_matrix, color = b.bijectors
        _, n = vec_to_matrix.ndim
        del vec_to_matrix

        return tfb.Chain([softclipexp, vec_to_matrix_bijector(m, n, constant=True), color])

    # Pick apart the bijector of the nonlinear coloring prior
    softclipexp, color = nonlinear_coloring_bijector.bijectors

    # Get the underlying MVN of the marginal (cross-sectional) prior
    shift, scale = color.bijectors

    marginal_mean = shift.parameters["shift"]
    marginal_tril = scale.parameters["scale_tril"]

    n = len(marginal_mean)

    # Get the envelope (longitudinal) correlations
    envelope_variance = 1.0
    envelope_kernel = isokernels.resolve(envelope_kernel_name)(
        envelope_variance, envelope_lengthscale
    )

    index_points = jnp.arange(m).astype(float)[:, None]

    envelope_K = envelope_kernel.matrix(index_points, index_points)
    envelope_tril = jnp.linalg.cholesky(stabilize(envelope_K, envelope_noise_sigma))

    # Construct the MVN with Kronecker kernel `k((r,i), (s,j)) = k(r,s) k(i,j)`
    # Note that `chol(A x B) = chol(A) x chol(B)`, where `x` is Kronecker product
    kron_mean = jnp.kron(marginal_mean, jnp.ones(m))
    kron_tril = jnp.kron(marginal_tril, envelope_tril)

    # Construct the bijector from white noise `z ~ N(0,1)` in `R^ndim`
    # where `ndim == n*envelope_points` to the desired trajectory
    # which is a matrix that lives in `R^(envelope_points,n)`
    color = color_bijector_tril(kron_mean, kron_tril)

    return tfb.Chain([softclipexp, vec_to_matrix_bijector(m, n, constant=False), color])


def list_to_batches(samples):
    """
    Convert list of samples of `n` variables shaped `(m, n)` where `m` varies
    to a list of "batch" arrays shaped `(b, m, n)` where `b` is batch size for
    samples of length `m`, and where `b` and `m` vary.
    """
    samples = sorted(samples, key=len)
    batches = [jnp.stack(list(g)) for _, g in itertools.groupby(samples, key=len)]
    return batches


def batches_to_list(batches):
    """Inverse of `list_to_batches()`"""
    return [
        sample[0, :] for batch in batches for sample in jnp.split(batch, len(batch))
    ]


def fit_nonlinear_coloring_trajectory_bijector(
    samples,
    bounds,
    envelope_kernel_name,
    cacheid,
    samplerargs=SAMPLERARGS,
    runargs=RUNARGS,
    return_fit_results=False,
):
    """
    **NOTE:** This function memoizes fit results primarily based on the `cacheid`,
    NOT on the `samples` and `bounds`! Suppying unique `cacheid`s is essential.

    This function works like `fit_nonlinear_coloring_bijector()`, but now
    an envelope GP with `envelope_kernel_name` is fitted to the samples
    and a bijector with the maximum likelihood values of the parameters `s`
    (rescaling factors) and the `envelope_lengthscale` and `envelope_noise_sigma`
    is returned.

    The `samples` is a list of observed trajectories of the `n` variables, and
    can have varying lengths `m` -- thus `samples[0].shape == [m1, n]`,
    `samples[1].shape == [m2, n]`, etc. The `bounds` are an `(n, 2)` shaped array
    as in `fit_nonlinear_coloring_bijector()`.

    The MVN underlying the `samples` has a kernel

        k([x,i], [y,j]) = k(x,y) k(i,j)

    and the first kernel is supplied by `fit_nonlinear_coloring_bijector()`;
    the second integer-indexed kernel is defined by the GP fitted here.
    This is in effect a Multi-Output GP (also called multi-task GP), where
    each output (task) in our case is the trajectory of a variable.
    """
    # Collect marginal statistics
    mlogstats = get_log_stats(jnp.vstack(samples), bounds)
    n = len(mlogstats["mean"])

    # Organize the `samples` into a list of batches with equal
    # trajectory lengths `m` for vectorized computation
    batches = list_to_batches(samples)

    def loglike(x):
        logprob = np.sum([loglike_batch(x, batch) for batch in batches])
        return logprob

    @jax.jit
    def loglike_batch(x, batch):
        """
        Calculate the log likelihood of a `batch` shaped `(b, m, n)` where
            b: Batch length (number of samples with trajectory length of `m`)
            m: Trajectory length
            n: Amount of marginal variables (equal to the global `n`)
        """
        b, m, n = batch.shape  # del b, n
        logprob = jnp.sum(get_distribution(x, m).log_prob(batch))
        return logprob

    def rescaled_normal_tril(s):
        """
        Get `cholesky(Sigma(s))` where `Sigma(s)` is the rescaled
        covariance matrix by the rescaling factors `s`
        """
        return jnp.diag(s * mlogstats["sigma"]) @ mlogstats["L_corr"]

    def unpack(x):
        return x[:n], x[n], x[n + 1]

    def get_bijector(x, m):
        s, envelope_lengthscale, envelope_noise_sigma = unpack(x)

        marginal_bijector = nonlinear_coloring_bijector(
            mlogstats["mean"],
            rescaled_normal_tril(s),
            mlogstats["bounds"],
            mlogstats["sigma"],
        )

        return nonlinear_coloring_trajectory_bijector(
            marginal_bijector,
            m,
            envelope_kernel_name,
            envelope_lengthscale,
            envelope_noise_sigma,
        )

    def get_standardnormals(m):
        return tfd.MultivariateNormalDiag(scale_diag=jnp.ones(n * m))

    def get_distribution(x, m):
        return tfd.TransformedDistribution(
            distribution=get_standardnormals(m), bijector=get_bijector(x, m)
        )

    # Define the prior
    def ptform(
        u,
        rescale_prior=scipy.stats.expon(scale=1.0),
        envelope_lengthscale_prior=scipy.stats.lognorm(np.log(10)),
        envelope_noise_sigma_prior=scipy.stats.lognorm(np.log(10)),
    ):
        us = unpack(u)
        return np.array(
            [
                *rescale_prior.ppf(us[0]),
                envelope_lengthscale_prior.ppf(us[1]),
                envelope_noise_sigma_prior.ppf(us[2]),
            ]
        )

    # Run the sampler and cache results based on `cacheid`
    ndim = n + 2
    if "nlive" not in samplerargs:
        samplerargs = samplerargs.copy()
        samplerargs["nlive"] = ndim * 5

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

    # Get the maximum likelihood fit...
    x_ML = results.samples[-1, :]

    # ... and use them to create the best bijector `N(0,I)`to `p(samples|m)`
    def Bijector(m):
        return get_bijector(x_ML, m)

    return (Bijector, results) if return_fit_results else Bijector


def estimate_observation_noise_cov(
    nonlinear_coloring_trajectory_bijector,
    true_samples,
    observed_samples,
    return_mean=False,
):
    """
    **Note: the covariance is estimate in the "colored domain", i.e., where
    the samples are assumed MVN or GP.**

    Estimate the `(n, n)` covariance matrix given a list of `(m, n)` samples.
    Optionally return the `(n,)` mean of the errors. The errors are defined
    as estimate MINUS truth, i.e.

        (observed estimate) = (true) + (error)

    """
    true_stacked = jnp.vstack(true_samples)
    observed_stacked = jnp.vstack(observed_samples)

    softclipexp, vec_to_matrix, color = nonlinear_coloring_trajectory_bijector.bijectors

    error = softclipexp.inverse(observed_stacked) - softclipexp.inverse(true_stacked)

    observation_noise_cov = jnp.atleast_2d(jnp.cov(error.T))

    if return_mean:
        observation_noise_mean = jnp.atleast_1d(jnp.mean(error, axis=0))
        return observation_noise_mean, observation_noise_cov
    else:
        return observation_noise_cov


def _firstnonzero(a):
    return a[(a > 0)][0]


def _maskshape(a):
    nrow = _firstnonzero(jnp.sum(a, axis=0))
    ncol = _firstnonzero(jnp.sum(a, axis=1))
    return (nrow, ncol)


def _matrix_mask(a, mask):
    flat = a[mask]
    return flat.reshape(_maskshape(mask))


def condition_nonlinear_coloring_trajectory_bijector(
    nonlinear_coloring_trajectory_bijector,
    observation,
    observation_noise_cov,
    observation_noise_mean=None,
    constant=False,
):
    """
    Condition a given nonlinear coloring trajectory bijector on
    `observation` (shaped as `(m, n)`) with the observation noise
    covariance matrix (shaped as `(n, n)`). Here `n` is the number
    of variables (also known as "tasks" in the multi-output GP literature)
    and `m` is the trajectory length. This function returns a new bijector
    that sends N(0,I) to the new GP conditional on the (possibly incomplete)
    `observation`s.

    Missing observations in `observation` can be input as NaNs, such
    that the training set (i.e., the observed time and task indices) is always a
    subset of the test set (i.e., the complete set of time indices and tasks).
    It is best to supply the full `observation_noise_cov` matrix without
    missing values, even though they will be masked out -- this is because
    the said mask can be complicated.

    In general, unless all tasks are observed at all times (the so-called
    block design case or "Kronecker GP"), there is no special structure that
    can be exploited; this is called the "Hadamard case". The basic formulas
    given in Rasmussen & Williams (2006) still apply in this case, and we
    simply use them here. This costs at most `O(m^3 n^3)`, but this cost
    is amortized because the conditional mean and covariances are precalculated
    before inference time. In the block design case, (Maddox+ 2021) and
    (Rakitsch+ 2013) give more efficient formulas; specifically [Maddox+ 2021]
    shows how to sample in `O(m^3 + n^3)` time.

    Note: get `observation_noise_cov` (and perhaps `observation_noise_mean`)
    from `estimate_observation_noise_cov()`, because these need to
    be calculated in the colored domain.
    """
    if constant:
        # To condition on one (marginal) value, take mean over all rows
        observation = jnp.array([jnp.nanmean(observation, axis=0)])
    
    m, n = observation.shape
    del n

    if jnp.all(jnp.isnan(observation)):
        warnings.warn(
            "No observations to condition bijector on; returning original bijector"
        )
        return nonlinear_coloring_trajectory_bijector


    # Pick apart the bijector of the nonlinear coloring prior
    softclipexp, vec_to_matrix, color = nonlinear_coloring_trajectory_bijector.bijectors

    # Define the vec() and matrix() operators for this multitask GP
    vec = tfb.Invert(vec_to_matrix).forward
    matrix = vec_to_matrix.forward

    # We assume that the test points (X*) are complete and the observation
    # points (X) are a subset of the test points X*. This means we can get
    # the K(X*, X ) and K(X*, X ) from masking the full test matrix K(X*, X*).
    # Because K(X*, X*) is complete, it can be calculated by a Kronecker
    # product, as is done in `nonlinear_coloring_trajectory_bijector()`.
    # Note: we use `t` to denote test and `x` to denote training inputs.
    shift, scale = color.bijectors
    kron_tril = scale.parameters["scale_tril"]

    mean_t = shift.parameters["shift"]
    K_tt = kron_tril @ kron_tril.T  # == K(X*, X*)

    # Get the masks of observed indices
    test = vec(jnp.ones_like(observation).astype(bool))
    train = vec(~jnp.isnan(observation))

    #   tt_mask = jnp.outer(test, test)
    tx_mask = jnp.outer(test, train)
    xx_mask = jnp.outer(train, train)

    # Derive the other kernel matrices from K_tt
    #   K_tt = _matrix_mask(K_tt, tt_mask) # == K(X*, X*)
    K_tx = _matrix_mask(K_tt, tx_mask)  # == K(X*, X )
    K_xx = _matrix_mask(K_tt, xx_mask)  # == K(X , X )
    K_xt = K_tx.T  # == K(X,  X*)

    # Construct the full observation noise kernel matrix: Rakitsch+ (2013), Eq. 3
    C_tt = jnp.kron(observation_noise_cov, jnp.eye(m))  # == C(X*, X*)
    C_xx = _matrix_mask(C_tt, xx_mask)

    # If the noise does not have zero mean, add the nonzero noise mean to the
    # GP prior GP(mean_t, K_tt) to recover the standard zero-mean noise setting
    if observation_noise_mean is not None:
        observation_noise_mean_t = jnp.kron(observation_noise_mean, jnp.ones(m))
        mean_t = mean_t + observation_noise_mean_t

    mean_x = mean_t[train]

    # Use Cholesky to solve the linear systems involved in the conditional GP
    KC_xx = stabilize(K_xx + C_xx)
    KCL_xx, lower = jax.scipy.linalg.cho_factor(KC_xx, lower=True)

    def solve_N_xx(B):
        """Return `Y = KC_xx^(-1) B` that solves `KC_xx Y = B`"""
        Y = jax.scipy.linalg.cho_solve((KCL_xx, lower), B)
        return Y

    # Calculate the conditional mean: Eq. (2.38) from Rasmussen (2006)
    y_x = vec(softclipexp.inverse(observation))[train]
    f_t = mean_t + K_tx @ solve_N_xx(y_x - mean_x)

    # Calculate the conditional covariance: Eq. (2.24) from Rasmussen (2006)
    # Note: calculation does not depend on prior mean `mean_{x,t}`
    cov_tt = K_tt - K_tx @ solve_N_xx(K_xt)

    # Assemble the bijector
    conditioned_color = color_bijector(f_t, cov_tt)
    return tfb.Chain([softclipexp, vec_to_matrix, conditioned_color])


def _keepmask(m, n, dropdims):
    dummy = np.ones((m, n))
    dummy[:, dropdims] = np.nan
    return ~np.isnan(dummy)


def drop_dimensions(nonlinear_coloring_trajectory_bijector, dropdims=[], constant=False):
    """
    Drop the dimensions `dropdims` from the `(m,n)` trajectory bijector.
    Each dimension in `dropdims` must be in `range(n)`. This corresponds
    to marginalizing out the underlying MVN, i.e., dropping the rows and
    columns appropriately in the covariance matrix.
    """
    softclipexp, vec_to_matrix, color = nonlinear_coloring_trajectory_bijector.bijectors

    # Get original dimensions
    m, n = vec_to_matrix.ndim

    # Calculate dimensions to keep
    keepdims = [idx for idx in range(n) if idx not in dropdims]
    new_n = len(keepdims)
    if new_n == n:
        warnings.warn(f"Keeping all {n} dimensions")

    # Define the vec() operator for this multitask GP
    vec = tfb.Invert(vec_to_matrix).forward

    # Get a mask to select dimensions to keep
    keep = vec(_keepmask(m, n, dropdims))  # 1D
    keep_matrix = jnp.outer(keep, keep)  # 2D

    #############################
    # Adjust the color bijector #
    #############################
    shift, scale = color.bijectors
    new_shift = shift.parameters["shift"][keep]

    tril = scale.parameters["scale_tril"]
    cov = tril @ tril.T
    new_scale = _matrix_mask(cov, keep_matrix)

    new_color = color_bijector(new_shift, new_scale)

    #####################################
    # Adjust the vec_to_matrix bijector #
    #####################################
    new_vec_to_matrix = vec_to_matrix_bijector(m, jnp.int32(new_n), constant=constant)

    ###################################
    # Adjust the softclipexp bijector #
    ###################################
    exp, softclip = softclipexp.bijectors
    del exp

    p = softclip.parameters

    bounds = jnp.column_stack((p["low"], p["high"]))
    new_bounds = bounds[keepdims, :]
    new_sigma = p["hinge_softness"][keepdims]

    new_softclipexp = softclipexp_bijector(new_bounds, new_sigma)

    # Done. Reassemble the Frankenstein and get it out of here
    return tfb.Chain([new_softclipexp, new_vec_to_matrix, new_color])
