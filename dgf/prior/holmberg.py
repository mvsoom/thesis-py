from init import __datadir__
from dgf import bijectors
from dgf import constants

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import numpy as np
import pandas as pd
import scipy.stats

def read(s):
    df = pd.read_csv(
        __datadir__(f'holmberg/{s}-normal-loudness.csv')
    )
    df['sex'] = s
    df['lower'] = [float(s.split('-')[0]) for s in df['range']]
    df['upper'] = [float(s.split('-')[1]) for s in df['range']]
    return df

def process_sexes(df, s, return_dict=False):
    """Aggregate the data over 'men' and 'women' data"""
    a = df[df['parameter'] == s]
    x = np.mean(a['x'])
    sd = np.sqrt(np.sum(a['sd']**2))
    lower = np.min(a['lower'])
    upper = np.max(a['upper'])
    d = dict(parameter=s, x=x, sd=sd, lower=lower, upper=upper)
    return d if return_dict else pd.DataFrame([d])

def truncnorm(df, s):
    """Convert the data for parameter `s` into a TruncatedNormal"""
    d = process_sexes(df, s, return_dict=True)
    bounds = (d['lower'], d['upper'])
    loc = d['x']
    scale = d['sd']
    return scipy.stats.truncnorm(
        (bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale
    )

def declination_time_prior(cacheid=45870):
    """Prior for the declination time `Td` in msec based on Holmberg (1988)
    
    The declination `Td` is usually in the range of [0.5 to 1] msec (Fant 1994).
    """
    df = pd.concat([read('men'), read('women')])

    # Simulate `Td` using the fact that `Td = U0/Ee` (Fant 1994, p. 1451)
    rng = np.random.default_rng(cacheid)
    numsamples = int(1e5)
    
    U0 = truncnorm(df, 'ac flow (l/s)')
    Ee = truncnorm(df, 'maximum airflow declination rate (l/s²)')
    Td = U0.rvs(numsamples, rng)/Ee.rvs(numsamples, rng)*1000 # msec
    
    # Impose bounds
    lower = constants.MIN_DECLINATION_TIME_MSEC
    upper = constants.MAX_DECLINATION_TIME_MSEC
    bounds = np.array([lower, upper])
    Td_clipped = Td[(lower < Td) & (Td < upper)]
    
    # Fit bijector and construct prior
    bijector = bijectors.fit_nonlinear_coloring_bijector(
        Td_clipped[:,None], bounds[None,:], cacheid
    )

    standardnormal = tfd.MultivariateNormalDiag(scale_diag=jnp.ones(1))
    prior = tfd.TransformedDistribution(
        standardnormal,
        bijector,
        name='LFDeclinationTimePrior'
    )
    
    return prior