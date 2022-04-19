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

def fit_holmberg_z(numsamples=10000, seed=4587):
    """Model the declination time in the z domain based on Holmberg (1988)"""
    df = pd.concat([read('men'), read('women')])

    # Simulate `Td` using the fact that `Td = U0/Ee` (Fant 1994, p. 1451)
    rng = np.random.default_rng(seed)
    U0 = truncnorm(df, 'ac flow (l/s)')
    Ee = truncnorm(df, 'maximum airflow declination rate (l/s²)')
    Td = U0.rvs(numsamples, rng)/Ee.rvs(numsamples, rng)*1000 # msec
    
    # Convert the `Td` samples to the z domain
    lower = constants.MIN_DECLINATION_TIME_MSEC
    upper = constants.MAX_DECLINATION_TIME_MSEC
    
    Td_clipped = Td[(lower < Td) & (Td < upper)]
    z = bijectors.declination_time_bijector().inverse(Td_clipped)
    
    # Fit a Gaussian in the z domain
    Td_mean_z, Td_std_z = np.mean(z), np.std(z)
    return Td_mean_z, Td_std_z

def declination_time_prior():
    """Prior for the declination time `Td` in msec based on Holmberg (1988)
    
    The declination `Td` is usually in the range of [0.5 to 1] msec (Fant 1994).
    """
    Td_mean_z, Td_std_z = fit_holmberg_z()

    prior = tfd.TransformedDistribution(
        tfd.Normal(Td_mean_z, Td_std_z),
        bijectors.declination_time_bijector(),
        name='DeclinationTimePrior'
    )
    
    return prior