from init import __memory__
from vtr.prior import formant
from dgf.prior import source
from vtr.prior import filter
from dgf import isokernels
from dgf import bijectors
from lib import util
from lib import constants
from lib import praat
from dgf.prior import period

import numpy as np
import jax.numpy as jnp

import scipy.signal
import parselmouth
import warnings

def resample_data(data, fs, fs_new):
    N = len(data)
    N_new = int(N*fs_new/fs)
    new_data = scipy.signal.resample(data, N_new)
    return new_data

def standardize_data(d, fs, standard_fs=constants.FS_HZ):
    d = resample_data(d, fs, standard_fs)
    d = util.normalize_power(d)
    return d, standard_fs

def get_pulse_estimate_idx(d, fs):
    try:
        return praat.get_pulses(d, fs)
    except parselmouth.PraatError as e:
        raise ValueError("Data too short for Praat pulse estimation") from e

def _get_middle_n_elements(a, n):
    start = (len(a) // 2) - (n // 2)
    end = (len(a) // 2) + (n // 2) + (n % 2)
    return a[start:end]

def _num_pitch_periods(pulse_estimate):
    return len(pulse_estimate) - 1

def process_data(
    fulld,
    fs,
    pulse_estimate_idx=None,
    F_estimate=None,
    anchor=None,
    prepend=constants.PREPEND_PITCH_PERIODS,
    max_NP=np.inf,
    return_full=False
):
    fulld, fs = standardize_data(fulld, fs)
    dt = 1/fs*1000.
    
    ##########
    # Pulses #
    ##########
    # Get estimate from Praat if not supplied
    if pulse_estimate_idx is None:
        pulse_estimate_idx = get_pulse_estimate_idx(fulld, fs)
    
    # Possibly limit the number of pitch periods
    NP = prepend + _num_pitch_periods(pulse_estimate_idx)
    if NP > max_NP:
        assert max_NP > prepend
        pulse_estimate_idx = _get_middle_n_elements(pulse_estimate_idx, max_NP-prepend+1)
        NP = max_NP
    
    # Discard all data outside first and last pulse
    first, last = pulse_estimate_idx[0], pulse_estimate_idx[-1]
    d = fulld[first:last]
    d = util.normalize_power(d)
    
    # Get the period estimate per pitch period
    def to_msec(idx):
        return idx*(1000./fs)
    
    pulse_estimate = to_msec(pulse_estimate_idx)
    
    Ts = np.diff(pulse_estimate)
    T_estimate = np.concatenate(([np.nan]*prepend, Ts))
    assert T_estimate.shape == (NP,)
    
    # Define the time and the time origin
    fullt = to_msec(np.arange(len(fulld)))
    t = fullt[first:last]
    
    # Define the anchor, i.e., "time origin"
    if anchor is None:
        anchor = 0
    anchort = to_msec(pulse_estimate_idx[anchor])
    
    ############################
    # Reference formant tracks #
    ############################
    if F_estimate is None:
        reference_tracks = praat.get_formant_tracks(
            fulld, fs, num_tracks=3
        )
        F_estimate = formant.average_F_over_periods(
            pulse_estimate_idx, reference_tracks
        )
        F_estimate = np.vstack([
            np.full((prepend, 3), np.nan),
            F_estimate,
        ])
    
    data = dict(
        fulld=fulld,
        fullt=fullt,
        fs=fs,
        dt=dt,
        d=d,
        t=t,
        prepend=prepend,
        anchor=anchor,
        anchort=anchort,
        NP=NP,
        T_estimate=T_estimate,
        F_estimate=F_estimate,
        pulse_estimate=pulse_estimate
    )
    
    if return_full:
        return data, locals()
    else:
        return data

def get_source_amplitudes_tril(NP, M):
    """Get the cholesky of the covariance matrix that encodes the trajectory structure of the source `f` amplitudes"""
    
    # Get the envelope (NP x NP) covariance matrix
    envelope_kernel, envelope_noise_sigma = source.get_source_envelope_kernel()
    
    index_points = jnp.arange(NP).astype(float)[:,None]
    envelope_K = envelope_kernel.matrix(index_points, index_points)
    envelope_tril = jnp.linalg.cholesky(
        bijectors.stabilize(envelope_K, envelope_noise_sigma)
    )
    
    # Get the marginal (M x M) covariance matrix
    I = jnp.eye(M)
    
    # Get Cholesky of their kronecker product (= kron of cholesky)
    L = jnp.kron(envelope_tril, I) # Not the other way around!
    return L # (NP*M, NP*M)

def make_rand_hyper(triple, K=None, process_data_kwargs={}):
    import random; random.seed()
    import jax
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_praat, F_true_periods, F_praat_periods, context = random.choice(list(formant.yield_training_data(*triple, return_full=True)))
    
    segment = context['segment']
    
    data_config = process_data(
        segment, constants.TIMIT_FS_HZ, **process_data_kwargs
    )
    
    source_config = random.choice(list(source._yield_all_configs()))
    
    ftril = get_source_amplitudes_tril(
        data_config['NP'], source_config['kernel_M']
    )
    
    if K is None:
        K = random.choice(filter.PZ.K_RANGE)
    if random.random() < 0.5:
        vtfilter = filter.AP(K, numpy_backend=jax.numpy, scipy_backend=jax.scipy)
    else:
        vtfilter = filter.PZ(K, numpy_backend=jax.numpy, scipy_backend=jax.scipy)
    
    hyper = dict(
        meta = dict(
            noiseless_estimates = False,
            bf = constants.BOUNDARY_FACTOR,
            rho = .5, # Peak picking
            inference_method = "nested_sampling",
            inference_method_options = {}
        ),
        ftril = ftril,
        data = data_config,
        source = source_config,
        filter = vtfilter
    )
    return hyper

def hyper_fullt(hyper):
    c = hyper.copy()
    c['data'] = hyper['data'].copy()
    c['data']['t'] = c['data']['fullt']
    return c