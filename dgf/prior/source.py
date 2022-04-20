"""Fitting the parameters of the source"""
import numpy as np

def _determine_GCI_index(t, u, Te, treshold):
    mask = (t > Te) & (np.abs(u) < treshold)
    return np.argmax(mask) # Returns 0 if mask is `False` everywhere

def _swap_dgf(t, u, Te, treshold):
    i = _determine_GCI_index(t, u, Te, treshold)
    return np.concatenate((u[i:], u[:i]))

def _nonzero_dgf_indices(t, T0):
    return np.flatnonzero((0 <= t) & (t < T0))

def closed_phase_first(t, u, p, treshold, offset=0.):
    """Take a DGF waveform and put its closed phase before the GOI"""
    i = _nonzero_dgf_indices(t - offset, p['T0'])
    t_nonzero = t[i]
    u_nonzero = u[i]
    u_nonzero_swapped = _swap_dgf(t_nonzero - offset, u_nonzero, p['Te'], treshold)
    return np.concatenate((u[:i[0]], u_nonzero_swapped, u[i[-1]+1:]))