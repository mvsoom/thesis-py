from vtr.prior import bandwidth
from vtr.prior import polezero

import numpy as np

class AllPoleFilter(polezero.PoleZeroFilter):
    K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)
    
    def __init__(self, K):
        super().__init__(K)
    
    def randw(self, rng=np.random.default_rng()):
        return np.atleast_1d(rng.normal())
    
    def randws(self, size, rng=np.random.default_rng()):
        return rng.normal(size=(size, 1))

    def excluded_pole_product(self, p):
        ps = np.concatenate([p, np.conj(p)])
        diff = ps[None,:] - ps[:,None]
        diff[np.diag_indices_from(diff)] = 1.
        denom = np.prod(diff, axis=0)
        return (1./denom)[:len(p)]

    def unscaled_pole_coefficients(self, x, y):
        p = self.poles(x, y)
        normalization = np.prod(np.abs(p)**2)
        return normalization*self.excluded_pole_product(p) # Hz if x, y in Hz

    def pole_coefficients(self, x, y, w):
        """(x, y) Hz, w ~ N(0,1) dimensionless"""
        S = self.overlap_matrix(x, y) # sec
        
        # Unscaled just means that the transfer function has not been
        # multiplied by `g` yet
        unscaled_c = self.unscaled_pole_coefficients(x, y) # Hz
        unscaled_alpha = self.complex_to_real_amplitudes(unscaled_c) # Hz
        unscaled_energy = (unscaled_alpha.T @ S @ unscaled_alpha)/1000 # kHz
        
        gsigma = np.sqrt(self.impulse_response_energy_msec/unscaled_energy)
        g = w*gsigma/1000 # sec
        c = g*unscaled_c
        return c # dimensionless
    
    def analytical_tilt(self):
        """Let s -> infty such that the all the poles look like zeros"""
        tilt = -20.*self.K*np.log10(4) # = (12*K) dB/octave
        return tilt

def get_fitted_TFB_samples(n_jobs=1):
    return bandwidth.get_fitted_TFB_samples(
        n_jobs,
        vtfilter=AllPoleFilter,
        seed=6667890,
        Ks=AllPoleFilter.K_RANGE
    )