from vtr.prior import bandwidth
from lib import constants

import numpy as np
import scipy.special

def _nan_like(a):
    return np.nan*a

def _transform_by_precision_matrix(w, precision_matrix):
    """Transform w ~ N(0, 1) to z ~ N(0, precision_matrix)
    
    Implementation: http://www.statsathome.com/2018/10/19/sampling-from-multivariate-normal-precision-and-covariance-parameterizations
    """
    try:
        U = scipy.linalg.cholesky(precision_matrix) # Upper triangular
        z = scipy.linalg.solve_triangular(U, w) # Backward substition
        return z
    except np.linalg.LinAlgError:
        return _nan_like(w)

class PoleZeroFilter:
    K_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)

    def __init__(self, K):
        self.K = K
        self.impulse_response_energy_msec = constants.IMPULSE_RESPONSE_ENERGY_MSEC

    def _cos_overlap_matrix(self, x, y):
        x1, x2 = x[:,None], x[None,:]
        y1, y2 = y[:,None], y[None,:]

        num = (y1 + y2)*(x1**2 + x2**2 + (y1 + y2)**2)
        den = ((x1 - x2)**2 + (y1 + y2)**2)*((x1 + x2)**2 + (y1 + y2)**2)
        return num/den

    def _sin_overlap_matrix(self, x, y):
        x1, x2 = x[:,None], x[None,:]
        y1, y2 = y[:,None], y[None,:]

        num = 2*x1*x2*(y1 + y2)
        den = ((x1 - x2)**2 + (y1 + y2)**2)*((x1 + x2)**2 + (y1 + y2)**2)
        return num/den

    def _cos_sin_overlap_matrix(self, x, y):
        x1, x2 = x[:,None], x[None,:]
        y1, y2 = y[:,None], y[None,:]

        num = x2 *(-x1**2 + x2**2 + (y1 + y2)**2)
        den = (x1**2 - x2**2)**2 + 2*(x1**2 + x2**2)*(y1 + y2)**2 + (y1 + y2)**4
        return num/den

    def overlap_matrix(self, x, y):
        X = 2*np.pi*x
        Y = np.pi*y

        c = self._cos_overlap_matrix(X, Y)
        s = self._sin_overlap_matrix(X, Y)
        cs = self._cos_sin_overlap_matrix(X, Y)

        S = np.block([
            [c,    cs],
            [cs.T, s ]
        ])
        return S # has units conjugate to (x, y)
    
    def _amplitude_precision_matrix_from_overlap(self, S):
        E = self.impulse_response_energy_msec/1000. # Convert to sec
        precision_matrix = (2*self.K)/E*S
        return precision_matrix 
    
    def amplitude_precision_matrix(self, x, y):
        S = self.overlap_matrix(x, y) # sec
        return self._amplitude_precision_matrix_from_overlap(S)
    
    def real_to_complex_amplitudes(self, g):
        """Calculate the complex pole coefficients corresponding to the real `a` (cos) and `b` (sin) amplitudes"""
        a, b = np.split(g, 2)
        c = (a - (1j)*b)/2
        return c
    
    def complex_to_real_amplitudes(self, c):
        """Calculate the a (cos) and b (sin) amplitudes corresponding with the pole coefficients c. This is the inverse of real_to_complex_amplitudes()"""
        a = np.real(c + np.conj(c))
        b = np.real((1j)*(c - np.conj(c)))
        g = np.concatenate((a, b))
        return g
    
    def ndim(self):
        return 2*self.K
    
    def ndim_g(self):
        return self.ndim()
    
    def randw(self, rng=np.random.default_rng()):
        return rng.normal(size=self.ndim_g())
    
    def randws(self, size, rng=np.random.default_rng()):
        return rng.normal(size=(size, self.ndim_g()))
    
    def poles(self, x, y):
        return -np.pi*y + 2*np.pi*(1j)*x
    
    def pole_coefficients(self, x, y, w):
        """Compute pole coefficients for poles(x, y) and w ~ N(0, I_{2K})
        
        Returns NaNs if semi-definite overlap matrix S(x, y) (e.g. when some
        of the x's and y's are very close together).
        """
        precision_matrix = self.amplitude_precision_matrix(x, y)
        amplitudes = _transform_by_precision_matrix(w, precision_matrix)
        return self.real_to_complex_amplitudes(amplitudes)

    def impulse_response(self, t, x, y, w):
        """t is in msec, (x, y) in Hz, w ~ N(0, I_{2K})"""
        c = self.pole_coefficients(x, y, w)
        p = self.poles(x, y)/1000. # kHz
        h = self.impulse_response_cp(t, c, p)
        return h

    def impulse_response_cp(self, t, c, p):
        return np.real(2.*c[None,:]*np.exp(t[:,None]*p[None,:])).sum(axis=1)
    
    def impulse_response_energy(self, x, y, w):
        """Return the analytical impulse response energy in msec"""
        c = self.pole_coefficients(x, y, w)
        g = self.complex_to_real_amplitudes(c)
        S = self.overlap_matrix(x, y)
        energy = (g.T @ S @ g)*1000. # msec
        return energy
    
    def transfer_function_power_dB(self, f, x, y, w):
        """Calculate the PZ power spectrum of the impulse response in dB
        at the frequencies `f`. `f, x, y` given in Hz. `w` ~ N(0,I_{2K}) can be
        shaped (2K,) or (size, 2K). The output is shaped (len(f),) or (size,len(f))
        respectively.
        """
        P = self.poles(x, y)[:,None]
        C = self.pole_coefficients(x, y, w.T).T[...,None]
        S = (2.*np.pi*1j)*f[None,:]
        terms = C/(S - P) + np.conj(C)/(S - np.conj(P))
        H = np.sum(terms, axis=-2)
        power = 20.*np.log10(np.abs(H))
        return power

    def analytical_tilt(self):
        tilt = -10.*np.log10(4) # = -6 dB/octave
        return tilt

def get_fitted_TFB_samples(n_jobs=1, **kwargs):
    return bandwidth.get_fitted_TFB_samples(
        n_jobs,
        vtfilter=PoleZeroFilter,
        seed=842329,
        Ks=PoleZeroFilter.K_RANGE,
        **kwargs
    )