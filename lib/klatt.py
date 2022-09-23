import numpy as np
import scipy
import pandas
from IPython.display import Audio

from init import __datadir__
vowels = pandas.read_csv(__datadir__('klatt/klatt.csv'), comment='#')

def poles(v, αscale=np.pi/1000, ωcscale=2*np.pi/1000):
    """Return the formants of vowel v in pole form (but not conjugate)
    
    Units are defined by the two scaling arguments.
    """
    def α(i): return -vowels.loc[v][f"B{i}"]*αscale
    def ωc(i): return vowels.loc[v][f"F{i}"]*ωcscale*(1j)
    return np.fromiter((α(i)+ωc(i) for i in (1,2,3)), complex)

def gen_vowel(v, u_prime, t, normalize_power=True):
    p = poles(v)

    zeros = np.array([])
    poles_ = np.hstack([p, np.conj(p)])
    gain = np.abs(p)**2

    T, y, _ = scipy.signal.lsim((zeros, poles_, gain), u_prime, t)
    
    if normalize_power:
        power = np.mean(y**2)
        y = y/np.sqrt(power)

    return T, y # all(t==T) == True

def freqresp(v, w=None):
    p = poles(v)
    conjp = np.hstack([p, np.conj(p)])
    k = np.abs(np.prod(conjp))
    w, H = scipy.signal.freqresp(([], conjp, k), w)
    return w, H

def play(t, y, autoplay=True):
    dt = t[1] - t[0]
    rate = int(1000./dt)
    return Audio(y, rate=rate, autoplay=autoplay, normalize=True)