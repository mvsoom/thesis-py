from init import __memory__
from vtr.prior import bandwidth
from vtr import spectrum
from lib import constants

import numpy as np

import warnings
import dynesty
import scipy.stats
import scipy.linalg
import scipy.special

import sys

def get_module(vtfilter):
    return sys.modules[vtfilter.__class__.__module__]