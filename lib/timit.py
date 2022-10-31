"""Utilities for TIMIT and TIMIT-derived VTRFormants datasets"""
from init import __datadir__
from lib import praat
from dgf.prior import period
from lib import htkio

import pathlib
import parselmouth
import soundfile
import warnings
import datatable

TIMIT = __datadir__('TIMIT')
VTRFORMANTS = __datadir__('VTRFormants')

VOWEL_LABELS = [
    'iy',  
    'ih',  
    'eh',  
    'ey',  
    'ae',  
    'aa',  
    'aw',  
    'ay',  
    'ah',  
    'ao',  
    'oy',  
    'ow',  
    'uh',  
    'uw',  
    'ux',  
    'er',  
    'ax',  
    'ix',  
    'axr', 
    'ax-h'
]

PHN_COLUMNS = ("BEGIN_SAMPLE", "END_SAMPLE", "PHONETIC_LABEL")

FRAME_LENGTH_SEC = 10/1000. # 10 msec

def training_set(path):
    return path / "TRAIN"

def test_set(path):
    return path / "TEST"

def getsampleid(file, root):
    path = file.relative_to(root)
    id = path.parent / path.stem
    return id

def corresponding_timit_id(fb_file, root, timit_root):
    id = getsampleid(fb_file, root)
    return timit_root / id

def corresponding_wav(fb_file, root, timit_root):
    return corresponding_timit_id(fb_file, root, timit_root).with_suffix(".WAV")

def corresponding_phn(fb_file, root, timit_root):
    return corresponding_timit_id(fb_file, root, timit_root).with_suffix(".PHN")

def yield_file_triples(vtr_root, timit_root):
    """Yield a triple of FB (VTRFormants) and PHN, WAV (TIMIT) files"""
    for fb_file in vtr_root.rglob("*.FB"):
        wav_file = corresponding_wav(fb_file, vtr_root, timit_root)
        if not wav_file.is_file():
            warnings.warn(f"No WAV file at {wav_file} found for {fb_file}; skipping")
            continue
        phn_file = corresponding_phn(fb_file, vtr_root, timit_root)
        if not phn_file.is_file():
            warnings.warn(f"No PHN file at {phn_file} found for {fb_file}; skipping")
            continue
        yield fb_file, phn_file, wav_file

def read_fb_file(fb_file):
    """Return F1-4 and B1-4 tracks in kHz at 10 msec frames"""
    FB, _, _ = htkio.htkread(fb_file)
    return FB.T # (rows = frame indices, cols 0-3 = F1-4, cols 4-7 = B1-B4)

def read_phn_file(phn_file):
    return datatable.fread(phn_file, sep=" ", columns=PHN_COLUMNS)

def read_wav_file(wav_file):
    d, fs = soundfile.read(wav_file)
    return d, fs # Hz