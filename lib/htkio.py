"""Adapted from https://github.com/lemnzhou/htkIO/blob/master/htkIO.py"""
import struct
import numpy as np

def htkread(filename):
    '''
    nframe -- frame number
    frate -- sample ratio
    ndim -- feature dimension
    feakind -- fea kind
    '''
    fid = open(filename, 'rb')
    readbytes = fid.read()
    fid.close()
    nframe = readbytes[0:4]
    #unpack return a tuple,whether it's necessarily to reversed the byte array depend on your machine.
    nframe,= struct.unpack('i',bytes(reversed(nframe)))
    frate  = readbytes[4:8]
    frate, = struct.unpack('i',bytes(reversed(frate)))
    ndim  = readbytes[8:10]
    ndim, = struct.unpack('h',bytes(reversed(ndim)))
    ndim /= 4
    ndim = int(ndim)
    nframe = nframe
    data = np.zeros((ndim,nframe))
    feakind = readbytes[10:12]
    feakind = struct.unpack('h',bytes(reversed(feakind)))
    feakind = int(feakind[0])
    startIndex = 12
    for i in range(nframe):
        for j in range(ndim):
            value = readbytes[startIndex:startIndex+4]
            value, = struct.unpack('f',bytes(reversed(value)))
            data[j][i] = value
            startIndex += 4
    return [data,frate,feakind]