# -*- coding: utf-8 -*-
def genEmitter(MeanPhot,PSFsigma=1,winsize=7,noiseRMS=1):
    """ function genEmitter Function to generate a single emitter for localization"""

    import numpy as np

    # Generate 5 sigma of random numbers for emission times
    sample = int(MeanPhot + np.round(np.sqrt(MeanPhot)*5))
    rndvals = np.random.rand(sample)
    # Turn random values into exponentially distributed wait times
    times = -np.log(1-rndvals)/MeanPhot
    times = np.cumsum(times)
    numvals = np.sum(times<1)
    # generate an x-y list of photon positions
    x = PSFsigma*np.random.randn(numvals)+winsize/2
    y = PSFsigma*np.random.randn(numvals)+winsize/2
    # convert x-y list into a matrix
    photbins = np.column_stack((x.astype(int),y.astype(int)))
    # create a blank matrix
    im = 0*np.empty([winsize,winsize])
    # loop over each photon and populate the empty matrix with counts
    for ii in range(0,numvals):
        indvar = photbins[ii]
        if indvar[0] < 0 or indvar[0] > winsize-1 or indvar[1] < 0 or indvar[1] > winsize-1:
            continue
        im[indvar[0],indvar[1]]+=1
    # Add read noise
    noyz = noiseRMS*np.random.rand(winsize,winsize)
    noyz[noyz<0] = 0.01
    out = im + noyz
    return out

