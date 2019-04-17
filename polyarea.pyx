# polyarea.pyx
# cython code to calculate the areas of a voronoi polygon
# to speed up the visualization code
# Author PKR, UPENN March 2019

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs

from scipy import spatial as sp

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# define function for pulling out areas
def PolyArea(const double[:,:] Pos):
    vor = sp.Voronoi(Pos,qhull_options='Qbb Qc Qx')
    cdef int nPoints = int(Pos.shape[0])
    cdef double[:] Area = np.zeros(nPoints,dtype=np.float64)
    cdef double[:,:] X = np.float64(vor.vertices)
    cdef int counter, index
    for counter, index in enumerate(vor.point_region):
        vertIndex = vor.regions[index]
        getAreas(X,vertIndex,Area,counter)
    return np.asarray(Area)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# define 2D Polygon area calculation function
cdef int getAreas(double[:,:] X, list vertIndex,double[:] Area, int Adex):
    cdef double tempArea = 0
    cdef double half = 0.5
    cdef int counter
    cdef int value
    cdef int ncounter, nindex, cindex
    cdef int arraylen = len(vertIndex)
    for counter in range(0,arraylen):
        ncounter = int((counter+1)%arraylen)
        cindex = int(vertIndex[counter])
        nindex = int(vertIndex[ncounter])
        tempArea += ( (X[nindex,0]+X[cindex,0]) *
            (X[cindex,1]-X[nindex,1]) )
    tempArea = half*fabs(tempArea)
    Area[Adex] = tempArea 
    return 0
