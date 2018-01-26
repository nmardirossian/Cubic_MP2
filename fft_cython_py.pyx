import cython
cimport cython
import numpy
cimport numpy

cdef extern void getK1_c(int dim, long int ngs, double* gfunc, double* gbarfunc, double* f2func, double* coulGsmall)
cdef extern void getJ_c(int dim, long int ngs, double* ffunc, double* f2func, double* coulGsmall)

@cython.boundscheck(False)
@cython.wraparound(False)
def getK1(int dim, long int ngs, double[:,:] gfunc, double[:,:] gbarfunc, double[:,:] f2func, double[:] coulGsmall):

    getK1_c(dim, ngs, &gfunc[0,0], &gbarfunc[0,0], &f2func[0,0], &coulGsmall[0])

    return None

def getJ(int dim, long int ngs, double[:,:] ffunc, double[:,:] f2func, double[:] coulGsmall):

    getJ_c(dim, ngs, &ffunc[0,0], &f2func[0,0], &coulGsmall[0])

    return None
