# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def angdistcut_cython_par(double[:,::1] vec_obj, double[:,::1] vec_ps, double cos_maxsep, int num_threads):
    """
    added extra compiler flags (NB: now specified at top of file!)
    also added a new argument for the number of threads
    """
    
    cdef:
        int nps = vec_ps.shape[0]
        int nobj = vec_obj.shape[0]
        int dim = vec_obj.shape[1]
        #use int array instead of bool
        int[:] found = np.zeros(nobj, np.int32)
        int i,j
        double cos

    #every object is independent, so can parallelize the outer loop easily
    # need to release the GIL for that (see above)
    with nogil, parallel(num_threads=num_threads):
        #extra keyword arguments of prange control OMP settings (can fine-tune this based on our problem)
        for i in prange(nobj): #,schedule='static',chunksize=1):
            for j in range(nps):
                cos = (vec_obj[i,0]*vec_ps[j,0] + 
                       vec_obj[i,1]*vec_ps[j,1] + 
                       vec_obj[i,2]*vec_ps[j,2])
                if cos > cos_maxsep:
                    found[i] = 1
                    break
                
    return np.flatnonzero(np.asarray(found))