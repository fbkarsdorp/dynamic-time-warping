import numpy as np
cimport numpy as np
np.import_array()
cimport cython
from libc.math cimport fabs, sqrt, ceil
from libc.stdlib cimport *
from cdtw cimport *


ctypedef np.double_t DTYPE_t


cdef class Distance:
    cpdef double dist(self, double a, double b) except *:
        return 0

cdef class Euclidean(Distance):
    cpdef double dist(self, double a, double b) except *:
        return (a - b) ** 2

cdef class SquaredEuclidean(Distance):
    cpdef double dist(self, double a, double b) except *:
        return sqrt((a - b) ** 2)

cdef class Manhattan(Distance):
    cpdef double dist(self, double a, double b) except *:
        return fabs(a - b)


cdef retrace_path(int n, int m, np.ndarray[DTYPE_t, ndim=2] dtw):
    cdef Path p 
    cdef np.ndarray[np.int_t, ndim=1] px_arr
    cdef np.ndarray[np.int_t, ndim=1] py_arr

    path(<double *> dtw.data, n, m, -1, -1, &p)
    px_arr = np.empty(p.k, dtype=np.int)
    py_arr = np.empty(p.k, dtype=np.int)
    for i in range(p.k):
        px_arr[i] = p.px[i]
        py_arr[i] = p.py[i]
    free (p.px)
    free (p.py)
    return px_arr, py_arr


@cython.boundscheck(False)
cdef void fill_cost_matrix_unconstrained(np.ndarray[DTYPE_t, ndim=2] dtw,
                                         np.ndarray[DTYPE_t, ndim=1] s,
                                         np.ndarray[DTYPE_t, ndim=1] t,
                                         int n, int m,
                                         double step_pattern,
                                         Distance cost_fn):
    
    cdef unsigned int i, j
    cdef DTYPE_t cost, insert, delete, replace

    dtw[0, 0] = cost_fn.dist(s[0], t[0])
    for i in range(1, n):
        dtw[i, 0] = cost_fn.dist(s[i], t[0]) + dtw[i-1, 0]

    for j in range(1, m):
        dtw[0, j] = cost_fn.dist(s[0], t[j]) + dtw[0, j-1]

    for i in range(1, n):
        for j in range(1, m):
            cost = cost_fn.dist(s[i], t[j])
            insert = cost + dtw[i-1, j]
            delete = cost + dtw[i, j-1]
            replace = cost * step_pattern + dtw[i-1, j-1]
            dtw[i, j] = min3(insert, delete, replace)

# slanted band constraint as in R's dtw package
@cython.cdivision(True)
cdef inline int slanted_band_constraint(int i, int j, double slant, int w):
    return fabs((j+1) - ((i+1) * slant)) <= w

@cython.boundscheck(False)
cdef void fill_cost_matrix_with_slanted_band_constraint(np.ndarray[DTYPE_t, ndim=2] dtw,
                                                        np.ndarray[DTYPE_t, ndim=1] s,
                                                        np.ndarray[DTYPE_t, ndim=1] t,
                                                        int n, int m,
                                                        double step_pattern,
                                                        Distance cost_fn,
                                                        int window):

    cdef unsigned int i, j
    cdef DTYPE_t cost, insert, delete, replace
    cdef double slant = float(m) / float(n)

    dtw.fill(np.inf)
    dtw[0, 0] = cost_fn.dist(s[0], t[0])

    for i in range(1, n):
        if slanted_band_constraint(i, 0, slant, window):
            dtw[i, 0] = cost_fn.dist(s[i], t[0]) + dtw[i-1, 0]

    for j in range(1, m):
        if slanted_band_constraint(0, j, slant, window):
            dtw[0, j] = cost_fn.dist(s[0], t[j]) + dtw[0, j-1]

    for i in range(1, n):
        for j in range(1, m):
            if slanted_band_constraint(i, j, slant, window):
                cost = cost_fn.dist(s[i], t[j])
                delete = cost + dtw[i-1, j]
                insert = cost + dtw[i, j-1]
                replace = cost * step_pattern + dtw[i-1, j-1]
                dtw[i, j] = min3(delete, replace, insert)



def dtw_distance(np.ndarray[DTYPE_t, ndim=1] s, np.ndarray[DTYPE_t, ndim=1] t, 
                 int step_pattern=1, str metric='euclidean', constraint='None',
                 normalized=False, int window=0, dist_only=True):
    """Implementation of the Dynamic Time Warping algorithm. The implementation 
    expects two numpy arrays as input and returns the DTW distance. This distance can 
    be normzalized in case the step patterns equals 2 (d / (m + n)). Two metrics are 
    supported: euclidean distance and manhattan distance. It is also possible to 
    contrain the band through the matrix using a slanted band constraint with a 
    particular window size k.
    """

    cdef double dist
    cdef int n = s.shape[0]
    cdef int m = t.shape[0]
    
    transpose_cost = False
    if n < m:
        transpose_cost = True
        n, m = m, n
        s, t = t, s 

    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.empty((n, m), dtype=np.float)

    cdef Distance cost_fn
    if metric == 'euclidean':
        cost_fn = Euclidean()
    elif metric == 'manhattan':
        cost_fn = Manhattan()
    elif metric == 'seuclidean':
        cost_fn = SquaredEuclidean()
    else:
        raise ValueError("Metric '%s' is not supported." % metric)
    
    if constraint is None or constraint.lower() == 'none':
        fill_cost_matrix_unconstrained(dtw, s, t, n, m, step_pattern, cost_fn)
    elif constraint == 'slanted_band':
        if window < 0:
            raise ValueError("Window size must be greater than or equal to 0.")
        fill_cost_matrix_with_slanted_band_constraint(dtw, s, t, n, m, step_pattern, cost_fn, window)
    else:
        raise ValueError("Constraint '%s' is not supported." % constraint)

    if transpose_cost:
        n, m = m, n
        dtw = dtw.T

    dist = dtw[n-1, m-1]
    if normalized and step_pattern == 2:
        dist = dist / (n + m)
    if dist_only:
        return dist
    px_arr, py_arr = retrace_path(n, m, dtw)
    return dist, dtw, (px_arr, py_arr)

def dtw_slanted_band(s, t, window, metric='euclidean', normalized=False, step_pattern=1):
    """DTW constrained by slanted band of width 2k+1. The warping path is 
    constrained by |i*len(x)/len(k)-j| <= k."""

    return dtw_distance(s, t, step_pattern, metric, 'slanted_band', normalized, window)
