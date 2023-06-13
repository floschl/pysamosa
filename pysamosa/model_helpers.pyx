import os
import pickle

import cython
import numpy as np


def load_samosa_luts():
    with open(os.path.dirname(__file__) + f'/luts_samosa.pickle', 'rb') as handle:
        return pickle.load(handle)

from libc.math cimport M_PI, sqrt


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef get_f_from_lut(double[:] xi, double[:,:] f_lut, int degree):
    ''' Reads the F0/F1 values from the LUT as described in DPM doc v2.5.2 4.2.3 SL7

    :param xi: the value to look up
    :param f_lut: either LUT_F0 or LUT_F1
    :param degree: 0: for LUT_F0, 1: LUT_F1
    :return: the looked up value
    '''
    farr = np.zeros(xi.shape[0])
    cdef double[:] farr_view = farr

    set_lut_val(xi, f_lut, degree, farr_view)

    return farr

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double[:] set_lut_val(double[:] xi, double[:,:] f_lut, int degree, double[:] farr_view) nogil:

    cdef double[:] f_lut_x = f_lut[:, 0]
    cdef double[:] f_lut_y = f_lut[:, 1]

    cdef double f = 0.0
    cdef int j = 0

    for i in range(xi.shape[0]):
        if xi[i] >= f_lut_x[0] and xi[i] <= f_lut_x[-1]:
            j = <int>((f_lut_x.shape[0] - 1) * (xi[i] - f_lut_x[0]) / (f_lut_x[-1] - f_lut_x[0]))
            f = (xi[i] - f_lut_x[j]) * (f_lut_y[j + 1] - f_lut_y[j]) / (f_lut_x[j + 1] - f_lut_x[j]) + f_lut_y[j]
        elif xi[i] < f_lut_x[0]:
            f = 0.0
        else:
            if degree == 0:
                f = 1.0 / 2.0 * sqrt(2.0 * M_PI / xi[i])
            elif degree == 1:
                f = 1.0 / 4.0 * sqrt(2.0 * M_PI / (xi[i] ** 3))

        farr_view[i] = f

    return farr_view