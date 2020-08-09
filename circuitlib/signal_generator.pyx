#cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport M_PI
from libc.math cimport sin as csin

cdef DC(double t, double period, double mod, double x_offset, double y_offset):
    pass

cdef sin(double t, double period, double mod, double x_offset, double y_offset):
    return csin

cdef double sawtooth(double t, double period, double mod, double x_offset, double y_offset) nogil:
    cdef double tmod = t % (2 * M_PI)
    if tmod < mod * 2 * M_PI:
        return tmod / (M_PI * mod) - 1
    else:
        return (M_PI * (mod + 1) - tmod) / (M_PI * (1 - mod))

cdef double square(double t, double period, double mod, double x_offset, double y_offset) nogil:
    cdef double tmod = x % (2 * M_PI)
    if tmod <= M_PI:
        return 1.0
    else:
        return -1.0
