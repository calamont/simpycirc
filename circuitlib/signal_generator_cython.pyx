#cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport M_PI
from libc.math cimport sin as csin

cdef double DC(double t, double value, double period, double mod, double x_offset, double y_offset) nogil:
    return value

cdef double sin(double t, double value, double period, double mod, double x_offset, double y_offset) nogil:
    return value * csin((t - x_offset) * 2 * M_PI / period) + y_offset

cdef double sawtooth(double t, double value, double period, double mod, double x_offset, double y_offset) nogil:
    cdef double tmod = t % (2 * M_PI)
    if tmod < mod * 2 * M_PI:
        return (value * tmod / (M_PI * mod) - 1) + y_offset
    else:
        return (value * (M_PI * (mod + 1) - tmod) / (M_PI * (1 - mod))) + y_offset

cdef double square(double t, double value, double period, double mod, double x_offset, double y_offset) nogil:
    cdef double tmod = t % (2 * M_PI)
    if tmod <= M_PI:
        return value
    else:
        return -value
