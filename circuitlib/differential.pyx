import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free, PyMem_RawRealloc
from libc.math cimport sin, fabs, isnan, M_PI
from scipy.linalg.cython_lapack cimport dgesvx, dgesv
from libc.stdint cimport uintptr_t

cpdef DEA_solve(double[::1,:] a1, double[::1,:] a2, double[::1] init, double end, double h_n, comp_dict):
    """Solves the differential algebriac equation of the circuits transient 
    response.

    Args:
        a1:
        a2:
        init:
        end:
        h_n:
        comp_dict:
    """
    # Populate struct with component values
    cdef size_t n_components = len(comp_dict)
    cdef comp * components = <comp *>malloc(n_components * sizeof(comp))
    if not (components):
        raise MemoryError()
    parse_components(comp_dict, components)  # parse component dict into struct

    # Assume the solver will complete the simulation in two times the number
    # of steps given by the time span and initial step size.
    cdef int max_success_steps = int(end / h_n), max_steps = max_success_steps * 2

    # Setting up variables for LAPACK solver
    cdef int n=a1.shape[0], nrhs=1, lda=a1.shape[0], ldx=init.shape[0], ldaf=a1.shape[0], ldx_n1=init.shape[0], info
    cdef int *ipiv = <int *>malloc(n * sizeof(int))
    if not ipiv:
        raise MemoryError()

    # Instantiating arrays to hold solutions to the MNA at a given time step
    cdef np.ndarray[double, ndim=2] A = np.empty((n,n), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_n = np.empty((n,2), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_n1 = np.empty((ldx,2), dtype=np.double, order="F")

    for i in range(init.shape[0]):
        x_n[i,0] = init[i]
        x_n[i,1] = init[i]

    # Setting up variables for integration
    cdef double h_n0=h_n, h_n1=h_n, h_            # previous prevous step sizes
    cdef double plte=0.0, error=0.0, alpha, p=1   # error of integration
    cdef int step=-1, succesful_steps=-1, j       # count iterations
    cdef double i_n=0.0, i_n0, i_, v_             # currents/voltages at the present step
    cdef double t=0.0  # time at step

    # Views for arrays to improve speed
    cdef double [::1,:] x_n_view = x_n
    cdef double [::1,:] x_n1_view = x_n1
    cdef double [::1,:] a1_view=a1
    cdef double [::1,:] a2_view=a2
    cdef double [::1,:] A_view=A

    # Arrays to store results of voltages at the selected time steps
    cdef double *time_steps = <double *>PyMem_Malloc(max_success_steps * sizeof(double))
    cdef double *voltages = <double *>PyMem_Malloc(max_success_steps * ldx * sizeof(double))
    if not (time_steps and voltages):
        raise MemoryError()

    try:
        # Iterate through time steps and solve for the currents/voltages passing
        # through the circuit at the _next_ time step until reaching the end
        # of the transient period.
        while t < end:

            step += 1
            # We use Richardson extrapolation, meaning we must solve for the
            # next step twice under different time steps to estimate the local
            # truncation error. With this we can determine if we should accept
            # or increase/decrease the time step.
            for j in range(2): 

                if j==0:
                    h_ = h_n  # use current time step length
                else:
                    # Use the current step plus the previous step length
                    h_ = h_n + h_n0

                # Create MNA stamps and solve for x
                stamp(x_n1_view[:,j], x_n_view[:,j], components, n_components, t, h_, j)
                addDivide(a1_view, a2_view, h_, A_view)
                # TODO: Check info from dgesv to ensure it succesfully completed
                dgesv(&n, &nrhs, &A_view[0,0], &lda, &ipiv[0], &x_n1_view[0,j], &ldx, &info)

            # Calculate error between the two solutions
            alpha = h_n0 / h_n
            plte = 0.0
            for i in range(ldx):
                error = ((x_n1_view[i,0] - x_n1_view[i,1])
                          / ((1.0 + alpha)**(p+1) - 1.0))
                # Take the largest absolute error of all the nodes
                if fabs(error) > plte:
                    plte = fabs(error)

            if (step < 2) or (1e-12 <= plte <= 1e-5):
                # Step succesful. Increment steps and save results.

                update_prev(x_n1_view[:,0], x_n_view[:,0], components, n_components, h_n)
                update_xs(x_n1_view, x_n_view)

                h_n0 = h_n
                t += h_n
                succesful_steps += 1

                time_steps[succesful_steps] = t
                addResults(&voltages[succesful_steps*ldx], x_n_view[:,0])

            elif 0 <= plte < 1e-12:
                # plte uneccesarily small. Increase step size and repeat step.
                h_n *= 2.0
            elif plte > 1e-5:
                # plte too large. Decrease step size and repeat step.
                h_n = h_n / 2.0

            if step > max_steps:  # TODO: raise error if over 50% of steps unsuccesful
                break
            elif succesful_steps+1 == max_success_steps:
                # Have hit length of arrays. Need to reallocate memory to
                # continue simulation.
                max_steps *= 2
                max_success_steps *= 2
                time_steps = <double *>PyMem_RawRealloc(time_steps, max_success_steps * sizeof(double))
                voltages = <double *>PyMem_RawRealloc(voltages, max_success_steps * ldx * sizeof(double))
                if not (time_steps and voltages):
                    raise MemoryError()

        return (np.asarray(<np.float64_t[:succesful_steps]> time_steps),
                np.asarray(<np.float64_t[:succesful_steps, :3]> voltages))

    finally:
        PyMem_Free(ipiv)
        PyMem_Free(components)
        PyMem_Free(time_steps)
        PyMem_Free(voltages)
    
    # TODO: Check if the full time range has been simulated. Perhaps do a sense check
    # at the start to ensure appropriate initial starting step size, then potentially
    # rerun the simulation again if it has failed in such a manner.
#     if info < 0:
#         return info
    
# Create typedef to represent voltage/source signal functions
ctypedef double (*func)(double x, double y) nogil

# TODO: Need to handle group 2 components using `group2_idx`
ctypedef struct comp:
    int type;  # type of component (resistor, capacitor etc.)
    int node1;  # the positive node
    int node2;  # the positive node
    func source;  # source signal (only relevant for V/I sourcses)
    double val1;
    double val2;  # TODO: work out if this is still needed
    double prev[2];  # voltage/current for component in previous time step
 

cdef double square(double x, double y) nogil:

    cdef double tmod = x % (2 * M_PI)
    if tmod <= M_PI:
        return 1.0
    else:
        return -1.0


cdef double sawtooth(double t, double w) nogil:
    cdef double tmod = t % (2 * M_PI)
    if tmod < w * 2 * M_PI:
        return tmod / (M_PI * w) - 1
    else:
        return (M_PI * (w + 1) - tmod) / (M_PI * (1 - w))


cdef parse_components(comp_dict, comp* comp_list):

    for i, val in enumerate(comp_dict.values()):
        comp_list[i].node1 = val["nodes"][0] - 1
        comp_list[i].node2 = val["nodes"][1] - 1
        comp_list[i].prev[0] = 0
        comp_list[i].prev[1] = 0
        comp_list[i].val1 = val["value"]
        comp_list[i].val2 = 0

        if val["type"] == "V":
            comp_list[i].node1 = val["dependent"]
            comp_list[i].type = 0
            comp_list[i].source = sawtooth  # TODO: how are we auto allocating this? Could use big if/elif statement if can't think of anything else.

        if val["type"] == "R":
            comp_list[i].type = 1

        if val["type"] == "C":
            comp_list[i].type = 2

        if val["type"] == "L":
            comp_list[i].type = 3


cdef int stamp(double[::1] x, double[::1] x_prev, comp* c, int n, double t, double step, int it) nogil:

    fillZeros(x)  # remove previous values stored in x

    for i in range(n):

        if c[i].type == 0:  # if voltage source
            x[c[i].node1] += c[i].source(t, 1)

        # TODO: Create a resource in the docs that explains these equations below
        # TODO: does the below work if it is a group 2 capacitor?
        elif c[i].type == 2:  # if capacitor
            if c[i].node1 == -1:  # if node1 connected to ground
                x[c[i].node2] += c[i].prev[it] + (2.0*c[i].val1/step)*x_prev[c[i].node2]
            elif c[i].node2 == -1:  # if node2 connected to ground
                x[c[i].node1] += c[i].prev[it] + (2.0*c[i].val1/step)*x_prev[c[i].node1]
            else:
                x[c[i].node1] += (c[i].prev[it] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))
                x[c[i].node2] -= (c[i].prev[it] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))
        elif c[i].type == 3:  # if inductor
                # TODO: Handle if node connected to ground?
                x[c[i].node1] += (c[i].prev[i] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))
                x[c[i].node2] -= (c[i].prev[i] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))

    return n


cdef void update_prev(double[::1] x, double[::1] x_prev, comp* c, int n, double step) nogil:

    # Iterate through each component and stamp value onto x
    cdef double u = 0, u_prev = 0
    for i in range(n):
        c[i].prev[1] = c[i].prev[0]

        if c[i].type == 2:  # if capacitor
            # TODO: This will get very bloated very quick! Need to think of a better way for this.
            if c[i].node2 == -1:
                u = x[c[i].node1]
                u_prev = x_prev[c[i].node1]
            elif c[i].node1 == -1:
                u = x[c[i].node2]
                u_prev = x_prev[c[i].node2]
            else: 
                u = x[c[i].node1] - x[c[i].node2]
                u_prev = x_prev[c[i].node1] - x_prev[c[i].node2]
                
            c[i].prev[0] = ((2.0*c[i].val1/step)*u - (c[i].prev[1] + ((2.0*c[i].val1/step)*u_prev)))

            

cdef void fillZeros(double[::1] x) nogil:
    cdef int dim=x.shape[0]
    for i in range(dim):
        x[i] = 0
                                                                                   
cdef void update_xs(double[::1,:] x_new, double[::1,:] x) nogil:
    cdef int dim=x.shape[0]
    for i in range(dim):
        x[i,1] = x[i,0]       
        x[i,0] = x_new[i,0]
        

cdef void addResults(double* results, double[::1] x) nogil:
    cdef int dim=x.shape[0]
    for i in range(dim):
        results[i] = x[i]

        
cdef void addDivide(double[::1,:] a1, double[::1,:] a2, double h, double[::1,:] out) nogil:
    cdef int dim1=a1.shape[0], dim2=a1.shape[1], idx, result=0
    for i in range(dim1):
        for j in range(dim2):
            out[i,j] = a1[i,j] + (a2[i,j] / h)


