"""
skais_mapper.raytrace module:
Optimized raytracing functions for astrophysical SPH and mesh-hybrid simulations

@author: phdenzel
Note: Adapted from https://github.com/franciscovillaescusa/Pylians3
"""
from scipy.integrate import quad
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, M_PI, sin, cos, floor, fabs
from libc.stdio cimport printf
from libc.time cimport time_t, time, difftime

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void PCS(np.float32_t[:, :] pos, np.float32_t[:, :, :] number, float box_size):
    """Compute the density field of a cubic distribution of particles.

    Args:
        pos: Particle 3D/2D position array (contiguous)
        number: Density field array (contiguous)
        box_size: Size of the region (edge length)
    """
    cdef int axis, dims, minimum, j, l, m, n, coord
    cdef long i, particles
    cdef float inv_cell_size, dist, diff
    cdef float C[3][4]
    cdef int index[3][4]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0]
    coord = pos.shape[1]
    dims = number.shape[0]
    inv_cell_size = dims/box_size

    # define arrays: for 2D set we have C[2, :] = 1.0 and index[2, :] = 0
    for i in range(3):
        for j in range(4):
            C[i][j] = 1.0
            index[i][j] = 0

    # do a loop over all particles
    for i in range(particles):
        # do a loop over the three axes of the particle
        for axis in range(coord):
            dist = pos[i, axis] * inv_cell_size
            minimum = <int>floor(dist - 2.0)
            for j in range(4):  # only 4 cells/dimension can contribute
                index[axis][j] = (minimum + j + 1 + dims) % dims
                diff = fabs(minimum + j+1 - dist)
                if diff<1.0:
                    C[axis][j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:
                    C[axis][j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:
                    C[axis][j] = 0.0

        for l in range(4):
            for m in range(4):
                for n in range(4):
                    number[index[0][l], index[1][m], index[2][n]] += C[0][l]*C[1][m]*C[2][n]


###############################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void voronoi_NGP_2D(
    double[:, ::1] field,
    np.float32_t[:, :] pos,
    float[::1] mass,
    float[::1] radius,
    float x_min, float y_min,
    float box_size,
    long tracers, int r_divisions,
    bint periodic,
    bint verbose=1
):
    """
    Compute the density field in a 2D region from a set of Voronoi
    cells that have masses and radii. Each particle is assumed to be a
    uniform sphere which is associated to the grid itself, divided
    into shells that have the same area. Shells are then associated to
    a number (particles-per-cell) of tracers equally distributed in
    angle. Grid cells are then assigned subparticles according to the
    NGP (nearest grid point) mass assignment scheme.

    Args:
        field: The column density field array (contiguous)
        pos: Particle 3D/2D position array (contiguous)
        mass: Particle mass array (contiguous)
        radius: SPH particle radius array (contiguous)
        x_min: Minimum coordinate along the first axis
        y_min: Minimum coordinate along the second axis
        box_size: Size of the region (edge length)
        tracers: TODO
        r_divisions: TODO
        periodic: If True, periodic boundary conditions are applied
        verbose: Print debug info.
    """
    cdef long i, j, k, particles, dims, count
    cdef double dtheta, angle, R, R1, R2, area, length, V_sphere, norm
    cdef double R_cell, W_cell, x_cell, y_cell
    cdef np.float32_t[:, :] pos_tracer
    cdef np.float32_t[:] w_tracer
    cdef double x, y, w, inv_cell_size
    cdef int index_x, index_y, index_xp, index_xm, index_yp, index_ym
    cdef int theta_divisions
    cdef time_t start
    cdef double duration

    # verbose
    if verbose:
        printf("Computing projected mass of the Voronoi tracers...\n")
    start = time(NULL)
    # find the number of particles, dimensions of the grid
    particles     = pos.shape[0]
    dims          = field.shape[0]
    inv_cell_size = dims * 1.0 / box_size
    # compute the number of particles in each shell and the angle between them
    theta_divisions = tracers // r_divisions
    dtheta          = 2.0 * M_PI / theta_divisions
    V_sphere        = 4.0/3.0 * M_PI * 1.0**3
    # define the arrays with the properties of the tracers; positions and weights
    pos_tracer = np.zeros((tracers, 2), dtype=np.float32)
    w_tracer   = np.zeros(tracers, dtype=np.float32)
    # define and fill the array containing pos_tracer
    count = 0
    for i in range(r_divisions):
        R1 = i * 1.0 / r_divisions
        R2 = (i + 1.0) / r_divisions
        R = 0.5 * (R1 + R2)
        area = M_PI * (R2**2 - R1**2) / theta_divisions
        length = 2.0 * sqrt(1.0**2 - R**2)
        for j in range(theta_divisions):
            angle = 2.0 * M_PI * (j + 0.5) / theta_divisions
            pos_tracer[count, 0] = R * cos(angle)
            pos_tracer[count, 1] = R * sin(angle)
            w_tracer[count] = area * length / V_sphere
            count += 1
    # normalize weights of tracers
    norm = np.sum(w_tracer, dtype=np.float64)
    for i in range(tracers):
        w_tracer[i] = w_tracer[i] / norm
    if periodic:
        for i in range(particles):
            R_cell = radius[i]
            W_cell = mass[i]
            x_cell = pos[i, 0]
            y_cell = pos[i, 1]
            # see if we need to split the particle into tracers or not
            index_xm = <int>((x_cell - R_cell - x_min) * inv_cell_size + 0.5)
            index_xp = <int>((x_cell + R_cell - x_min) * inv_cell_size + 0.5)
            index_ym = <int>((y_cell - R_cell - y_min) * inv_cell_size + 0.5)
            index_yp = <int>((y_cell + R_cell - y_min) * inv_cell_size + 0.5)
            if (index_xm == index_xp) and (index_ym == index_yp):
                index_x = (index_xm + dims) % dims
                index_y = (index_ym + dims) % dims
                field[index_x, index_y] += W_cell
            else:
                for j in range(tracers):
                    x = x_cell + R_cell * pos_tracer[j, 0]
                    y = y_cell + R_cell * pos_tracer[j, 1]
                    w = W_cell * w_tracer[j]
                    index_x = <int>((x - x_min) * inv_cell_size + 0.5)
                    index_y = <int>((y - y_min) * inv_cell_size + 0.5)
                    index_x = (index_x + dims) % dims
                    index_y = (index_y + dims) % dims
                    field[index_x, index_y] += w
    # no boundary conditions
    else:
        for i in range(particles):
            R_cell = radius[i]
            W_cell = mass[i]
            x_cell = pos[i, 0]
            y_cell = pos[i, 1]
            for j in range(tracers):
                x = x_cell + R_cell * pos_tracer[j, 0]
                y = y_cell + R_cell * pos_tracer[j, 1]
                w = W_cell * w_tracer[j]
                index_x = <int>((x - x_min) * inv_cell_size + 0.5)
                index_y = <int>((y - y_min) * inv_cell_size + 0.5)
                if (index_x < 0) or (index_x >= dims):
                    continue
                if (index_y < 0) or (index_y >= dims):
                    continue
                field[index_x, index_y] += w
    if verbose:
        duration = difftime(time(NULL), start)
        printf("Time taken = %.2f seconds\n", duration)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void voronoi_RT_2D(
    double[:, ::1] density,
    float[:, ::1] pos,
    float[::1] mass,
    float[::1] radius,
    float x_min, float y_min,
    float box_size,
    int axis_x, int axis_y,
    bint periodic,
    bint verbose=1
):
    """
    Compute the 2D density field from a set of Voronoi cells in 3D
    that have masses and radii assuming they represent uniform
    spheres.  A cell that intersects with a cell will increase its
    value by the column density of the cell along the sphere. The
    density array contains the column densities in M/(L/h)^2 units if
    the quantities in mass units (M) are given in M/h and length units
    (L) in L/h.

    Args:
        density: The column density field array (contiguous)
        pos: Particle 3D/2D position array (contiguous)
        mass: Particle mass array (contiguous)
        radius: SPH particle radius array (contiguous)
        x_min: Minimum coordinate along the first axis
        y_min: Minimum coordinate along the second axis
        box_size: Size of the region (edge length)
        axis_x: Coordinate projected along the x-axis [x=0, y=1, z=2]
        axis_x: Coordinate projected along the y-axis [x=0, y=1, z=2]
        periodic: If True, periodic boundary conditions are applied
        verbose: Print debug info.
    """
    cdef long particles, i
    cdef int dims, index_x, index_y, index_R, ii, jj, i_cell, j_cell
    cdef float x, y, rho, cell_size, inv_cell_size, radius2
    cdef float dist2, dist2_x
    cdef time_t start
    cdef double duration
    start = time(NULL)

    if verbose:
        printf("Computing column densities of the particles...\n")
    # find the number of particles and the dimensions of the grid
    particles = pos.shape[0]
    dims      = density.shape[0]
    # define cell size and the inverse of the cell size
    cell_size     = box_size * 1.0 / dims
    inv_cell_size = dims * 1.0 / box_size
    if periodic:
        for i in range(particles):
            # find the density of the particle and the square of its radius
            rho     = 3.0 * mass[i] / (4.0 * M_PI * radius[i]**3)  # h^2 M/L^3
            radius2 = radius[i]**2  # (L/h)^2
            # find cell where the particle center is and its radius in cell units
            index_x = <int>((pos[i, axis_x] - x_min) * inv_cell_size)
            index_y = <int>((pos[i, axis_y] - y_min) * inv_cell_size)
            index_R = <int>(radius[i] * inv_cell_size) + 1
            # loop over the cells that contribute in the x-direction
            for ii in range(-index_R, index_R+1):
                x       = (index_x + ii) * cell_size + x_min
                i_cell  = ((index_x + ii + dims) % dims)
                dist2_x = (x - pos[i, axis_x])**2
                # loop over the cells that contribute in the y-direction
                for jj in range(-index_R, index_R+1):
                    y      = (index_y + jj) * cell_size + y_min
                    j_cell = ((index_y + jj + dims) % dims)
                    dist2 = dist2_x + (y - pos[i, axis_y])**2
                    if dist2 < radius2:
                        density[i_cell, j_cell] += 2.0 * rho * sqrt(radius2 - dist2)
    # no boundary conditions
    else:
        for i in range(particles):
            # find the density of the particle and the square of its radius
            rho     = 3.0 * mass[i] / (4.0 * M_PI * radius[i]**3)  # h^2 M/L^3
            radius2 = radius[i]**2  # (L/h)^2
            # find cell where the particle center is and its radius in cell units
            index_x = <int>((pos[i, axis_x] - x_min) * inv_cell_size)
            index_y = <int>((pos[i, axis_y] - y_min) * inv_cell_size)
            index_R = <int>(radius[i] * inv_cell_size) + 1
            # contribution in the x-direction
            for ii in range(-index_R, index_R+1):
                i_cell = index_x + ii
                if i_cell >= 0 and i_cell < dims:
                    x = i_cell * cell_size + x_min
                    dist2_x = (x - pos[i, axis_x])**2 
                else:
                    continue        
                # contribution in the y-direction
                for jj in range(-index_R, index_R+1):
                    j_cell = index_y + jj
                    if j_cell >= 0 and j_cell < dims:
                        y = j_cell * cell_size + y_min
                    else:
                        continue
                    dist2 = dist2_x + (y - pos[i, axis_y])**2
                    if dist2 < radius2:
                        density[i_cell, j_cell] += 2.0 * rho * sqrt(radius2 - dist2)

    if verbose:
        duration = difftime(time(NULL), start)
        printf("Time taken = %.2f seconds\n", duration)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float kernel_SPH(float r, float R):
    """The SPH kernel function."""
    cdef float u, prefact
    u = r / R
    prefact = 8.0 / (M_PI * R**3)
    if u < 0.5:
        return prefact * (1 + (6.0 * u - 6.0) * u**2)
    elif u <= 1.0:
        return prefact * 2.0 * (1.0 - u)**3
    else:
        return 0.0

    
# def py_kernel_SPH(r, R):
#     """
#     The SPH kernel function in pure Python code
#     """
#     u = r/R
#     prefact = 8.0 / (np.pi * R**3)
#     if u < 0.5:
#         return prefact * (1 + (6*u - 6) * u**2)
#     elif u <= 1.0:
#         return prefact * 2.0 * (1.0 - u)**3
#     else:
#         return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float integrand(float x, float b2):
    """Integral kernel for scipy.integrate.quad."""
    cdef float r
    r = sqrt(b2 + x**2)
    return kernel_SPH(r, 1.0)


# def py_integrand(x, b2):
#     r = sqrt(b2 + x**2)
#     return py_kernel_SPH(r, 1.0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef NHI_table(int bins):
    """Compute the integral of the SPH kernel.

    \int_{0}^{lmax} W(r) dl, where b^2 + l^2 = r^2 (b is the impact parameter).

    Args:
        bins (int): bins
    """
    # arrays with impact parameter^2 and the column densities
    cdef Py_ssize_t i
    cdef float b2, lmax, I, dI
    cdef np.ndarray[np.float64_t, ndim=1] b2s = np.linspace(0, 1, bins, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] NHI = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        b2 = b2s[i]
        if b2 == 1.0:
            continue
        lmax = sqrt(1.0 - b2)
        I, dI = quad(integrand, 0.0, lmax, args=(b2,), epsabs=1e-12, epsrel=1e-12)
        NHI[i] = 2.0 * I
    return b2s, NHI

# # This function computes the integral of the SPH kernel
# def py_NHI_table(bins):
#     """
#     Compute the integral of the SPH kernel
#     \int_{0}^{lmax} W(r) dl, where b^2 + l^2 = r^2 (b is the impact parameter).

#     Args:
#       bins (int):
#         bins
#     """
#     # arrays with impact parameter^2 and the column densities
#     b2s = np.linspace(0, 1, bins, dtype=np.float64)
#     NHI = np.zeros(bins,          dtype=np.float64)
#     for i, b2 in enumerate(b2s):
#         if b2 == 1.0:
#             continue
#         lmax = sqrt(1.0 - b2)
#         I, dI = quad(integrand, 0.0, lmax,
#                     args=(b2,),
#                     epsabs=1e-12,
#                     epsrel=1e-12)
#         NHI[i] = 2.0*I

#     return b2s, NHI

###############################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void SPH_RT_2D(double[:,::1] density,
                     float[:,::1] pos,
                     float[::1] mass,
                     float[::1] radius,
                     float x_min, float y_min,
                     int axis_x, int axis_y,
                     float box_size,
                     bint periodic,
                     bint verbose=1):
    """
    Compute the 2D density field from a set of SPH particles that have
    masses and radii assuming they represent uniform spheres.  A cell
    that intersects with a cell will increase its value by the column
    density of the cell along the sphere. The density array contains
    the column densities in M/(L/h)^2 units if the quantities in mass
    units (M) are given in M/h and length units (L) in L/h.

    Args:
        density: The column density field array (contiguous)
        pos: Particle 3D/2D position array (contiguous)
        mass: Particle mass array (contiguous)
        radius: SPH particle radius array (contiguous)
        x_min: Minimum coordinate along the first axis
        y_min: Minimum coordinate along the second axis
        axis_x: Coordinate projected along the x-axis [x=0, y=1, z=2]
        axis_x: Coordinate projected along the y-axis [x=0, y=1, z=2]
        box_size: Size of the region (edge length)
        periodic: If True, periodic boundary conditions are applied
        verbose: Print debug info
    """
    cdef long particles, i, num, bins = 1000
    cdef int dims, index_x, index_y, index_R, ii, jj, i_cell, j_cell
    cdef float x, y, cell_size, inv_cell_size, radius2
    cdef float dist2, dist2_x, mass_part
    cdef double[::1] b2, NHI
    cdef time_t start
    cdef double duration
    start = time(NULL)
    if verbose:
        printf("Computing column densities of the particles...\n")
    # find the number of particles and the dimensions of the grid
    particles = pos.shape[0]
    dims      = density.shape[0]
    # define cell size and the inverse of the cell size
    cell_size     = box_size * 1.0 / dims
    inv_cell_size = dims * 1.0 / box_size
    # compute the normalized column density for normalized radii^2
    b2, NHI = NHI_table(bins)
    # periodic boundary conditions
    if periodic:
        for i in range(particles):
            # find the particle mass and the square of its radius
            radius2   = radius[i]**2  # (L/h)^2
            mass_part = mass[i]
            # find cell where the particle center is and its radius in cell units
            index_x = <int>((pos[i, axis_x] - x_min) * inv_cell_size)
            index_y = <int>((pos[i, axis_y] - y_min) * inv_cell_size)
            index_R = <int>(radius[i] * inv_cell_size) + 1
            # do a loop over the cells that contribute in the x-direction
            for ii in range(-index_R, index_R+1):
                x       = (index_x + ii) * cell_size + x_min
                i_cell  = ((index_x + ii + dims) % dims)
                dist2_x = (x - pos[i, axis_x])**2
                # do a loop over the cells that contribute in the y-direction
                for jj in range(-index_R, index_R+1):
                    y      = (index_y + jj) * cell_size + y_min
                    j_cell = ((index_y + jj + dims) % dims)
                    dist2 = dist2_x + (y - pos[i, axis_y])**2
                    if dist2 < radius2:
                        num = <int>(dist2 / radius2) * bins
                        density[i_cell, j_cell] += (mass_part * NHI[num])
    # no periodic boundary conditions
    else:
        for i in range(particles):
            # find the particle mass and the square of its radius
            radius2   = radius[i]**2  # (L/h)^2
            mass_part = mass[i]
            # find cell where the particle center is and its radius in cell units
            index_x = <int>((pos[i, axis_x] - x_min) * inv_cell_size)
            index_y = <int>((pos[i, axis_y] - y_min) * inv_cell_size)
            index_R = <int>(radius[i] * inv_cell_size) + 1
            # do a loop over the cells that contribute in the x-direction
            for ii in range(-index_R, index_R+1):
                i_cell = index_x + ii
                if (i_cell >= 0) and (i_cell < dims):
                    x = i_cell * cell_size + x_min
                else:
                    continue
                dist2_x = (x - pos[i, axis_x])**2
                # do a loop over the cells that contribute in the y-direction
                for jj in range(-index_R, index_R+1):
                    j_cell = index_y + jj
                    if j_cell >= 0 and j_cell < dims:
                        y = j_cell * cell_size + y_min
                    else:
                        continue
                    dist2 = dist2_x + (y - pos[i, axis_y])**2
                    if dist2 < radius2:
                        num = <int>(dist2 / radius2) * bins
                        density[i_cell, j_cell] += (mass_part * NHI[num])
    if verbose:
        duration = difftime(time(NULL), start)
        printf('Time taken = %.2f seconds', duration)
