#pragma once

#include <omp.h>
#include <cmath>

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include "fourier-methods.h"
#include "nse2d.h"
#include "nse-out2d.h"
#include "pois2d-fft.h"

using namespace nse;

// --- main definitions --- //
#define OPENMP_CORES		1		// number of OpenMP cores to use

// - data type precision: float[double] //
#define Real double

// --- computational domain --- //
const Real domain_length = (Real) 2.0 * (Real)M_PI;        // - [x]
const Real domain_width = (Real) 2.0 * (Real)M_PI;         // - [y]

const int domain_nx = 64, domain_ny = 64;		// # of grid cells in -[x,y] directions

const int domain_gcx = 2, domain_gcy = 2, domain_gcz = 2;

const Real print_xmin = (Real) 0.0, print_xmax = domain_length;
const Real print_ymin = (Real) 0.0, print_ymax = domain_width;

uniGrid2d< Real > grid;

void set_ones_inside(Real* w, const uniGrid2d< Real >& grid)
{
    int i, j, idx;

    null(w, grid.size);
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++) 
    {
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) 
        {
            w[idx] = (Real) 1.0;    
		}
	}
}

void get_random_field(Real* w, const uniGrid2d< Real >& grid)
{
    int i, j, idx;

    null(w, grid.size);
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++) 
    {
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) 
        {
            w[idx] = uniform_distribution((Real)0.0, (Real)1.0);    
		}
	}
}

Real error_inside(Real* w1, Real* w2, const uniGrid2d< Real >& grid)
{
    int i, j, idx;

    Real err_l = (Real)0.0;
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++) 
    {
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) 
        {
            err_l += std::fabs(w1[idx] - w2[idx]);
		}
	}

    return mpi_allreduce(err_l, MPI_SUM) / ((grid.mpi_nx-2*grid.gcx) * (grid.mpi_ny-2*grid.gcy));
}

Real error_relative(Real* input, Real* target, const uniGrid2d< Real >& grid)
{
    int i, j, idx;

    Real err_l = (Real)0.0;
    Real norm_l = (Real)0.0;
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++) 
    {
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) 
        {
            err_l += std::fabs(input[idx] - target[idx]);
            norm_l += std::fabs(target[idx]);
		}
	}

    return mpi_allreduce(err_l, MPI_SUM) / mpi_allreduce(norm_l, MPI_SUM);
}

Real correlation(Real* x, Real* y, const uniGrid2d< Real >& grid)
{
    int i, j, idx;
    
    Real mean_x = 0.0, mean_y = 0.0, mean_x2 = 0.0, mean_y2 = 0.0, mean_xy = 0.0;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++) 
    {
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) 
        {
            mean_x += x[idx];
            mean_y += y[idx];
            mean_x2 += x[idx] * x[idx];
            mean_y2 += y[idx] * y[idx];
            mean_xy += x[idx] * y[idx];
		}
	}

    mean_x = mpi_allreduce(mean_x, MPI_SUM);
    mean_y = mpi_allreduce(mean_y, MPI_SUM);
    mean_x2 = mpi_allreduce(mean_x2, MPI_SUM);
    mean_y2 = mpi_allreduce(mean_y2, MPI_SUM);
    mean_xy = mpi_allreduce(mean_xy, MPI_SUM);

    return (mean_xy - mean_x * mean_y) / std::sqrt((mean_x2 - mean_x * mean_x) * (mean_y2 - mean_y * mean_y));
}

void pfield (Real* model_field)
{
	write_tecplot("-debug_field-.plt", 0, model_field, "ly", 
				print_xmin, print_xmax, print_ymin, 
				print_ymax,grid, (Real)0.0);
}