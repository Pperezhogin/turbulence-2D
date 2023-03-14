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

const int domain_nx = 64, domain_ny = 128;		// # of grid cells in -[x,y] directions

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

void pfield (Real* model_field)
{
	write_tecplot("-debug_field-.plt", 0, model_field, "ly", 
				print_xmin, print_xmax, print_ymin, 
				print_ymax,grid, (Real)0.0);
}