#pragma once

#include <omp.h>
#include <cmath>

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include "dynamic-model.h"
#include "fourier-methods.h"
#include "nse2d.h"
#include "nse-out2d.h"
#include "pois2d-fft.h"

using namespace nse;
using namespace std;

// --- main definitions --- //
#define OPENMP_CORES		1		// number of OpenMP cores to use

// - data type precision: float[double] //
#define Real double

// --- computational domain --- //
const Real domain_length = (Real) 2.0 * (Real)M_PI;        // - [x]
const Real domain_width = (Real) 2.0 * (Real)M_PI;         // - [y]

const int domain_nx = 4096, domain_ny = 4096;		// # of grid cells in -[x,y] directions

const int domain_gcx = 2, domain_gcy = 2, domain_gcz = 2;

const Real print_xmin = (Real) 0.0, print_xmax = domain_length;
const Real print_ymin = (Real) 0.0, print_ymax = domain_width;

uniGrid2d< Real > grid;

Real *Psi;

dynamic_model< Real > dyn_model_lap_128;
dynamic_model< Real > dyn_model_bilap_128;
dynamic_model< Real > dyn_model_ssm_bilap_128;
dynamic_model< Real > dyn_model_ssm_bilap_keb_128;

dynamic_model< Real > dyn_model_lap_256;
dynamic_model< Real > dyn_model_bilap_256;
dynamic_model< Real > dyn_model_ssm_bilap_256;
dynamic_model< Real > dyn_model_ssm_bilap_keb_256;

dynamic_model< Real > dyn_model_lap_512;
dynamic_model< Real > dyn_model_bilap_512;
dynamic_model< Real > dyn_model_ssm_bilap_512;
dynamic_model< Real > dyn_model_ssm_bilap_keb_512;

void psi_bc(Real* Psi, const uniGrid2d< Real >& grid)
{
        const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(Psi,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}

void w_bc(Real* w, const uniGrid2d< Real >& grid)
{
        const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(w,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);

}

void velocity_bc(Real* U, Real* V, const uniGrid2d< Real >& grid)
{
	const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(U, V,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}