#pragma once

#include <omp.h>

#include "unigrid2d.h"

#include "nse2d.h" 
#include "nse-bc2d.h"
#include "nse-out2d.h"

#include "pois2d.h"
#include "pois2d-fft.h"
#include "vecmath.h"

#include "fourier-methods.h"
#include "dynamic-model.h"

#include "time-series.h"
#include "str-com.h"

#include <stdio.h>
#include <cassert>

using namespace nse;

// --- main definitions --- //
#define OPENMP_CORES		1		// number of OpenMP cores to use

//#define DUMP_CONTINUE				// define for startup from dump 

#define POISSON_FFT

//#define DECAYING_TURBULENCE
#define TEST_TAYLOR_GREEN

#define DIFFUSION
//#define DYNAMIC_MODEL

#define NUM        3 // number of arakawa jacobians
#define J1
#define J2
#define J3

// - data type precision: float[double] //
#define Real double

#ifdef DUMP_CONTINUE
const int dump_continue_mark = 11;	// dump control point index
#endif
// ----------------------------------------------------- //

// --- computational domain --- //
const Real domain_length = (Real) 2.0 * (Real)M_PI;        // - [x]
const Real domain_width = (Real) 2.0 * (Real)M_PI;         // - [y]

const int domain_nx = 256, domain_ny = 256;		// # of grid cells in -[x,y] directions

const int domain_gcx = 2, domain_gcy = 2, domain_gcz = 2;

Real begin_time = (Real) 0.0;
const Real end_time = (Real) 10.0;

//spectrum parameters
const Real CFL = (Real) 0.8; //CFL = 1 is a maximum
Real dt;

// balance of energy and enstrophy
s_balance< Real > balance;

// --- fluid parameters --- //

const Real c_Reynolds = (Real) 1000;

// ----------------------------------------------------- //
// --- output parameters --- //
#define OUTPUT_DIR	           "output/" 

// --- dynamic model params --- //
Real tf_width = sqrt(6.0);
Real bf_width = sqrt(6.0);
int viscosity_model = lap_smag;
int averaging_method = lagrangian;
bool mixed_model = true;
int mixed_type = mixed_ssm;
bool negvisc_backscatter = false;
bool reynolds_backscatter = false;
bool adm_model = false;
int adm_order = 10;
int filter_iterations = 1;        // only for clipping
Real lagrangian_time = (Real)10.0; // only for lagrangian averaging

int mpi_rank;
Real abs_error;
Real max_abs_error = (Real)0.2;

const Real print_begin = (Real) -1e-16;
const Real print_dt = (Real) 1.0;

const Real print_xmin = (Real) 0.0, print_xmax = domain_length;
const Real print_ymin = (Real) 0.0, print_ymax = domain_width;

const int ndebug = 30;
// ----------------------------------------------------- //

// --- dump parameters --- //
#define DUMP_DIR							"dump/"

const Real dump_begin = (Real) 100.0;
const Real dump_dt = (Real) 100.0;
// ----------------------------------------------------- //

// --- poisson solver --- //
#ifdef POISSON_FFT
fft_mpi_poisson2d_data< Real > fft_data;
#else
mg_mpi_poisson2d_data< Real > mg_data;
#endif

dynamic_model< Real > dyn_model;

const Real pois_retol = (Real) 1e-4, pois_abstol = (Real) 1e-5;
const int pois_piters = 1;
const int pois_mg_min = 4;			// min grid size in multigrid sequence
int pois_ngrid = 0;					// auto definition: 0 value
const int pois_maxiters = 200;

const int pois_bc_type = c_pois_bc_periodic_xy;

Real poisson_norm;
int poisson_status;
// ----------------------------------------------------- // 

int print_index, dump_index;
Real print_mark, dump_mark;

const int c_seq_max_length = 100 * 1024;
// ----------------------------------------------------- //

// --- main data --- //
uniGrid2d< Real > grid;

Real *U, *V, *Psi, *w;

Real *Psi_rk, *w_rk;

Real *Usol, *Vsol, *Psisol, *wsol;

Real *Uerr, *Verr, *Psierr, *werr;
Real *wim;

Real *rhs;
Real *memory;					// poisson solver additional memory

Real u_max = (Real)1.0, v_max = (Real)1.0;				// maximum velocity values //

// ----------------------------------------------------- //

// --- additional data variables --- //
Real current_time;
int time_index;

double cpu_run_time = (double) 0.0,
cpu_nse_eq_time = (double) 0.0,
cpu_pois_time = (double) 0.0;
// ----------------------------------------------------- //

// - function declarations //
bool model_setup();								// user defined parameter setup
bool model_init();								// model initialization
bool model_print(const char* msg_status);		// print model parameters
void model_clear();								// clear model run data

void init_nse_eq();								// init nse eq. integration
bool advance_nse_eq();							// nse eq. integration step
bool advance_nse_eq_runge_kutta();

bool advance_time();							// advance time and post-processing
// ----------------------------------------------------- //

bool model_setup()
{
	// - nse time //
	dt = CFL * min(grid.dx / u_max, grid.dy / v_max,
		(grid.dxh * grid.dxh * c_Reynolds), 
		(grid.dyh * grid.dyh * c_Reynolds));
	
#ifdef DUMP_CONTINUE
	double* cpu_stamp;
	int cpu_stamp_size;

	read_binary_stamp(DUMP_NSE_STAMP_FILE, dump_continue_mark,
		&cpu_stamp, &cpu_stamp_size,
		grid, &current_time, &time_index);

	if (cpu_stamp_size >= 4) {
		//cpu_run_time += cpu_stamp[0];
		cpu_nse_eq_time += cpu_stamp[1];
		cpu_pois_time += cpu_stamp[2];
		grid.mpi_com.cpu_time_exch += cpu_stamp[3];
	}
	delete[] cpu_stamp;
	begin_time = current_time;
#else
	current_time = begin_time;
	time_index = 0;
#endif
	
// - output setup //
	print_mark = print_begin;
	print_index = 0;
#ifdef DUMP_CONTINUE
	while (current_time >= print_mark) {
		print_mark += print_dt;
		print_index++;
	}
#endif

	// - dump setup //
	dump_mark = dump_begin;
	dump_index = 1;
#ifdef DUMP_CONTINUE
	while (current_time >= dump_mark) {
		dump_mark += dump_dt;
	}
	dump_index = dump_continue_mark + 1;
#endif

	return true;
}

void velocity_bc(Real* U, Real* V, const uniGrid2d< Real >& grid)
{
	const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(U, V,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}


void intermediate_bc(Real* Uinterm, Real* Vinterm, const uniGrid2d< Real >& grid)
{
#ifndef SCHEME_X4
	const int exch_width = 1;
#else
	const int exch_width = 2;
#endif

	grid.mpi_com.exchange_halo(Uinterm, Vinterm,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}

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

void init_solution(Real* Usol, Real* Vsol, Real* Psisol, Real* wsol, 
	const uniGrid2d< Real >& grid)
{
	null(Usol, grid.size);
	null(Vsol, grid.size);
	null(Psisol, grid.size);
	null(wsol, grid.size);
	
#ifdef TEST_TAYLOR_GREEN
   	Real px, py;
	int i, j, idx;
    
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
        px = grid.x + (i - grid.gcx) * grid.dx + grid.dxh; // center of the cell
		py = grid.y + grid.dyh;

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, py += grid.dy)
		{
            Psisol[idx] = sin(px) * sin(py);
        }
    }
    psi_bc(Psisol, grid);
    velocity_stag(Usol, Vsol, Psisol, grid); 
    velocity_bc(Usol, Vsol, grid);
    vorticity(wsol, Usol, Vsol, grid);
#endif
    
#ifdef DECAYING_TURBULENCE
    power_exp_spectra(Psisol, (Real)10.0, grid); // power spectra from Maulik 2017
    psi_bc(Psisol, grid);
    velocity_stag(Usol, Vsol, Psisol, grid); 
    velocity_bc(Usol, Vsol, grid);
    vorticity(wsol, Usol, Vsol, grid);
    
    // unit energy per unit area
    Real KE = kinetic_energy_collocated(Usol, Vsol, grid);
    
    if (grid.mpi_com.rank == 0) printf("KE = %E \n", KE);
    mul(Usol,1.0/sqrt(KE),grid.size);
    mul(Vsol,1.0/sqrt(KE),grid.size);
    
    mul(Psisol,1.0/sqrt(KE),grid.size);
    mul(wsol  ,1.0/sqrt(KE),grid.size);
#endif
}

void init_velocity(Real* U, Real* V, 
	const Real* Usol, const Real* Vsol, const uniGrid2d< Real >& grid)
{
#ifndef DUMP_CONTINUE
	null(U, grid.size);
	null(V, grid.size);

	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			U[idx] = Usol[idx];
			V[idx] = Vsol[idx];
		}
	}
#else
	read_binary(DUMP_VELOCITY_FILE, dump_continue_mark,
		U, V, grid);
#endif
}

void init_psi(Real* Psi, 
	const Real* Psisol, const uniGrid2d< Real >& grid)
{
#ifndef DUMP_CONTINUE
	null(Psi, grid.size);

	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
			Psi[idx] = Psisol[idx];
	}
#endif
}

void init_w(Real* w,
	const Real* wsol, const uniGrid2d< Real >& grid)
{
#ifndef DUMP_CONTINUE
	null(w, grid.size);
	
	int i, j, idx;
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
			w[idx] = wsol[idx];
	}
#else
	read_binary(DUMP_VORTEX_FILE, dump_continue_mark,
		w, grid);
#endif
}


void model_error(Real* Uerr, Real* Verr, Real* Psierr, Real* werr,
	const Real* U, const Real* V, const Real* Psi, const Real* w, const Real* Usol, const Real* Vsol, const Real* Psisol, const Real* wsol,
	const Real c_Reynolds, const Real current_time, const uniGrid2d< Real >& grid)
{
	const Real F_diss =
		exp((-(Real)2.0 * current_time) / c_Reynolds);
	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Uerr[idx] = Usol[idx] * F_diss - U[idx];
			Verr[idx] = Vsol[idx] * F_diss - V[idx];
            
            Psierr[idx] = Psisol[idx] * F_diss - Psi[idx];
            werr  [idx] = wsol  [idx] * F_diss - w  [idx];
		}
	}
}

float pfield (Real* model_field)
{
	write_tecplot("-debug_field-.plt", 0, model_field, "ly", 
				print_xmin, print_xmax, print_ymin, 
				print_ymax,grid, current_time);
	return (float)current_time;
}
