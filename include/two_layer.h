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

using namespace nse;

// --- main definitions --- //
#define OPENMP_CORES		1		// number of OpenMP cores to use

//#define DUMP_CONTINUE				// define for startup from dump 

#define POISSON_FFT

//#define DIFFUSION
#define Q_PARAMETERIZATION
#define DYNAMIC_MODEL
//#define SIMPLE_MODEL
//#define ADM
#define BETA_PLANE

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
const Real end_time = (Real) 20.0;

//spectrum parameters
const Real CFL = (Real) 0.8; //CFL = 1 is a maximum
Real dt;

// balance of energy and enstrophy
s_balance< Real > balance;

// --- fluid parameters from Grooms2015 --- //
const Real c_Reynolds = (Real) 20000000000;
const Real kd = (Real) 50.0;
//const Real Relaigh = (Real)16.0;
// ------ my parameter to make jet on wavenumber 10 ----- //
const Real Beta_effect = (Real)1000.0;
const Real Relaigh = (Real)0.5;
// ----------------------------------------------------- //
// --- output parameters --- //
#define OUTPUT_DIR	           "output/two_layer/beta_effect/bilap_smag/" 

// --- dynamic model params --- //
Real tf_width = sqrt(6.0);
Real bf_width = sqrt(6.0);
int viscosity_model = bilap_smag;
int averaging_method = averaging_global;
bool mixed_model = false;
bool negvisc_backscatter = false;
bool reynolds_backscatter = false;
int filter_iterations = 1;        // only for clipping
Real lagrangian_time = (Real)2.0; // only for lagrangian averaging

// --- simple model params --- //
Real Cs2 = (Real)0.06;

const Real print_begin = (Real) -1e-16;
const Real print_dt = (Real) 0.1;

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

dynamic_model< Real > dyn_model1,  dyn_model2;

const Real pois_retol = (Real) 1e-4, pois_abstol = (Real) 1e-5;
const int pois_piters = 1;
const int pois_mg_min = 4;			// min grid size in multigrid sequence
int pois_ngrid = 0;					// auto definition: 0 value
const int pois_maxiters = 200;

const int pois_bc_type = c_pois_bc_periodic_xy;

Real poisson_norm;
int poisson_status;
// ----------------------------------------------------- // 


// --- output definitions and data --- //
#define NSE_SEQ_FILE           OUTPUT_DIR"nse.dsq"

#define DATA_FILE              OUTPUT_DIR"nse-data.txt"

#define DUMP_NSE_STAMP_FILE			DUMP_DIR"nse-stamp-.dsq"
#define DUMP_NSE_SEQ_FILE			DUMP_DIR"nse-.dsq"

#define DUMP_VELOCITY_FILE			DUMP_DIR"velocity-dump-.nsx"
#define DUMP_VORTEX_FILE			DUMP_DIR"vortex-dump-.nsx"
#define DUMP_VORTEX_IMP_FILE		DUMP_DIR"vortex-imp-dump-.nsx"

#define DUMP_PSI_FILE			DUMP_DIR"psi-dump-.nsx"

int print_index, dump_index;
Real print_mark, dump_mark;

timeSeries nse_series;

const int c_seq_max_length = 100 * 1024;
// ----------------------------------------------------- //

// --- main data --- //
uniGrid2d< Real > grid;

Real *U1, *V1, *Psi1, *w1, *q1;
Real *U2, *V2, *Psi2, *w2, *q2;

Real *Psi1_rk, *q1_rk;
Real *Psi2_rk, *q2_rk;

Real *qim1, *qim2;

Real *eddy_rhs_1,  *eddy_rhs_2;
Real *shear_rhs_1, *shear_rhs_2;
Real *fric_rhs_1,  *fric_rhs_2;

Real *rhs;
Real *memory;					// poisson solver additional memory
Real *rhs_visc;

Real u_max = (Real)150.0, v_max = (Real)150.0;				// maximum velocity values //

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
bool advance_nse_eq_runge_kutta();

bool advance_time();							// advance time and post-processing

inline Real jet_function(const Real y);
inline Real perturbation_function(const Real x, const Real y);
// ----------------------------------------------------- //

bool model_setup()
{
	// - grid //
	if (!grid.set(
		(Real) 0.0, (Real) 0.0, domain_length, domain_width,
#ifdef POISSON_FFT
		domain_nx, domain_ny, domain_gcx, domain_gcy, 1))
#else
		domain_nx, domain_ny, domain_gcx, domain_gcy, 2))	  
#endif
	{
		return false;
	}

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

void velocity_bc(Real* U1, Real* V1, Real* U2, Real* V2, const uniGrid2d< Real >& grid)
{
	const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(U1, V1,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);

	grid.mpi_com.exchange_halo(U2, V2,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}

void scalar_bc(Real* q1, Real* q2, const uniGrid2d< Real >& grid)
{
	const int exch_width = domain_gcx;

	grid.mpi_com.exchange_halo(q1, q2,
		grid.nx, grid.ny, grid.gcx, grid.gcy,
		exch_width, exch_width, 1, 1);
}

void initial_condition(Real* U1, Real* V1, Real* U2, Real* V2, Real* q1, Real* q2, Real* w1, Real* w2, Real* Psi1, Real* Psi2, 
	const uniGrid2d< Real >& grid)
{
	Real px, py;
	int i, j, idx;

	null(q1, grid.size);
	null(q2, grid.size);

	/*
	px = grid.x;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++, px += grid.dx)
	{
		py = grid.y;

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, py += grid.dy)
		{
            q1[idx] = perturbation_function(px, py);
        }
    }
	*/
	power_exp_spectra(q1, (Real)10.0, grid);

	remove_const(q1, grid);
	scalar_bc(q1, q2, grid);

	two_layer_streamfunction(Psi1, Psi2, q1, q2, kd, fft_data, grid);
	scalar_bc(Psi1, Psi2, grid);

	velocity_stag(U1, V1, Psi1, grid); 
	velocity_stag(U2, V2, Psi2, grid); 
	
	velocity_bc(U1, V1, U2, V2, grid);
	vorticity(w1, U1, V1, grid);
	vorticity(w2, U2, V2, grid);

	scalar_bc(w1, w2, grid);
}

inline Real jet_function(const Real y) {
	Real ajet  = (Real)8.0 * pow((Real)M_PI, (Real)4.0);
	Real c, eps, ans;
	c = (Real)1.0 / exp( - ajet / ((Real)M_PI * (Real)M_PI));
	eps = (Real)1.e-10;
	if ((y > eps) && (y < (Real)2.0 * (Real)M_PI - eps)) 
		ans = c * exp( ajet / (y * (y - (Real)2.0 * (Real)M_PI)));
	else
		ans = (Real)0.0;
	return ans;
}

inline Real perturbation_function(const Real x, const Real y) {
	Real ans;
	Real std_xy, amp, x2y2;
	std_xy = (Real)1.0; // as for jet (see Perezhogin2020a)
	amp = (Real)1.0;
	x2y2 = sqr(x-(Real)M_PI) + sqr(y-(Real)M_PI);

	ans = amp * exp(- x2y2 / ((Real)2.0 * std_xy * std_xy));
	return ans;
}

float pfield (Real* model_field)
{
	write_tecplot("-debug_field-.plt", 0, model_field, "ly", 
				print_xmin, print_xmax, print_ymin, 
				print_ymax,grid, current_time);
	return (float)current_time;
}

Real check_solver(const Real* q1, const Real* q2, const Real* Psi1, const Real* Psi2, const Real* w1, const Real* w2)
{
	int i, j, idx;
	Real err_l = (Real)0.0;
	Real err;
	Real q_norm_l = (Real)0.0;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			err_l += fabs(q1[idx] - w1[idx] - (Real)0.5 * sqr(kd) * (Psi2[idx] - Psi1[idx]))
				   + fabs(q2[idx] - w2[idx] + (Real)0.5 * sqr(kd) * (Psi2[idx] - Psi1[idx]));
			q_norm_l += fabs(q1[idx]) + fabs(q2[idx]);
		}
	}
	err = mpi_allreduce(err_l, MPI_SUM) / mpi_allreduce(q_norm_l, MPI_SUM);
}

Real two_layer_KE(const Real* U1, const Real* V1, const Real* U2, const Real* V2)
{
	int i, j, idx;
	Real KE = 0;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			KE += sqr(U1[idx]) + sqr(V1[idx]) + sqr(U2[idx]) + sqr(V2[idx]);
		}
	}
	KE = mpi_allreduce(KE, MPI_SUM) * grid.dxdy  / (grid.mpi_length * grid.mpi_width) * (Real)0.5;
	return KE;
}

Real two_layer_PE(const Real* Psi1, const Real* Psi2)
{
	int i, j, idx;
	Real PE = 0;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			PE += sqr(Psi1[idx] - Psi2[idx]);
		}
	}
	PE = mpi_allreduce(PE, MPI_SUM) * grid.dxdy  / (grid.mpi_length * grid.mpi_width) * sqr(kd) * (Real)0.25;
	return PE;
}

Real two_layer_enstrophy(const Real* q1, const Real* q2)
{
	int i, j, idx;
	Real Z = 0;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Z += sqr(q1[idx]) + sqr(q2[idx]);
		}
	}
	Z = mpi_allreduce(Z, MPI_SUM) * grid.dxdy  / (grid.mpi_length * grid.mpi_width) * (Real)0.5;
	return Z;
}

Real E_dissipation(const Real* rhs1, const Real* rhs2, const Real* Psi1, const Real* Psi2)
{
	int i, j, idx;
	Real dE = 0;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			dE += Psi1[idx] * rhs1[idx] + Psi2[idx] * rhs2[idx];
		}
	}
	dE = mpi_allreduce(dE, MPI_SUM) * grid.dxdy  / (grid.mpi_length * grid.mpi_width);
	return dE;
}

Real Z_dissipation(const Real* rhs1, const Real* rhs2, const Real* q1, const Real* q2)
{
	int i, j, idx;
	Real dZ = 0;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			dZ -= q1[idx] * rhs1[idx] + q2[idx] * rhs2[idx];
		}
	}
	dZ = mpi_allreduce(dZ, MPI_SUM) * grid.dxdy  / (grid.mpi_length * grid.mpi_width);
	return dZ;
}

Real heat_flux(const Real* V1, const Real* V2, const Real* Psi1, const Real* Psi2)
{
	int i, j, idx;
	Real dH = 0;
	Real v, v_left, v_center;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			v = V1[idx] + V2[idx];
			v_left = V1[idx - grid.ny] + V2[idx - grid.ny];
			v_center = (v + v_left) * (Real)0.5;
			dH += 0.25 * v_center * (Psi1[idx] - Psi2[idx]);
		}
	}
	dH = mpi_allreduce(dH, MPI_SUM) * grid.dxdy; // integral over the domain (As Grooms Majda suggest), not mean value
	return dH;
}