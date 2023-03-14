#pragma once

#include "unigrid2d.h"
#include "mg-data2d.h"

#include "pois-const2d.h"
// *[pois2d.h]: full //

namespace nse
{

	//#define _POIS2D_INIT_NULL               // init solvers with x = 0 //
	//#define _POIS2D_USE_SSE               // use sse calls //
	#define _POIS2D_NORM		1			// poisson error norm [C]=1, [L2]=2 //

	// -------------------------------------------------------------------- //
	// N - grid size

	// * CG-SGS (Red-Black) memory req.: [4 * N] * //
	template< typename T >
	int cg_sgs_redblack(T* x, const T* rhs, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid, const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //


	// * CG-Multigrid SGS (Red-Black) memory req.: [ 4 * N + MG(INSIDE) ] * // 
	template< typename T >
	int cg_mg_sgs_redblack(T* x, const T* rhs, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid,
		mg_poisson2d_data< T >& mg_data,
		const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //


	// * CG-Multigrid(MPI) SGS (Red-Black) memory req.: [ 4 * N + MG(INSIDE) ] * // 
	template< typename T >
	int cg_mg_sgs_redblack(T* x, const T* rhs, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid,
		mg_mpi_poisson2d_data< T >& mg_data,
		const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //

	// * CG-Multigrid(MPI) SGS (Red-Black) memory req.: [ 4 * N + MG(INSIDE) ] * // 
	template< typename T >
	int cg_mg_sgs_redblack_x4(T* x, const T* rhs, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid,
		mg_mpi_poisson2d_data< T >& mg_data,
		const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //


	// * variable density poisson equations * //
	// * CG-SGS (Red-Black) memory req.: [5 * N] * //
	template< typename T >
	int cg_sgs_redblack(T* x, const T* rhs, const T* i_density, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid, const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //


	// * variable density poisson equation * //
	// * CG-Multigrid SGS (Red-Black) memory req.: [ 4 * N + MG(INSIDE) ] * // 
	template< typename T >
	int cg_mg_sgs_redblack(T* x, const T* rhs, const T* i_density, T* memory,

		const int piters,
		const T retol, const T abstol, const int maxiters,
		const uniGrid2d< T >& grid,
		mg_var_poisson2d_data< T >& mg_data,
		const int pois_bc_type,
		T* resnorm);
	// -------------------------------------------------------------------- //
}
