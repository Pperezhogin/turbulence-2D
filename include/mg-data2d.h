#pragma once

#include "unigrid2d.h"

// *[mg-data2d.h]: full //

//#define _MG_PROLONGATE_EX    // include extended cells in prolongation

// * Multigrid data * //
namespace nse
{
	const int mg_max_grids = 16;

	const int mg_coarse_null = -1;
	const int mg_coarse_x = 1;
	const int mg_coarse_y = 2;
	const int mg_coarse_xy = 3;
	const double mg_coarse_aspect = (double) 1.95;

	const int mg_mpi_min_proc_size = 1024;      // mpi minimum size for gathering in bytes //

	const int c_mg_sgs_down_iters = 1;
	const int c_mg_sgs_up_iters = 2;
	const int c_mg_sgs_direct_iters = 3;

	template< typename T >
	struct mg_poisson2d_data {
		T *mg_memory;
		T *x[mg_max_grids], *rhs[mg_max_grids];

		T dx[mg_max_grids], dy[mg_max_grids];
		T dx2i[mg_max_grids], dy2i[mg_max_grids];
		T idg[mg_max_grids];

		int mg_memory_size;
		int num_grids;

		int coarse_type[mg_max_grids];
		int nx[mg_max_grids], ny[mg_max_grids];
		int gcx, gcy;

		int sgs_down_iters[mg_max_grids],
			sgs_up_iters[mg_max_grids];

		int sgs_color_shift[mg_max_grids];


		mg_poisson2d_data() : mg_memory_size(0), num_grids(0) { }
		~mg_poisson2d_data() {
			clear();
		}

		void init(const uniGrid2d< T >& grid, const int _num_grids);
		void clear();

		int memory_size(const uniGrid2d< T >& grid, const int _num_grids);
	};

	template< typename T >
	struct mg_var_poisson2d_data {
		T *mg_memory;
		T *x[mg_max_grids], *rhs[mg_max_grids], *res[mg_max_grids];
		T *idg[mg_max_grids];
		T *i_density[mg_max_grids];

		T dx[mg_max_grids], dy[mg_max_grids];
		T dx2i[mg_max_grids], dy2i[mg_max_grids];
		T dx2ih[mg_max_grids], dy2ih[mg_max_grids];

		int mg_memory_size;
		int num_grids;

		int coarse_type[mg_max_grids];
		int nx[mg_max_grids], ny[mg_max_grids];
		int gcx, gcy;

		int sgs_down_iters[mg_max_grids],
			sgs_up_iters[mg_max_grids];

		int sgs_color_shift[mg_max_grids];


		mg_var_poisson2d_data() : mg_memory_size(0), num_grids(0) { }
		~mg_var_poisson2d_data() {
			clear();
		}

		void init(T* i_density, const int bc_type,
			const uniGrid2d< T >& grid, const int _num_grids);
		void clear();

		int memory_size(const uniGrid2d< T >& grid, const int _num_grids);
	};

	template< typename T >
	struct mg_mpi_poisson2d_data {
		// MPI communicators //
		nse::mpiCom2d mpi_com[mg_max_grids];

		T *mg_memory;
		T *x[mg_max_grids], *rhs[mg_max_grids];

		T dx[mg_max_grids], dy[mg_max_grids];
		T dx2i[mg_max_grids], dy2i[mg_max_grids];
		T idg[mg_max_grids];

		int mg_memory_size;
		int num_grids;

		int coarse_type[mg_max_grids];
		int local_nx[mg_max_grids], local_ny[mg_max_grids];
		int mpi_nx[mg_max_grids], mpi_ny[mg_max_grids];
		int gcx, gcy;

		int sgs_down_iters[mg_max_grids],
			sgs_up_iters[mg_max_grids];

		int sgs_color_shift[mg_max_grids];

		int mpi_run[mg_max_grids];        // [0,1] run smoother flag
		int mpi_combine[mg_max_grids];    // [0,level] > 0 gather grids on k-th step using mpi_com[ k - 1 ]
		int mpi_level[mg_max_grids];      // [2^k] number of grid division levels

		mg_mpi_poisson2d_data() : mg_memory_size(0), num_grids(0) { }
		~mg_mpi_poisson2d_data() {
			clear();
		}

		void init(const uniGrid2d< T >& grid, const int _num_grids);
		void clear();

		int memory_size(const uniGrid2d< T >& grid, const int _num_grids);
	};
}

namespace poisson2d
{
	// * restrict operator * //
	template< typename T >
	void mg_restrict(T* coarse, const T* fine,
		const int type,

		const int cnx, const int cny,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const int icb, const int ice,
		const int jcb, const int jce);
	// ----------------------------------------------------------------- //

	// * restrict residual operator * //
	template< typename T >
	void mg_restrict_residual(T* coarse, const T* x, const T* rhs,
		const int type,

		const int cnx, const int cny,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const int icb, const int ice,
		const int jcb, const int jce,

		const T dx2i, const T dy2i      // fine grid parameters
		);
	// ----------------------------------------------------------------- //

	// * prolongation operator * //
	template< typename T >
	void mg_prolongate(T* fine, const T* coarse,
		const int type,

		const int nx, const int ny,
		const int cnx, const int cny,
		const int gcx, const int gcy);

	//      - no prolongation extension inside //
	template< typename T >
	void mg_prolongate(T* fine, const T* coarse,
		const int type,

		const int nx, const int ny,
		const int cnx, const int cny,
		const int gcx, const int gcy,

		const int icb, const int ice,
		const int jcb, const int jce);
	// ----------------------------------------------------------------- //
}
