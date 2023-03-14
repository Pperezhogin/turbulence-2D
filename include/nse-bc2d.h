#pragma once

// *    2D Navier-Stokes boundary-conditions module  * //

// *[nse-bc2d.h]: full //

#include "unigrid2d.h"

#include <time.h>
#include "mtrand.h"

namespace nse
{
	const int _west_bc = 0,
		_east_bc = 1,
		_south_bc = 2,
		_north_bc = 3;

	// * ghost cell extrapolation * //
	template< typename T, int side_bc >
	void c_ghost_extrapolation(T* X,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int gcx_sh, const int gcy_sh,
		const mpiCom2d& mpi_com);

	template< typename T, int side_bc >
	void u_ghost_extrapolation(T* U,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int gcx_sh, const int gcy_sh,
		const mpiCom2d& mpi_com);

	template< typename T, int side_bc >
	void v_ghost_extrapolation(T* V,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int gcx_sh, const int gcy_sh,
		const mpiCom2d& mpi_com);



	// * dirichlet boundary condition for (u,v,c) grid * //   
	template< typename T, int side_bc >
	void c_dirichlet_bc(T* X, const T rhs, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void u_dirichlet_bc(T* U, const T rhs, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void v_dirichlet_bc(T* V, const T rhs, const uniGrid2d< T >& grid);
	// ---------------------------------------------------------------------  //

	// * dirichlet boundary condition for (u,v,c) grid * //
	// *     F( p ) = 4 * F_{max} * ( p - p_{min} ) * ( p_{max} - p ) / L^{2} 
	//                           , where L = ( p_{max} - p_{min} )
	template< typename T, int side_bc >
	void c_dirichlet_bc(T* X, const T x_max, const T p_min, const T p_max,
		const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void u_dirichlet_bc(T* U, const T u_max, const T p_min, const T p_max,
		const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void v_dirichlet_bc(T* V, const T v_max, const T p_min, const T p_max,
		const uniGrid2d< T >& grid);
	// ---------------------------------------------------------------------  //

	// * neumann boundary condition for (u,v,c) grid * //
	template< typename T, int side_bc >
	void c_neumann_bc(T* X, const T rhs, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void u_neumann_bc(T* U, const T rhs, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	void v_neumann_bc(T* V, const T rhs, const uniGrid2d< T >& grid);
	// ---------------------------------------------------------------------  //

	// * convective boundary condition for (u,v,c) grid * //
	template< typename T, int side_bc >
	void c_convective_bc(T* X, const T* X_p, const T c_velocity,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T, int side_bc >
	void u_convective_bc(T* U, const T* U_p, const T c_velocity,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T, int side_bc >
	void v_convective_bc(T* V, const T* V_p, const T c_velocity,
		const uniGrid2d< T >& grid, const T dt);
	// ---------------------------------------------------------------------  //

	// * open boundary conditions for (u,v,c) grid * //
	template< typename T, int side_bc >
	void c_open_bc(T* X, const T* X_p,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T, int side_bc >
	void u_open_bc(T* U, const T* U_p,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T, int side_bc >
	void v_open_bc(T* V, const T* V_p,
		const uniGrid2d< T >& grid, const T dt);
	// ---------------------------------------------------------------------  //

	// * boundary average operators for (u,v,c) grid * //
	template< typename T, int side_bc >
	T c_external_average(const T* X, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	T u_external_average(const T* U, const uniGrid2d< T >& grid);

	template< typename T, int side_bc >
	T v_external_average(const T* V, const uniGrid2d< T >& grid);
	// ---------------------------------------------------------------------  //

	// * init stratification * //
	template< typename T >
	void c_init_stratification(T* X,
		const T x_min, const T x_max, const T y_level, const T y_eps,
		const uniGrid2d< T >& grid);

	// * add random disturbance * //
	template< typename T >
	void add_disturbance(T* X, const T variance, const long int seed,
		const uniGrid2d< T >& grid);
}


template< typename T, int side_bc >
void nse::c_ghost_extrapolation(T* X,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const int gcx_sh, const int gcy_sh,
	const mpiCom2d& mpi_com)
{
	int i, j, idx;

	const int is_west = (side_bc == _west_bc) && (mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (mpi_com.rank_x == mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (mpi_com.rank_y == mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
#pragma omp parallel for private( i, j, idx ) shared( X )
		for (j = gcy; j < ny - gcy; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (gcx - i - 1) * ny + j;

			X[idx] =
				(T) 3.0 * (X[idx + ny] - X[idx + (ny << 1)])
				+ X[idx + (ny << 1) + ny];
		}
	}

	// * east side * //
	if (is_east)
	{
#pragma omp parallel for private( i, j, idx ) shared( X )
		for (j = gcy; j < ny - gcy; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (nx - gcx + i) * ny + j;

			X[idx] =
				(T) 3.0 * (X[idx - ny] - X[idx - (ny << 1)])
				+ X[idx - (ny << 1) - ny];
		}
	}

	// * south side * //
	if (is_south)
	{
#pragma omp parallel for private( i, j, idx ) shared( X )
		for (i = gcx; i < nx - gcx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (gcy - j - 1);

			X[idx] =
				(T) 3.0 * (X[idx + 1] - X[idx + 2])
				+ X[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
#pragma omp parallel for private( i, j, idx ) shared( X )
		for (i = gcx; i < nx - gcx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (ny - gcy + j);

			X[idx] =
				(T) 3.0 * (X[idx - 1] - X[idx - 2])
				+ X[idx - 3];
		}
	}
}

template< typename T, int side_bc >
void nse::u_ghost_extrapolation(T* U,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const int gcx_sh, const int gcy_sh,
	const mpiCom2d& mpi_com)
{
	int i, j, idx;

	const int is_west = (side_bc == _west_bc) && (mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (mpi_com.rank_x == mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (mpi_com.rank_y == mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
#pragma omp parallel for private( i, j, idx ) shared( U )
		for (j = gcy; j < ny - gcy; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (gcx - i) * ny + j;

			U[idx] = (T) 3.0 * (U[idx + ny] - U[idx + (ny << 1)])
				+ U[idx + (ny << 1) + ny];
		}
	}

	// * east side * //
	if (is_east)
	{
#pragma omp parallel for private( i, j, idx ) shared( U )
		for (j = gcy; j < ny - gcy; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (nx - gcx + i) * ny + j;

			U[idx] = (T) 3.0 * (U[idx - ny] - U[idx - (ny << 1)])
				+ U[idx - (ny << 1) - ny];
		}
	}

	// * south side * //
	if (is_south)
	{
#pragma omp parallel for private( i, j, idx ) shared( U )
		for (i = 0; i < nx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (gcy - j - 1);

			U[idx] =
				(T) 3.0 * (U[idx + 1] - U[idx + 2])
				+ U[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
#pragma omp parallel for private( i, j, idx ) shared( U )
		for (i = 0; i < nx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (ny - gcy + j);

			U[idx] =
				(T) 3.0 * (U[idx - 1] - U[idx - 2])
				+ U[idx - 3];
		}
	}
}

template< typename T, int side_bc >
void nse::v_ghost_extrapolation(T* V,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const int gcx_sh, const int gcy_sh,
	const mpiCom2d& mpi_com)
{
	int i, j, idx;

	const int is_west = (side_bc == _west_bc) && (mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (mpi_com.rank_x == mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (mpi_com.rank_y == mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
#pragma omp parallel for private( i, j, idx ) shared( V )
		for (j = 0; j < ny; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (gcx - i - 1) * ny + j;

			V[idx] =
				(T) 3.0 * (V[idx + ny] - V[idx + (ny << 1)])
				+ V[idx + (ny << 1) + ny];
		}
	}

	// * east side * //
	if (is_east)
	{
#pragma omp parallel for private( i, j, idx ) shared( V )
		for (j = 0; j < ny; j++)
		for (i = gcx_sh; i < gcx; i++) {
			idx = (nx - gcx + i) * ny + j;

			V[idx] =
				(T) 3.0 * (V[idx - ny] - V[idx - (ny << 1)])
				+ V[idx - (ny << 1) - ny];
		}
	}

	// * south side * //
	if (is_south)
	{
#pragma omp parallel for private( i, j, idx ) shared( V )
		for (i = gcx; i < nx - gcx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (gcy - j);

			V[idx] =
				(T) 3.0 * (V[idx + 1] - V[idx + 2])
				+ V[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
#pragma omp parallel for private( i, j, idx ) shared( V )
		for (i = gcx; i < nx - gcx; i++)
		for (j = gcy_sh; j < gcy; j++) {
			idx = i * ny + (ny - gcy + j);

			V[idx] =
				(T) 3.0 * (V[idx - 1] - V[idx - 2])
				+ V[idx - 3];
		}
	}
}

// ------------------------------------------------------------------------- //
// Dirichlet Type Boundary Conditions (u,v,c)
// ============================================
template< typename T, int side_bc >
void nse::c_dirichlet_bc(T* X, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			X[idx] = c1 * Rhs +
				c2 * X[idx + grid.ny] + c3 * X[idx + (grid.ny << 1)];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			X[idx] = c1 * Rhs +
				c2 * X[idx - grid.ny] + c3 * X[idx - (grid.ny << 1)];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			X[idx] = c1 * Rhs +
				c2 * X[idx + 1] + c3 * X[idx + 2];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			X[idx] = c1 * Rhs +
				c2 * X[idx - 1] + c3 * X[idx - 2];
		}
	}

	c_ghost_extrapolation< T, side_bc >(X,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::u_dirichlet_bc(T* U, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;
			U[idx] = Rhs;
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			U[idx] = Rhs;
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			U[idx] = c1 * Rhs +
				c2 * U[idx + 1] + c3 * U[idx + 2];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			U[idx] = c1 * Rhs +
				c2 * U[idx - 1] + c3 * U[idx - 2];
		}
	}

	u_ghost_extrapolation< T, side_bc >(U,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::v_dirichlet_bc(T* V, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			V[idx] = c1 * Rhs +
				c2 * V[idx + grid.ny] + c3 * V[idx + (grid.ny << 1)];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			V[idx] = c1 * Rhs +
				c2 * V[idx - grid.ny] + c3 * V[idx - (grid.ny << 1)];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;

			V[idx] = Rhs;
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			V[idx] = Rhs;
		}
	}

	v_ghost_extrapolation< T, side_bc >(V,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::c_dirichlet_bc(T* X, const T x_max, const T p_min, const T p_max,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const T p_value = (T) 4.0 * x_max /
		((p_max - p_min) * (p_max - p_min));

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy + grid.dyh;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			X[idx] = c1 * Rhs +
				c2 * X[idx + grid.ny] + c3 * X[idx + (grid.ny << 1)];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy + grid.dyh;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			X[idx] = c1 * Rhs +
				c2 * X[idx - grid.ny] + c3 * X[idx - (grid.ny << 1)];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);
			p_x = grid.x + (i - grid.gcx) * grid.dx + grid.dxh;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			X[idx] = c1 * Rhs +
				c2 * X[idx + 1] + c3 * X[idx + 2];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);
			p_x = grid.x + (i - grid.gcx) * grid.dx + grid.dxh;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			X[idx] = c1 * Rhs +
				c2 * X[idx - 1] + c3 * X[idx - 2];
		}
	}

	c_ghost_extrapolation< T, side_bc >(X,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::u_dirichlet_bc(T* U, const T u_max, const T p_min, const T p_max,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const T p_value = (T) 4.0 * u_max /
		((p_max - p_min) * (p_max - p_min));

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy + grid.dyh;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			U[idx] = Rhs;
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy + grid.dyh;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			U[idx] = Rhs;
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);
			p_x = grid.x + (i - grid.gcx) * grid.dx;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			U[idx] = c1 * Rhs +
				c2 * U[idx + 1] + c3 * U[idx + 2];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);
			p_x = grid.x + (i - grid.gcx) * grid.dx;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			U[idx] = c1 * Rhs +
				c2 * U[idx - 1] + c3 * U[idx - 2];
		}
	}

	u_ghost_extrapolation< T, side_bc >(U,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::v_dirichlet_bc(T* V, const T v_max, const T p_min, const T p_max,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 8.0 / (T) 3.0,
		c2 = -(T) 2.0,
		c3 = (T) 1.0 / (T) 3.0;

	const T p_value = (T) 4.0 * v_max /
		((p_max - p_min) * (p_max - p_min));

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			V[idx] = c1 * Rhs +
				c2 * V[idx + grid.ny] + c3 * V[idx + (grid.ny << 1)];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
		T p_y, Rhs;

#pragma omp parallel for private( j, idx, p_y, Rhs ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;
			p_y = grid.y + (j - grid.gcy) * grid.dy;

			Rhs = ((p_y < p_min) || (p_y > p_max)) ? (T) 0.0 :
				p_value * (p_y - p_min) * (p_max - p_y);

			V[idx] = c1 * Rhs +
				c2 * V[idx - grid.ny] + c3 * V[idx - (grid.ny << 1)];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;
			p_x = grid.x + (i - grid.gcx) * grid.dx + grid.dxh;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			V[idx] = Rhs;
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
		T p_x, Rhs;

#pragma omp parallel for private( i, idx, p_x, Rhs ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);
			p_x = grid.x + (i - grid.gcx) * grid.dx + grid.dxh;

			Rhs = ((p_x < p_min) || (p_x > p_max)) ? (T) 0.0 :
				p_value * (p_x - p_min) * (p_max - p_x);

			V[idx] = Rhs;
		}
	}

	v_ghost_extrapolation< T, side_bc >(V,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

// ------------------------------------------------------------------------- //
// Neumann Type Boundary Conditions (u,v,c)
// ============================================
template< typename T, int side_bc >
void nse::c_neumann_bc(T* X, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			X[idx] = X[idx + grid.ny] - grid.dx * Rhs;
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			X[idx] = X[idx - grid.ny] + grid.dx * Rhs;
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			X[idx] = X[idx + 1] - grid.dy * Rhs;
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			X[idx] = X[idx - 1] + grid.dy * Rhs;
		}
	}

	c_ghost_extrapolation< T, side_bc >(X,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::u_neumann_bc(T* U, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 2.0 / (T) 3.0,
		c2 = (T) 4.0 / (T) 3.0,
		c3 = -(T) 1.0 / (T) 3.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;

			U[idx] = -c1 * Rhs * grid.dx +
				c2 * U[idx + grid.ny] + c3 * U[idx + (grid.ny << 1)];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			U[idx] = c1 * Rhs * grid.dx +
				c2 * U[idx - grid.ny] + c3 * U[idx - (grid.ny << 1)];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			U[idx] = U[idx + 1] - grid.dy * Rhs;
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			U[idx] = U[idx - 1] + grid.dy * Rhs;
		}
	}

	u_ghost_extrapolation< T, side_bc >(U,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::v_neumann_bc(T* V, const T Rhs,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 2.0 / (T) 3.0,
		c2 = (T) 4.0 / (T) 3.0,
		c3 = -(T) 1.0 / (T) 3.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			V[idx] = V[idx + grid.ny] - grid.dx * Rhs;
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			V[idx] = V[idx - grid.ny] + grid.dx * Rhs;
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;

			V[idx] = -c1 * Rhs * grid.dy +
				c2 * V[idx + 1] + c3 * V[idx + 2];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			V[idx] = c1 * Rhs * grid.dy +
				c2 * V[idx - 1] + c3 * V[idx - 2];
		}
	}

	v_ghost_extrapolation< T, side_bc >(V,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

// ------------------------------------------------------------------------- //
// Convective Type Boundary Conditions (u,v,c)
// ============================================
template< typename T, int side_bc >
void nse::c_convective_bc(T* X, const T* X_p, const T c_velocity,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			X[idx] = (((T) 1.0 + (T) 2.0 * courant) * X[idx + grid.ny] -
				X_p[idx] - X_p[idx + grid.ny]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			X[idx] = (((T) 2.0 * courant - (T) 1.0) * X[idx - grid.ny] +
				X_p[idx] + X_p[idx - grid.ny]) * del;
		}
	}

	// * south side * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			X[idx] = (((T) 1.0 + (T) 2.0 * courant) * X[idx + 1] -
				X_p[idx] - X_p[idx + 1]) * del;
		}
	}

	// * north side * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			X[idx] = (((T) 2.0 * courant - (T) 1.0) * X[idx - 1] +
				X_p[idx] + X_p[idx - 1]) * del;
		}
	}

	c_ghost_extrapolation< T, side_bc >(X,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::u_convective_bc(T* U, const T* U_p, const T c_velocity,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 - (T) 3.0 * courant);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;

			U[idx] = ((T) 2.0 * U_p[idx] -
				(T) 4.0 * courant * U[idx + grid.ny] +
				courant * U[idx + (grid.ny << 1)]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 + (T) 3.0 * courant);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			U[idx] = ((T) 2.0 * U_p[idx] +
				(T) 4.0 * courant * U[idx - grid.ny] -
				courant * U[idx - (grid.ny << 1)]) * del;
		}
	}

	// * south side * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			U[idx] = (((T) 1.0 + (T) 2.0 * courant) * U[idx + 1] -
				U_p[idx] - U_p[idx + 1]) * del;
		}
	}

	// * north side * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			U[idx] = (((T) 2.0 * courant - (T) 1.0) * U[idx - 1] +
				U_p[idx] + U_p[idx - 1]) * del;
		}
	}

	u_ghost_extrapolation< T, side_bc >(U,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::v_convective_bc(T* V, const T* V_p, const T c_velocity,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	// * west side * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			V[idx] = (((T) 1.0 + (T) 2.0 * courant) * V[idx + grid.ny] -
				V_p[idx] - V_p[idx + grid.ny]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			V[idx] = (((T) 2.0 * courant - (T) 1.0) * V[idx - grid.ny] +
				V_p[idx] + V_p[idx - grid.ny]) * del;
		}
	}

	// * south side * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 - (T) 3.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;

			V[idx] = ((T) 2.0 * V_p[idx] -
				(T) 4.0 * courant * V[idx + 1] +
				courant * V[idx + 2]) * del;
		}
	}

	// * north side * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 + (T) 3.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			V[idx] = ((T) 2.0 * V_p[idx] +
				(T) 4.0 * courant * V[idx - 1] -
				courant * V[idx - 2]) * del;
		}
	}

	v_ghost_extrapolation< T, side_bc >(V,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

// ------------------------------------------------------------------------- //
// Convective Type Boundary Conditions (u,v,c)
// ============================================
template< typename T, int side_bc >
void nse::c_open_bc(T* X, const T* X_p,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	const T c_velocity = (T) 1.0;

	// * west side [DO!] * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			X[idx] = (((T) 1.0 + (T) 2.0 * courant) * X[idx + grid.ny] -
				X_p[idx] - X_p[idx + grid.ny]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		T phase;
		int j, idx;
#pragma omp parallel for private( j, idx, phase ) shared( X, X_p )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			phase = -(grid.dx / dt) * (
				(X[idx - grid.ny] - X_p[idx - grid.ny]) /
				(X_p[idx - grid.ny] - X_p[idx - 2 * grid.ny]));
			if (phase < (T)0) phase = (T)0;
			if (phase > grid.dx / dt) phase = grid.dx / dt;

			X[idx] = ((T)1 - phase * (dt / grid.dx)) * X_p[idx] +
				phase * (dt / grid.dx) * X_p[idx - grid.ny];
		}
	}

	// * south side [DO!] * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			X[idx] = (((T) 1.0 + (T) 2.0 * courant) * X[idx + 1] -
				X_p[idx] - X_p[idx + 1]) * del;
		}
	}

	// * north side [DO!] * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			X[idx] = (((T) 2.0 * courant - (T) 1.0) * X[idx - 1] +
				X_p[idx] + X_p[idx - 1]) * del;
		}
	}

	c_ghost_extrapolation< T, side_bc >(X,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::u_open_bc(T* U, const T* U_p,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	const T c_velocity = (T) 1.0;

	// * west side [DO!] * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 - (T) 3.0 * courant);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;

			U[idx] = ((T) 2.0 * U_p[idx] -
				(T) 4.0 * courant * U[idx + grid.ny] +
				courant * U[idx + (grid.ny << 1)]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		T phase;
		int j, idx;
#pragma omp parallel for private( j, idx, phase ) shared( U, U_p )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			phase = -(grid.dx / dt) * (
				(U[idx - grid.ny] - U_p[idx - grid.ny]) /
				(U_p[idx - grid.ny] - U_p[idx - 2 * grid.ny]));
			if (phase < (T)0) phase = (T)0;
			if (phase > grid.dx / dt) phase = grid.dx / dt;

			U[idx] = ((T)1 - phase * (dt / grid.dx)) * U_p[idx] +
				phase * (dt / grid.dx) * U_p[idx - grid.ny];
		}
	}

	// * south side [DO!] * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			U[idx] = (((T) 1.0 + (T) 2.0 * courant) * U[idx + 1] -
				U_p[idx] - U_p[idx + 1]) * del;
		}
	}

	// * north side [DO!] * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 1.0 + (T) 2.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U )
		for (i = 0; i < grid.nx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			U[idx] = (((T) 2.0 * courant - (T) 1.0) * U[idx - 1] +
				U_p[idx] + U_p[idx - 1]) * del;
		}
	}

	u_ghost_extrapolation< T, side_bc >(U,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

template< typename T, int side_bc >
void nse::v_open_bc(T* V, const T* V_p,
	const uniGrid2d< T >& grid, const T dt)
{
	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	const T c_velocity = (T) 1.0;

	// * west side [DO!] * //
	if (is_west)
	{
		const T courant = c_velocity * dt / grid.dx;
		const T del = (T) 1.0 / (
			(T) 2.0 * courant - (T) 1.0);

		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			V[idx] = (((T) 1.0 + (T) 2.0 * courant) * V[idx + grid.ny] -
				V_p[idx] - V_p[idx + grid.ny]) * del;
		}
	}

	// * east side * //
	if (is_east)
	{
		T phase;
		int j, idx;
#pragma omp parallel for private( j, idx, phase ) shared( V, V_p )
		for (j = 0; j < grid.ny; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			phase = -(grid.dx / dt) * (
				(V[idx - grid.ny] - V_p[idx - grid.ny]) /
				(V_p[idx - grid.ny] - V_p[idx - 2 * grid.ny]));
			if (phase < (T)0) phase = (T)0;
			if (phase > grid.dx / dt) phase = grid.dx / dt;

			V[idx] = ((T)1 - phase * (dt / grid.dx)) * V_p[idx] +
				phase * (dt / grid.dx) * V_p[idx - grid.ny];
		}
	}

	// * south side [DO!] * //
	if (is_south)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 - (T) 3.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;

			V[idx] = ((T) 2.0 * V_p[idx] -
				(T) 4.0 * courant * V[idx + 1] +
				courant * V[idx + 2]) * del;
		}
	}

	// * north side [DO!] * //
	if (is_north)
	{
		const T courant = c_velocity * dt / grid.dy;
		const T del = (T) 1.0 / (
			(T) 2.0 + (T) 3.0 * courant);

		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			V[idx] = ((T) 2.0 * V_p[idx] +
				(T) 4.0 * courant * V[idx - 1] -
				courant * V[idx - 2]) * del;
		}
	}

	v_ghost_extrapolation< T, side_bc >(V,
		grid.nx, grid.ny, grid.gcx, grid.gcy, 1, 1,
		grid.mpi_com);
}

// ------------------------------------------------------------------------- //
// External boundary average (u,v,c)
// ============================================
template< typename T, int side_bc >
T nse::c_external_average(const T* X,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 15.0 / (T) 8.0,
		c2 = -(T) 5.0 / (T) 4.0,
		c3 = (T) 3.0 / (T) 8.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	T avg = (T) 0.0;

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			avg += c1 * X[idx + grid.ny] +
				c2 * X[idx + (grid.ny << 1)] +
				c3 * X[idx + (grid.ny << 1) + grid.ny];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( X ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			avg += c1 * X[idx - grid.ny] +
				c2 * X[idx - (grid.ny << 1)] +
				c3 * X[idx - (grid.ny << 1) - grid.ny];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			avg += c1 * X[idx + 1] +
				c2 * X[idx + 2] +
				c3 * X[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( X ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			avg += c1 * X[idx - 1] +
				c2 * X[idx - 2] +
				c3 * X[idx - 3];
		}
	}

	// reduction for average boundary conditions
	avg = mpi_allreduce(avg, MPI_SUM);
	if ((side_bc == _west_bc) || (side_bc == _east_bc))
		return avg / (grid.mpi_ny - 2 * grid.gcy);
	if ((side_bc == _south_bc) || (side_bc == _north_bc))
		return avg / (grid.mpi_nx - 2 * grid.gcx);
}

template< typename T, int side_bc >
T nse::u_external_average(const T* U,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 15.0 / (T) 8.0,
		c2 = -(T) 5.0 / (T) 4.0,
		c3 = (T) 3.0 / (T) 8.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	T avg = (T) 0.0;

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = grid.gcx * grid.ny + j;

			avg += (T) 3.0 * (U[idx + grid.ny] - U[idx + (grid.ny << 1)])
				+ U[idx + (grid.ny << 1) + grid.ny];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( U ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			avg += (T) 3.0 * (U[idx - grid.ny] - U[idx - (grid.ny << 1)])
				+ U[idx - (grid.ny << 1) - grid.ny];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.gcy - 1);

			avg += c1 * U[idx + 1] +
				c2 * U[idx + 2] +
				c3 * U[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( U ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			avg += c1 * U[idx - 1] +
				c2 * U[idx - 2] +
				c3 * U[idx - 3];
		}
	}

	// reduction for average boundary conditions
	avg = mpi_allreduce(avg, MPI_SUM);
	if ((side_bc == _west_bc) || (side_bc == _east_bc))
		return avg / (grid.mpi_ny - 2 * grid.gcy);
	if ((side_bc == _south_bc) || (side_bc == _north_bc))
		return avg / (grid.mpi_nx - 2 * grid.gcx);
}

template< typename T, int side_bc >
T nse::v_external_average(const T* V,
	const uniGrid2d< T >& grid)
{
	const T c1 = (T) 15.0 / (T) 8.0,
		c2 = -(T) 5.0 / (T) 4.0,
		c3 = (T) 3.0 / (T) 8.0;

	const int is_west = (side_bc == _west_bc) && (grid.mpi_com.rank_x == 0);
	const int is_east = (side_bc == _east_bc) && (grid.mpi_com.rank_x == grid.mpi_com.size_x - 1);
	const int is_south = (side_bc == _south_bc) && (grid.mpi_com.rank_y == 0);
	const int is_north = (side_bc == _north_bc) && (grid.mpi_com.rank_y == grid.mpi_com.size_y - 1);

	T avg = (T) 0.0;

	// * west side * //
	if (is_west)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.gcx - 1) * grid.ny + j;

			avg += c1 * V[idx + grid.ny] +
				c2 * V[idx + (grid.ny << 1)] +
				c3 * V[idx + (grid.ny << 1) + grid.ny];
		}
	}

	// * east side * //
	if (is_east)
	{
		int j, idx;
#pragma omp parallel for private( j, idx ) shared( V ) reduction( + : avg )
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = (grid.nx - grid.gcx) * grid.ny + j;

			avg += c1 * V[idx - grid.ny] +
				c2 * V[idx - (grid.ny << 1)] +
				c3 * V[idx - (grid.ny << 1) - grid.ny];
		}
	}

	// * south side * //
	if (is_south)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + grid.gcy;

			avg += (T) 3.0 * (V[idx + 1] - V[idx + 2])
				+ V[idx + 3];
		}
	}

	// * north side * //
	if (is_north)
	{
		int i, idx;
#pragma omp parallel for private( i, idx ) shared( V ) reduction( + : avg )
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
			idx = i * grid.ny + (grid.ny - grid.gcy);

			avg += (T) 3.0 * (V[idx - 1] - V[idx - 2])
				+ V[idx - 3];
		}
	}

	// reduction for average boundary conditions
	avg = mpi_allreduce(avg, MPI_SUM);
	if ((side_bc == _west_bc) || (side_bc == _east_bc))
		return avg / (grid.mpi_ny - 2 * grid.gcy);
	if ((side_bc == _south_bc) || (side_bc == _north_bc))
		return avg / (grid.mpi_nx - 2 * grid.gcx);
}

template< typename T >
void nse::c_init_stratification(T* X,
	const T x_min, const T x_max, const T y_level, const T y_eps,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T py, value;

#pragma omp parallel for private( i, j, idx, py, value ) shared( X )
	for (j = grid.gcy; j < grid.ny - grid.gcy; j++)
	{
		py = grid.y + (j - grid.gcy) * grid.dy + grid.dyh;
		value = x_min + (x_max - x_min) * linear_step(py - y_level, y_eps);

		idx = grid.gcx * grid.ny + j;
		for (i = grid.gcx; i < grid.nx - grid.gcx; i++, idx += grid.ny) {
			X[idx] = value;
		}
	}
}

template< typename T >
void nse::add_disturbance(T* X,
	const T variance, const long int seed,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	GaussRand gen;

	gen.set((double)0, fabs((double)variance), seed);

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++) {
			idx = i * grid.ny + j;

			// gauss noise
			X[idx] += (T)gen.mt_rand();
		}
	}
}