#include "pois-base2d.h"

template< typename T >
void poisson2d::matvec( // matrix vector multiplication
	T* y, const T* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++) {
			y[idx] =
				(x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) * dx2i +
				(x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) * dy2i;
		}
	}
}

template< typename T >
void poisson2d::matvec_x4( // matrix vector multiplication
	T* y, const T* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x )
	for (i = ib; i <= ie; i++)
	{
		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++)
		{
			y[idx] =
				(C1 * C1 * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) +
				C2 * C2 * (x[idx + 3 * ny] - x[idx] - x[idx] + x[idx - 3 * ny]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2 * ny] - x[idx + ny] - x[idx - ny] + x[idx - 2 * ny])) * dx2i +

				(C1 * C1 * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) +
				C2 * C2 * (x[idx + 3] - x[idx] - x[idx] + x[idx - 3]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2] - x[idx + 1] - x[idx - 1] + x[idx - 2])) * dy2i;
		}
	}
}

template< typename T >
void poisson2d::matvec( // matrix vector multiplication
	T* y, const T* x, const T* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2ih, const T dy2ih)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x, i_density )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++) {

			y[idx] =
				(
				i_density[idx + ny] * (x[idx + ny] - x[idx]) -
				i_density[idx - ny] * (x[idx] - x[idx - ny]) +
				i_density[idx] * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny])
				) * dx2ih

				+

				(
				i_density[idx + 1] * (x[idx + 1] - x[idx]) -
				i_density[idx - 1] * (x[idx] - x[idx - 1]) +
				i_density[idx] * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1])
				) * dy2ih;
		}
	}
}

template< typename T >
T poisson2d::matvec_dp( // matrix vector multiplication and dot product
	T* y, const T* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	const int jsh = (je - jb + 1) & 1;

	int i, j, idx;
	T dp = (T)0;

#pragma omp parallel for private( i, j, idx ) shared( y, x ) reduction( + : dp )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;

		for (j = jb; j <= je - jsh; j += 2, idx += 2) {
			y[idx] =
				(x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) * dx2i +
				(x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) * dy2i;

			y[idx + 1] =
				(x[idx + ny + 1] - x[idx + 1] - x[idx + 1] + x[idx - ny + 1]) * dx2i +
				(x[idx + 2] - x[idx + 1] - x[idx + 1] + x[idx]) * dy2i;

			dp += y[idx] * x[idx] + y[idx + 1] * x[idx + 1];
		}

		if (jsh) {
			idx = i * ny + je;

			y[idx] =
				(x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) * dx2i +
				(x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) * dy2i;


			dp += y[idx] * x[idx];
		}
	}

	return dp;
}

template< typename T >
T poisson2d::matvec_dp_x4( // matrix vector multiplication and dot product
	T* y, const T* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;

	const int jsh = (je - jb + 1) & 1;

	int i, j, idx;
	T dp = (T)0;

#pragma omp parallel for private( i, j, idx ) shared( y, x ) reduction( + : dp )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;

		for (j = jb; j <= je - jsh; j += 2, idx += 2) {
			y[idx] =
				(C1 * C1 * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) +
				C2 * C2 * (x[idx + 3 * ny] - x[idx] - x[idx] + x[idx - 3 * ny]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2 * ny] - x[idx + ny] - x[idx - ny] + x[idx - 2 * ny])) * dx2i +

				(C1 * C1 * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) +
				C2 * C2 * (x[idx + 3] - x[idx] - x[idx] + x[idx - 3]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2] - x[idx + 1] - x[idx - 1] + x[idx - 2])) * dy2i;

			
			y[idx + 1] =
				(C1 * C1 * (x[idx + ny + 1] - x[idx + 1] - x[idx + 1] + x[idx - ny + 1]) +
				C2 * C2 * (x[idx + 3 * ny + 1] - x[idx + 1] - x[idx + 1] + x[idx - 3 * ny + 1]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2 * ny + 1] - x[idx + ny + 1] - x[idx - ny + 1] + x[idx - 2 * ny + 1])) * dx2i +

				(C1 * C1 * (x[idx + 2] - x[idx + 1] - x[idx + 1] + x[idx]) +
				C2 * C2 * (x[idx + 4] - x[idx + 1] - x[idx + 1] + x[idx - 2]) -
				(T) 2.0 * C1 * C2 * (x[idx + 3] - x[idx + 2] - x[idx] + x[idx - 1])) * dy2i;

			dp += y[idx] * x[idx] + y[idx + 1] * x[idx + 1];
		}

		if (jsh) {
			idx = i * ny + je;

			y[idx] =
				(C1 * C1 * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) +
				C2 * C2 * (x[idx + 3 * ny] - x[idx] - x[idx] + x[idx - 3 * ny]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2 * ny] - x[idx + ny] - x[idx - ny] + x[idx - 2 * ny])) * dx2i +

				(C1 * C1 * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) +
				C2 * C2 * (x[idx + 3] - x[idx] - x[idx] + x[idx - 3]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2] - x[idx + 1] - x[idx - 1] + x[idx - 2])) * dy2i;


			dp += y[idx] * x[idx];
		}
	}

	return dp;
}

template< typename T >
T poisson2d::matvec_dp( // matrix vector multiplication
	T* y, const T* x, const T* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2ih, const T dy2ih)
{
	int i, j, idx;
	T dp = (T)0;

#pragma omp parallel for private( i, j, idx ) shared( y, x, i_density ) reduction( + : dp )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;

		for (j = jb; j <= je; j++, idx++) {

			y[idx] =
				(
				i_density[idx + ny] * (x[idx + ny] - x[idx]) -
				i_density[idx - ny] * (x[idx] - x[idx - ny]) +
				i_density[idx] * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny])
				) * dx2ih

				+

				(
				i_density[idx + 1] * (x[idx + 1] - x[idx]) -
				i_density[idx - 1] * (x[idx] - x[idx - 1]) +
				i_density[idx] * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1])
				) * dy2ih;

			dp += y[idx] * x[idx];
		}
	}

	return dp;
}

template< typename T >
void poisson2d::resvec( // poisson residual
	T* y, const T* x, const T* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x, rhs )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++) {
			y[idx] = rhs[idx] -
				(x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) * dx2i -
				(x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) * dy2i;
		}
	}
}

template< typename T >
void poisson2d::resvec_x4( // poisson residual
	T* y, const T* x, const T* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2i, const T dy2i)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x, rhs )
	for (i = ib; i <= ie; i++)
	{
		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++)
		{
			y[idx] = rhs[idx] -
				(C1 * C1 * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny]) +
				C2 * C2 * (x[idx + 3 * ny] - x[idx] - x[idx] + x[idx - 3 * ny]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2 * ny] - x[idx + ny] - x[idx - ny] + x[idx - 2 * ny])) * dx2i -

				(C1 * C1 * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1]) +
				C2 * C2 * (x[idx + 3] - x[idx] - x[idx] + x[idx - 3]) -
				(T) 2.0 * C1 * C2 * (x[idx + 2] - x[idx + 1] - x[idx - 1] + x[idx - 2])) * dy2i;
		}
	}
}

template< typename T >
void poisson2d::resvec( // poisson residual
	T* y, const T* x, const T* rhs, const T* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2ih, const T dy2ih)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y, x, rhs, i_density )
	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++) {

			y[idx] = rhs[idx] -
				(
				i_density[idx + ny] * (x[idx + ny] - x[idx]) -
				i_density[idx - ny] * (x[idx] - x[idx - ny]) +
				i_density[idx] * (x[idx + ny] - x[idx] - x[idx] + x[idx - ny])
				) * dx2ih

				-

				(
				i_density[idx + 1] * (x[idx + 1] - x[idx]) -
				i_density[idx - 1] * (x[idx] - x[idx - 1]) +
				i_density[idx] * (x[idx + 1] - x[idx] - x[idx] + x[idx - 1])
				) * dy2ih;
		}
	}
}


template< typename T >
void poisson2d::set_diagonal_inverse( // diagonal inverse
	T* idg, const T* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const T dx2ih, const T dy2ih)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( idg, i_density )
	for (i = ib; i <= ie; i++)
	{
		idx = i * ny + jb;
		for (j = jb; j <= je; j++, idx++)
		{
			idg[idx] = -(T) 1.0 /
				(
				(i_density[idx + ny] + i_density[idx - ny] +
				(T) 2.0 * i_density[idx]) * dx2ih
				+
				(i_density[idx + 1] + i_density[idx - 1] +
				(T) 2.0 * i_density[idx]) * dy2ih
				);
		}
	}
}

// * Boundary Conditions * //
template< typename T >
void poisson2d::put_exch_bc( // poisson boundary conditions
	T* x,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	put_bc(x,
		nx, ny,
		gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,

		pois_bc_type);

	// * MPI exchanges * //
	mpi_com.exchange_cross_halo(x,
		nx, ny, gcx, gcy, 1, 1,
		x_periodic, y_periodic);
}

template< typename T >
void poisson2d::put_bc( // poisson boundary conditions
	T* x,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int rank_x, const int rank_y,
	const int size_x, const int size_y,

	const int pois_bc_type)
{
	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) || 
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) || 
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	int i, j, idx;
	int jb, je, ib, ie;

#pragma omp parallel shared( x )
	{
		// west boundary condition //
		if ((rank_x == 0) && (!x_periodic)) {

			// x = 0, y = 0 --> j = gcy to ny
			// x = 0, y = size_y - 1 --> j = 0 to ny - gcy
			// x = 0, 0 < y < size_y - 1 --> j = 0 to ny

			jb = ((rank_y > 0) || ((rank_y == 0) && (y_periodic))) ? 0 : gcy;
			je = ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (y_periodic))) ? ny : ny - gcy;

			idx = (gcx - 1) * ny;
			if (pois_bc_type == nse::c_pois_bc_west_ext) {  // west outflow //
#pragma omp for private( j ) nowait
				for (j = jb; j < je; j++)
					x[idx + j] = -x[idx + ny + j];
			}
			else                                                // neumann //
			{
#pragma omp for private( j ) nowait
				for (j = jb; j < je; j++)
					x[idx + j] = x[idx + ny + j];
			}

#ifdef _POIS2D_BC_DIRICHLET_POINT
			// special handler for -y periodic bc //
			if ((rank_y == 0) && (pois_bc_type == nse::c_pois_bc_periodic_y))
				x[idx + gcy] = -x[idx + ny + gcy];
#endif
		}

		// east boundary conditions //
		if ((rank_x == size_x - 1) && (!x_periodic)) {

			jb = ((rank_y > 0) || ((rank_y == 0) && (y_periodic))) ? 0 : gcy;
			je = ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (y_periodic))) ? ny : ny - gcy;

			idx = (nx - gcx) * ny;
			if (pois_bc_type == nse::c_pois_bc_east_ext) {  // east outflow //
#pragma omp for private( j ) nowait
				for (j = jb; j < je; j++)
					x[idx + j] = -x[idx - ny + j];
			}
			else                                                // neumann //
			{
#pragma omp for private( j ) nowait
				for (j = jb; j < je; j++)
					x[idx + j] = x[idx - ny + j];
			}

		}

		// south boundary conditions //
		if ((rank_y == 0) && (!y_periodic)) {

			// x = 0, y = 0 --> i = gcx to nx
			// x = size_x - 1, y = 0 --> i = 0 to nx - gcx
			// 0 < x < size_x - 1, y = 0 --> i = 0 to nx

			ib = ((rank_x > 0) || ((rank_x == 0) && (x_periodic))) ? 0 : gcx;
			ie = ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (x_periodic))) ? nx : nx - gcx;

			if (pois_bc_type == nse::c_pois_bc_south_ext) { // south outflow //

#pragma omp for private( i, idx ) nowait
				for (i = ib; i < ie; i++) {
					idx = i * ny + gcy - 1;
					x[idx] = -x[idx + 1];
				}
			}
			else
			{
#pragma omp for private( i, idx ) nowait
				for (i = ib; i < ie; i++) {                   // neumann //
					idx = i * ny + gcy - 1;
					x[idx] = x[idx + 1];
				}
			}

#ifdef _POIS2D_BC_DIRICHLET_POINT
			// special handler for -x periodic bc //
			if ((rank_x == 0) && (pois_bc_type == nse::c_pois_bc_periodic_x))
			{
				idx = gcx * ny + gcy - 1;
				x[idx] = -x[idx + 1];
			}
#endif
		}

		// north boundary conditions //
		if ((rank_y == size_y - 1) && (!y_periodic)) {

			ib = ((rank_x > 0) || ((rank_x == 0) && (x_periodic))) ? 0 : gcx;
			ie = ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (x_periodic))) ? nx : nx - gcx;

			if (pois_bc_type == nse::c_pois_bc_north_ext) { // north outflow //
#pragma omp for private( i, idx ) nowait
				for (i = ib; i < ie; i++) {
					idx = i * ny + ny - gcy;
					x[idx] = -x[idx - 1];
				}
			}
			else
			{
#pragma omp for private( i, idx ) nowait
				for (i = ib; i < ie; i++) {                   // neumann //
					idx = i * ny + ny - gcy;
					x[idx] = x[idx - 1];
				}
			}
		}
	}
}
// ------------------------------------------------------------------------ //


// * initialize: matvec operations * //
template void poisson2d::matvec(float* y, const float* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template void poisson2d::matvec(double* y, const double* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template void poisson2d::matvec_x4(float* y, const float* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template void poisson2d::matvec_x4(double* y, const double* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template void poisson2d::matvec(float* y, const float* x, const float* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2ih, const float dy2ih);
template void poisson2d::matvec(double* y, const double* x, const double* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2ih, const double dy2ih);


template float poisson2d::matvec_dp(float* y, const float* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template double poisson2d::matvec_dp(double* y, const double* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template float poisson2d::matvec_dp_x4(float* y, const float* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template double poisson2d::matvec_dp_x4(double* y, const double* x,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template float poisson2d::matvec_dp(float* y, const float* x, const float* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2ih, const float dy2ih);
template double poisson2d::matvec_dp(double* y, const double* x, const double* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2ih, const double dy2ih);


// * initialize: resvec operations * //
template void poisson2d::resvec(float* y, const float* x, const float* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template void poisson2d::resvec(double* y, const double* x, const double* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template void poisson2d::resvec_x4(float* y, const float* x, const float* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2i, const float dy2i);
template void poisson2d::resvec_x4(double* y, const double* x, const double* rhs,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2i, const double dy2i);

template void poisson2d::resvec(float* y, const float* x, const float* rhs, const float* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2ih, const float dy2ih);
template void poisson2d::resvec(double* y, const double* x, const double* rhs, const double* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2ih, const double dy2ih);


// * initialize: diagonal inverse * //
template void poisson2d::set_diagonal_inverse(float* idg, const float* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const float dx2ih, const float dy2ih);
template void poisson2d::set_diagonal_inverse(double* idg, const double* i_density,
	const int nx, const int ny,
	const int ib, const int ie,
	const int jb, const int je,

	const double dx2ih, const double dy2ih);


// * initialize: boundary conditions * //
template void poisson2d::put_exch_bc(float* x,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const nse::mpiCom2d& mpi_com, const int pois_bc_type);
template void poisson2d::put_exch_bc(double* x,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const nse::mpiCom2d& mpi_com, const int pois_bc_type);

template void poisson2d::put_bc(float* x,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int rank_x, const int rank_y,
	const int size_x, const int size_y,

	const int pois_bc_type);
template void poisson2d::put_bc(double* x,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int rank_x, const int rank_y,
	const int size_x, const int size_y,

	const int pois_bc_type);
