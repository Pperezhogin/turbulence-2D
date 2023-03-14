#pragma once

#include "pois-const2d.h"
#include "mpi-com2d.h"

// *[pois-base2d.h]: full //

namespace poisson2d
{
	// * matvec * //
	template< typename T >
	void matvec(T* y, const T* x,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	void matvec_x4(T* y, const T* x,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	void matvec(T* y, const T* x, const T* i_density,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2ih, const T dy2ih);


	template< typename T >
	T matvec_dp(T* y, const T* x,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	T matvec_dp_x4(T* y, const T* x,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	T matvec_dp(T* y, const T* x, const T* i_density,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2ih, const T dy2ih);


	// * resvec * //
	template< typename T >
	void resvec(T* y, const T* x, const T* rhs,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	void resvec_x4(T* y, const T* x, const T* rhs,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	template< typename T >
	void resvec(T* y, const T* x, const T* rhs, const T* i_density,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2ih, const T dy2ih);


	// * diagonal inverse * //
	template< typename T >
	void set_diagonal_inverse(T* idg, const T* i_density,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2ih, const T dy2ih);


	// * symmetric gauss-seidel poisson init * //
	template< typename T > inline
		void sgs_init(T* x, const T* rhs, const T idg,
		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy);

	// * symmetric gauss-seidel poisson cycle * //
	template< typename T > inline
		void sgs_cycle(T* x, const T* rhs, const T idg,
		const int color,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);

	// * symmetric sor poisson cycle * //
	template< typename T > inline
		void ssor_cycle(T* x, const T* rhs, const T omega_x, const T omega_rhs,
		const int color,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2i, const T dy2i);


	// * symmetric gauss-seidel var poisson init * //
	template< typename T > inline
		void sgs_init(T* x, const T* rhs, const T* idg,
		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy);

	// * symmetric gauss-seidel var poisson cycle * //
	template< typename T > inline
		void sgs_cycle(T* x, const T* rhs, const T* idg, const T* i_density,
		const int color,
		const int nx, const int ny,
		const int ib, const int ie,
		const int jb, const int je,

		const T dx2ih, const T dy2ih);

	// * boundary conditions * //
	template< typename T >          // pure BC //
	void put_bc(T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const int rank_x, const int rank_y,
		const int size_x, const int size_y,

		const int pois_bc_type);

	template< typename T >  // BC and MPI cross exchanges //
	void put_exch_bc(T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //
}



template< typename T > inline
void poisson2d::sgs_init(T* x, const T* rhs, const T idg,
const int color,
const int nx, const int ny,
const int gcx, const int gcy)
{
	int i, j, idx;

#pragma omp parallel shared( x, rhs ) 
	{
		// - x columns null boundary conditions
#pragma omp for private( i, j, idx ) nowait
		for (i = 0; i < gcx; i++) {

			idx = i * ny;
			for (j = 0; j < ny; j++, idx++)
				x[idx] = (T)0;

			idx = (nx - gcx + i) * ny;
			for (j = 0; j < ny; j++, idx++)
				x[idx] = (T)0;
		}

		// [red] and -y rows null boundary conditions //
#pragma omp for private( i, j, idx )
		for (i = gcx; i < nx - gcx; i++) {

			idx = i * ny + gcy + ((i + gcy + color) & 1);
			for (j = gcy + ((i + gcy + color) & 1); j < ny - gcy; j += 2, idx += 2)
				x[idx] = idg * rhs[idx];

			idx = i * ny;
			for (j = 0; j < gcy; j++, idx++) {
				x[idx] = (T)0;
				x[idx + ny - gcy] = (T)0;
			}
		}
	} // omp parallel end
}

template< typename T > inline
void poisson2d::sgs_cycle(T* x, const T* rhs, const T idg,
const int color,
const int nx, const int ny,
const int ib, const int ie,
const int jb, const int je,

const T dx2i, const T dy2i)
{
	int i, j, idx, shc;

	if ((je - jb + 1 <= 2) && (je - jb + 1 < ie - ib + 1))
	{

#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs )
		for (j = jb; j <= je; j++) {

			shc = ((ib + j + color) & 1);
			idx = (ib + shc) * ny + j;
			for (i = ib + shc; i <= ie; i += 2, idx += (ny << 1)) {
				x[idx] = idg * (rhs[idx] -
					(x[idx - ny] + x[idx + ny]) * dx2i -
					(x[idx - 1] + x[idx + 1]) * dy2i);
			}
		}

		return;
	}

#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs )
	for (i = ib; i <= ie; i++) {

		shc = ((i + jb + color) & 1);
		idx = i * ny + jb + shc;

		for (j = jb + shc; j <= je; j += 2, idx += 2) {
			x[idx] = idg * (rhs[idx] -
				(x[idx - ny] + x[idx + ny]) * dx2i -
				(x[idx - 1] + x[idx + 1]) * dy2i);
		}
	}
}

template< typename T > inline
void poisson2d::ssor_cycle(T* x, const T* rhs, const T omega_x, const T omega_rhs,
const int color,
const int nx, const int ny,
const int ib, const int ie,
const int jb, const int je,

const T dx2i, const T dy2i)
{
	int i, j, idx, shc;

	if ((je - jb + 1 <= 2) && (je - jb + 1 < ie - ib + 1))
	{

#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs )
		for (j = jb; j <= je; j++) {

			shc = ((ib + j + color) & 1);
			idx = (ib + shc) * ny + j;
			for (i = ib + shc; i <= ie; i += 2, idx += (ny << 1)) {
				x[idx] = omega_x * x[idx] + omega_rhs * (rhs[idx] -
					(x[idx - ny] + x[idx + ny]) * dx2i -
					(x[idx - 1] + x[idx + 1]) * dy2i);
			}
		}

		return;
	}

#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs )
	for (i = ib; i <= ie; i++) {

		shc = (i + jb + color) & 1;
		idx = i * ny + jb + shc;
		for (j = jb + shc; j <= je; j += 2, idx += 2) {
			x[idx] = omega_x * x[idx] + omega_rhs * (rhs[idx] -
				(x[idx - ny] + x[idx + ny]) * dx2i -
				(x[idx - 1] + x[idx + 1]) * dy2i);
		}
	}
}


template< typename T > inline
void poisson2d::sgs_init(T* x, const T* rhs, const T* idg,
const int color,
const int nx, const int ny,
const int gcx, const int gcy)
{
	int i, j, idx;

#pragma omp parallel shared( x, rhs, idg ) 
	{
		// - x columns null boundary conditions
#pragma omp for private( i, j, idx )
		for (i = 0; i < gcx; i++) {

			idx = i * ny;
			for (j = 0; j < ny; j++, idx++)
				x[idx] = (T)0;

			idx = (nx - gcx + i) * ny;
			for (j = 0; j < ny; j++, idx++)
				x[idx] = (T)0;
		}

		// [red] and -y rows null boundary conditions //
#pragma omp for private( i, j, idx )
		for (i = gcx; i < nx - gcx; i++) {

			idx = i * ny + gcy + ((i + gcy + color) & 1);
			for (j = gcy + ((i + gcy + color) & 1); j < ny - gcy; j += 2, idx += 2)
				x[idx] = idg[idx] * rhs[idx];

			idx = i * ny;
			for (j = 0; j < gcy; j++, idx++) {
				x[idx] = (T)0;
				x[idx + ny - gcy] = (T)0;
			}
		}
	} // omp parallel end
}

template< typename T > inline
void poisson2d::sgs_cycle(T* x, const T* rhs, const T* idg, const T* i_density,
const int color,
const int nx, const int ny,
const int ib, const int ie,
const int jb, const int je,

const T dx2ih, const T dy2ih)
{
	int i, j, idx, shc;

	if ((je - jb + 1 <= 2) && (je - jb + 1 < ie - ib + 1))
	{
#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs, idg, i_density )
		for (j = jb; j <= je; j++) {

			shc = ((ib + j + color) & 1);
			idx = (ib + shc) * ny + j;
			for (i = ib + shc; i <= ie; i += 2, idx += (ny << 1)) {
				x[idx] = idg[idx] * (rhs[idx] -
					(
					x[idx + ny] * (i_density[idx + ny] + i_density[idx]) +
					x[idx - ny] * (i_density[idx - ny] + i_density[idx])

					) * dx2ih

					-

					(
					x[idx + 1] * (i_density[idx + 1] + i_density[idx]) +
					x[idx - 1] * (i_density[idx - 1] + i_density[idx])

					) * dy2ih);
			}
		}

		return;
	}

#pragma omp parallel for private( i, j, idx, shc ) shared( x, rhs, idg, i_density )
	for (i = ib; i <= ie; i++) {

		shc = (i + jb + color) & 1;
		idx = i * ny + jb + shc;
		for (j = jb + shc; j <= je; j += 2, idx += 2) {
			x[idx] = idg[idx] * (rhs[idx] -
				(
				x[idx + ny] * (i_density[idx + ny] + i_density[idx]) +
				x[idx - ny] * (i_density[idx - ny] + i_density[idx])

				) * dx2ih

				-

				(
				x[idx + 1] * (i_density[idx + 1] + i_density[idx]) +
				x[idx - 1] * (i_density[idx - 1] + i_density[idx])

				) * dy2ih);
		}
	}
}
