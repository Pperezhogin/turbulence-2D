#include "pois2d.h"
#include "pois-base2d.h"

#include <string.h>
#include "vecmath.h"

namespace poisson2d {

	// *** Laplas and Dot Product (MPI inside) *** //
	template< typename T >
	T laplas_dp(T* y, T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	T laplas_dp_x4(T* y, T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	T laplas_dp(T* y, T* x, const T* i_density,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2ih, const T dy2ih,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** Residual (MPI inside) *** //
	template< typename T >
	void laplas_residual(T* y, T* x, const T* rhs,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void laplas_residual_x4(T* y, T* x, const T* rhs,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void laplas_residual(T* y, T* x, const T* rhs, const T* i_density,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2ih, const T dy2ih,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** Restrict Residual (MPI inside) *** //
	template< typename T >
	void laplas_restrict_residual(T* y_coarse, T* x_fine, const T* rhs_fine,
		const int type,

		const int cnx, const int cny,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i, // grid params on fine grid //

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** Prolongate X (MPI inside) *** //
	template< typename T >
	void laplas_prolongate(T* x_fine, T* x_coarse,
		const int type,

		const int nx, const int ny,
		const int cnx, const int cny,
		const int gcx, const int gcy,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** SGS(SSOR) Red-Black Preconditioners *** //
	const int c_sgs_init = 0;
	const int c_sgs_continue = 1;

	template< typename T >
	void sgs_start(T* x, T* rhs, const T idg, const int type,

		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com, const int pois_bc_type);

	template< typename T >
	void sgs_run(T* x, const T* rhs, const T idg,

		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com, const int pois_bc_type);

	template< typename T >
	void sgs_run_cache(const int niters,
		T* x, const T* rhs, const T idg,

		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy,

		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com, const int pois_bc_type);

	template< typename T >
	void sgs_redblack(T* x, T* rhs, const T idg,

		const int type, const int color, const int piters,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void sgs_redblack(T* x, const T* rhs, const T* idg, const T* i_density,

		const int type, const int color, const int piters,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const T dx2ih, const T dy2ih,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void ssor_redblack(T* x, const T* rhs, const T idg, const T omega,

		const int type, const int color, const int piters,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** Jacobi Preconditioner *** //
	template< typename T >
	void jacobi(T* x, const T* rhs, T* mem, const T idg,
		const int piters,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	// *** Approximate Inverse Preconditioner *** //
	template< typename T >
	void aip(T* x, const T* rhs, T* mem, const T idg,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const T dx2i, const T dy2i,

		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //

	template< typename T >
	void mg_sgs_redblack(T* x, T* rhs, const int piters,
		nse::mg_poisson2d_data< T >& mg_data,
		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void mg_sgs_redblack(T* x, T* rhs, const int piters,
		nse::mg_var_poisson2d_data< T >& mg_data,
		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);

	template< typename T >
	void mg_sgs_redblack(T* x, T* rhs, const int piters,
		nse::mg_mpi_poisson2d_data< T >& mg_data,
		const nse::mpiCom2d& mpi_com,
		const int pois_bc_type);
	// -------------------------------------------------------------------- //
}


// * CG SGS (Red-Black) * //
template< typename T >
int nse::cg_sgs_redblack(
	T* x, const T* rhs, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid, const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T idg = (T) 1.0 /
		(-(T) 2.0 * grid.dx2i - (T) 2.0 * grid.dy2i);

	int sgs_mode = ((grid.mpi_com.offset_x(grid.nx, grid.gcx) +
		grid.mpi_com.offset_y(grid.ny, grid.gcy)) & 1);

	T *residual = memory,
		*p = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*w = &memory[(grid.size << 1) + grid.size];

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual(residual, x, rhs,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}
	if (norm_star < abstol * (T)1.0E-10) { // in case of zero initial conditions
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::sgs_redblack(w, residual, idg,
		poisson2d::c_sgs_init, sgs_mode,
		piters,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,
		grid.mpi_com, pois_bc_type);

	// - additional preconditioners //
	/*
	poisson2d :: jacobi( w, residual, q, idg,
	piters,
	grid.nx, grid.ny,
	grid.gcx, grid.gcy,
	grid.dx2i, grid.dy2i,
	grid.mpi_com, pois_bc_type );

	poisson2d :: aip( w, residual, q, idg,
	grid.nx, grid.ny,
	grid.gcx, grid.gcy,
	grid.dx2i, grid.dy2i,
	grid.mpi_com, pois_bc_type );
	*/
	// =========================== //

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp(q, p,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2i, grid.dy2i,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::sgs_redblack(w, residual, idg,
			poisson2d::c_sgs_init, sgs_mode,
			piters,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2i, grid.dy2i,
			grid.mpi_com, pois_bc_type);

		// - additional preconditioners //
		/*
		poisson2d :: jacobi( w, residual, q, idg,
		piters,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,
		grid.mpi_com, pois_bc_type );

		poisson2d :: aip( w, residual, q, idg,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,
		grid.mpi_com, pois_bc_type );
		*/
		// ============================ //

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) && 
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //


// * CG Multigrid SGS (Red-Black) * //
template< typename T >
int nse::cg_mg_sgs_redblack(
	T* x, const T* rhs, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid,
	mg_poisson2d_data< T >& mg_data,
	const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *w = memory,
		*residual = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*p = &memory[(grid.size << 1) + grid.size];

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual(residual, x, rhs,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::mg_sgs_redblack(
		w, residual, piters,
		mg_data, grid.mpi_com, pois_bc_type);

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp(q, p,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2i, grid.dy2i,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::mg_sgs_redblack(
			w, residual, piters,
			mg_data, grid.mpi_com, pois_bc_type);

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) &&
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //

// * CG MPI Multigrid SGS (Red-Black) * //
template< typename T >
int nse::cg_mg_sgs_redblack(
	T* x, const T* rhs, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid,
	mg_mpi_poisson2d_data< T >& mg_data,
	const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*p = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*w = &memory[(grid.size << 1) + grid.size];

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual(residual, x, rhs,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::mg_sgs_redblack(
		w, residual, piters,
		mg_data, grid.mpi_com, pois_bc_type);

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp(q, p,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2i, grid.dy2i,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::mg_sgs_redblack(
			w, residual, piters,
			mg_data, grid.mpi_com, pois_bc_type);

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) &&
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //

// * CG MPI Multigrid SGS (Red-Black) * //
template< typename T >
int nse::cg_mg_sgs_redblack_x4(
	T* x, const T* rhs, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid,
	mg_mpi_poisson2d_data< T >& mg_data,
	const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*p = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*w = &memory[(grid.size << 1) + grid.size];

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual_x4(residual, x, rhs,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2i, grid.dy2i,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::mg_sgs_redblack(
		w, residual, piters,
		mg_data, grid.mpi_com, pois_bc_type);

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp_x4(q, p,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2i, grid.dy2i,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::mg_sgs_redblack(
			w, residual, piters,
			mg_data, grid.mpi_com, pois_bc_type);

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) &&
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //

// * CG SGS (Red-Black) (variable density) * //
template< typename T >
int nse::cg_sgs_redblack(
	T* x, const T* rhs, const T* i_density, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid, const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	int sgs_mode = ((grid.mpi_com.offset_x(grid.nx, grid.gcx) +
		grid.mpi_com.offset_y(grid.ny, grid.gcy)) & 1);

	T *residual = memory,
		*p = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*w = &memory[(grid.size << 1) + grid.size];

	T *idg = &memory[(grid.size << 2)];
	poisson2d::set_diagonal_inverse(idg, i_density,
		grid.nx, grid.ny,
		grid.gcx, grid.nx - grid.gcx - 1,
		grid.gcy, grid.ny - grid.gcy - 1,

		grid.dx2ih, grid.dy2ih);

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual(residual, x, rhs, i_density,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2ih, grid.dy2ih,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::sgs_redblack(w, residual, idg, i_density,
		poisson2d::c_sgs_init, sgs_mode,
		piters,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2ih, grid.dy2ih,
		grid.mpi_com, pois_bc_type);

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp(q, p, i_density,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2ih, grid.dy2ih,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::sgs_redblack(w, residual, idg, i_density,
			poisson2d::c_sgs_init, sgs_mode,
			piters,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2ih, grid.dy2ih,
			grid.mpi_com, pois_bc_type);

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) &&
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //


// * CG Multigrid SGS (Red-Black) (variable density) * //
template< typename T >
int nse::cg_mg_sgs_redblack(
	T* x, const T* rhs, const T* i_density, T* memory,

	const int piters,
	const T retol, const T abstol, const int maxiters,
	const uniGrid2d< T >& grid,
	mg_var_poisson2d_data< T >& mg_data,
	const int pois_bc_type,
	T* resnorm)
{
	T alpha, beta, gamma, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*p = &memory[grid.size],
		*q = &memory[(grid.size << 1)],
		*w = &memory[(grid.size << 1) + grid.size];

#ifdef _POIS2D_INIT_NULL
	memcpy(residual, rhs, grid.size * sizeof(T));
#else
	poisson2d::laplas_residual(residual, x, rhs, i_density,
		grid.nx, grid.ny,
		grid.gcx, grid.gcy,
		grid.dx2ih, grid.dy2ih,

		grid.mpi_com, pois_bc_type);
#endif

#if (_POIS2D_NORM == 2)
	norm_star = mpi_lnorm(residual, grid.size);
#else
	norm_star = mpi_cnorm(residual, grid.size);
#endif
	if ((norm_star < abstol) && (c_pois_miniters <= 0)) {
		(*resnorm) = norm_star;
		return 0;
	}

	poisson2d::mg_sgs_redblack(w, residual, piters,
		mg_data,
		grid.mpi_com, pois_bc_type);

	rho = mpi_dot_product(residual, w, grid.size);
	memcpy(p, w, sizeof(T)* grid.size);

	for (int k = 1; k <= maxiters; k++)
	{
		gamma = poisson2d::laplas_dp(q, p, i_density,
			grid.nx, grid.ny,
			grid.gcx, grid.gcy,
			grid.dx2ih, grid.dy2ih,
			grid.mpi_com, pois_bc_type);

		alpha = rho / gamma;

#ifdef _POIS2D_USE_SSE
		update_sse(x, residual,
			alpha, -alpha, p, q, grid.size);
#else
		update(x, residual,
			alpha, -alpha, p, q, grid.size);
#endif

		poisson2d::mg_sgs_redblack(w, residual, piters,
			mg_data,
			grid.mpi_com, pois_bc_type);

		rho_star = rho;
#if (_POIS2D_NORM == 2)
		mpi_lnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#else
		mpi_cnorm_and_dp(residual, w, grid.size, &norm_current, &rho);
#endif
		if (((norm_current < retol * norm_star) || (norm_current < abstol)) &&
			(k >= c_pois_miniters))
		{
			(*resnorm) = norm_current;
			return k;
		}

		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, grid.size);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}
// ------------------------------------------------------------------------ //

// * [laplas + dp] for poisson equation with async exchanges * //
template< typename T >
T poisson2d::laplas_dp(T* y, T* x,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];
	T dp;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,

		pois_bc_type);

	dp = matvec_dp(y, x, nx, ny,
		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2i, dy2i);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	if (p_west)
		dp += matvec_dp(y, x, nx, ny,
		ib, ib,
		jb, je,

		dx2i, dy2i);

	if (p_east)
		dp += matvec_dp(y, x, nx, ny,
		ie, ie,
		jb, je,

		dx2i, dy2i);

	if (p_south)
		dp += matvec_dp(y, x, nx, ny,
		ib + p_west, ie - p_east,
		jb, jb,

		dx2i, dy2i);

	if (p_north)
		dp += matvec_dp(y, x, nx, ny,
		ib + p_west, ie - p_east,
		je, je,

		dx2i, dy2i);

	nse::mpi_allreduce(&dp, MPI_SUM);

	return dp;
}

// * [laplas + dp] for poisson equation with async exchanges * //
template< typename T >
T poisson2d::laplas_dp_x4(T* y, T* x,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];
	T dp;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		3, 3, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,

		pois_bc_type);

	dp = matvec_dp_x4(y, x, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		jb + 3 * p_south, je - 3 * p_north,

		dx2i, dy2i);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		3, 3, x_periodic, y_periodic, mpi_req);

	if (p_west)
		dp += matvec_dp_x4(y, x, nx, ny,
		ib, ib + 3 * p_west - 1,
		jb, je,

		dx2i, dy2i);

	if (p_east)
		dp += matvec_dp_x4(y, x, nx, ny,
		ie - 3 * p_east + 1, ie,
		jb, je,

		dx2i, dy2i);

	if (p_south)
		dp += matvec_dp_x4(y, x, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		jb, jb + 3 * p_south - 1,

		dx2i, dy2i);

	if (p_north)
		dp += matvec_dp_x4(y, x, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		je - 3 * p_north + 1, je,

		dx2i, dy2i);

	nse::mpi_allreduce(&dp, MPI_SUM);

	return dp;
}

// * [laplas + dp] for var poisson equation with async exchanges * //
template< typename T >
T poisson2d::laplas_dp(T* y, T* x, const T* i_density,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2ih, const T dy2ih,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];
	T dp;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	dp = matvec_dp(y, x, i_density, nx, ny,
		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2ih, dy2ih);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	if (p_west)
		dp += matvec_dp(y, x, i_density, nx, ny,
		ib, ib,
		jb, je,

		dx2ih, dy2ih);

	if (p_east)
		dp += matvec_dp(y, x, i_density, nx, ny,
		ie, ie,
		jb, je,

		dx2ih, dy2ih);

	if (p_south)
		dp += matvec_dp(y, x, i_density, nx, ny,
		ib + p_west, ie - p_east,
		jb, jb,

		dx2ih, dy2ih);

	if (p_north)
		dp += matvec_dp(y, x, i_density, nx, ny,
		ib + p_west, ie - p_east,
		je, je,

		dx2ih, dy2ih);

	nse::mpi_allreduce(&dp, MPI_SUM);

	return dp;
}
// ------------------------------------------------------------------------ //

// * [laplas-residual] for poisson equation with async exchanges * //
template< typename T >
void poisson2d::laplas_residual(T* y, T* x, const T* rhs,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	resvec(y, x, rhs, nx, ny,
		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2i, dy2i);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	if (p_west)
		resvec(y, x, rhs, nx, ny,
		ib, ib,
		jb, je,

		dx2i, dy2i);

	if (p_east)
		resvec(y, x, rhs, nx, ny,
		ie, ie,
		jb, je,

		dx2i, dy2i);

	if (p_south)
		resvec(y, x, rhs, nx, ny,
		ib + p_west, ie - p_east,
		jb, jb,

		dx2i, dy2i);

	if (p_north)
		resvec(y, x, rhs, nx, ny,
		ib + p_west, ie - p_east,
		je, je,

		dx2i, dy2i);
}

// * [laplas-residual] for poisson equation with async exchanges * //
template< typename T >
void poisson2d::laplas_residual_x4(T* y, T* x, const T* rhs,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		3, 3, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	resvec_x4(y, x, rhs, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		jb + 3 * p_south, je - 3 * p_north,

		dx2i, dy2i);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		3, 3, x_periodic, y_periodic, mpi_req);

	if (p_west)
		resvec_x4(y, x, rhs, nx, ny,
		ib, ib + 3 * p_west - 1,
		jb, je,

		dx2i, dy2i);

	if (p_east)
		resvec_x4(y, x, rhs, nx, ny,
		ie - 3 * p_east + 1, ie,
		jb, je,

		dx2i, dy2i);

	if (p_south)
		resvec_x4(y, x, rhs, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		jb, jb + 3 * p_south - 1,

		dx2i, dy2i);

	if (p_north)
		resvec_x4(y, x, rhs, nx, ny,
		ib + 3 * p_west, ie - 3 * p_east,
		je - 3 * p_north + 1, je,

		dx2i, dy2i);
}

// * [laplas-residual] for var poisson equation with async exchanges * //
template< typename T >
void poisson2d::laplas_residual(T* y, T* x, const T* rhs, const T* i_density,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2ih, const T dy2ih,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	resvec(y, x, rhs, i_density, nx, ny,
		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2ih, dy2ih);

	mpi_com.pop_exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	if (p_west)
		resvec(y, x, rhs, i_density, nx, ny,
		ib, ib,
		jb, je,

		dx2ih, dy2ih);

	if (p_east)
		resvec(y, x, rhs, i_density, nx, ny,
		ie, ie,
		jb, je,

		dx2ih, dy2ih);

	if (p_south)
		resvec(y, x, rhs, i_density, nx, ny,
		ib + p_west, ie - p_east,
		jb, jb,

		dx2ih, dy2ih);

	if (p_north)
		resvec(y, x, rhs, i_density, nx, ny,
		ib + p_west, ie - p_east,
		je, je,

		dx2ih, dy2ih);
}
// ------------------------------------------------------------------------ //

// * [laplas-restrict-residual] for poisson equation with async exchanges * //
template< typename T >
void poisson2d::laplas_restrict_residual(T* y_coarse, T* x_fine, const T* rhs_fine,

	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[8];

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = cnx - gcx - 1;
	const int jb = gcy, je = cny - gcy - 1;

	mpi_com.push_exchange_cross_halo(x_fine, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	put_bc(x_fine, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	mg_restrict_residual(y_coarse, x_fine, rhs_fine, type,
		cnx, cny, nx, ny, gcx, gcy,

		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2i, dy2i);

	mpi_com.pop_exchange_cross_halo(x_fine, nx, ny, gcx, gcy,
		1, 1, x_periodic, y_periodic, mpi_req);

	if (p_west)
		mg_restrict_residual(y_coarse, x_fine, rhs_fine, type,
		cnx, cny, nx, ny, gcx, gcy,

		ib, ib,
		jb, je,

		dx2i, dy2i);

	if (p_east)
		mg_restrict_residual(y_coarse, x_fine, rhs_fine, type,
		cnx, cny, nx, ny, gcx, gcy,

		ie, ie,
		jb, je,

		dx2i, dy2i);

	if (p_south)
		mg_restrict_residual(y_coarse, x_fine, rhs_fine, type,
		cnx, cny, nx, ny, gcx, gcy,

		ib + p_west, ie - p_east,
		jb, jb,

		dx2i, dy2i);

	if (p_north)
		mg_restrict_residual(y_coarse, x_fine, rhs_fine, type,
		cnx, cny, nx, ny, gcx, gcy,

		ib + p_west, ie - p_east,
		je, je,

		dx2i, dy2i);
}
// ------------------------------------------------------------------------ //

// * [laplas-prolongate] for poisson equation with async exchanges * //
template< typename T >
void poisson2d::laplas_prolongate(T* x_fine, T* x_coarse,

	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	MPI_Request mpi_req[4];

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = cnx - gcx - 1;
	const int jb = gcy, je = cny - gcy - 1;

	int x_exch = 0;

	put_bc(x_coarse, cnx, cny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	// push exchanges on -x //
	mpi_com.push_exchange_halo_x(x_coarse, cnx, cny, gcx, gcy,
		1, 1, x_periodic, mpi_req);

	// init exchanges on -y //
	mpi_com.init_exchange_halo_y(x_coarse, cnx, cny, gcx, gcy,
		1, 1, y_periodic);

	// check if -x exchanges completed //
	if ((x_exch = mpi_com.test_exchange(mpi_req, 4))) {

		// pop exchanges on -x
		mpi_com.pop_exchange_halo_x(x_coarse, cnx, cny, gcx, gcy,
			1, 1, x_periodic, mpi_req);

		// push exchanges on -y
		mpi_com.push_exchange_halo_y(x_coarse, 1, cnx, cny, gcx, gcy,
			1, 1, y_periodic, mpi_req);

		// - do main block //
		mg_prolongate(x_fine, x_coarse, type,
			nx, ny, cnx, cny, gcx, gcy,

			ib + p_west, ie - p_east,
			jb + p_south, je - p_north);
	}
	else
	{
		int shx = (ie - ib - p_east - p_west + 1) / 2;

		// - do first half block //
		mg_prolongate(x_fine, x_coarse, type,
			nx, ny, cnx, cny, gcx, gcy,

			ib + p_west, ie - p_east - shx,
			jb + p_south, je - p_north);

		// pop exchanges on -x
		mpi_com.pop_exchange_halo_x(x_coarse, cnx, cny, gcx, gcy,
			1, 1, x_periodic, mpi_req);

		// push exchanges on -y
		mpi_com.push_exchange_halo_y(x_coarse, 1, cnx, cny, gcx, gcy,
			1, 1, y_periodic, mpi_req);

		// - do second half block //
		mg_prolongate(x_fine, x_coarse, type,
			nx, ny, cnx, cny, gcx, gcy,

			ie - p_east - shx + 1, ie - p_east,
			jb + p_south, je - p_north);
	}

	// pop exchanges on -y
	mpi_com.pop_exchange_halo_y(x_coarse, cnx, cny, gcx, gcy,
		1, 1, y_periodic, mpi_req);

	if (p_west)
		mg_prolongate(x_fine, x_coarse, type,
		nx, ny, cnx, cny, gcx, gcy,

		ib, ib,
		jb, je);

	if (p_east)
		mg_prolongate(x_fine, x_coarse, type,
		nx, ny, cnx, cny, gcx, gcy,

		ie, ie,
		jb, je);

	if (p_south)
		mg_prolongate(x_fine, x_coarse, type,
		nx, ny, cnx, cny, gcx, gcy,

		ib + p_west, ie - p_east,
		jb, jb);

	if (p_north)
		mg_prolongate(x_fine, x_coarse, type,
		nx, ny, cnx, cny, gcx, gcy,

		ib + p_west, ie - p_east,
		je, je);
}
// ------------------------------------------------------------------------ //

// * SGS Red-Black for Poisson equation * //
//
// (starting routine - some optimizations for case x = 0) //
template< typename T >
void poisson2d::sgs_start(
	T* x, T* rhs, const T i_diag, const int type,

	const int color,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com, const int pois_bc_type)
{
	MPI_Request mpi_req[4];

	const int c_black = color;
	const int c_red = !color;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	int x_exch = 0; // X exchange status //

	if (type == c_sgs_init) {

		// push [RHS] X exchange //
		mpi_com.push_exchange_halo_x(rhs, nx, ny, gcx, gcy,
			1, 1, x_periodic, mpi_req);

		// init [RHS] Y exchange //
		mpi_com.init_exchange_halo_y(rhs, nx, ny, gcx, gcy,
			1, 1, y_periodic);

		// check [RHS] X exchange //
		if ((x_exch = mpi_com.test_exchange(mpi_req, 4))) {
			// pop [RHS] X exchange //
			mpi_com.pop_exchange_halo_x(rhs, nx, ny, gcx, gcy,
				1, 1, x_periodic, mpi_req);

			// push [RHS] Y exchange //
			mpi_com.push_exchange_halo_y(rhs, 1, nx, ny, gcx, gcy,
				1, 1, y_periodic, mpi_req);
		}

		// sgs [red] //
		sgs_init(x, rhs, i_diag, c_red, nx, ny, gcx, gcy);

		if (!x_exch) {
			// pop [RHS] X exchange //
			mpi_com.pop_exchange_halo_x(rhs, nx, ny, gcx, gcy,
				1, 1, x_periodic, mpi_req);

			// push [RHS] Y exchange //
			mpi_com.push_exchange_halo_y(rhs, 1, nx, ny, gcx, gcy,
				1, 1, y_periodic, mpi_req);
		}

		// pop [RHS] Y exchange //
		mpi_com.pop_exchange_halo_y(rhs, nx, ny, gcx, gcy,
			1, 1, y_periodic, mpi_req);

		return;
	}

	if (type == c_sgs_continue) {

		put_bc(x, nx, ny, gcx, gcy,
			mpi_com.rank_x, mpi_com.rank_y,
			mpi_com.size_x, mpi_com.size_y,
			pois_bc_type);

		// push [black] X exchange //
		mpi_com.push_exchange_color_halo_x(x, c_black, nx, ny, gcx, gcy,
			1, 1, x_periodic, mpi_req);

		// check if X exchange completed //
		if ((x_exch = mpi_com.test_exchange(mpi_req, 4))) {
			// pop [black] X exchange //
			mpi_com.pop_exchange_color_halo_x(x, c_black, nx, ny, gcx, gcy,
				1, 1, x_periodic, mpi_req);

			// push [black] Y exchange //
			mpi_com.push_exchange_color_halo_y(x, c_black, nx, ny, gcx, gcy,
				1, 1, y_periodic, mpi_req);
		}

		// red shift dependent on exch X completeness //
		int shx = (x_exch) ? 0 : (ie - ib - p_east - p_west + 1) / 2;

		// sgs [red] //
		sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ib + p_west, ie - p_east - shx,
			jb + p_south, je - p_north,

			dx2i, dy2i);

		if (!x_exch) {
			// pop [black] X exchange //
			mpi_com.pop_exchange_color_halo_x(x, c_black, nx, ny, gcx, gcy,
				1, 1, x_periodic, mpi_req);

			// push [black] Y exchange //
			mpi_com.push_exchange_color_halo_y(x, c_black, nx, ny, gcx, gcy,
				1, 1, y_periodic, mpi_req);

			// sgs [red] //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				ie - p_east - shx + 1, ie - p_east,
				jb + p_south, je - p_north,

				dx2i, dy2i);
		}

		// pop [black] X exchange //
		mpi_com.pop_exchange_color_halo_y(x, c_black, nx, ny, gcx, gcy,
			1, 1, y_periodic, mpi_req);

		// sgs [red] parts //
		if (p_west)   // west //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ib, ib,
			jb, je,

			dx2i, dy2i);
		if (p_east)   // east //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ie, ie,
			jb, je,

			dx2i, dy2i);
		if (p_south)  // south //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ib + p_west, ie - p_east,
			jb, jb,

			dx2i, dy2i);
		if (p_north)  // north //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ib + p_west, ie - p_east,
			je, je,

			dx2i, dy2i);

		return;
	}
}

// (main in cycle iteration) //
template< typename T >
void poisson2d::sgs_run(
	T* x, const T* rhs, const T i_diag,

	const int color,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com, const int pois_bc_type)
{
	MPI_Request mpi_req[4];

	const int c_black = color;
	const int c_red = !color;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	int x_exch = 0;

	if (mpi_com.size == 1) // single MPI process algorithm //
	{
		// only periodic boundary conditions //
		mpi_com.exchange_color_halo(x, c_red, nx, ny, gcx, gcy,
			2, 2, x_periodic, y_periodic);

		// sgs [black] with shift = - 1 //
		sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
			ib - p_west, ie + p_east,
			jb - p_south, je + p_north,

			dx2i, dy2i);

		// boundary conditions [red (previous), black]
		put_bc(x, nx, ny, gcx, gcy,
			mpi_com.rank_x, mpi_com.rank_y,
			mpi_com.size_x, mpi_com.size_y,
			pois_bc_type);

		// sgs [red] with shift = 0 //
		sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
			ib, ie,
			jb, je,

			dx2i, dy2i);

		return;
	}

	// push X exchange [red] cells with width = 2 //
	mpi_com.push_exchange_color_halo_x(x, c_red, nx, ny, gcx, gcy,
		2, 2, x_periodic, mpi_req);
	// check if X echange completed //
	if ((x_exch = mpi_com.test_exchange(mpi_req, 4))) {
		// pop X exchange [red] cells with width = 2 //
		mpi_com.pop_exchange_color_halo_x(x, c_red, nx, ny, gcx, gcy,
			2, 2, x_periodic, mpi_req);

		// push Y exchange [red] cells with width = 2 //
		mpi_com.push_exchange_color_halo_y(x, c_red, nx, ny, gcx, gcy,
			2, 2, y_periodic, mpi_req);
	}

	// sgs [black] with shift = + 1 //
	sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
		ib + p_west, ie - p_east,
		jb + p_south, je - p_north,

		dx2i, dy2i);

	// if X exchanges has not completed //
	if (!x_exch) {
		// pop X exchange [red] cells with width = 2 //
		mpi_com.pop_exchange_color_halo_x(x, c_red, nx, ny, gcx, gcy,
			2, 2, x_periodic, mpi_req);

		// push Y exchange [red] cells with width = 2 //
		mpi_com.push_exchange_color_halo_y(x, c_red, nx, ny, gcx, gcy,
			2, 2, y_periodic, mpi_req);
	}

	// sgs [red] with shift = + 2 //
	sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
		ib + p_west + 1, ie - p_east - 1,
		jb + p_south + 1, je - p_north - 1,

		dx2i, dy2i);

	// pop Y exchange [red] cells with width = 2 //
	mpi_com.pop_exchange_color_halo_y(x, c_red, nx, ny, gcx, gcy,
		2, 2, y_periodic, mpi_req);


	// sgs [black] finishing //
	// west part [- 1, 0] //
	if (p_west)
		sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
		ib - 1, ib,
		jb - p_south, je + p_north,

		dx2i, dy2i);
	// east part [0, 1] //
	if (p_east)
		sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
		ie, ie + 1,
		jb - p_south, je + p_north,

		dx2i, dy2i);
	// south part [- 1, 0] //
	if (p_south)
		sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
		ib + p_west, ie - p_east,
		jb - 1, jb,

		dx2i, dy2i);
	// north part [0, 1] //
	if (p_north)
		sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
		ib + p_west, ie - p_east,
		je, je + 1,

		dx2i, dy2i);

	// boundary conditions [red (previous), black]
	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);

	// sgs [red] finishing //
	// west part [0, 1] //
	sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
		ib, ib + p_west,
		jb, je,

		dx2i, dy2i);
	// east part [-1, 0] //
	sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
		ie - p_east, ie,
		jb, je,

		dx2i, dy2i);
	// south part [0, 1] //
	sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
		ib + p_west + 1, ie - p_east - 1,
		jb, jb + p_south,

		dx2i, dy2i);
	// north part [-1, 0] //
	sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
		ib + p_west + 1, ie - p_east - 1,
		je - p_north, je,

		dx2i, dy2i);
}

//
// (main in cycle iteration) //
template< typename T >
void poisson2d::sgs_run_cache(const int niters,
	T* x, const T* rhs, const T i_diag,

	const int color,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com, const int pois_bc_type)
{
	// Working on single processor //
	// --------------------------- //

	const int c_black = color;
	const int c_red = !color;

	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int p_west = ((mpi_com.rank_x > 0) || x_periodic) ? 1 : 0;
	const int p_east = ((mpi_com.rank_x < mpi_com.size_x - 1) || x_periodic) ? 1 : 0;
	const int p_south = ((mpi_com.rank_y > 0) || y_periodic) ? 1 : 0;
	const int p_north = ((mpi_com.rank_y < mpi_com.size_y - 1) || y_periodic) ? 1 : 0;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	const int cx = ie - ib + 1, cy = je - jb + 1;
	const int p_size = cx * cy;                    // problem size;
	const int small_size = 128 * 1024;              // < - 'small' problem size

	// Small Problem //
	if (p_size <= small_size)
	{
		for (int k = 0; k < niters; k++) {

			// only periodic boundary conditions //
			mpi_com.exchange_color_halo(x, c_red, nx, ny, gcx, gcy,
				2, 2, x_periodic, y_periodic);

			// sgs [black] with shift = - 1 //
			sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
				ib - p_west, ie + p_east,
				jb - p_south, je + p_north,

				dx2i, dy2i);

			// boundary conditions [red (previous), black]
			put_bc(x, nx, ny, gcx, gcy,
				mpi_com.rank_x, mpi_com.rank_y,
				mpi_com.size_x, mpi_com.size_y,
				pois_bc_type);

			// sgs [red] with shift = 0 //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				ib, ie,
				jb, je,

				dx2i, dy2i);
		}

		return;
	}

	// Huge problem for L2 cache //

	// - Divide problem 
	int idiv = 1, jdiv = 1;

	int task_cx = cx, task_cy = cy;
	int task_size = p_size;
	while (task_size > small_size) {

		if (task_cx >= task_cy) {
			idiv *= 2;
			task_cx = (task_cx / 2) + (task_cx % 2);
		}
		else
		{
			jdiv *= 2;
			task_cy = (task_cy / 2) + (task_cy % 2);
		}

		task_size = task_cx * task_cy;
	}

	const int ish = cx / idiv;
	const int jsh = cy / jdiv;

	// - Main iters for each block //
	int iblock, jblock, k;
	int ibb, jbb, iee, jee;

	for (iblock = 0; iblock < idiv; iblock++)
	{
		for (jblock = 0; jblock < jdiv; jblock++)
		{
			ibb = ib + iblock * ish;
			jbb = jb + jblock * jsh;

			iee = ib + (iblock + 1) * ish - 1;
			jee = jb + (jblock + 1) * jsh - 1;

			for (k = 0; k < niters; k++) {

				// sgs [black] //
				sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
					ibb + 2 * k, iee - 2 * k,
					jbb + 2 * k, jee - 2 * k,

					dx2i, dy2i);

				// sgs [red] //
				sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
					ibb + 2 * k + 1, iee - 2 * k - 1,
					jbb + 2 * k + 1, jee - 2 * k - 1,

					dx2i, dy2i);
			}
		}
	}

	// boundary conditions [red (previous), black]
	put_bc(x, nx, ny, gcx, gcy,
		mpi_com.rank_x, mpi_com.rank_y,
		mpi_com.size_x, mpi_com.size_y,
		pois_bc_type);


	// do remaining sgs [red] //
	for (iblock = 0; iblock < idiv; iblock++)
	{
		for (jblock = 0; jblock < jdiv; jblock++)
		{
			ibb = ib + iblock * ish;
			jbb = jb + jblock * jsh;

			iee = ib + (iblock + 1) * ish - 1;
			jee = jb + (jblock + 1) * jsh - 1;

			// sgs [red] //
			// west //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				ibb, ibb,
				jbb, jee,

				dx2i, dy2i);
			// east //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				iee, iee,
				jbb, jee,

				dx2i, dy2i);

			// south //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				ibb + 1, iee - 1,
				jbb, jbb,

				dx2i, dy2i);

			// north //
			sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
				ibb + 1, iee - 1,
				jee, jee,

				dx2i, dy2i);
		}
	}

	// Remaining work 
	for (k = 1; k < niters; k++)
	{
		// k = 1.
		//      black (shift=2)
		//      boundary conditions
		//      red (shift=3)
		// k = 2.
		//      black (shift=4)
		//      boundary conditions
		//      red (shift=5)



		// do remaining sgs [black] //
		for (iblock = 0; iblock < idiv; iblock++)
		{
			for (jblock = 0; jblock < jdiv; jblock++)
			{
				ibb = ib + iblock * ish;
				jbb = jb + jblock * jsh;

				iee = ib + (iblock + 1) * ish - 1;
				jee = jb + (jblock + 1) * jsh - 1;

				// sgs [black] //
				// west //
				sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
					ibb, ibb + 2 * k - 1,
					jbb, jee,

					dx2i, dy2i);
				// east //
				sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
					iee - 2 * k + 1, iee,
					jbb, jee,

					dx2i, dy2i);
				// south //
				sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
					ibb + 2 * k, iee - 2 * k,
					jbb, jbb + 2 * k - 1,

					dx2i, dy2i);
				// north //
				sgs_cycle(x, rhs, i_diag, c_black, nx, ny,
					ibb + 2 * k, iee - 2 * k,
					jee - 2 * k + 1, jee,

					dx2i, dy2i);
			}
		}

		// boundary conditions [red (previous), black]
		put_bc(x, nx, ny, gcx, gcy,
			mpi_com.rank_x, mpi_com.rank_y,
			mpi_com.size_x, mpi_com.size_y,
			pois_bc_type);


		// do remaining sgs [red] //
		for (iblock = 0; iblock < idiv; iblock++)
		{
			for (jblock = 0; jblock < jdiv; jblock++)
			{
				ibb = ib + iblock * ish;
				jbb = jb + jblock * jsh;

				iee = ib + (iblock + 1) * ish - 1;
				jee = jb + (jblock + 1) * jsh - 1;

				// sgs [red] //
				// west //
				sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
					ibb, ibb + 2 * k,
					jbb, jee,

					dx2i, dy2i);
				// east //
				sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
					iee - 2 * k, iee,
					jbb, jee,

					dx2i, dy2i);
				// south //
				sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
					ibb + 2 * k + 1, iee - 2 * k - 1,
					jbb, jbb + 2 * k,

					dx2i, dy2i);
				// north //
				sgs_cycle(x, rhs, i_diag, c_red, nx, ny,
					ibb + 2 * k + 1, iee - 2 * k - 1,
					jee - 2 * k, jee,

					dx2i, dy2i);
			}
		}
	}
}

// * SGS Red-Black preconditioner * //
template< typename T >
void poisson2d::sgs_redblack(
	T* x, T* rhs, const T i_diag,
	const int type, const int color_mode,
	const int piters,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const T dx2i, const T dy2i,
	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	sgs_start(x, rhs, i_diag, type, color_mode,
		nx, ny, gcx, gcy,
		dx2i, dy2i,

		mpi_com, pois_bc_type);

	for (int k = 0; k < piters; k++)
		sgs_run(x, rhs, i_diag, color_mode,
		nx, ny, gcx, gcy,
		dx2i, dy2i,

		mpi_com, pois_bc_type);

	// Cache version //
	//sgs_run_cache( piters, x, rhs, i_diag, color_mode, 
	//    nx, ny, gcx, gcy,
	//    dx2i, dy2i,

	//    mpi_com, pois_bc_type );
}
// ------------------------------------------------------------------------ //

// * SGS Red-Black var preconditioner * //
template< typename T >
void poisson2d::sgs_redblack(
	T* x, const T* rhs, const T* idg, const T* i_density,

	const int type, const int color, const int piters,

	const int nx, const int ny,
	const int gcx, const int gcy,
	const T dx2ih, const T dy2ih,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	const int c_red = !color;
	const int c_black = color;

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	if (type == c_sgs_init) {

		// sgs [red] init //
		sgs_init(x, rhs, idg, c_red, nx, ny, gcx, gcy);
	}
	if (type == c_sgs_continue)
	{
		// exchange [black] [width=1, periodic=yes] //
		put_exch_bc(x, nx, ny, gcx, gcy, mpi_com, pois_bc_type);

		// sgs [red] //
		sgs_cycle(x, rhs, idg, i_density, c_red, nx, ny,
			ib, ie,
			jb, je,

			dx2ih, dy2ih);
	}

	for (int k = 0; k < piters; k++)
	{
		// exchange [red] [width=1, periodic=no] //
		mpi_com.exchange_cross_halo(x, nx, ny, gcx, gcy,
			1, 1, 0, 0);

		// sgs [black] //
		sgs_cycle(x, rhs, idg, i_density, c_black, nx, ny,
			ib, ie,
			jb, je,

			dx2ih, dy2ih);

		// boundary conditions [red,black] //
		// exchange [black] [width=1, periodic=yes] //
		put_exch_bc(x, nx, ny, gcx, gcy,
			mpi_com, pois_bc_type);

		// sgs [red] //
		sgs_cycle(x, rhs, idg, i_density, c_red, nx, ny,
			ib, ie,
			jb, je,

			dx2ih, dy2ih);
	}
}
// -------------------------------------------------------------------- //

template< typename T >
void poisson2d::ssor_redblack(
	T* x, const T* rhs, const T idg, const T omega,

	const int type, const int color, const int piters,

	const int nx, const int ny,
	const int gcx, const int gcy,
	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	const int c_red = !color;
	const int c_black = color;

	const T c_alpha = idg * omega;
	const T c_beta = ((T) 2.0 - omega) * c_alpha;
	const T c_gamma = ((T) 1.0 - omega) * ((T) 1.0 - omega);

	const int ib = gcx, ie = nx - gcx - 1;
	const int jb = gcy, je = ny - gcy - 1;

	if (type == c_sgs_init) {

		// sgs [red] init //
		sgs_init(x, rhs, c_alpha, c_red, nx, ny, gcx, gcy);
	}
	if (type == c_sgs_continue)
	{
		put_exch_bc(x, nx, ny, gcx, gcy,
			mpi_com, pois_bc_type);

		// sgs [red] //
		ssor_cycle(x, rhs, (T) 1.0 - omega, c_alpha, c_red, nx, ny,
			ib, ie,
			jb, je,

			dx2i, dy2i);
	}

	for (int k = 0; k < piters - 1; k++)
	{
		// exchange [red] [width=1, periodic=no] //
		mpi_com.exchange_cross_halo(x, nx, ny, gcx, gcy,
			1, 1, 0, 0);

		// sgs [black] //
		ssor_cycle(x, rhs, c_gamma, c_beta, c_black, nx, ny,
			ib, ie,
			jb, je,

			dx2i, dy2i);

		// boundary conditions [red,black] //
		// exchange [black] [width=1, periodic=yes] //
		put_exch_bc(x, nx, ny, gcx, gcy,
			mpi_com, pois_bc_type);

		// sgs [red] //
		ssor_cycle(x, rhs, c_gamma, c_beta, c_red, nx, ny,
			ib, ie,
			jb, je,

			dx2i, dy2i);

	}

	// exchange [red] [width=1, periodic=no] //
	mpi_com.exchange_cross_halo(x, nx, ny, gcx, gcy,
		1, 1, 0, 0);

	// sgs [black] //
	ssor_cycle(x, rhs, c_gamma, c_beta, c_black, nx, ny,
		ib, ie,
		jb, je,

		dx2i, dy2i);

	// boundary conditions [red,black] //
	// exchange [black] [width=1, periodic=yes] //
	put_exch_bc(x, nx, ny, gcx, gcy,
		mpi_com, pois_bc_type);

	// sgs [red] //
	ssor_cycle(x, rhs, (T) 1.0 - omega, c_alpha, c_red, nx, ny,
		ib, ie,
		jb, je,

		dx2i, dy2i);
}

// * Approximate inverse preconditioner * //
template< typename T >
void poisson2d::aip(T* x, const T* rhs, T* mem, const T idg,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	int i, j, idx;
	const T i_dx2i = idg * dx2i;
	const T i_dy2i = idg * dy2i;

	T *interm = mem;

#pragma omp parallel for private( i, j, idx ) shared( x, rhs, interm )
	for (i = gcx - 1; i < nx - gcx; i++) {
		idx = i * ny + gcy - 1;
		for (j = gcy - 1; j < ny - gcy; j++, idx++)
			interm[idx] = rhs[idx] - (
			rhs[idx + ny] * i_dx2i + rhs[idx + 1] * i_dy2i);
	}

#pragma omp parallel for private( i, j, idx ) shared( x, rhs, interm )
	for (i = gcx; i < nx - gcx; i++) {
		idx = i * ny + gcy;
		for (j = gcy; j < ny - gcy; j++, idx++) {
			x[idx] = interm[idx] - (
				interm[idx - ny] * i_dx2i + interm[idx - 1] * i_dy2i);
		}
	}
}


// * Jacobi preconditioner * //
template< typename T >
void poisson2d::jacobi(T* x, const T* rhs, T* mem, const T idg,
	const int piters,

	const int nx, const int ny,
	const int gcx, const int gcy,
	const T dx2i, const T dy2i,

	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	int i, j, idx;

	T *ptr_x = x, *ptr_y = mem;
	T *ptr_ref;

#pragma omp parallel for private( i, j, idx ) shared( x, rhs )
	for (i = gcx; i < nx - gcx; i++) {
		idx = i * ny + gcy;
		for (j = gcy; j < ny - gcy; j++, idx++)
			x[idx] = idg * rhs[idx];
	}

	for (int k = 0; k < piters - 1; k++)
	{
		put_exch_bc(ptr_x, nx, ny, gcx, gcy,
			mpi_com, pois_bc_type);

#pragma omp parallel for private( i, j, idx ) shared( ptr_y, ptr_x, rhs )
		for (i = gcx; i < nx - gcx; i++) {
			idx = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, idx++)
				ptr_y[idx] = idg * (rhs[idx] -
				(ptr_x[idx - ny] + ptr_x[idx + ny]) * dx2i -
				(ptr_x[idx - 1] + ptr_x[idx + 1]) * dy2i);
		}

		ptr_ref = ptr_x;

		ptr_x = ptr_y;
		ptr_y = ptr_ref;
	}

	if ((piters & 1) == 0)
		memcpy(x, mem, nx * ny * sizeof(T));
}
// -------------------------------------------------------------------- //

// * Multigrid SGS Red-Black preconditioner * //
template< typename T >
void poisson2d::mg_sgs_redblack(
	T* x, T* rhs, const int piters,
	nse::mg_poisson2d_data< T >& mg,
	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	const int fine = 0;
	int i, k;

	mg.x[fine] = x;
	mg.rhs[fine] = rhs;

	sgs_redblack(mg.x[fine], mg.rhs[fine], mg.idg[fine],
		c_sgs_init, mg.sgs_color_shift[fine], mg.sgs_down_iters[fine],
		mg.nx[fine], mg.ny[fine],
		mg.gcx, mg.gcy,
		mg.dx2i[fine], mg.dy2i[fine],
		mpi_com, pois_bc_type);

	for (k = 0; k < piters; k++)
	{
		for (i = fine + 1; i < mg.num_grids; i++) {

			// MPI: cross exchange inside //
			// MPI: no corner cells exchanges which could be needed for odd number of grid cells!
			laplas_restrict_residual(mg.rhs[i], mg.x[i - 1], mg.rhs[i - 1],
				mg.coarse_type[i],

				mg.nx[i], mg.ny[i],
				mg.nx[i - 1], mg.ny[i - 1],
				mg.gcx, mg.gcy,

				mg.dx2i[i - 1], mg.dy2i[i - 1],

				mpi_com, pois_bc_type);

			sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i],
				c_sgs_init, mg.sgs_color_shift[i], mg.sgs_down_iters[i],
				mg.nx[i], mg.ny[i],
				mg.gcx, mg.gcy,
				mg.dx2i[i], mg.dy2i[i],
				mpi_com, pois_bc_type);
		}

		for (i = mg.num_grids - 2; i >= fine; i--) {

			// prolongate extension is not included for odd cells //
			laplas_prolongate(mg.x[i], mg.x[i + 1],
				mg.coarse_type[i + 1],

				mg.nx[i], mg.ny[i],
				mg.nx[i + 1], mg.ny[i + 1],
				mg.gcx, mg.gcy,

				mpi_com,
				pois_bc_type);

			sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i],
				c_sgs_continue, mg.sgs_color_shift[i], mg.sgs_up_iters[i],
				mg.nx[i], mg.ny[i],
				mg.gcx, mg.gcy,
				mg.dx2i[i], mg.dy2i[i],
				mpi_com, pois_bc_type);
		}
	}

	nse::null_halo(mg.rhs[fine],
		mg.nx[fine], mg.ny[fine],
		mg.gcx, mg.gcy);
}

template< typename T >
void poisson2d::mg_sgs_redblack(
	T* x, T* rhs, const int piters,
	nse::mg_var_poisson2d_data< T >& mg,
	const nse::mpiCom2d& mpi_com, const int pois_bc_type)
{
	const int fine = 0;
	int i, k;

	mg.x[fine] = x;
	mg.rhs[fine] = rhs;

	sgs_redblack(mg.x[fine], mg.rhs[fine], mg.idg[fine], mg.i_density[fine],
		c_sgs_init, mg.sgs_color_shift[fine], mg.sgs_down_iters[fine],
		mg.nx[fine], mg.ny[fine],
		mg.gcx, mg.gcy,
		mg.dx2ih[fine], mg.dy2ih[fine],
		mpi_com, pois_bc_type);

	for (k = 0; k < piters; k++)
	{
		for (i = fine + 1; i < mg.num_grids; i++) {

			laplas_residual(mg.res[i - 1], mg.x[i - 1], mg.rhs[i - 1], mg.i_density[i - 1],
				mg.nx[i - 1], mg.ny[i - 1],
				mg.gcx, mg.gcy,

				mg.dx2ih[i - 1], mg.dy2ih[i - 1],

				mpi_com, pois_bc_type);

			// MPI: possible additional exchange needed for corner cells in case of odd number of grid cells!
			mg_restrict(mg.rhs[i], mg.res[i - 1],
				mg.coarse_type[i],

				mg.nx[i], mg.ny[i],
				mg.nx[i - 1], mg.ny[i - 1],
				mg.gcx, mg.gcy,

				mg.gcx, mg.nx[i] - mg.gcx - 1,
				mg.gcy, mg.ny[i] - mg.gcy - 1);

			sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i], mg.i_density[i],
				c_sgs_init, mg.sgs_color_shift[i], mg.sgs_down_iters[i],
				mg.nx[i], mg.ny[i],
				mg.gcx, mg.gcy,
				mg.dx2ih[i], mg.dy2ih[i],
				mpi_com, pois_bc_type);
		}

		for (i = mg.num_grids - 2; i >= fine; i--) {

			// prolongate extension is not included for odd cells //
			laplas_prolongate(mg.x[i], mg.x[i + 1],
				mg.coarse_type[i + 1],

				mg.nx[i], mg.ny[i],
				mg.nx[i + 1], mg.ny[i + 1],
				mg.gcx, mg.gcy,

				mpi_com,
				pois_bc_type);

			sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i], mg.i_density[i],
				c_sgs_continue, mg.sgs_color_shift[i], mg.sgs_up_iters[i],
				mg.nx[i], mg.ny[i],
				mg.gcx, mg.gcy,
				mg.dx2ih[i], mg.dy2ih[i],
				mpi_com, pois_bc_type);

		}
	}

	nse::null_halo(mg.rhs[fine],
		mg.nx[fine], mg.ny[fine],
		mg.gcx, mg.gcy);
}
// ------------------------------------------------------------------------ //

template< typename T >
void poisson2d::mg_sgs_redblack(
	T* x, T* rhs, const int piters,
	nse::mg_mpi_poisson2d_data< T >& mg,
	const nse::mpiCom2d& mpi_com,
	const int pois_bc_type)
{
	const int x_periodic = (pois_bc_type == nse::c_pois_bc_periodic_x) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);
	const int y_periodic = (pois_bc_type == nse::c_pois_bc_periodic_y) ||
		(pois_bc_type == nse::c_pois_bc_periodic_xy);

	const int fine = 0;
	int i, k;

	mg.x[fine] = x;
	mg.rhs[fine] = rhs;

	// do fine grid on all mpi processors //
	sgs_redblack(mg.x[fine], mg.rhs[fine], mg.idg[fine],
		c_sgs_init, mg.sgs_color_shift[fine], mg.sgs_down_iters[fine],
		mg.mpi_nx[fine], mg.mpi_ny[fine],
		mg.gcx, mg.gcy,
		mg.dx2i[fine], mg.dy2i[fine],
		mg.mpi_com[fine], pois_bc_type);

	for (k = 0; k < piters; k++)
	{

		for (i = fine + 1; i < mg.num_grids; i++) {

			// MPI RUN( on finer grid )
			if (mg.mpi_run[i - 1]) {

				// MPI: cross exchange inside //
				// MPI: no corner cells exchanges which could be needed for odd number of grid cells!
				laplas_restrict_residual(mg.rhs[i], mg.x[i - 1], mg.rhs[i - 1],
					mg.coarse_type[i],

					mg.local_nx[i], mg.local_ny[i],
					mg.mpi_nx[i - 1], mg.mpi_ny[i - 1],
					mg.gcx, mg.gcy,

					mg.dx2i[i - 1], mg.dy2i[i - 1],

					mg.mpi_com[i - 1], pois_bc_type);
			}

			// MPI COMBINE( on coarser grid ) - GATHER
			if (mg.mpi_combine[i]) {

				// use communicator on finer grid: mg.mpi_com[ i - 1 ] //
				mg.mpi_com[i - 1].c_gather_odd_grid(mg.rhs[i], mg.rhs[i],
					mg.mpi_nx[i], mg.mpi_ny[i],
					mg.local_nx[i], mg.local_ny[i],
					mg.gcx, mg.gcy);
			}

			// MPI RUN( on coarser grid )
			if (mg.mpi_run[i]) {
				sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i],
					c_sgs_init, mg.sgs_color_shift[i], mg.sgs_down_iters[i],
					mg.mpi_nx[i], mg.mpi_ny[i],
					mg.gcx, mg.gcy,
					mg.dx2i[i], mg.dy2i[i],
					mg.mpi_com[i], pois_bc_type);
			}
		}

		for (i = mg.num_grids - 2; i >= fine; i--) {

			// MPI RUN( on coarser grid )
			if (mg.mpi_run[i + 1]) {

				// prolongation requires mpi-exchanges with corners //
				put_bc(mg.x[i + 1],
					mg.mpi_nx[i + 1], mg.mpi_ny[i + 1],
					mg.gcx, mg.gcy,
					mg.mpi_com[i + 1].rank_x, mg.mpi_com[i + 1].rank_y,
					mg.mpi_com[i + 1].size_x, mg.mpi_com[i + 1].size_y,
					pois_bc_type);

				mg.mpi_com[i + 1].exchange_halo(mg.x[i + 1],
					mg.mpi_nx[i + 1], mg.mpi_ny[i + 1],
					mg.gcx, mg.gcy,
					1, 1,
					x_periodic, y_periodic);
				// -------------------------------------------------- //
			}

			// MPI COMBINE( on coarser grid ) - SCATTER
			if (mg.mpi_combine[i + 1]) {

				// use communicator on finer grid: mg.mpi_com[ i ] //
				mg.mpi_com[i].c_scatter_odd_grid(mg.x[i + 1], mg.x[i + 1],
					mg.mpi_nx[i + 1], mg.mpi_ny[i + 1],
					mg.local_nx[i + 1], mg.local_ny[i + 1],
					mg.gcx, mg.gcy);
			}

			// MPI RUN( on finer grid )
			if (mg.mpi_run[i]) {

				mg_prolongate(mg.x[i], mg.x[i + 1],
					mg.coarse_type[i + 1],

					mg.mpi_nx[i], mg.mpi_ny[i],
					mg.local_nx[i + 1], mg.local_ny[i + 1],
					mg.gcx, mg.gcy);


				sgs_redblack(mg.x[i], mg.rhs[i], mg.idg[i],
					c_sgs_continue, mg.sgs_color_shift[i], mg.sgs_up_iters[i],
					mg.mpi_nx[i], mg.mpi_ny[i],
					mg.gcx, mg.gcy,
					mg.dx2i[i], mg.dy2i[i],
					mg.mpi_com[i], pois_bc_type);
			}
		}
	}

	nse::null_halo(mg.rhs[fine],
		mg.mpi_nx[fine], mg.mpi_ny[fine],
		mg.gcx, mg.gcy);
}
// ------------------------------------------------------------------------ //

// initialize: CG SGS Red-Black
template int nse::cg_sgs_redblack(
	float* x, const float* rhs, float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid, const int pois_bc_type,
	float* resnorm);

template int nse::cg_sgs_redblack(
	double* x, const double* rhs, double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid, const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //

// initialize: CG Multigrid SGS Red-Black
template int nse::cg_mg_sgs_redblack(
	float* x, const float* rhs, float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid,
	mg_poisson2d_data< float >& mg_data,
	const int pois_bc_type,
	float* resnorm);

template int nse::cg_mg_sgs_redblack(
	double* x, const double* rhs, double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid,
	mg_poisson2d_data< double >& mg_data,
	const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //

// initialize: CG Multigrid(MPI) SGS Red-Black
template int nse::cg_mg_sgs_redblack(
	float* x, const float* rhs, float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid,
	mg_mpi_poisson2d_data< float >& mg_data,
	const int pois_bc_type,
	float* resnorm);

template int nse::cg_mg_sgs_redblack(
	double* x, const double* rhs, double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid,
	mg_mpi_poisson2d_data< double >& mg_data,
	const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //

// initialize: CG Multigrid(MPI) SGS Red-Black
template int nse::cg_mg_sgs_redblack_x4(
	float* x, const float* rhs, float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid,
	mg_mpi_poisson2d_data< float >& mg_data,
	const int pois_bc_type,
	float* resnorm);

template int nse::cg_mg_sgs_redblack_x4(
	double* x, const double* rhs, double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid,
	mg_mpi_poisson2d_data< double >& mg_data,
	const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //


// initialize: CG SGS Red-Black (variable density)
template int nse::cg_sgs_redblack(
	float* x, const float* rhs, const float* i_density, float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid, const int pois_bc_type,
	float* resnorm);

template int nse::cg_sgs_redblack(
	double* x, const double* rhs, const double* i_density, double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid, const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //

// initialize: CG Multigrid SGS Red-Black (variable density)
template int nse::cg_mg_sgs_redblack(
	float* x, const float* rhs, const float* i_density,
	float* memory,

	const int piters,
	const float retol, const float abstol, const int maxiters,
	const uniGrid2d< float >& grid,
	mg_var_poisson2d_data< float >& mg_data,
	const int pois_bc_type,
	float* resnorm);
template int nse::cg_mg_sgs_redblack(
	double* x, const double* rhs, const double* i_density,
	double* memory,

	const int piters,
	const double retol, const double abstol, const int maxiters,
	const uniGrid2d< double >& grid,
	mg_var_poisson2d_data< double >& mg_data,
	const int pois_bc_type,
	double* resnorm);
// ------------------------------------------------------------------------ //
