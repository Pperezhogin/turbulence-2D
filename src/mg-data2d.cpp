#include "mg-data2d.h"

#include "vecmath.h"
#include "pois-base2d.h"

template< typename T >
int nse::mg_poisson2d_data< T > ::memory_size( // multigrid size
	const uniGrid2d< T >& grid, const int _num_grids)
{
	const int fine = 0;
	int coarse_type = mg_coarse_null;
	int nx = grid.nx, ny = grid.ny;
	int gcx = grid.gcx, gcy = grid.gcy;
	int size = 0;

	T dx = grid.dx, dy = grid.dy;

	for (int i = fine + 1; i < _num_grids; i++) {

		int shx = ((nx - (gcx << 1)) & 1);
		int shy = ((ny - (gcy << 1)) & 1);

		coarse_type = mg_coarse_null;
		// - define coarse type
		if ((dx / dy) >(T) mg_coarse_aspect) {
			// - semi-coarsening by y
			coarse_type = mg_coarse_y;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dy *= (T) 2.0;
		}
		if ((dy / dx) > (T)mg_coarse_aspect) {
			// - semi-coarsening by x
			coarse_type = mg_coarse_x;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			dx *= (T) 2.0;
		}
		if (coarse_type == mg_coarse_null) {
			// - full coarsening by x and y
			coarse_type = mg_coarse_xy;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dx *= (T) 2.0;
			dy *= (T) 2.0;
		}

		size += 2 * nx * ny;
	}

	return size;
}

template< typename T >
void nse::mg_poisson2d_data< T > ::init(
	const nse::uniGrid2d< T >& grid, const int _num_grids)
{
	clear();

	num_grids = _num_grids;
	mg_memory_size = memory_size(grid, num_grids);
	allocate(&mg_memory, mg_memory_size);

	const int fine = 0;

	x[fine] = NULL;       // defined in mg solver
	rhs[fine] = NULL;     // defined in mg solver

	nx[fine] = grid.nx;
	ny[fine] = grid.ny;
	gcx = grid.gcx;
	gcy = grid.gcy;

	dx[fine] = grid.dx;
	dy[fine] = grid.dy;
	dx2i[fine] = grid.dx2i;
	dy2i[fine] = grid.dy2i;

	idg[fine] = -(T) 1.0 / (grid.dx2id + grid.dy2id);

	sgs_down_iters[fine] = c_mg_sgs_down_iters;
	sgs_up_iters[fine] = c_mg_sgs_up_iters;

	sgs_color_shift[fine] = ((grid.mpi_com.offset_x(nx[fine], gcx) +
		grid.mpi_com.offset_y(ny[fine], gcy)) & 1);

	coarse_type[fine] = mg_coarse_null;

	int mem_shift = 0;
	for (int i = fine + 1; i < num_grids; i++) {

		int shx = ((nx[i - 1] - (gcx << 1)) & 1);
		int shy = ((ny[i - 1] - (gcy << 1)) & 1);

		coarse_type[i] = mg_coarse_null;

		// - define coarse type
		if ((dx[i - 1] / dy[i - 1]) >(T) mg_coarse_aspect)
		{
			// - semi-coarsening by y
			coarse_type[i] = mg_coarse_y;

			nx[i] = nx[i - 1];
			ny[i] = ((ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			idg[i] = -(T) 1.0 / ((T) 2.0 * dx2i[i] + (T) 2.0 * dy2i[i]);

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}
		if ((dy[i - 1] / dx[i - 1]) >(T) mg_coarse_aspect)
		{
			// - semi-coarsening by x
			coarse_type[i] = mg_coarse_x;
			nx[i] = ((nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny[i] = ny[i - 1];

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = dy2i[i - 1];
			idg[i] = -(T) 1.0 / ((T) 2.0 * dx2i[i] + (T) 2.0 * dy2i[i]);

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}


		if (coarse_type[i] == mg_coarse_null) {
			// - full coarsening by x and y
			coarse_type[i] = mg_coarse_xy;
			nx[i] = ((nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny[i] = ((ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			idg[i] = (T) 4.0 * idg[i - 1];

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}

		sgs_color_shift[i] = ((grid.mpi_com.offset_x(nx[i], gcx) +
			grid.mpi_com.offset_y(ny[i], gcy)) & 1);

		x[i] = &mg_memory[mem_shift];
		rhs[i] = &mg_memory[mem_shift + nx[i] * ny[i]];

		mem_shift += 2 * (nx[i] * ny[i]);
	}
}

template< typename T >
void nse::mg_poisson2d_data< T > ::clear()
{
	num_grids = 0;

	if (mg_memory_size > 0) {
		deallocate(mg_memory);

		mg_memory_size = 0;

	}
}
// ------------------------------------------------------------------------ //

template< typename T >
int nse::mg_var_poisson2d_data< T > ::memory_size( // multigrid size
	const uniGrid2d< T >& grid, const int _num_grids)
{
	const int fine = 0;
	int coarse_type = mg_coarse_null;
	int nx = grid.nx, ny = grid.ny;
	int gcx = grid.gcx, gcy = grid.gcy;
	int size = 2 * nx * ny;

	T dx = grid.dx, dy = grid.dy;

	for (int i = fine + 1; i < _num_grids; i++) {

		int shx = ((nx - (gcx << 1)) & 1);
		int shy = ((ny - (gcy << 1)) & 1);

		coarse_type = mg_coarse_null;
		// - define coarse type
		if ((dx / dy) >(T) mg_coarse_aspect) {
			// - semi-coarsening by y
			coarse_type = mg_coarse_y;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dy *= (T) 2.0;
		}
		if ((dy / dx) > (T)mg_coarse_aspect) {
			// - semi-coarsening by x
			coarse_type = mg_coarse_x;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			dx *= (T) 2.0;
		}
		if (coarse_type == mg_coarse_null) {
			// - full coarsening by x and y
			coarse_type = mg_coarse_xy;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dx *= (T) 2.0;
			dy *= (T) 2.0;
		}

		size += 5 * nx * ny;
	}

	return size;
}

template< typename T >
void nse::mg_var_poisson2d_data< T > ::init(
	T* i_density_fine,
	const int bc_type,
	const nse::uniGrid2d< T >& grid, const int _num_grids)
{
	// check <memory is sufficient>
	int m_size = memory_size(grid, _num_grids);
	if (m_size > mg_memory_size)
	{
		clear();
		mg_memory_size = m_size;
		allocate(&mg_memory, mg_memory_size);
	}

	num_grids = _num_grids;
	null(mg_memory, mg_memory_size);

	const int fine = 0;

	x[fine] = NULL;       // defined in mg solver
	rhs[fine] = NULL;     // defined in mg solver
	res[fine] = mg_memory;
	i_density[fine] = i_density_fine;
	idg[fine] = &mg_memory[grid.size];

	nx[fine] = grid.nx;
	ny[fine] = grid.ny;
	gcx = grid.gcx;
	gcy = grid.gcy;

	dx[fine] = grid.dx;
	dy[fine] = grid.dy;
	dx2i[fine] = grid.dx2i;
	dy2i[fine] = grid.dy2i;
	dx2ih[fine] = grid.dx2ih;
	dy2ih[fine] = grid.dy2ih;

	poisson2d::set_diagonal_inverse(idg[fine], i_density[fine],
		nx[fine], ny[fine],
		gcx, nx[fine] - gcx - 1,
		gcy, ny[fine] - gcy - 1,

		dx2ih[fine], dy2ih[fine]);

	sgs_down_iters[fine] = c_mg_sgs_down_iters;
	sgs_up_iters[fine] = c_mg_sgs_up_iters;

	sgs_color_shift[fine] = ((grid.mpi_com.offset_x(nx[fine], gcx) +
		grid.mpi_com.offset_y(ny[fine], gcy)) & 1);

	coarse_type[fine] = mg_coarse_null;

	int mem_shift = 2 * grid.size;

	for (int i = fine + 1; i < num_grids; i++) {

		int shx = ((nx[i - 1] - (gcx << 1)) & 1);
		int shy = ((ny[i - 1] - (gcy << 1)) & 1);

		coarse_type[i] = mg_coarse_null;
		// - define coarse type
		if ((dx[i - 1] / dy[i - 1]) >(T) mg_coarse_aspect) {
			// - semi-coarsening by y
			coarse_type[i] = mg_coarse_y;

			nx[i] = nx[i - 1];
			ny[i] = ((ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			dx2ih[i] = dx2ih[i - 1];
			dy2ih[i] = (T) 0.25 * dy2ih[i - 1];

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}

		}
		if ((dy[i - 1] / dx[i - 1]) > (T)mg_coarse_aspect) {
			// - semi-coarsening by x
			coarse_type[i] = mg_coarse_x;
			nx[i] = ((nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny[i] = ny[i - 1];

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = dy2i[i - 1];
			dx2ih[i] = (T) 0.25 * dx2ih[i - 1];
			dy2ih[i] = dy2ih[i - 1];

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}

		if (coarse_type[i] == mg_coarse_null) {
			// - full coarsening by x and y
			coarse_type[i] = mg_coarse_xy;
			nx[i] = ((nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny[i] = ((ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			dx2ih[i] = (T) 0.25 * dx2ih[i - 1];
			dy2ih[i] = (T) 0.25 * dy2ih[i - 1];

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}

		sgs_color_shift[i] = ((grid.mpi_com.offset_x(nx[i], gcx) +
			grid.mpi_com.offset_y(ny[i], gcy)) & 1);

		x[i] = &mg_memory[mem_shift];
		rhs[i] = &mg_memory[mem_shift + nx[i] * ny[i]];
		res[i] = &mg_memory[mem_shift + 2 * nx[i] * ny[i]];
		i_density[i] = &mg_memory[mem_shift + 3 * nx[i] * ny[i]];
		idg[i] = &mg_memory[mem_shift + 4 * nx[i] * ny[i]];

		poisson2d::mg_restrict(i_density[i], i_density[i - 1],
			coarse_type[i],

			nx[i], ny[i],
			nx[i - 1], ny[i - 1],
			gcx, gcy,

			gcx, nx[i] - gcx - 1,
			gcy, ny[i] - gcy - 1);

		poisson2d::put_exch_bc(i_density[i],
			nx[i], ny[i], gcx, gcy, grid.mpi_com, bc_type);

		poisson2d::set_diagonal_inverse(idg[i], i_density[i],
			nx[i], ny[i],
			gcx, nx[i] - gcx - 1,
			gcy, ny[i] - gcy - 1,

			dx2ih[i], dy2ih[i]);

		mem_shift += 5 * (nx[i] * ny[i]);
	}
}

template< typename T >
void nse::mg_var_poisson2d_data< T > ::clear()
{
	num_grids = 0;

	if (mg_memory_size > 0) {
		deallocate(mg_memory);

		mg_memory_size = 0;

	}
}
// ------------------------------------------------------------------------ //

template< typename T >
int nse::mg_mpi_poisson2d_data< T > ::memory_size( // multigrid size
	const uniGrid2d< T >& grid, const int _num_grids)
{
	const int fine = 0;
	int coarse_type = mg_coarse_null;

	int mpi_run = 1;
	int mpi_level = 1;
	int rank = grid.mpi_com.rank;
	int rank_x = grid.mpi_com.rank_x, rank_y = grid.mpi_com.rank_y;
	int size_x = grid.mpi_com.size_x, size_y = grid.mpi_com.size_y;

	int nx = grid.nx, ny = grid.ny;
	int gcx = grid.gcx, gcy = grid.gcy;

	int size = 0;

	T dx = grid.dx, dy = grid.dy;

	for (int i = fine + 1; i < _num_grids; i++) {

		int shx = ((nx - (gcx << 1)) & 1);
		int shy = ((ny - (gcy << 1)) & 1);

		coarse_type = mg_coarse_null;
		// - define coarse type
		if (((dx / dy) >(T) mg_coarse_aspect) && (mpi_run)) {
			// - semi-coarsening by y
			coarse_type = mg_coarse_y;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dy *= (T) 2.0;
		}
		if (((dy / dx) > (T)mg_coarse_aspect) && (mpi_run)) {
			// - semi-coarsening by x
			coarse_type = mg_coarse_x;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			dx *= (T) 2.0;
		}
		if ((coarse_type == mg_coarse_null) && (mpi_run)) {
			// - full coarsening by x and y
			coarse_type = mg_coarse_xy;
			nx = ((nx - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			ny = ((ny - (gcy << 1)) >> 1) + (gcy << 1) + shy;
			dx *= (T) 2.0;
			dy *= (T) 2.0;
		}

		int max_processor_size = mpi_run * sizeof(T)*
			((nx - 2 * gcx) * (ny - 2 * gcy));
		nse::mpi_allreduce(&max_processor_size, MPI_MAX);

		if ((max_processor_size <= mg_mpi_min_proc_size) && (mpi_run)) {
			mpi_level *= 2;
			mpi_run = (((rank_x % mpi_level) == 0) && ((rank_y % mpi_level) == 0));

			int rank_sh = (mpi_level >> 1);
			if (mpi_run) {
				// get size addition from adjacent processors //

				int acx = 0, acy = 0;
				if (rank_x + rank_sh < size_x)
					MPI_Recv(&acx, 1, MPI_INT, rank + rank_sh, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				if (rank_y + rank_sh < size_y)
					MPI_Recv(&acy, 1, MPI_INT, rank + rank_sh * size_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				nx += acx;
				ny += acy;
			}
			else
			{
				int acx = nx - 2 * gcx;
				int acy = ny - 2 * gcy;

				if ((rank_x - rank_sh >= 0) &&
					((rank_x % mpi_level) == rank_sh) && ((rank_y % mpi_level) == 0))
				{
					// send acx //
					MPI_Send(&acx, 1, MPI_INT, rank - rank_sh, 0, MPI_COMM_WORLD);
				}
				if ((rank_y - rank_sh >= 0) &&
					((rank_x % mpi_level) == 0) && ((rank_y % mpi_level) == rank_sh))
				{
					// send acy //
					MPI_Send(&acy, 1, MPI_INT, rank - rank_sh * size_x, 0, MPI_COMM_WORLD);
				}

				size += 2 * nx * ny;
			}
		}

		size += mpi_run * (2 * nx * ny);
	}

	return size;
}

template< typename T >
void nse::mg_mpi_poisson2d_data< T > ::init(
	const nse::uniGrid2d< T >& grid, const int _num_grids)
{
	clear();

	num_grids = _num_grids;
	mg_memory_size = memory_size(grid, num_grids);
	allocate(&mg_memory, mg_memory_size);

	const int fine = 0;

	x[fine] = NULL;       // defined in mg solver
	rhs[fine] = NULL;     // defined in mg solver

	mpi_run[fine] = 1;
	mpi_level[fine] = 1;
	mpi_combine[fine] = 0;

	mpi_com[fine].copy(grid.mpi_com);

	local_nx[fine] = grid.nx;
	local_ny[fine] = grid.ny;
	mpi_nx[fine] = grid.nx;
	mpi_ny[fine] = grid.ny;
	gcx = grid.gcx;
	gcy = grid.gcy;

	dx[fine] = grid.dx;
	dy[fine] = grid.dy;
	dx2i[fine] = grid.dx2i;
	dy2i[fine] = grid.dy2i;

	idg[fine] = -(T) 1.0 / (grid.dx2id + grid.dy2id);

	sgs_down_iters[fine] = c_mg_sgs_down_iters;
	sgs_up_iters[fine] = c_mg_sgs_up_iters;

	sgs_color_shift[fine] = ((mpi_com[fine].offset_x(mpi_nx[fine], gcx) +
		mpi_com[fine].offset_y(mpi_ny[fine], gcy)) & 1);

	coarse_type[fine] = mg_coarse_null;

	int mem_shift = 0;
	for (int i = fine + 1; i < num_grids; i++) {

		mpi_run[i] = mpi_run[i - 1];
		mpi_level[i] = mpi_level[i - 1];
		mpi_combine[i] = 0;

		mpi_com[i].copy(mpi_com[i - 1]);

		local_nx[i] = mpi_nx[i - 1];
		local_ny[i] = mpi_ny[i - 1];

		int shx = ((mpi_nx[i - 1] - (gcx << 1)) & 1);
		int shy = ((mpi_ny[i - 1] - (gcy << 1)) & 1);

		coarse_type[i] = mg_coarse_null;
		// - define coarse type
		// - divide only on running processors
		if ((mpi_run[i]) && ((dx[i - 1] / dy[i - 1]) > (T)mg_coarse_aspect)) {
			// - semi-coarsening by y
			coarse_type[i] = mg_coarse_y;
			local_nx[i] = mpi_nx[i - 1];
			local_ny[i] = ((mpi_ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			idg[i] = -(T) 1.0 / ((T) 2.0 * dx2i[i] + (T) 2.0 * dy2i[i]);

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}

		}
		if ((mpi_run[i]) && ((dy[i - 1] / dx[i - 1]) > (T)mg_coarse_aspect)) {
			// - semi-coarsening by x
			coarse_type[i] = mg_coarse_x;
			local_nx[i] = ((mpi_nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			local_ny[i] = mpi_ny[i - 1];

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = dy2i[i - 1];
			idg[i] = -(T) 1.0 / ((T) 2.0 * dx2i[i] + (T) 2.0 * dy2i[i]);

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}
		if ((mpi_run[i]) && (coarse_type[i] == mg_coarse_null)) {
			// - full coarsening by x and y
			coarse_type[i] = mg_coarse_xy;
			local_nx[i] = ((mpi_nx[i - 1] - (gcx << 1)) >> 1) + (gcx << 1) + shx;
			local_ny[i] = ((mpi_ny[i - 1] - (gcy << 1)) >> 1) + (gcy << 1) + shy;

			dx[i] = (T) 2.0 * dx[i - 1];
			dy[i] = (T) 2.0 * dy[i - 1];

			dx2i[i] = (T) 0.25 * dx2i[i - 1];
			dy2i[i] = (T) 0.25 * dy2i[i - 1];
			idg[i] = (T) 4.0 * idg[i - 1];

			if (i < num_grids - 1) {
				sgs_down_iters[i] = c_mg_sgs_down_iters;
				sgs_up_iters[i] = c_mg_sgs_up_iters;
			}
			else
			{
				sgs_down_iters[i] = c_mg_sgs_direct_iters;
				sgs_up_iters[i] = c_mg_sgs_direct_iters;
			}
		}

		mpi_nx[i] = local_nx[i];
		mpi_ny[i] = local_ny[i];

		// get max task size for all processors //
		int max_processor_size = mpi_run[i] * sizeof(T)*
			((local_nx[i] - 2 * gcx) * (local_ny[i] - 2 * gcy));
		nse::mpi_allreduce(&max_processor_size, MPI_MAX);

		// if task size small enough: begin divide-and-conquer //
		if ((max_processor_size <= mg_mpi_min_proc_size) &&
			(mpi_run[i]) && (mpi_com[i].size > 1))
		{

			mpi_level[i] *= 2;
			mpi_combine[i] = mpi_level[i];

			mpi_run[i] = (((grid.mpi_com.rank_x % mpi_level[i]) == 0) &&
				((grid.mpi_com.rank_y % mpi_level[i]) == 0));

			mpi_com[i].split_comm(mpi_com[i - 1], 2, 2);


			int rank_sh = (mpi_level[i] >> 1);
			if (mpi_run[i]) {
				// get size addition from adjacent processors

				int acx = 0, acy = 0;
				if (grid.mpi_com.rank_x + rank_sh < grid.mpi_com.size_x)
					MPI_Recv(&acx, 1, MPI_INT, grid.mpi_com.rank + rank_sh,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				if (grid.mpi_com.rank_y + rank_sh < grid.mpi_com.size_y)
					MPI_Recv(&acy, 1, MPI_INT, grid.mpi_com.rank + rank_sh * grid.mpi_com.size_x,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				mpi_nx[i] += acx;
				mpi_ny[i] += acy;
			}
			else
			{
				int acx = local_nx[i] - 2 * gcx;
				int acy = local_ny[i] - 2 * gcy;

				if ((grid.mpi_com.rank_x - rank_sh >= 0) &&
					((grid.mpi_com.rank_x % mpi_level[i]) == rank_sh) &&
					((grid.mpi_com.rank_y % mpi_level[i]) == 0))
				{
					// send acx //
					MPI_Send(&acx, 1, MPI_INT, grid.mpi_com.rank - rank_sh,
						0, MPI_COMM_WORLD);
				}
				if ((grid.mpi_com.rank_y - rank_sh >= 0) &&
					((grid.mpi_com.rank_x % mpi_level[i]) == 0) &&
					((grid.mpi_com.rank_y % mpi_level[i]) == rank_sh))
				{
					// send acy //
					MPI_Send(&acy, 1, MPI_INT, grid.mpi_com.rank - rank_sh * grid.mpi_com.size_x,
						0, MPI_COMM_WORLD);
				}

				mpi_nx[i] -= acx;
				mpi_ny[i] -= acy;
			}
		}

		if (mpi_run[i]) {
			x[i] = &mg_memory[mem_shift];
			rhs[i] = &mg_memory[mem_shift + mpi_nx[i] * mpi_ny[i]];

			sgs_color_shift[i] = ((mpi_com[i].offset_x(mpi_nx[i], gcx) +
				mpi_com[i].offset_y(mpi_ny[i], gcy)) & 1);

			mem_shift += 2 * (mpi_nx[i] * mpi_ny[i]);
		}
		if ((mpi_combine[i]) && (!mpi_run[i])) {
			x[i] = &mg_memory[mem_shift];
			rhs[i] = &mg_memory[mem_shift + local_nx[i] * local_ny[i]];

			mem_shift += 2 * (local_nx[i] * local_ny[i]);
		}
	}
}

template< typename T >
void nse::mg_mpi_poisson2d_data< T > ::clear()
{
	// free MPI communicators, excluding MPI_COMM_WORLD
	const int fine = 0;
	for (int i = fine + 1; i < num_grids; i++) {

		if (mpi_combine[i]) {   // combination step and split of communicators
			MPI_Comm_free(&mpi_com[i].comm);
		}
	}

	num_grids = 0;

	if (mg_memory_size > 0) {
		deallocate(mg_memory);

		mg_memory_size = 0;

	}
}
// ------------------------------------------------------------------------ //

// ------------------------------------------------------------------------ //
template< typename T >
void poisson2d::mg_restrict( // multigrid restrict: fine -> coarse
	T* coarse, const T* fine,
	const int type,
	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce)
{
	int i, j, cidx, idx;

	if (type == nse::mg_coarse_xy) {

		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid


#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, fine )
		for (i = icb; i <= ice; i++) {

			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2) {
				coarse[cidx] = (T) 0.25 * (
					fine[idx] + fine[idx + 1] +
					fine[idx + ny] + fine[idx + ny + 1]);
			}
		}

		return;
	}

	if (type == nse::mg_coarse_x) {

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, fine )
		for (i = icb; i <= ice; i++)
		{

			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jcb;
			for (j = jcb; j <= jce; j++, cidx++, idx++)
				coarse[cidx] = (T) 0.5 * (fine[idx] + fine[idx + ny]);
		}

		return;
	}

	if (type == nse::mg_coarse_y) {

		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, fine )
		for (i = icb; i <= ice; i++)
		{

			cidx = i * cny + jcb;
			idx = i * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2) {
				coarse[cidx] = (T) 0.5 * (fine[idx] + fine[idx + 1]);
			}
		}

		return;
	}
}
// ------------------------------------------------------------------------ //

template< typename T >
void poisson2d::mg_restrict_residual( // multigrid restrict: fine -> coarse
	T* coarse, const T* x, const T* rhs,

	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce,

	const T dx2i, const T dy2i)
{
	int i, j, cidx, idx;

	if (type == nse::mg_coarse_xy) {
		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid

		const T dx2iq = (T) 0.25 * dx2i;
		const T dy2iq = (T) 0.25 * dy2i;

		T b_value;

#pragma omp parallel for private( i, j, cidx, idx, b_value ) shared( coarse, x )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2) {

				b_value = x[idx + ny + 1] + x[idx + ny] +
					x[idx + 1] + x[idx];

				coarse[cidx] =
					(b_value -

					x[idx + (ny << 1)] - x[idx + (ny << 1) + 1] -
					x[idx - ny + 1] - x[idx - ny]
					) * dx2iq

					+

					(b_value -

					x[idx + ny + 2] - x[idx + ny - 1] -
					x[idx + 2] - x[idx - 1]
					) * dy2iq;
			}
		}

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, rhs )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2)
				coarse[cidx] += (T) 0.25 * (
				rhs[idx] + rhs[idx + 1] + rhs[idx + ny] + rhs[idx + ny + 1]);
		}

		return;
	}

	if (type == nse::mg_coarse_x) {
		const T dx2ih = (T) 0.5 * dx2i;
		const T dy2ih = (T) 0.5 * dy2i;

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, x )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jcb;
			for (j = jcb; j <= jce; j++, cidx++, idx++)
			{
				coarse[idx] =
					-(x[idx + (ny << 1)] + x[idx - ny] - x[idx + ny] - x[idx]) * dx2ih

					- (x[idx + ny + 1] - (T) 2.0 * x[idx + ny] + x[idx + ny - 1]
					+ x[idx + 1] - (T) 2.0 * x[idx] + x[idx - 1]) * dy2ih;
			}
		}

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, rhs )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = ((i << 1) - gcx) * ny + jcb;
			for (j = jcb; j <= jce; j++, cidx++, idx++)
				coarse[idx] += (T) 0.5 * (rhs[idx] + rhs[idx + ny]);
		}

		return;
	}

	if (type == nse::mg_coarse_y) {
		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid

		const T dx2ih = (T) 0.5 * dx2i;
		const T dy2ih = (T) 0.5 * dy2i;

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, x )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = i * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2) {
				coarse[cidx] =
					-(x[idx + ny] - (T) 2.0 * x[idx] + x[idx - ny]
					+ x[idx + ny + 1] - (T) 2.0 * x[idx + 1] + x[idx - ny + 1]) * dx2ih

					- (x[idx + 2] + x[idx - 1] - x[idx + 1] - x[idx]) * dy2ih;
			}
		}

#pragma omp parallel for private( i, j, cidx, idx ) shared( coarse, rhs )
		for (i = icb; i <= ice; i++)
		{
			cidx = i * cny + jcb;
			idx = i * ny + jsh;
			for (j = jcb; j <= jce; j++, cidx++, idx += 2)
				coarse[cidx] += (T) 0.5 * (rhs[idx] + rhs[idx + 1]);
		}

		return;
	}
}
// ------------------------------------------------------------------------ //

template< typename T >
void poisson2d::mg_prolongate( // multigrid prolongate: coarse -> fine
	T* fine, const T* coarse,

	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy)
{
	int i, j, idx, cidx;

	if (type == nse::mg_coarse_xy) {
		const T a = (T) 9.0 / (T) 16.0;
		const T b = (T) 3.0 / (T) 16.0;
		const T c = (T) 3.0 / (T) 16.0;
		const T d = (T) 1.0 / (T) 16.0;


		T C, Cxm, Cxp, Cym, Cyp;
#pragma omp parallel for private( i, j, idx, cidx, C, Cxm, Cxp, Cym, Cyp ) shared( fine, coarse )
		for (i = gcx; i < cnx - gcx; i++)
		{
			idx = ((i << 1) - gcx) * ny + gcy;
			cidx = i * cny + gcy;
			for (j = gcy; j < cny - gcy; j++, idx += 2, cidx++)
			{
				C = a * coarse[cidx];
				Cxm = c * coarse[cidx - cny];
				Cxp = c * coarse[cidx + cny];
				Cym = C + b * coarse[cidx - 1];
				Cyp = C + b * coarse[cidx + 1];

				fine[idx] += Cym + Cxm + d * coarse[cidx - cny - 1];
				fine[idx + 1] += Cyp + Cxm + d * coarse[cidx - cny + 1];
				fine[idx + ny] += Cym + Cxp + d * coarse[cidx + cny - 1];
				fine[idx + ny + 1] += Cyp + Cxp + d * coarse[cidx + cny + 1];
			}

#ifdef _MG_PROLONGATE_EX
			if (ny & 1)
				fine[((i << 1) - gcx) * ny + ny - gcy - 1] +=
				coarse[i * cny + cny - gcy - 1];
#endif    
		}


#ifdef _MG_PROLONGATE_EX
		if (nx & 1)
		{
			idx = (nx - gcx - 1) * ny + gcy;
			cidx = (cnx - gcx - 1) * cny + gcy;
			for (j = gcy; j < cny - gcy; j++, idx += 2, cidx++)
			{
				fine[idx] += coarse[cidx];
				fine[idx + 1] += coarse[cidx];
			}

			if (ny & 1)
				fine[(nx - gcx - 1) * ny + ny - gcy - 1] +=
				coarse[(cnx - gcx - 1) * cny + cny - gcy - 1];
		}
#endif

		return;
	}

	if (type == nse::mg_coarse_x) {

#pragma omp parallel for private( i, j, idx, cidx ) shared( fine, coarse )
		for (i = gcx; i < cnx - gcx; i++)
		{
			idx = ((i << 1) - gcx) * ny + gcy;
			cidx = i * cny + gcy;
			for (j = gcy; j < cny - gcy; j++, idx++, cidx++)
			{
				fine[idx] += coarse[cidx];
				fine[idx + ny] += coarse[cidx];
			}
		}

		return;
	}

	if (type == nse::mg_coarse_y) {

#pragma omp parallel for private( i, j, idx, cidx ) shared( fine, coarse )
		for (i = gcx; i < cnx - gcx; i++)
		{
			idx = i * ny + gcy;
			cidx = i * cny + gcy;
			for (j = gcy; j < cny - gcy; j++, idx += 2, cidx++)
			{
				fine[idx] += coarse[cidx];
				fine[idx + 1] += coarse[cidx];
			}
		}

		return;
	}
}

template< typename T >
void poisson2d::mg_prolongate( // multigrid prolongate: coarse -> fine
	T* fine, const T* coarse,

	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce)
{
	int i, j, idx, cidx;

	if (type == nse::mg_coarse_xy) {
		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid

		const T a = (T) 9.0 / (T) 16.0;
		const T b = (T) 3.0 / (T) 16.0;
		const T c = (T) 3.0 / (T) 16.0;
		const T d = (T) 1.0 / (T) 16.0;


		T C, Cxm, Cxp, Cym, Cyp;
#pragma omp parallel for private( i, j, idx, cidx, C, Cxm, Cxp, Cym, Cyp ) shared( fine, coarse )
		for (i = icb; i <= ice; i++)
		{
			idx = ((i << 1) - gcx) * ny + jsh;
			cidx = i * cny + jcb;
			for (j = jcb; j <= jce; j++, idx += 2, cidx++)
			{
				C = a * coarse[cidx];
				Cxm = c * coarse[cidx - cny];
				Cxp = c * coarse[cidx + cny];
				Cym = C + b * coarse[cidx - 1];
				Cyp = C + b * coarse[cidx + 1];

				fine[idx] += Cym + Cxm + d * coarse[cidx - cny - 1];
				fine[idx + 1] += Cyp + Cxm + d * coarse[cidx - cny + 1];
				fine[idx + ny] += Cym + Cxp + d * coarse[cidx + cny - 1];
				fine[idx + ny + 1] += Cyp + Cxp + d * coarse[cidx + cny + 1];
			}
		}

		return;
	}

	if (type == nse::mg_coarse_x) {

#pragma omp parallel for private( i, j, idx, cidx ) shared( fine, coarse )
		for (i = icb; i <= ice; i++)
		{
			idx = ((i << 1) - gcx) * ny + jcb;
			cidx = i * cny + jcb;
			for (j = jcb; j <= jce; j++, idx++, cidx++)
			{
				fine[idx] += coarse[cidx];
				fine[idx + ny] += coarse[cidx];
			}
		}

		return;
	}

	if (type == nse::mg_coarse_y) {
		const int jsh = ((jcb << 1) - gcy);    // j starting index on fine grid

#pragma omp parallel for private( i, j, idx, cidx ) shared( fine, coarse )
		for (i = icb; i <= ice; i++)
		{
			idx = i * ny + jsh;
			cidx = i * cny + jcb;
			for (j = jcb; j <= jce; j++, idx += 2, cidx++)
			{
				fine[idx] += coarse[cidx];
				fine[idx + 1] += coarse[cidx];
			}
		}

		return;
	}
}
// ------------------------------------------------------------------------ //

// initialize: multigrid data 
template struct nse::mg_poisson2d_data< float >;
template struct nse::mg_poisson2d_data< double >;

template struct nse::mg_var_poisson2d_data< float >;
template struct nse::mg_var_poisson2d_data< double >;

template struct nse::mg_mpi_poisson2d_data< float >;
template struct nse::mg_mpi_poisson2d_data< double >;

// initialize: multigrid restrict operator
template void poisson2d::mg_restrict(float* coarse, const float* fine,
	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce);
template void poisson2d::mg_restrict(double* coarse, const double* fine,
	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce);
// ------------------------------------------------------------------------ //

// initialize: multigrid restrict residual operator
template void poisson2d::mg_restrict_residual(float* coarse, const float* x, const float* rhs,
	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce,

	const float dx2i, const float dy2i);
template void poisson2d::mg_restrict_residual(double* coarse, const double* x, const double* rhs,
	const int type,

	const int cnx, const int cny,
	const int nx, const int ny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce,

	const double dx2i, const double dy2i);
// ------------------------------------------------------------------------ //

// initialize: multigrid prolongation operator
template void poisson2d::mg_prolongate(float* fine, const float* coarse,
	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy);
template void poisson2d::mg_prolongate(double* fine, const double* coarse,
	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy);

template void poisson2d::mg_prolongate(float* fine, const float* coarse,
	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce);
template void poisson2d::mg_prolongate(double* fine, const double* coarse,
	const int type,

	const int nx, const int ny,
	const int cnx, const int cny,
	const int gcx, const int gcy,

	const int icb, const int ice,
	const int jcb, const int jce);
// ------------------------------------------------------------------------ //
