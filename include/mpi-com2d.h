#pragma once

// * 2D MPI communicator * //

#include <mpi.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

#include "mpi-com.h"
#include "grid-common2d.h"

// *[mpi-com2d.h]: full //

// NOTES:
//      - direct corner exchanges are inefficient based on SKIF MSU tests
//      - it's assumed for periodic bc that ( mpi_nx[y] - 2 * gcx[y] ) is even
//      - 'init' exchange is only valid for last coordinate for now


namespace nse
{
	// * MPI communicator: mpiCom2d * //
	// =======================================================================
	class mpiCom2d
	{
	public:
		mpiCom2d();
		mpiCom2d(const mpiCom2d& mpi_com);
		~mpiCom2d();

		void set(const int ndim);
		void set(const int _size_x, const int _size_y);

		void copy(const mpiCom2d& mpi_com);
		void split_comm(const mpiCom2d& mpi_com, const int modx, const int mody);

		static void clear();

		// * gather data * //
		template< typename T >
		void gather_x(T* out, T* in, const int host, const int nx, const int gcx) const;
		template< typename T >
		void gather_y(T* out, T* in, const int host, const int ny, const int gcy) const;

		template< typename T >
		void gather(T* out, T* in, const int host,
			const int nx, const int ny, const int gcx, const int gcy) const;

		// * scatter data * //
		template< typename T >
		void scatter(T* out, T* in, const int host,
			const int nx, const int ny, const int gcx, const int gcy) const;


		// * gather-scatter data from(to) odd processors * //
		template< typename T >
		void c_gather_odd_grid(T* out, const T* in,    // out[nx, ny] <-- in[sub_nx, sub_ny]
			const int nx, const int ny,
			const int sub_nx, const int sub_ny,
			const int gcx, const int gcy) const;

		template< typename T >
		void c_scatter_odd_grid(T* out, const T* in,   // out[sub_nx, sub_ny] <-- in[nx, ny]
			const int nx, const int ny,
			const int sub_nx, const int sub_ny,
			const int gcx, const int gcy) const;

		template< typename T >
		void c_gather_odd_grid_x(T* out, const T* in,    // out[nx] <-- in[sub_nx]
			const int nx, const int sub_nx, const int gcx) const;

		template< typename T >
		void c_gather_odd_grid_y(T* out, const T* in,    // out[ny] <-- in[sub_ny]
			const int ny, const int sub_ny, const int gcy) const;

		// * exchange data * //
		// --------------------------------------------------------------------------------- //
		//
		// *** sync mpi: cross halo ( no corners exchanges )
		//          single sync for -x and -y
		template< typename T >
		void exchange_cross_halo(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y) const;

		// *** sync mpi: full halo( including corners )
		//          double sync for -x and -y
		template< typename T >
		void exchange_halo(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y) const;

		// *** sync mpi: full halo( including corners )
		//          single sync for -x
		template< typename T >
		void exchange_halo_x(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x) const;

		// *** sync mpi: full halo( including corners )
		//          single sync for -y
		template< typename T >
		void exchange_halo_y(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y) const;

		// *** sync mpi: full color halo (including corners)
		//          double sync for -x and -y
		template< typename T >
		void exchange_color_halo(T* x, const int color,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y) const;

		// *** sync mpi: full halo( including corners ), combined for u, v staggered components
		//          double sync for -x[ u ], -y[ v ] and -y[ u ], -x[ v ]
		template< typename T >
		void exchange_halo(T* u, T* v,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y) const;


		// *** async mpi exchange init: cross halo ( no corners exchanges )
		template< typename T >
		void push_exchange_cross_halo(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y,
			MPI_Request mpi_req[8]) const;

		// *** async mpi exchange finalize: cross halo ( no corners exchanges )
		template< typename T >
		void pop_exchange_cross_halo(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, const int periodic_y,
			MPI_Request mpi_req[8]) const;


		// *** async - X mpi exchange init: full halo ( including corners exchanges )
		template< typename T >
		void push_exchange_halo_x(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, MPI_Request mpi_req[4]) const;

		// *** async - X mpi exchange finalize: full halo ( including corners exchanges )
		template< typename T >
		void pop_exchange_halo_x(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x, MPI_Request mpi_req[4]) const;

		// *** async - Y mpi exchange init: full halo ( including corners exchanges )
		template< typename T >
		void init_exchange_halo_y(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y) const;

		// *** async - Y mpi exchange init: full halo ( including corners exchanges )
		template< typename T >
		void push_exchange_halo_y(T* x,
			const int init_flag,

			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y,
			MPI_Request mpi_req[4]) const;

		// *** async - Y mpi exchange finalize: full halo ( including corners exchanges )
		template< typename T >
		void pop_exchange_halo_y(T* x,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y,
			MPI_Request mpi_req[4]) const;


		// *** async - X mpi exchange push: full color halo ( including corners )
		template< typename T >
		void push_exchange_color_halo_x(T* x, const int color,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x,
			MPI_Request mpi_req[4]) const;

		// *** async - X mpi exchange finalize: full color halo ( including corners )
		template< typename T >
		void pop_exchange_color_halo_x(T* x, const int color,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_x,
			MPI_Request mpi_req[4]) const;

		// *** async - Y mpi exchange push: full color halo ( including corners )
		template< typename T >
		void push_exchange_color_halo_y(T* x, const int color,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y,
			MPI_Request mpi_req[4]) const;

		// *** async - Y mpi exchange finalize: full color halo ( including corners )
		template< typename T >
		void pop_exchange_color_halo_y(T* x, const int color,
			const int nx, const int ny,
			const int gcx, const int gcy,
			const int hx, const int hy,
			const int periodic_y,
			MPI_Request mpi_req[4]) const;

		// *** async - test mpi exchange ( any exchange operator )
		int test_exchange(MPI_Request* mpi_req, const int n_req) const;
		// --------------------------------------------------------------------------------- //

		int offset_x(const int nx, const int gcx) const;
		int offset_y(const int ny, const int gcy) const;

	public:     // communicator data declared public
		MPI_Comm comm;
				// additional line communicators
		MPI_Comm comm_x, comm_y;

		int rank, size;
		int rank_x, size_x;
		int rank_y, size_y;

		// MPI exchange timing
		static double cpu_time_exch;
		static double cpu_time_exch_x;
		static double cpu_time_exch_y;

	private:
		// -------------------------------------------------------------------------- //
		// buffer for exchange template operations
		static void *exch;
		// directional buffers for exchange template operations
		static void *exch_x, *exch_y;
		// corner buffer for exchange template operations
		static void *exch_xy;

		static size_t exch_size;
		static size_t exch_size_x, exch_size_y;
		static size_t exch_size_xy;

		void allocate_memory(void** mem, size_t* mem_size, const size_t new_size) const;
	};
	// =============================================================================== //

}


// [implementation]: mpiCom2d //
// =======================================================================
namespace nse
{
	inline
		mpiCom2d::mpiCom2d(
		) : rank(0), size(1),
		rank_x(0), size_x(1),
		rank_y(0), size_y(1)
	{
		}

	inline
		mpiCom2d::mpiCom2d(
		const mpiCom2d& mpi_com)
		: comm(mpi_com.comm),
		comm_x(mpi_com.comm_x), comm_y(mpi_com.comm_y),
		rank(mpi_com.rank), size(mpi_com.size),
		rank_x(mpi_com.rank_x), size_x(mpi_com.size_x),
		rank_y(mpi_com.rank_y), size_y(mpi_com.size_y)
	{
		}

	inline
		mpiCom2d :: ~mpiCom2d(
		)
	{
		}

	inline
		void mpiCom2d::set(
		const int ndim)
	{
			assert((ndim >= 0) && (ndim <= 2));
			if (ndim == 0) return;

			comm = MPI_COMM_WORLD;
			MPI_Comm_size(comm, &size);
			MPI_Comm_rank(comm, &rank);

			if (ndim == 1) {
				rank_x = rank; size_x = size;
				rank_y = 0;    size_y = 1;
			}
			if (ndim == 2) {
				mpi_com_dims(size, &size_x, &size_y);

				rank_x = rank % size_x;
				rank_y = (rank / size_x) % size_y;
			}

			// line communicators
			MPI_Comm_split(comm, rank_y, rank, &comm_x);
			MPI_Comm_split(comm, rank_x, rank, &comm_y);
		}

	inline
		void mpiCom2d::set(
		const int _size_x, const int _size_y)
	{
			comm = MPI_COMM_WORLD;
			MPI_Comm_size(comm, &size);
			MPI_Comm_rank(comm, &rank);

			if (_size_x * _size_y != size)
			{
				if (_size_y == 1) set(1);
				else
					set(2);

				return;
			}

			size_x = _size_x;
			size_y = _size_y;

			rank_x = rank % size_x;
			rank_y = (rank / size_x) % size_y;

			// line communicators
			MPI_Comm_split(comm, rank_y, rank, &comm_x);
			MPI_Comm_split(comm, rank_x, rank, &comm_y);
		}

	inline
		void mpiCom2d::copy(const mpiCom2d& mpi_com)
	{
			comm = mpi_com.comm;
			comm_x = mpi_com.comm_x;
			comm_y = mpi_com.comm_y;

			rank = mpi_com.rank;
			size = mpi_com.size;

			rank_x = mpi_com.rank_x; size_x = mpi_com.size_x;
			rank_y = mpi_com.rank_y; size_y = mpi_com.size_y;
		}

	inline
		void mpiCom2d::split_comm(const mpiCom2d& mpi_com, const int modx, const int mody)
	{
			int color = ((mpi_com.rank_x % modx == 0) &&
				(mpi_com.rank_y % mody == 0));

			MPI_Comm_split(mpi_com.comm, color, mpi_com.rank, &comm);
			MPI_Comm_size(comm, &size);
			MPI_Comm_rank(comm, &rank);

			size_x = (mpi_com.size_x + (modx - 1)) / modx;
			size_y = (mpi_com.size_y + (mody - 1)) / mody;

			rank_x = rank % size_x;
			rank_y = (rank / size_x) % size_y;

			// Note that we don't change line communicators
		}

	inline
		void mpiCom2d::clear() {
			if (exch_size > 0) {
				free(exch);
				exch_size = 0;
			}

			if (exch_size_x > 0) {
				free(exch_x);
				exch_size_x = 0;
			} 
			if (exch_size_y > 0) {
				free(exch_y);
				exch_size_y = 0;
			} 

			if (exch_size_xy > 0) {
				free(exch_xy);
				exch_size_xy = 0;
			} 
		}

	template< typename T >
	void mpiCom2d::gather_x(
		T* out, T* in, const int host,
		const int nx, const int gcx) const
	{
		int mpi_nx = mpi_allreduce_comm(nx - 2 * gcx, MPI_SUM, comm_x) + 2 * gcx;

		if (rank == host) // write array on processor with rank = host
		{
			int i, m;
			int prank, pdim;
			int posx = mpi_local_offset(mpi_nx, gcx, rank_x, size_x);
			int shxb = (rank_x == 0) ? 0 : gcx;
			int shxe = (rank_x == size_x - 1) ? 0 : gcx;

			for (i = shxb; i < nx - shxe; i++)
				out[i + posx] = in[i];

			T *x = NULL;
			for (m = 0; m < size; m++) {	// recieve sub array from each processor
				if (m == host) continue;

				MPI_Recv(&prank, 1, MPI_INT, m, 0, comm, MPI_STATUS_IGNORE);
				MPI_Recv(&pdim, 1, MPI_INT, m, 1, comm, MPI_STATUS_IGNORE);

				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* pdim);
				x = (T*)exch_x;
				MPI_Recv(x, pdim, mpi_type< T >(), m, 2, comm, MPI_STATUS_IGNORE);

				posx = mpi_local_offset(mpi_nx, gcx, prank, size_x);
				shxb = (prank == 0) ? 0 : gcx;
				shxe = (prank == size_x - 1) ? 0 : gcx;

				for (i = shxb; i < pdim - shxe; i++)
					out[i + posx] = x[i];
			}
		}
		else
		{
			int prank = rank_x, pdim = nx;

			MPI_Send(&prank, 1, MPI_INT, host, 0, comm);
			MPI_Send(&pdim, 1, MPI_INT, host, 1, comm);
			MPI_Send(in, pdim, mpi_type< T >(), host, 2, comm);
		}

		MPI_Barrier(comm);
	}

	template< typename T >
	void mpiCom2d::gather_y(
		T* out, T* in, const int host,
		const int ny, const int gcy) const
	{
		int mpi_ny = mpi_allreduce_comm(ny - 2 * gcy, MPI_SUM, comm_y) + 2 * gcy;

		if (rank == host) // write array on processor with rank = host
		{
			int j, m;
			int prank, pdim;
			int posy = mpi_local_offset(mpi_ny, gcy, rank_y, size_y);
			int shyb = (rank_y == 0) ? 0 : gcy;
			int shye = (rank_y == size_y - 1) ? 0 : gcy;

			for (j = shyb; j < ny - shye; j++)
				out[j + posy] = in[j];

			T *x = NULL;
			for (m = 0; m < size; m++) {	// recieve sub array from each processor
				if (m == host) continue;

				MPI_Recv(&prank, 1, MPI_INT, m, 0, comm, MPI_STATUS_IGNORE);
				MPI_Recv(&pdim, 1, MPI_INT, m, 1, comm, MPI_STATUS_IGNORE);

				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* pdim);
				x = (T*)exch_y;
				MPI_Recv(x, pdim, mpi_type< T >(), m, 2, comm, MPI_STATUS_IGNORE);

				posy = mpi_local_offset(mpi_ny, gcy, prank, size_y);
				shyb = (prank == 0) ? 0 : gcy;
				shye = (prank == size_y - 1) ? 0 : gcy;

				for (j = shyb; j < pdim - shye; j++)
					out[j + posy] = x[j];
			}
		}
		else
		{
			int prank = rank_y, pdim = ny;

			MPI_Send(&prank, 1, MPI_INT, host, 0, comm);
			MPI_Send(&pdim, 1, MPI_INT, host, 1, comm);
			MPI_Send(in, pdim, mpi_type< T >(), host, 2, comm);
		}

		MPI_Barrier(comm);
	}

	template< typename T >
	void mpiCom2d::gather(
		T* out, T* in, const int host,
		const int nx, const int ny, const int gcx, const int gcy) const
	{
		int mpi_nx = mpi_allreduce_comm(nx - 2 * gcx, MPI_SUM, comm_x) + 2 * gcx;
		int mpi_ny = mpi_allreduce_comm(ny - 2 * gcy, MPI_SUM, comm_y) + 2 * gcy;

		if (rank == host) // write array on processor with rank = host
		{
			int i, j, m, idx, odx;
			int prank[2], pdim[2];
			int posx = mpi_local_offset(mpi_nx, gcx, rank_x, size_x);
			int posy = mpi_local_offset(mpi_ny, gcy, rank_y, size_y);
			int shxb = (rank_x == 0) ? 0 : gcx;
			int shxe = (rank_x == size_x - 1) ? 0 : gcx;
			int shyb = (rank_y == 0) ? 0 : gcy;
			int shye = (rank_y == size_y - 1) ? 0 : gcy;

			for (i = shxb; i < nx - shxe; i++)
			for (j = shyb; j < ny - shye; j++) {
				odx = (i + posx) * mpi_ny + (j + posy);
				idx = i * ny + j;

				out[odx] = in[idx];
			}
 
			T *x = NULL;
			for (m = 0; m < size; m++) {	// recieve sub array from each processor
				if (m == host) continue;

				MPI_Recv(prank, 2, MPI_INT, m, 0, comm, MPI_STATUS_IGNORE);
				MPI_Recv(pdim, 2, MPI_INT, m, 1, comm, MPI_STATUS_IGNORE);

				allocate_memory(&exch, &exch_size, sizeof(T)* pdim[0] * pdim[1]);
				x = (T*)exch;
				MPI_Recv(x, pdim[0] * pdim[1], mpi_type< T >(), m, 2, 
					comm, MPI_STATUS_IGNORE);

				posx = mpi_local_offset(mpi_nx, gcx, prank[0], size_x);
				posy = mpi_local_offset(mpi_ny, gcy, prank[1], size_y);
				shxb = (prank[0] == 0) ? 0 : gcx;
				shxe = (prank[0] == size_x - 1) ? 0 : gcx;
				shyb = (prank[1] == 0) ? 0 : gcy;
				shye = (prank[1] == size_y - 1) ? 0 : gcy;

				for (i = shxb; i < pdim[0] - shxe; i++)
				for (j = shyb; j < pdim[1] - shye; j++) {
					odx = (i + posx) * mpi_ny + (j + posy);
					idx = i * pdim[1] + j;

					out[odx] = x[idx];
				}
			}
		}
		else
		{
			int prank[2], pdim[2];
			prank[0] = rank_x; prank[1] = rank_y;
			pdim[0] = nx; pdim[1] = ny;

			MPI_Send(prank, 2, MPI_INT, host, 0, comm);
			MPI_Send(pdim, 2, MPI_INT, host, 1, comm);
			MPI_Send(in, nx * ny, mpi_type< T >(), host, 2, comm);
		}

		MPI_Barrier(comm);
	}

	template< typename T >
	void mpiCom2d::scatter(
		T* out, T* in, const int host,
		const int nx, const int ny, const int gcx, const int gcy) const
	{
		int mpi_nx = mpi_allreduce_comm(nx - 2 * gcx, MPI_SUM, comm_x) + 2 * gcx;
		int mpi_ny = mpi_allreduce_comm(ny - 2 * gcy, MPI_SUM, comm_y) + 2 * gcy;

		if (rank == host) // write sub-array on processor with rank = host
		{
			int i, j, m;
			int posx, posy, pnx, pny, pdim[2], prank;
			T* mem;

			for (i = 0; i < size_x; i++) {

				// define -x [offset,size] for [i] processor
				posx = mpi_local_offset(mpi_nx, gcx, i, size_x);
				pnx = mpi_local_size(mpi_nx, gcx, i, size_x);

				for (j = 0; j < size_y; j++) {

					// define -y [offset,size] for [j] processor
					posy = mpi_local_offset(mpi_ny, gcy, j, size_y);
					pny = mpi_local_size(mpi_ny, gcy, j, size_y);

					// copy to temporary memory
					allocate_memory(&exch, &exch_size, sizeof(T)* pnx * pny);
					mem = (T*)exch;
					for (m = 0; m < pnx; m++)
						memcpy(&mem[m * pny],
						&in[(m + posx) * mpi_ny + posy], pny * sizeof(T));

					if ((i == rank_x) && (j == rank_y))
					{
						if ((pnx == nx) && (pny == ny))
							memcpy(out, mem, pnx * pny * sizeof(T));
						continue;
					}

					prank = j * size_x + i;
					pdim[0] = pnx; pdim[1] = pny;
					MPI_Send(pdim, 2, MPI_INT, prank, 0, comm);
					MPI_Send(mem, pnx * pny, mpi_type< T >(), prank, 0, comm);

				}
			}
		}
		else
		{
			int pdim[2];
			MPI_Recv(pdim, 2, MPI_INT, host, 0, comm, MPI_STATUS_IGNORE);
			if ((pdim[0] == nx) && (pdim[1] == ny))
				MPI_Recv(out, nx * ny, mpi_type< T >(), host, 0, comm, MPI_STATUS_IGNORE);
		}

		MPI_Barrier(comm);
	}

	template< typename T >
	void mpiCom2d::c_gather_odd_grid(T* out, const T* in,
		const int nx, const int ny,
		const int sub_nx, const int sub_ny,
		const int gcx, const int gcy) const
	{
		double start_time = omp_get_wtime();

		if ((rank_x % 2 == 0) && (rank_y % 2 == 0)) {

			MPI_Request mpi_req[3];
			for (int k = 0; k < 3; k++)
				mpi_req[k] = MPI_REQUEST_NULL;

			// *in from current rank //
			//      out: [0 ... sub_nx - 1], [0 ... sub_ny - 1]
			// *in from east rank //
			//      out: [sub_nx - gcx ... nx - 1], [0 ... sub_ny - 1]
			// *in from north rank //
			//      out: [0 ... sub_nx - 1], [sub_ny - gcy ... ny - 1]
			// *in from north-east rank //
			//      out: [sub_nx - gcx ... nx - 1], [sub_ny - gcy ... ny - 1]

			T* mrecv[4];
			if (rank_x < size_x - 1) {
				// * east * //
				const int e_nx = nx - sub_nx + gcx;
				const int e_ny = sub_ny;
				const int e_size = e_nx * e_ny;

				// memory allocation on demand //
				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* e_size);
				mrecv[1] = (T*)exch_x;

				MPI_Irecv(mrecv[1], e_size, mpi_type< T >(), rank + 1,
					0, comm, &mpi_req[0]);
			}

			if (rank_y < size_y - 1) {
				// * north * //
				const int n_nx = sub_nx;
				const int n_ny = ny - sub_ny + gcy;
				const int n_size = n_nx * n_ny;

				// memory allocation on demand //
				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* n_size);
				mrecv[2] = (T*)exch_y;

				MPI_Irecv(mrecv[2], n_size, mpi_type< T >(), rank + size_x,
					0, comm, &mpi_req[1]);
			}

			if ((rank_x < size_x - 1) && (rank_y < size_y - 1)) {
				// * north east * //
				const int ne_nx = nx - sub_nx + gcx;
				const int ne_ny = ny - sub_ny + gcy;
				const int ne_size = ne_nx * ne_ny;

				// memory allocation on demand //
				allocate_memory(&exch_xy, &exch_size_xy, sizeof(T)* ne_size);
				mrecv[3] = (T*)exch_xy;

				MPI_Irecv(mrecv[3], ne_size, mpi_type< T >(), rank + size_x + 1,
					0, comm, &mpi_req[2]);
			}

			// out <-- in could fail as pointers could overlap
			allocate_memory(&exch, &exch_size, sizeof(T)* sub_nx * sub_ny);
			mrecv[0] = (T*)exch;

			memcpy(mrecv[0], in, sizeof(T)* sub_nx * sub_ny);
			put_sub_array(out, nx, ny, 0, sub_nx - 1, 0, sub_ny - 1, mrecv[0]);

			MPI_Waitall(3, mpi_req, MPI_STATUSES_IGNORE);

			if (rank_x < size_x - 1)
				put_sub_array(out, nx, ny, sub_nx - gcx, nx - 1, 0, sub_ny - 1, mrecv[1]);
			if (rank_y < size_y - 1)
				put_sub_array(out, nx, ny, 0, sub_nx - 1, sub_ny - gcy, ny - 1, mrecv[2]);
			if ((rank_x < size_x - 1) && (rank_y < size_y - 1))
				put_sub_array(out, nx, ny, sub_nx - gcx, nx - 1, sub_ny - gcy, ny - 1, mrecv[3]);
		}
		else
		{

			if ((rank_x % 2 == 1) && (rank_y % 2 == 0)) {
				// send west [condition means that west processor exists]
				const int cx = sub_nx - gcx;
				const int cy = sub_ny;

				// memory allocation on demand //
				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* cx * cy);
				T *msend = (T*)exch_x;

				get_sub_array(in, sub_nx, sub_ny, gcx, sub_nx - 1, 0, sub_ny - 1, msend);
				MPI_Send(msend, cx * cy, mpi_type< T >(), rank - 1, 0, comm);
			}
			if ((rank_x % 2 == 0) && (rank_y % 2 == 1)) {
				// send south [condition means that south processor exists]
				const int cx = sub_nx;
				const int cy = sub_ny - gcy;

				// memory allocation on demand //
				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* cx * cy);
				T *msend = (T*)exch_y;

				get_sub_array(in, sub_nx, sub_ny, 0, sub_nx - 1, gcy, sub_ny - 1, msend);
				MPI_Send(msend, cx * cy, mpi_type< T >(), rank - size_x, 0, comm);
			}
			if ((rank_x % 2 == 1) && (rank_y % 2 == 1)) {
				// send south-west [condition means that south-west processor exists]
				const int cx = sub_nx - gcx;
				const int cy = sub_ny - gcy;

				// memory allocation on demand //
				allocate_memory(&exch_xy, &exch_size_xy, sizeof(T)* cx * cy);
				T *msend = (T*)exch_xy;

				get_sub_array(in, sub_nx, sub_ny, gcx, sub_nx - 1, gcy, sub_ny - 1, msend);
				MPI_Send(msend, cx * cy, mpi_type< T >(), rank - size_x - 1, 0, comm);
			}
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::c_scatter_odd_grid(T* out, const T* in,
		const int nx, const int ny,
		const int sub_nx, const int sub_ny,
		const int gcx, const int gcy) const
	{
		double start_time = omp_get_wtime();

		if ((rank_x % 2 == 0) && (rank_y % 2 == 0)) {

			MPI_Request mpi_req[3];
			for (int k = 0; k < 3; k++)
				mpi_req[k] = MPI_REQUEST_NULL;

			// *in to current rank //
			//      in: [0 ... sub_nx - 1], [0 ... sub_ny - 1]
			// *in to east rank (includes bc) //
			//      in: [sub_nx - 2 * gcx ... nx - 1], [0 ... sub_ny - 1]
			// *in to north rank (includes bc) //
			//      in: [0 ... sub_nx - 1], [sub_ny - 2 * gcy ... ny - 1]
			// *in to north-east rank (includes bc) //
			//      in: [sub_nx - 2 * gcx ... nx - 1], [sub_ny - 2 * gcy ... ny - 1]

			T *msend[4];
			if (rank_x < size_x - 1) {
				// * east * //
				const int e_nx = nx - sub_nx + 2 * gcx;
				const int e_ny = sub_ny;
				const int e_size = e_nx * e_ny;

				// memory allocation on demand //
				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* e_size);
				msend[1] = (T*)exch_x;

				get_sub_array(in, nx, ny, sub_nx - 2 * gcx, nx - 1, 0, sub_ny - 1, msend[1]);
				MPI_Isend(msend[1], e_size, mpi_type< T >(), rank + 1,
					0, comm, &mpi_req[0]);
			}

			if (rank_y < size_y - 1) {
				// * north * //
				const int n_nx = sub_nx;
				const int n_ny = ny - sub_ny + 2 * gcy;
				const int n_size = n_nx * n_ny;

				// memory allocation on demand //
				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* n_size);
				msend[2] = (T*)exch_y;

				get_sub_array(in, nx, ny, 0, sub_nx - 1, sub_ny - 2 * gcy, ny - 1, msend[2]);
				MPI_Isend(msend[2], n_size, mpi_type< T >(), rank + size_x,
					0, comm, &mpi_req[1]);
			}

			if ((rank_x < size_x - 1) && (rank_y < size_y - 1)) {
				// * north east * //
				const int ne_nx = nx - sub_nx + 2 * gcx;
				const int ne_ny = ny - sub_ny + 2 * gcy;
				const int ne_size = ne_nx * ne_ny;

				// memory allocation on demand //
				allocate_memory(&exch_xy, &exch_size_xy, sizeof(T)* ne_size);
				msend[3] = (T*)exch_xy;

				get_sub_array(in, nx, ny, sub_nx - 2 * gcx, nx - 1, sub_ny - 2 * gcy, ny - 1, msend[3]);
				MPI_Isend(msend[3], ne_size, mpi_type< T >(), rank + size_x + 1,
					0, comm, &mpi_req[2]);
			}

			// out <-- in could fail as pointers could overlap
			allocate_memory(&exch, &exch_size, sizeof(T)* sub_nx * sub_ny);
			msend[0] = (T*)exch;

			get_sub_array(in, nx, ny, 0, sub_nx - 1, 0, sub_ny - 1, msend[0]);
			memcpy(out, msend[0], sizeof(T)* sub_nx * sub_ny);

			MPI_Waitall(3, mpi_req, MPI_STATUSES_IGNORE);
		}
		else
		{
			// recieving full array //

			if ((rank_x % 2 == 1) && (rank_y % 2 == 0)) {
				// recv from west [condition means that west processor exists]
				MPI_Recv(out, sub_nx * sub_ny, mpi_type< T >(), rank - 1, 0, comm, MPI_STATUS_IGNORE);
			}
			if ((rank_x % 2 == 0) && (rank_y % 2 == 1)) {
				// recv from south [condition means that south processor exists]
				MPI_Recv(out, sub_nx * sub_ny, mpi_type< T >(), rank - size_x, 0, comm, MPI_STATUS_IGNORE);
			}
			if ((rank_x % 2 == 1) && (rank_y % 2 == 1)) {
				// recv from south-west [condition means that south-west processor exists]
				MPI_Recv(out, sub_nx * sub_ny, mpi_type< T >(), rank - size_x - 1, 0, comm, MPI_STATUS_IGNORE);
			}
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::c_gather_odd_grid_x(T* out, const T* in,
		const int nx, const int sub_nx, const int gcx) const
	{
		double start_time = omp_get_wtime();

		if ((rank_x % 2 == 0) && (rank_y % 2 == 0)) {

			// *in: [0 ... sub_nx-1] [sub_nx-gcx ... nx-1]
			MPI_Request mpi_req = MPI_REQUEST_NULL;
			T* mrecv[2];

			if (rank_x < size_x - 1) {	// east //
				const int e_size = nx - sub_nx + gcx;

				// memory allocation on demand //
				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* e_size);
				mrecv[1] = (T*)exch_x;

				MPI_Irecv(mrecv[1], e_size, mpi_type< T >(), rank + 1,
					0, comm, &mpi_req);
			}

			// out <-- in could fail as pointers could overlap
			allocate_memory(&exch, &exch_size, sizeof(T)* sub_nx);
			mrecv[0] = (T*)exch;

			memcpy(mrecv[0], in, sizeof(T)* sub_nx);
			put_sub_array(out, nx, 1, 0, sub_nx - 1, 0, 0, mrecv[0]);

			MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

			if (rank_x < size_x - 1)
				put_sub_array(out, nx, 1, sub_nx - gcx, nx - 1, 0, 0, mrecv[1]);
		}
		else
		{
			if ((rank_x % 2 == 1) && (rank_y % 2 == 0)) {
				// send west [condition means that west processor exists]
				const int cx = sub_nx - gcx;

				// memory allocation on demand //
				allocate_memory(&exch_x, &exch_size_x, sizeof(T)* cx);
				T *msend = (T*)exch_x;

				get_sub_array(in, sub_nx, 1, gcx, sub_nx - 1, 0, 0, msend);
				MPI_Send(msend, cx, mpi_type< T >(), rank - 1, 0, comm);
			}
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_x += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::c_gather_odd_grid_y(T* out, const T* in,
		const int ny, const int sub_ny, const int gcy) const
	{
		double start_time = omp_get_wtime();

		if ((rank_x % 2 == 0) && (rank_y % 2 == 0)) {

			// *in: [0 ... sub_ny-1] [sub_ny-gcy ... ny-1]
			MPI_Request mpi_req = MPI_REQUEST_NULL;
			T* mrecv[2];

			if (rank_y < size_y - 1) {	// north //
				const int n_size = ny - sub_ny + gcy;
				
				// memory allocation on demand //
				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* n_size);
				mrecv[1] = (T*)exch_y;

				MPI_Irecv(mrecv[1], n_size, mpi_type< T >(), rank + size_x,
					0, comm, &mpi_req);
			}

			// out <-- in could fail as pointers could overlap
			allocate_memory(&exch, &exch_size, sizeof(T)* sub_ny);
			mrecv[0] = (T*)exch;

			memcpy(mrecv[0], in, sizeof(T)* sub_ny);
			put_sub_array(out, 1, ny, 0, 0, 0, sub_ny - 1, mrecv[0]);

			MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

			if (rank_y < size_y - 1)
				put_sub_array(out, 1, ny, 0, 0, sub_ny - gcy, ny - 1, mrecv[1]);
		}
		else
		{
			if ((rank_x % 2 == 0) && (rank_y % 2 == 1)) {
				// send south [condition means that south processor exists]
				const int cy = sub_ny - gcy;

				// memory allocation on demand //
				allocate_memory(&exch_y, &exch_size_y, sizeof(T)* cy);
				T *msend = (T*)exch_y;

				get_sub_array(in, 1, sub_ny, 0, 0, gcy, sub_ny - 1, msend);
				MPI_Send(msend, cy, mpi_type< T >(), rank - size_x, 0, comm);
			}
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::exchange_cross_halo(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y) const
	{
		// degenerate case //
		if ((size == 1) &&
			(!periodic_x) && (!periodic_y)) return;

		MPI_Request mpi_req[8];

		push_exchange_cross_halo(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_x, periodic_y, mpi_req);
		pop_exchange_cross_halo(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_x, periodic_y, mpi_req);
	}

	template< typename T >
	void mpiCom2d::exchange_halo(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y) const
	{
		// degenerate case //
		if ((size == 1) &&
			(!periodic_x) && (!periodic_y)) return;

		MPI_Request mpi_req[4];

		push_exchange_halo_x(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_x, mpi_req);
		init_exchange_halo_y(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_y);
		pop_exchange_halo_x(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_x, mpi_req);

		push_exchange_halo_y(x, 1, nx, ny, 
			gcx, gcy, hx, hy, periodic_y, mpi_req);
		pop_exchange_halo_y(x, nx, ny, 
			gcx, gcy, hx, hy, periodic_y, mpi_req);
	}

	template< typename T >
	void mpiCom2d::exchange_halo(
		T* u, T* v,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y) const
	{
		// degenerate case //
		if ((size == 1) &&
			(!periodic_x) && (!periodic_y)) return;

		MPI_Request mpi_req[8];

		// initialize -x[u], -y[v] //
		push_exchange_halo_x(u, nx, ny,
			gcx, gcy, hx, hy, periodic_x, mpi_req);
		push_exchange_halo_y(v, 0, nx, ny,
			gcx, gcy, hx, hy, periodic_y, &mpi_req[4]);

		// finalize -x[u], -y[v] //
		pop_exchange_halo_x(u, nx, ny,
			gcx, gcy, hx, hy, periodic_x, mpi_req);
		pop_exchange_halo_y(v, nx, ny,
			gcx, gcy, hx, hy, periodic_y, &mpi_req[4]);

		// initialize -y[u], -x[v] //
		push_exchange_halo_y(u, 0, nx, ny,
			gcx, gcy, hx, hy, periodic_y, mpi_req);
		push_exchange_halo_x(v, nx, ny,
			gcx, gcy, hx, hy, periodic_x, &mpi_req[4]);

		// finalize -y[u], -x[v]
		pop_exchange_halo_y(u, nx, ny,
			gcx, gcy, hx, hy, periodic_y, mpi_req);
		pop_exchange_halo_x(v, nx, ny,
			gcx, gcy, hx, hy, periodic_x, &mpi_req[4]);
	}

	template< typename T >
	void mpiCom2d::exchange_color_halo(
		T* x,
		const int color,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y) const
	{
		// degenerate case //
		if ((size == 1) &&
			(!periodic_x) && (!periodic_y)) return;

		MPI_Request mpi_req[4];

		push_exchange_color_halo_x(x, color, nx, ny, gcx, gcy,
			hx, hy, periodic_x, mpi_req);
		pop_exchange_color_halo_x(x, color, nx, ny, gcx, gcy,
			hx, hy, periodic_x, mpi_req);

		push_exchange_color_halo_y(x, color, nx, ny, gcx, gcy,
			hx, hy, periodic_y, mpi_req);
		pop_exchange_color_halo_y(x, color, nx, ny, gcx, gcy,
			hx, hy, periodic_y, mpi_req);
	}

	template< typename T >
	void mpiCom2d::exchange_halo_x(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x) const
	{
		// degenerate case //
		if ((size_x == 1) && (!periodic_x)) return;

		MPI_Request mpi_req[4];

		push_exchange_halo_x(x, nx, ny, gcx, gcy,
			hx, hy, periodic_x, mpi_req);
		pop_exchange_halo_x(x, nx, ny, gcx, gcy,
			hx, hy, periodic_x, mpi_req);
	}

	template< typename T >
	void mpiCom2d::exchange_halo_y(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y) const
	{
		// degenerate case //
		if ((size_y == 1) && (!periodic_y)) return;

		MPI_Request mpi_req[4];

		push_exchange_halo_y(x, 0, nx, ny, gcx, gcy,
			hx, hy, periodic_y, mpi_req);
		pop_exchange_halo_y(x, nx, ny, gcx, gcy,
			hx, hy, periodic_y, mpi_req);
	}

	template< typename T >
	void mpiCom2d::push_exchange_cross_halo(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y,
		MPI_Request mpi_req[8]) const
	{
		if ((size == 1) &&
			(!periodic_x) && (!periodic_y))	// degenerate case //
		{
			for (int k = 0; k < 8; k++)
				mpi_req[k] = MPI_REQUEST_NULL;
			return;
		}

		// initialize -x, -y cross exchanges //
		push_exchange_halo_x(x, nx, ny, 
			gcx, gcy, hx, 0, periodic_x, &mpi_req[0]);
		push_exchange_halo_y(x, 0, nx, ny, 
			gcx, gcy, 0, hy, periodic_y, &mpi_req[4]);
	}


	template< typename T >
	void mpiCom2d::pop_exchange_cross_halo(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x, const int periodic_y,
		MPI_Request mpi_req[8]) const
	{
		if (size == 1) return;	// degenerate case //

		// finalize -x, -y cross exchanges //
		pop_exchange_halo_x(x, nx, ny, 
			gcx, gcy, hx, 0, periodic_x, &mpi_req[0]);
		pop_exchange_halo_y(x, nx, ny, 
			gcx, gcy, 0, hy, periodic_y, &mpi_req[4]);
	}

	template< typename T >
	void mpiCom2d::push_exchange_halo_x(
		T* x,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x,

		MPI_Request mpi_req[4]) const
	{
		for (int k = 0; k < 4; k++)
			mpi_req[k] = MPI_REQUEST_NULL;

		if (size_x == 1) {	// degenerate case //

			if (periodic_x)	// apply periodicity withing local array
				apply_periodic_x(x, nx, ny, gcx, gcy, hx, hy);
			return;
		}

		double start_time = omp_get_wtime();

		// initialize -x MPI exchanges //
		// --------------------------- //
		const int msx = hx * (ny - ((gcy - hy) << 1));    // message size in -x communicator 

		T *(msend_x[2]), *(mrecv_x[2]);          // -x message pointers //
		for (int k = 0; k < 2; k++) {
			msend_x[k] = NULL; mrecv_x[k] = NULL;
		}

		// memory allocation on demand //
		allocate_memory(&exch_x, &exch_size_x, 4 * sizeof(T)* msx);

		// west halo exchange //
		if ((rank_x > 0) || ((rank_x == 0) && (periodic_x)))
		{
			int pidx = (rank_x > 0) ? rank - 1 : rank + (size_x - 1);

			mrecv_x[0] = (T*)exch_x;
			msend_x[0] = &((T*)exch_x)[(msx << 1)];

			MPI_Irecv(mrecv_x[0], msx, mpi_type< T >(), pidx, 1, comm, &mpi_req[0]);

			get_sub_array(x, nx, ny, gcx, gcx + hx - 1, gcy - hy, ny - gcy + hy - 1, msend_x[0]);
			MPI_Isend(msend_x[0], msx, mpi_type< T >(), pidx, 0, comm, &mpi_req[2]);
		}
		// east halo exchange //
		if ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (periodic_x)))
		{
			int pidx = (rank_x < size_x - 1) ? rank + 1 : rank - (size_x - 1);

			mrecv_x[1] = &((T*)exch_x)[msx];
			msend_x[1] = &((T*)exch_x)[(msx << 1) + msx];

			MPI_Irecv(mrecv_x[1], msx, mpi_type< T >(), pidx, 0, comm, &mpi_req[1]);

			get_sub_array(x, nx, ny, nx - gcx - hx, nx - gcx - 1, gcy - hy, ny - gcy + hy - 1, msend_x[1]);
			MPI_Isend(msend_x[1], msx, mpi_type< T >(), pidx, 1, comm, &mpi_req[3]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_x += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::pop_exchange_halo_x(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x,

		MPI_Request mpi_req[4]) const
	{
		if (size_x == 1) return;	// degenerate case //

		double start_time = omp_get_wtime();

		// finalize -x MPI exchanges //
		// ------------------------- //
		MPI_Waitall(4, mpi_req, MPI_STATUSES_IGNORE);

		const int msx = hx * (ny - ((gcy - hy) << 1));    // message size in -x communicator

		T *(mrecv_x[2]);	// -x message pointers //
		// west halo exchange //
		if ((rank_x > 0) || ((rank_x == 0) && (periodic_x))) {
			mrecv_x[0] = (T*)exch_x;
			put_sub_array(x, nx, ny, gcx - hx, gcx - 1, gcy - hy, ny - gcy + hy - 1, mrecv_x[0]);
		}
		// east halo exchange //
		if ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (periodic_x))) {
			mrecv_x[1] = &((T*)exch_x)[msx];
			put_sub_array(x, nx, ny, nx - gcx, nx - gcx + hx - 1, gcy - hy, ny - gcy + hy - 1, mrecv_x[1]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_x += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::init_exchange_halo_y(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y) const
	{
		if (size_y == 1) return;	// degenerate case //

		double start_time = omp_get_wtime();

		// pre-initialize -y MPI exchanges //
		// ------------------------------- //
		const int msy = hy * (nx - ((gcx - hx) << 1));    // message size in -y communicator 

		T *(msend_y[2]);                 // -y message pointers //
		for (int k = 0; k < 2; k++) {
			msend_y[k] = NULL;
		}

		// memory allocation on demand //
		allocate_memory(&exch_y, &exch_size_y, 4 * sizeof(T)* msy);

		// south halo exchange //
		if ((rank_y > 0) || ((rank_y == 0) && (periodic_y)))
		{
			msend_y[0] = &((T*)exch_y)[(msy << 1)];
			get_sub_array(x, nx, ny, gcx, nx - gcx - 1, gcy, gcy + hy - 1, &msend_y[0][hx * hy]);
		}
		// north halo exchange //
		if ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (periodic_y)))
		{
			msend_y[1] = &((T*)exch_y)[(msy << 1) + msy];
			get_sub_array(x, nx, ny, gcx, nx - gcx - 1, ny - gcy - hy, ny - gcy - 1, &msend_y[1][hx * hy]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::push_exchange_halo_y(
		T* x,

		const int init_flag,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y,

		MPI_Request mpi_req[4]) const
	{
		for (int k = 0; k < 4; k++)
			mpi_req[k] = MPI_REQUEST_NULL;

		if (size_y == 1) {	// degenerate case //

			if (periodic_y)	// apply periodicity withing local array
				apply_periodic_y(x, nx, ny, gcx, gcy, hx, hy);
			return;
		}

		double start_time = omp_get_wtime();

		// initialize -y MPI exchanges //
		// --------------------------- //
		const int msy = hy * (nx - ((gcx - hx) << 1));    // message size in -y communicator 

		T *(msend_y[2]), *(mrecv_y[2]);          // -y message pointers //
		for (int k = 0; k < 2; k++) {
			msend_y[k] = NULL; mrecv_y[k] = NULL;
		}

		// memory allocation on demand //
		allocate_memory(&exch_y, &exch_size_y, 4 * sizeof(T)* msy);

		// south halo exchange //
		if ((rank_y > 0) || ((rank_y == 0) && (periodic_y)))
		{
			int pidx = (rank_y > 0) ? rank - size_x : rank + (size_y - 1) * size_x;

			mrecv_y[0] = (T*)exch_y;
			msend_y[0] = &((T*)exch_y)[(msy << 1)];

			MPI_Irecv(mrecv_y[0], msy, mpi_type< T >(), pidx, 1, comm, &mpi_req[0]);

			if (!init_flag)
				get_sub_array(x, nx, ny, gcx - hx, nx - gcx + hx - 1, gcy, gcy + hy - 1, msend_y[0]);
			else
			{
				get_sub_array(x, nx, ny, gcx - hx, gcx - 1, gcy, gcy + hy - 1,
					&msend_y[0][0]);
				get_sub_array(x, nx, ny, nx - gcx, nx - gcx + hx - 1, gcy, gcy + hy - 1,
					&msend_y[0][(nx - (gcx << 1) + hx) * hy]);
			}

			MPI_Isend(msend_y[0], msy, mpi_type< T >(), pidx, 0, comm, &mpi_req[2]);
		}
		// north halo exchange //
		if ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (periodic_y)))
		{
			int pidx = (rank_y < size_y - 1) ? rank + size_x : rank - (size_y - 1) * size_x;

			mrecv_y[1] = &((T*)exch_y)[msy];
			msend_y[1] = &((T*)exch_y)[(msy << 1) + msy];

			MPI_Irecv(mrecv_y[1], msy, mpi_type< T >(), pidx, 0, comm, &mpi_req[1]);

			if (!init_flag)
				get_sub_array(x, nx, ny, gcx - hx, nx - gcx + hx - 1, ny - gcy - hy, ny - gcy - 1, msend_y[1]);
			else
			{
				get_sub_array(x, nx, ny, gcx - hx, gcx - 1, ny - gcy - hy, ny - gcy - 1,
					&msend_y[1][0]);
				get_sub_array(x, nx, ny, nx - gcx, nx - gcx + hx - 1, ny - gcy - hy, ny - gcy - 1,
					&msend_y[1][(nx - (gcx << 1) + hx) * hy]);
			}

			MPI_Isend(msend_y[1], msy, mpi_type< T >(), pidx, 1, comm, &mpi_req[3]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::pop_exchange_halo_y(
		T* x,
		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y,

		MPI_Request mpi_req[4]) const
	{
		if (size_y == 1) return;	// degenerate case //

		double start_time = omp_get_wtime();

		// finalize -y MPI exchanges //
		// ------------------------- //
		MPI_Waitall(4, mpi_req, MPI_STATUSES_IGNORE);

		const int msy = hy * (nx - ((gcx - hx) << 1));    // message size in -y communicator 

		T *(mrecv_y[2]);	// -y message pointers //		
		// south halo exchange //
		if ((rank_y > 0) || ((rank_y == 0) && (periodic_y))) {
			mrecv_y[0] = (T*)exch_y;
			put_sub_array(x, nx, ny, gcx - hx, nx - gcx + hx - 1, gcy - hy, gcy - 1, mrecv_y[0]);
		}
		// north halo exchange //
		if ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (periodic_y))) {
			mrecv_y[1] = &((T*)exch_y)[msy];
			put_sub_array(x, nx, ny, gcx - hx, nx - gcx + hx - 1, ny - gcy, ny - gcy + hy - 1, mrecv_y[1]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::push_exchange_color_halo_x(
		T* x,
		const int color,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x,

		MPI_Request mpi_req[4]) const
	{
		for (int k = 0; k < 4; k++)
			mpi_req[k] = MPI_REQUEST_NULL;

		if (size_x == 1) {	// degenerate case //

			if (periodic_x)	// apply -x periodicity within local array
				apply_periodic_x(x, color, nx, ny, gcx, gcy, hx, hy);
			return;
		}

		double start_time = omp_get_wtime();

		// initialize -x MPI exchanges //
		// --------------------------- //

		T *(msend_x[2]), *(mrecv_x[2]);          // -x message pointers //
		for (int k = 0; k < 2; k++) {
			msend_x[k] = NULL; mrecv_x[k] = NULL;
		}

		// message size in -x communicator
		const int msx_send_west = get_num_colored(color, nx, ny,
			gcx, gcx + hx - 1, gcy - hy, ny - gcy + hy - 1);
		const int msx_send_east = get_num_colored(color, nx, ny,
			nx - gcx - hx, nx - gcx - 1, gcy - hy, ny - gcy + hy - 1);
		const int msx_recv_west = get_num_colored(color, nx, ny,
			gcx - hx, gcx - 1, gcy - hy, ny - gcy + hy - 1);
		const int msx_recv_east = get_num_colored(color, nx, ny,
			nx - gcx, nx - gcx + hx - 1, gcy - hy, ny - gcy + hy - 1);

		// send-receive message sizes in -x communicator
		const int msx_recv = msx_recv_west + msx_recv_east;
		const int msx_send = msx_send_west + msx_send_east;

		// memory allocation on demand //
		allocate_memory(&exch_x, &exch_size_x, sizeof(T)* (msx_recv + msx_send));

		// west halo exchange //
		if ((rank_x > 0) || ((rank_x == 0) && (periodic_x)))
		{
			int pidx = (rank_x > 0) ? rank - 1 : rank + (size_x - 1);

			mrecv_x[0] = (T*)exch_x;
			msend_x[0] = &((T*)exch_x)[msx_recv];

			MPI_Irecv(mrecv_x[0], msx_recv_west, mpi_type< T >(), pidx, 1, comm, &mpi_req[0]);

			get_sub_array(x, color, nx, ny, gcx, gcx + hx - 1, gcy - hy, ny - gcy + hy - 1, msend_x[0]);
			MPI_Isend(msend_x[0], msx_send_west, mpi_type< T >(), pidx, 0, comm, &mpi_req[2]);
		}
		// east halo exchange //
		if ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (periodic_x)))
		{
			int pidx = (rank_x < size_x - 1) ? rank + 1 : rank - (size_x - 1);

			mrecv_x[1] = &((T*)exch_x)[msx_recv_west];
			msend_x[1] = &((T*)exch_x)[msx_recv + msx_send_west];

			MPI_Irecv(mrecv_x[1], msx_recv_east, mpi_type< T >(), pidx, 0, comm, &mpi_req[1]);

			get_sub_array(x, color, nx, ny, nx - gcx - hx, nx - gcx - 1, gcy - hy, ny - gcy + hy - 1, msend_x[1]);
			MPI_Isend(msend_x[1], msx_send_east, mpi_type< T >(), pidx, 1, comm, &mpi_req[3]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_x += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::pop_exchange_color_halo_x(
		T* x,
		const int color,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_x,

		MPI_Request mpi_req[4]) const
	{
		if (size_x == 1) return;	// degenerate case //

		double start_time = omp_get_wtime();

		// message size in -x communicator
		const int msx_recv_west = get_num_colored(color, nx, ny,
			gcx - hx, gcx - 1, gcy - hy, ny - gcy + hy - 1);
		const int msx_recv_east = get_num_colored(color, nx, ny,
			nx - gcx, nx - gcx + hx - 1, gcy - hy, ny - gcy + hy - 1);

		// receive message size in -x communicator
		const int msx_recv = msx_recv_west + msx_recv_east;

		// finalize -x MPI exchanges //
		// ------------------------- //
		MPI_Waitall(4, mpi_req, MPI_STATUSES_IGNORE);

		T *(mrecv_x[2]);	// -x message pointers //
		// west halo exchange //
		if ((rank_x > 0) || ((rank_x == 0) && (periodic_x))) {
			mrecv_x[0] = (T*)exch_x;
			put_sub_array(x, color, nx, ny, gcx - hx, gcx - 1, gcy - hy, ny - gcy + hy - 1, mrecv_x[0]);
		}
		// east halo exchange //
		if ((rank_x < size_x - 1) || ((rank_x == size_x - 1) && (periodic_x))) {
			mrecv_x[1] = &((T*)exch_x)[msx_recv_west];
			put_sub_array(x, color, nx, ny, nx - gcx, nx - gcx + hx - 1, gcy - hy, ny - gcy + hy - 1, mrecv_x[1]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_x += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::push_exchange_color_halo_y(
		T* x,
		const int color,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y,

		MPI_Request mpi_req[4]) const
	{
		for (int k = 0; k < 4; k++)
			mpi_req[k] = MPI_REQUEST_NULL;

		if (size_y == 1) {	// degenerate case //

			if (periodic_y)	// apply -y periodicity within local array
				apply_periodic_y(x, color, nx, ny, gcx, gcy, hx, hy);
			return;
		}

		double start_time = omp_get_wtime();

		// initialize -y MPI exchanges //
		// --------------------------- //

		T *(msend_y[2]), *(mrecv_y[2]);          // -y message pointers //
		for (int k = 0; k < 2; k++) {
			msend_y[k] = NULL; mrecv_y[k] = NULL;
		}

		// message size in -y communicator
		const int msy_send_south = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, gcy, gcy + hy - 1);
		const int msy_send_north = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, ny - gcy - hy, ny - gcy - 1);
		const int msy_recv_south = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, gcy - hy, gcy - 1);
		const int msy_recv_north = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, ny - gcy, ny - gcy + hy - 1);

		// send-recieve message sizes in -y communicator
		const int msy_recv = msy_recv_south + msy_recv_north;
		const int msy_send = msy_send_south + msy_send_north;

		// memory allocation on demand //
		allocate_memory(&exch_y, &exch_size_y, sizeof(T)* (msy_recv + msy_send));

		// south halo exchange //
		if ((rank_y > 0) || ((rank_y == 0) && (periodic_y)))
		{
			int pidx = (rank_y > 0) ? rank - size_x : rank + (size_y - 1) * size_x;

			mrecv_y[0] = (T*)exch_y;
			msend_y[0] = &((T*)exch_y)[msy_recv];

			MPI_Irecv(mrecv_y[0], msy_recv_south, mpi_type< T >(), pidx, 1, comm, &mpi_req[0]);

			get_sub_array(x, color, nx, ny, gcx - hx, nx - gcx + hx - 1, gcy, gcy + hy - 1, msend_y[0]);
			MPI_Isend(msend_y[0], msy_send_south, mpi_type< T >(), pidx, 0, comm, &mpi_req[2]);
		}
		// north halo exchange //
		if ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (periodic_y)))
		{
			int pidx = (rank_y < size_y - 1) ? rank + size_x : rank - (size_y - 1) * size_x;

			mrecv_y[1] = &((T*)exch_y)[msy_recv_south];
			msend_y[1] = &((T*)exch_y)[msy_recv + msy_send_south];

			MPI_Irecv(mrecv_y[1], msy_recv_north, mpi_type< T >(), pidx, 0, comm, &mpi_req[1]);

			get_sub_array(x, color, nx, ny, gcx - hx, nx - gcx + hx - 1, ny - gcy - hy, ny - gcy - 1, msend_y[1]);
			MPI_Isend(msend_y[1], msy_send_north, mpi_type< T >(), pidx, 1, comm, &mpi_req[3]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	template< typename T >
	void mpiCom2d::pop_exchange_color_halo_y(
		T* x,
		const int color,

		const int nx, const int ny,
		const int gcx, const int gcy,
		const int hx, const int hy,
		const int periodic_y,

		MPI_Request mpi_req[4]) const
	{
		if (size_y == 1) return;	// degenerate case //

		double start_time = omp_get_wtime();

		// message size in -y communicator
		const int msy_recv_south = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, gcy - hy, gcy - 1);
		const int msy_recv_north = get_num_colored(color, nx, ny,
			gcx - hx, nx - gcx + hx - 1, ny - gcy, ny - gcy + hy - 1);

		// recieve message size in -y communicator
		const int msy_recv = msy_recv_south + msy_recv_north;

		// finalize -y MPI exchanges //
		// ------------------------- //
		MPI_Waitall(4, mpi_req, MPI_STATUSES_IGNORE);

		T *(mrecv_y[2]);	// -y message pointers //
		// south halo exchange //
		if ((rank_y > 0) || ((rank_y == 0) && (periodic_y))) {
			mrecv_y[0] = (T*)exch_y;
			put_sub_array(x, color, nx, ny, gcx - hx, nx - gcx + hx - 1, gcy - hy, gcy - 1, mrecv_y[0]);
		}
		// north halo exchange //
		if ((rank_y < size_y - 1) || ((rank_y == size_y - 1) && (periodic_y))) {
			mrecv_y[1] = &((T*)exch_y)[msy_recv_south];
			put_sub_array(x, color, nx, ny, gcx - hx, nx - gcx + hx - 1, ny - gcy, ny - gcy + hy - 1, mrecv_y[1]);
		}

		double end_time = omp_get_wtime();
		cpu_time_exch += end_time - start_time;
		cpu_time_exch_y += end_time - start_time;
	}

	inline int mpiCom2d::test_exchange(MPI_Request* mpi_req, const int n_req) const
	{
		int flag;
		MPI_Testall(n_req, mpi_req, &flag, MPI_STATUSES_IGNORE);

		return flag;
	}
	// ================================================================================== //

	inline int mpiCom2d::offset_x(const int nx, const int gcx) const
	{
		int cx = nx - 2 * gcx, in_cx = cx;
		int mpi_cx = mpi_allreduce_comm(cx, MPI_SUM, comm) / size_y;

		int offset;
		MPI_Scan(&in_cx, &offset, 1, MPI_INT, MPI_SUM, comm);

		return offset - mpi_cx * rank_y - cx;
	}

	inline int mpiCom2d::offset_y(const int ny, const int gcy) const
	{
		int cy = ny - 2 * gcy, in_cy = (rank_x == 0) ? cy : 0;

		int offset;
		MPI_Scan(&in_cy, &offset, 1, MPI_INT, MPI_SUM, comm);

		return offset - cy;
	}

	// * Private * //
	// ---------------------------------------------------------------------------------- //
	inline void mpiCom2d::allocate_memory(
		void** memory, size_t* memory_size, const size_t size) const
	{
		if ((*memory_size) < size) {
			if ((*memory_size) > 0) free((*memory));

			(*memory_size) = size;
			(*memory) = malloc((*memory_size));
		}
	}
}
