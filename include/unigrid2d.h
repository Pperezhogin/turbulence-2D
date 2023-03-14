#pragma once

// *    2D uniform grid  * //

#include "vecmath.h"
#include "mpi-com2d.h"
#include "grid-common2d.h"

// *[unigrid2d.h]: full //

namespace nse
{
	// * 2D uniform grid: unigrid2d< T > [ T = float, double ] * //
	// =======================================================================
	template< typename T = double >
	class uniGrid2d
	{
	public:
		uniGrid2d();
		uniGrid2d(const uniGrid2d& grid);
		~uniGrid2d();

		bool set(const T _x, const T _y, const T _length, const T _width,
			const int _cx, const int _cy, const int _gcx, const int _gcy,

			const int mpi_ndim);

		bool set(const T _x, const T _y, const T _length, const T _width,
			const int _cx, const int _cy, const int _gcx, const int _gcy,

			const int mpi_size_x, const int mpi_size_y);

		T get_x() const { return x; }
		T get_y() const { return y; }
		T get_length() const { return length; }
		T get_width() const { return width; }

		int get_nx() const { return nx; }
		int get_ny() const { return ny; }
		int get_size() const { return size; }

		T get_mpi_x() const { return mpi_x; }
		T get_mpi_y() const { return mpi_y; }
		T get_mpi_length() const { return mpi_length; }
		T get_mpi_width() const { return mpi_width; }

		int get_mpi_nx() const { return mpi_nx; }
		int get_mpi_ny() const { return mpi_ny; }
		int get_mpi_size() const { return mpi_size; }

		// MPI global cell index [== -1 - on failure]
		int mpi_locate_x(const T x) const;
		int mpi_locate_y(const T y) const;

		// local cell index [== -1 - on failure]
		//  - only one MPI process returns >=0 for (x,y) calls
		int locate_x(const T x) const;
		int locate_y(const T y) const;

		// locate cell index [== -1 on failure]
		// - include MPI & domain ghost cells [non unique MPI call] >=0
		int locate_inc_x(const T x) const;
		int locate_inc_y(const T y) const;

		// (i,j) MPI-local coordinates [input - global index]
		//	- only one MPI process returns >= 0
		int i_local_coord(const int i) const;
		int j_local_coord(const int j) const;

		// interpolation
		T c_interp(const T* X, const T x, const T y) const;
		T u_interp(const T* U, const T x, const T y) const;
		T v_interp(const T* V, const T x, const T y) const;

		// profiling (C,U,V) -> C node array
		// -x profile
		void c_profile_at_y(T* Px, const T* X, const T y) const;
		void u_profile_at_y(T* Px, const T* U, const T y) const;
		void v_profile_at_y(T* Px, const T* V, const T y) const;
		// -y profile
		void c_profile_at_x(T* Py, const T* X, const T x) const;
		void u_profile_at_x(T* Py, const T* U, const T x) const;
		void v_profile_at_x(T* Py, const T* V, const T x) const;

		// averaging (C,U,V) -> C node array
		// -y average [ X(nx,ny)->Px(nx) ]
		void c_average_by_y(T* Px, const T* X) const;
		void u_average_by_y(T* Px, const T* U) const;
		void v_average_by_y(T* Px, const T* V) const;
		// -x average [ X(nx,ny)->Py(ny) ]
		void c_average_by_x(T* Py, const T* X) const;
		void u_average_by_x(T* Py, const T* U) const;
		void v_average_by_x(T* Py, const T* V) const;
		// -xy average [ X(nx,ny)-> return value ]
		T c_average(const T* X) const;
		T u_average(const T* U) const;
		T v_average(const T* V) const;

		// precise averaging (C,U,V) -> (C,U,V) node array
		// note*: functions dont perform any sub averaging e.g.: (Xi + Xi+1)/2
		// note*: averaging for U,V is okey if we assume that
		//			boundary conditions are periodic or inflow is set
		//			at (west,south) part of boundaries
		// -y average [ X(nx,ny)->Px(nx) ]
		void average_by_y(T* Px, const T* X) const;	
		// -x average [ X(nx,ny)->Py(ny) ]
		void average_by_x(T* Py, const T* X) const;
		// -xy average [ X(nx,ny)-> return value ]
		T average(const T* X) const;

		// grid re-interpolation out(current grid), in(input grid)
		void c_grid_reinterp(T* Xout, const T* Xin, const uniGrid2d< T >& grid) const;
		void u_grid_reinterp(T* Uout, const T* Uin, const uniGrid2d< T >& grid) const;
		void v_grid_reinterp(T* Vout, const T* Vin, const uniGrid2d< T >& grid) const;

	private:

		// main grid parameters init routine, mpi communicator assumed initialized //
		bool init_grid(const T _x, const T _y, const T _length, const T _width,
			const int _cx, const int _cy, const int _gcx, const int _gcy);

		// additional memory for profile manipulation size = max( nx, ny ) //
		T *profile_memory;
		int profile_memory_size;

	public:
		mpiCom2d mpi_com;

		T x, y;
		T length, width;

		T mpi_x, mpi_y;
		T mpi_length, mpi_width;

		T dx, dy, dxh, dyh;
		T dxi, dyi, dxih, dyih, dxiq, dyiq, dxiqh, dyiqh;
		T dx2i, dy2i, dx2ih, dy2ih, dx2iq, dy2iq;
		T dx2id, dy2id;

		T dxdy;

		int nx, ny, size;
		int mpi_nx, mpi_ny, mpi_size;

		int gcx, gcy;
	};
}

namespace nse
{
	template< typename T >
	uniGrid2d< T > ::uniGrid2d(
		) : profile_memory_size(0),

		mpi_com(),
		x((T)0), y((T)0),
		mpi_x((T)0), mpi_y((T)0),

		length((T)0), width((T)0),
		mpi_length((T)0), mpi_width((T)0),

		nx(0), ny(0), size(0),
		mpi_nx(0), mpi_ny(0), mpi_size(0),

		gcx(0), gcy(0)
	{
	}

	template< typename T >
	uniGrid2d< T > ::uniGrid2d(
		const uniGrid2d< T >& grid)
		: profile_memory_size(0),

		mpi_com(grid.mpi_com),
		x(grid.x), y(grid.y),
		mpi_x(grid.mpi_x), y(grid.mpi_y),

		length(grid.length), width(grid.width),
		mpi_length(grid.mpi_length), mpi_width(grid.mpi_width),

		nx(grid.nx), ny(grid.ny), size(grid.size),
		mpi_nx(grid.mpi_nx), mpi_ny(grid.mpi_ny), mpi_size(grid.mpi_size),

		gcx(grid.gcx), gcy(grid.gcy),

		dx(grid.dx), dy(grid.dy),
		dxh(grid.dxh), dyh(grid.dyh),
		dxi(grid.dxi), dyi(grid.dyi),
		dxih(grid.dxih), dyih(grid.dyih),
		dxiq(grid.dxiq), dyiq(grid.dyiq),
		dxiq(grid.dxiqh), dyiq(grid.dyiqh),
		dx2i(grid.dx2i), dy2i(grid.dy2i),
		dx2ih(grid.dx2ih), dy2ih(grid.dy2ih),
		dx2iq(grid.dx2iq), dy2iq(grid.dy2iq),
		dx2id(grid.dx2id), dy2id(grid.dy2id),

		dxdy(grid.dxdy)
	{
		if (size > 0) {
			profile_memory_size = (nx > ny) ? nx : ny;
			allocate(&profile_memory, profile_memory_size);
		}
	}

	template< typename T >
	uniGrid2d< T > :: ~uniGrid2d(
		)
	{
		if (profile_memory_size > 0)
			deallocate(profile_memory);
		
	}

	template< typename T >
	bool uniGrid2d< T > ::set(
		const T _x, const T _y, const T _length, const T _width,
		const int _cx, const int _cy, const int _gcx, const int _gcy,
		const int mpi_ndim)
	{
		mpi_com.set(mpi_ndim);

		return init_grid(_x, _y, _length, _width,
			_cx, _cy, _gcx, _gcy);
	}

	template< typename T >
	bool uniGrid2d< T > ::set(
		const T _x, const T _y, const T _length, const T _width,
		const int _cx, const int _cy, const int _gcx, const int _gcy,
		const int mpi_size_x, const int mpi_size_y)
	{
		mpi_com.set(mpi_size_x, mpi_size_y);

		return init_grid(_x, _y, _length, _width,
			_cx, _cy, _gcx, _gcy);
	}

	template< typename T >
	int uniGrid2d< T > ::mpi_locate_x(const T _x) const
	{
		int i;
		T px = mpi_x;
		for (i = gcx; i < mpi_nx - gcx; i++, px += dx)
		if ((_x >= px) && (_x <= px + dx)) { return i; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::mpi_locate_y(const T _y) const
	{
		int j;
		T py = mpi_y;
		for (j = gcy; j < mpi_ny - gcy; j++, py += dy)
		if ((_y >= py) && (_y <= py + dy)) { return j; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::locate_x(const T _x) const
	{
		int i;
		T px = x;

		if (mpi_com.rank_x == 0) {
			for (i = gcx; i < nx - gcx; i++, px += dx)
			if ((_x >= px) && (_x <= px + dx)) { return i; }
		}
		else
		for (i = gcx; i < nx - gcx; i++, px += dx)
		if ((_x > px) && (_x <= px + dx)) { return i; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::locate_y(const T _y) const
	{
		int j;
		T py = y;

		if (mpi_com.rank_y == 0) {
			for (j = gcy; j < ny - gcy; j++, py += dy)
			if ((_y >= py) && (_y <= py + dy)) { return j; }
		}
		else
		for (j = gcy; j < ny - gcy; j++, py += dy)
		if ((_y > py) && (_y <= py + dy)) { return j; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::locate_inc_x(const T _x) const
	{
		int i;
		T px = x - gcx * dx;

		for (i = 0; i < nx; i++, px += dx)
		if ((_x >= px) && (_x <= px + dx)) { return i; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::locate_inc_y(const T _y) const
	{
		int j;
		T py = y - gcy * dy;

		for (j = 0; j < ny; j++, py += dy)
		if ((_y >= py) && (_y <= py + dy)) { return j; }

		return -1;
	}

	template< typename T >
	int uniGrid2d< T > ::i_local_coord(const int i) const
	{
		if ((i < 0) & (i > mpi_nx - 1)) return -1;

		const int
			bx = (mpi_com.rank_x == 0) ? gcx : 0,
			ex = (mpi_com.rank_x == mpi_com.size_x - 1) ? gcx : 0;

		int mpi_cx = mpi_nx - 2 * gcx;
		int ip, ish = 0;
		for (int m = 0; m < mpi_com.rank_x; m++)
		{
			ish += mpi_cx / mpi_com.size_x;
			if (m < mpi_cx % mpi_com.size_x) ish++;
		}

		ip = i - ish;
		return ((ip >= gcx - bx) && (ip < nx - gcx + ex)) ? ip : -1;
	}

	template< typename T >
	int uniGrid2d< T > ::j_local_coord(const int j) const
	{
		if ((j < 0) && (j > mpi_ny - 1)) return -1;

		const int
			by = (mpi_com.rank_y == 0) ? gcy : 0,
			ey = (mpi_com.rank_y == mpi_com.size_y - 1) ? gcy : 0;

		int mpi_cy = mpi_ny - 2 * gcy;
		int jp, jsh = 0;
		for (int m = 0; m < mpi_com.rank_y; m++)
		{
			jsh += mpi_cy / mpi_com.size_y;
			if (m < mpi_cy % mpi_com.size_y) jsh++;
		}

		jp = j - jsh;
		return ((jp >= gcy - by) && (jp < ny - gcy + ey)) ? jp : -1;
	}

	template< typename T >
	T uniGrid2d< T > ::c_interp(const T* X, const T _px, const T _py) const
	{

		int i = locate_x(_px),
			j = locate_y(_py);
		int index = i * ny + j;

		T p_value = (T) 0.0;

		if ((i >= 0) && (j >= 0)) {

			T _cx = x + (i - gcx) * dx + dxh,
				_cy = y + (j - gcy) * dy + dyh;

			if (_px < _cx) {
				if (_py < _cy)
					p_value = interp_bilinear(
					_px, _py, _cx - dx, _cy - dy, dx, dy,
					X[index - ny - 1], X[index - 1],
					X[index - ny], X[index]);
				else
					p_value = interp_bilinear(
					_px, _py, _cx - dx, _cy, dx, dy,
					X[index - ny], X[index],
					X[index - ny + 1], X[index + 1]);
			}
			else
			{
				if (_py < _cy)
					p_value = interp_bilinear(
					_px, _py, _cx, _cy - dy, dx, dy,
					X[index - 1], X[index + ny - 1],
					X[index], X[index + ny]);
				else
					p_value = interp_bilinear(
					_px, _py, _cx, _cy, dx, dy,
					X[index], X[index + ny],
					X[index + 1], X[index + ny + 1]);
			}
		}

		return mpi_allreduce_comm(p_value, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	T uniGrid2d< T > ::u_interp(const T* U, const T _px, const T _py) const
	{
		int i = locate_x(_px),
			j = locate_y(_py);
		int index = i * ny + j;

		T p_value = (T) 0.0;

		if ((i >= 0) && (j >= 0)) {

			T _cx = x + (i - gcx) * dx + dxh,
				_cy = y + (j - gcy) * dy + dyh;

			if (_py > _cy)
				p_value = interp_bilinear(
				_px, _py, _cx - dxh, _cy, dx, dy,
				U[index], U[index + ny],
				U[index + 1], U[index + ny + 1]);

			else
				p_value = interp_bilinear(
				_px, _py, _cx - dxh, _cy - dy, dx, dy,
				U[index - 1], U[index + ny - 1],
				U[index], U[index + ny]);
		}

		return mpi_allreduce_comm(p_value, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	T uniGrid2d< T > ::v_interp(const T* V, const T _px, const T _py) const
	{
		int i = locate_x(_px),
			j = locate_y(_py);
		int index = i * ny + j;

		T p_value = (T) 0.0;

		if ((i >= 0) && (j >= 0)) {

			T _cx = x + (i - gcx) * dx + dxh,
				_cy = y + (j - gcy) * dy + dyh;

			if (_px > _cx)
				p_value = interp_bilinear(
				_px, _py, _cx, _cy - dyh, dx, dy,
				V[index], V[index + ny],
				V[index + 1], V[index + ny + 1]);
			else
				p_value = interp_bilinear(
				_px, _py, _cx - dx, _cy - dyh, dx, dy,
				V[index - ny], V[index],
				V[index - ny + 1], V[index + 1]);
		}

		return mpi_allreduce_comm(p_value, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	void uniGrid2d< T > ::c_profile_at_y(T* Px, const T* X, const T _py) const
	{
		int i, j = locate_y(_py);
		int index;

		T *mpi_Px = profile_memory;
		null(mpi_Px, nx);

		if ((j >= gcy) && (j < ny - gcy)) {

			T _cy = y + (j - gcy) * dy + dyh;

			// bilinear interpolation == linear interpolation ( _px == _cx )
			if (_py < _cy)  {

				T alpha = (_py - (_cy - dy)) * dyi;

#pragma omp parallel for private( i, index ) shared( mpi_Px, X, alpha, j )
				for (i = gcx; i < nx - gcx; i++) {
					index = i * ny + j;
					mpi_Px[i] = alpha * (X[index] - X[index - 1]) + X[index - 1];
				}
			}
			else
			{
				T alpha = (_py - _cy) * dyi;

#pragma omp parallel for private( i, index ) shared( mpi_Px, X, alpha, j )
				for (i = gcx; i < nx - gcx; i++) {
					index = i * ny + j;
					mpi_Px[i] = alpha * (X[index + 1] - X[index]) + X[index];
				}
			}
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::u_profile_at_y(T* Px, const T* U, const T _py) const
	{
		int i, j = locate_y(_py);
		int index;

		T *mpi_Px = profile_memory;
		null(mpi_Px, nx);

		if ((j >= gcy) && (j < ny - gcy)) {

			T _cy = y + (j - gcy) * dy + dyh;

			// bilinear interpolation == linear interpolation ( _px == _cx )
			if (_py < _cy) {

				T alpha = (_py - (_cy - dy)) * dyi;
				T Cij, Cijm;

#pragma omp parallel for private( i, index, Cij, Cijm ) shared( mpi_Px, U, alpha, j )
				for (i = gcx; i < nx - gcx; i++) {
					index = i * ny + j;
					Cij = (T) 0.5 * (U[index] + U[index + ny]);
					Cijm = (T) 0.5 * (U[index - 1] + U[index + ny - 1]);

					mpi_Px[i] = alpha * (Cij - Cijm) + Cijm;
				}
			}
			else
			{
				T alpha = (_py - _cy) * dyi;
				T Cij, Cijp;

#pragma omp parallel for private( i, index, Cij, Cijp ) shared( mpi_Px, U, alpha, j )
				for (i = gcx; i < nx - gcx; i++) {
					index = i * ny + j;
					Cij = (T) 0.5 * (U[index] + U[index + ny]);
					Cijp = (T) 0.5 * (U[index + 1] + U[index + ny + 1]);

					mpi_Px[i] = alpha * (Cijp - Cij) + Cij;
				}
			}
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::v_profile_at_y(T* Px, const T* V, const T _py) const
	{
		int i, j = locate_y(_py);
		int index;

		T *mpi_Px = profile_memory;
		null(mpi_Px, nx);

		if ((j >= gcy) && (j < ny - gcy)) {

			T _cy = y + (j - gcy) * dy + dyh;
			T alpha = (_py - (_cy - dyh)) * dyi;

#pragma omp parallel for private( i, index ) shared( mpi_Px, V, alpha, j )
			for (i = gcx; i < nx - gcx; i++) {
				index = i * ny + j;
				mpi_Px[i] = alpha * (V[index + 1] - V[index]) + V[index];
			}
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::c_profile_at_x(T* Py, const T* X, const T _px) const
	{
		int i = locate_x(_px), j;
		int index;

		T *mpi_Py = profile_memory;
		null(mpi_Py, ny);

		if ((i >= gcx) && (i < nx - gcx)) {

			T _cx = x + (i - gcx) * dx + dxh;

			// bilinear interpolation == linear interpolation ( _py == _cy )
			if (_px < _cx) {

				T alpha = (_px - (_cx - dx)) * dxi;

#pragma omp parallel for private( j, index ) shared( mpi_Py, X, alpha, i )
				for (j = gcy; j < ny - gcy; j++) {
					index = i * ny + j;
					mpi_Py[j] = alpha * (X[index] - X[index - ny]) + X[index - ny];
				}
			}
			else
			{
				T alpha = (_px - _cx) * dxi;

#pragma omp parallel for private( j, index ) shared( mpi_Py, X, alpha, i )
				for (j = gcy; j < ny - gcy; j++) {
					index = i * ny + j;
					mpi_Py[j] = alpha * (X[index + ny] - X[index]) + X[index];
				}
			}
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	void uniGrid2d< T > ::u_profile_at_x(T* Py, const T* U, const T _px) const
	{
		int i = locate_x(_px), j;
		int index;

		T *mpi_Py = profile_memory;
		null(mpi_Py, ny);

		if ((i >= gcx) && (i < nx - gcx)) {

			T _cx = x + (i - gcx) * dx + dxh;
			T alpha = (_px - (_cx - dxh)) * dxi;

#pragma omp parallel for private( j, index ) shared( mpi_Py, U, alpha, i )
			for (j = gcy; j < ny - gcy; j++) {
				index = i * ny + j;
				mpi_Py[j] = alpha * (U[index + ny] - U[index]) + U[index];
			}
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	void uniGrid2d< T > ::v_profile_at_x(T* Py, const T* V, const T _px) const
	{
		int i = locate_x(_px), j;
		int index;

		T *mpi_Py = profile_memory;
		null(mpi_Py, ny);

		if ((i >= gcx) && (i < nx - gcx)) {

			T _cx = x + (i - gcx) * dx + dxh;

			// bilinear interpolation == linear interpolation ( _py == _cy )
			if (_px < _cx) {

				T alpha = (_px - (_cx - dx)) * dxi;
				T Cij, Cimj;

#pragma omp parallel for private( j, index, Cij, Cimj ) shared( mpi_Py, V, alpha, i )
				for (j = gcy; j < ny - gcy; j++) {
					index = i * ny + j;
					Cij = (T) 0.5 * (V[index] + V[index + 1]);
					Cimj = (T) 0.5 * (V[index - ny] + V[index - ny + 1]);

					mpi_Py[j] = alpha * (Cij - Cimj) + Cimj;
				}
			}
			else
			{
				T alpha = (_px - _cx) * dxi;
				T Cij, Cipj;

#pragma omp parallel for private( j, index, Cij, Cipj ) shared( mpi_Py, V, alpha, i )
				for (j = gcy; j < ny - gcy; j++) {
					index = i * ny + j;
					Cij = (T) 0.5 * (V[index] + V[index + 1]);
					Cipj = (T) 0.5 * (V[index + ny] + V[index + ny + 1]);

					mpi_Py[j] = alpha * (Cipj - Cij) + Cij;
				}
			}
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	void uniGrid2d< T > ::c_average_by_y(T* Px, const T* X) const
	{
		const T i_div_y = (T) 1.0 / (mpi_ny - 2 * gcy);
		int i, j, index;

		T *mpi_Px = profile_memory;
		T px_value;

#pragma omp parallel for private( i, j, index, px_value ) shared( mpi_Px, X )
		for (i = gcx; i < nx - gcx; i++) {

			px_value = (T)0;
			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++) {
				px_value += X[index];
			}

			mpi_Px[i] = px_value * i_div_y;
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::u_average_by_y(T* Px, const T* U) const
	{
		const T i_div_y = (T) 0.5 / (mpi_ny - 2 * gcy);
		int i, j, index;

		T *mpi_Px = profile_memory;
		T px_value;

#pragma omp parallel for private( i, j, index, px_value ) shared( mpi_Px, U )
		for (i = gcx; i < nx - gcx; i++) {

			px_value = (T)0;
			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++) {
				px_value += (U[index] + U[index + ny]);
			}

			mpi_Px[i] = px_value * i_div_y;
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::v_average_by_y(T* Px, const T* V) const
	{
		const T i_div_y = (T) 0.5 / (mpi_ny - 2 * gcy);
		int i, j, index;

		T *mpi_Px = profile_memory;
		T px_value;

#pragma omp parallel for private( i, j, index, px_value ) shared( mpi_Px, V )
		for (i = gcx; i < nx - gcx; i++) {

			px_value = (T)0;
			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++) {
				px_value += (V[index] + V[index + 1]);
			}

			mpi_Px[i] = px_value * i_div_y;
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::c_average_by_x(T* Py, const T* X) const
	{
		const T i_div_x = (T) 1.0 / (mpi_nx - 2 * gcx);
		int i, j, index;

		T *mpi_Py = profile_memory;
		T py_value;

#pragma omp parallel for private( i, j, index, py_value ) shared( mpi_Py, X )
		for (j = gcy; j < ny - gcy; j++) {

			py_value = (T)0;
			index = gcx * ny + j;
			for (i = gcx; i < nx - gcx; i++, index += ny) {
				py_value += X[index];
			}

			mpi_Py[j] = py_value * i_div_x;
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	void uniGrid2d< T > ::u_average_by_x(T* Py, const T* U) const
	{
		const T i_div_x = (T) 0.5 / (mpi_nx - 2 * gcx);
		int i, j, index;

		T *mpi_Py = profile_memory;
		T py_value;

#pragma omp parallel for private( i, j, index, py_value ) shared( mpi_Py, U )
		for (j = gcy; j < ny - gcy; j++) {

			py_value = (T)0;
			index = gcx * ny + j;
			for (i = gcx; i < nx - gcx; i++, index += ny) {
				py_value += (U[index] + U[index + ny]);
			}

			mpi_Py[j] = py_value * i_div_x;
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	void uniGrid2d< T > ::v_average_by_x(T* Py, const T* V) const
	{
		const T i_div_x = (T) 0.5 / (mpi_nx - 2 * gcx);
		int i, j, index;

		T *mpi_Py = profile_memory;
		T py_value;

#pragma omp parallel for private( i, j, index, py_value ) shared( mpi_Py, V )
		for (j = gcy; j < ny - gcy; j++) {

			py_value = (T)0;
			index = gcx * ny + j;
			for (i = gcx; i < nx - gcx; i++, index += ny) {
				py_value += (V[index] + V[index + 1]);
			}

			mpi_Py[j] = py_value * i_div_x;
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	T uniGrid2d< T > ::c_average(const T* X) const
	{
		const T i_div_xy = (T) 1.0 / ((mpi_nx - 2 * gcx) * (mpi_ny - 2 * gcy));
		int i, j, index;
		
		T avg = (T)0;
#pragma omp parallel for private( i, j, index ) shared( X ) reduction( + : avg )
		for (i = gcx; i < nx - gcx; i++) {

			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++)
				avg += X[index];
		}

		avg *= i_div_xy;
		return mpi_allreduce_comm(avg, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	T uniGrid2d< T > ::u_average(const T* U) const
	{
		const T i_div_xy = (T) 0.5 / ((mpi_nx - 2 * gcx) * (mpi_ny - 2 * gcy));
		int i, j, index;
		
		T avg = (T)0;
#pragma omp parallel for private( i, j, index ) shared( U ) reduction( + : avg )
		for (i = gcx; i < nx - gcx; i++) {

			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++)
				avg += (U[index] + U[index + ny]);
		}

		avg *= i_div_xy;
		return mpi_allreduce_comm(avg, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	T uniGrid2d< T > ::v_average(const T* V) const
	{
		const T i_div_xy = (T) 0.5 / ((mpi_nx - 2 * gcx) * (mpi_ny - 2 * gcy));
		int i, j, index;

		T avg = (T)0;
#pragma omp parallel for private( i, j, index ) shared( V ) reduction( + : avg )
		for (i = gcx; i < nx - gcx; i++) {

			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++)
				avg += (V[index] + V[index + 1]);
		}

		avg *= i_div_xy;
		return mpi_allreduce_comm(avg, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	void uniGrid2d< T > ::average_by_y(T* Px, const T* X) const
	{
		const T i_div_y = (T) 1.0 / (mpi_ny - 2 * gcy);
		int i, j, index;

		T *mpi_Px = profile_memory;
		T px_value;

#pragma omp parallel for private( i, j, index, px_value ) shared( mpi_Px, X )
		for (i = gcx; i < nx - gcx; i++) {

			px_value = (T)0;
			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++) {
				px_value += X[index];
			}

			mpi_Px[i] = px_value * i_div_y;
		}

		mpi_allreduce_comm(mpi_Px, Px, nx, MPI_SUM, mpi_com.comm_y);
	}

	template< typename T >
	void uniGrid2d< T > ::average_by_x(T* Py, const T* X) const
	{
		const T i_div_x = (T) 1.0 / (mpi_nx - 2 * gcx);
		int i, j, index, jb, je, jsh;
		int omp_id, omp_num;

		T *mpi_Py = profile_memory;

#pragma omp parallel private( i, j, index, omp_id, omp_num, jb, je, jsh ) shared( mpi_Py, X )
		{
			omp_num = omp_get_num_threads();
			omp_id = omp_get_thread_num();

			jsh = (ny - 2 * gcy) / omp_num;

			jb = gcy + jsh * omp_id;
			je = (omp_id < omp_num - 1) ? jb + jsh
				: jb + jsh + ((ny - 2 * gcy) % omp_num);

			for (j = jb; j < je; j++)
				mpi_Py[j] = (T)0;

			for (i = gcx; i < nx - gcx; i++)
			{
				index = i * ny + jb;
				for (j = jb; j < je; j++, index++)
					mpi_Py[j] += X[index];
			}

			for (j = jb; j < je; j++)
				mpi_Py[j] *= i_div_x;
		}

		mpi_allreduce_comm(mpi_Py, Py, ny, MPI_SUM, mpi_com.comm_x);
	}

	template< typename T >
	T uniGrid2d< T > ::average(const T* X) const
	{
		const T i_div_xy = (T) 1.0 / ((mpi_nx - 2 * gcx) * (mpi_ny - 2 * gcy));
		int i, j, index;

		T avg = (T)0;
#pragma omp parallel for private( i, j, index ) shared( X ) reduction( + : avg )
		for (i = gcx; i < nx - gcx; i++) {

			index = i * ny + gcy;
			for (j = gcy; j < ny - gcy; j++, index++)
				avg += X[index];
		}

		avg *= i_div_xy;
		return mpi_allreduce_comm(avg, MPI_SUM, mpi_com.comm);
	}

	template< typename T >
	void uniGrid2d< T > ::c_grid_reinterp(T* Xout, const T* Xin, const uniGrid2d< T >& grid) const
	{
		int i, j;
		int ip, jp;
		T _px, _py, value;

		for (i = gcx; i < mpi_nx - gcx; i++)
		for (j = gcy; j < mpi_ny - gcy; j++)
		{
			// define local (i,j) coordinates //
			ip = i_local_coord(i);
			jp = j_local_coord(j);

			// define global (x,y) coordinates //
			_px = mpi_x + (i - gcx) * dx + dxh;
			_py = mpi_y + (j - gcy) * dy + dyh;

			// interpolation on input grid //
			value = grid.c_interp(Xin, _px, _py);

			if ((ip >= 0) && (jp >= 0))
				Xout[ip * ny + jp] = value;
		}
	}

	template< typename T >
	void uniGrid2d< T > ::u_grid_reinterp(T* Uout, const T* Uin, const uniGrid2d< T >& grid) const
	{
		int i, j;
		int ip, jp;
		T _px, _py, value;

		for (i = gcx; i < mpi_nx - gcx; i++)
		for (j = gcy; j < mpi_ny - gcy; j++)
		{
			// define local (i,j) coordinates //
			ip = i_local_coord(i);
			jp = j_local_coord(j);

			// define global (x,y) coordinates //
			_px = mpi_x + (i - gcx) * dx;
			_py = mpi_y + (j - gcy) * dy + dyh;

			// interpolation on input grid //
			value = grid.u_interp(Uin, _px, _py);

			if ((ip >= 0) && (jp >= 0))
				Uout[ip * ny + jp] = value;
		}
	}

	template< typename T >
	void uniGrid2d< T > ::v_grid_reinterp(T* Vout, const T* Vin, const uniGrid2d< T >& grid) const
	{
		int i, j;
		int ip, jp;
		T _px, _py, value;

		for (i = gcx; i < mpi_nx - gcx; i++)
		for (j = gcy; j < mpi_ny - gcy; j++)
		{
			// define local (i,j) coordinates //
			ip = i_local_coord(i);
			jp = j_local_coord(j);

			// define global (x,y) coordinates //
			_px = mpi_x + (i - gcx) * dx + dxh;
			_py = mpi_y + (j - gcy) * dy;

			// interpolation on input grid //
			value = grid.v_interp(Vin, _px, _py);

			if ((ip >= 0) && (jp >= 0))
				Vout[ip * ny + jp] = value;
		}
	}

	template< typename T >
	bool uniGrid2d< T > ::init_grid(
		const T _x, const T _y, const T _length, const T _width,
		const int _cx, const int _cy, const int _gcx, const int _gcy)
	{
		dx = _length / _cx; dy = _width / _cy;

		dxh = (T) 0.5 * dx; dyh = (T) 0.5 * dy;
		dxi = _cx / _length; dyi = _cy / _width;
		dxih = (T) 0.5 * dxi; dyih = (T) 0.5 * dyi;
		dxiq = (T) 0.25 * dxi; dyiq = (T) 0.25 * dyi;
		dxiqh = (T) 0.125 * dxi; dyiqh = (T) 0.125 * dyi;
		dx2i = dxi * dxi; dy2i = dyi * dyi;
		dx2ih = (T) 0.5 * dx2i; dy2ih = (T) 0.5 * dy2i;
		dx2iq = (T) 0.25 * dx2i; dy2iq = (T) 0.25 * dy2i;
		dx2id = (T) 2.0 * dx2i; dy2id = (T) 2.0 * dy2i;

		dxdy = dx * dy;

		nx = _cx / mpi_com.size_x;
		if (mpi_com.rank_x < _cx % mpi_com.size_x) nx++;
		gcx = _gcx;

		ny = _cy / mpi_com.size_y;
		if (mpi_com.rank_y < _cy % mpi_com.size_y) ny++;
		gcy = _gcy;

		int px = _cx / mpi_com.size_x, py = _cy / mpi_com.size_y;

		x = _x;
		for (int k = 0; k < mpi_com.rank_x; k++) {
			x += px * dx;
			if (k < _cx % mpi_com.size_x) x += dx;
		}

		y = _y;
		for (int k = 0; k < mpi_com.rank_y; k++) {
			y += py * dy;
			if (k < _cy % mpi_com.size_y) y += dy;
		}

		length = nx * dx;
		width = ny * dy;

		nx += 2 * gcx; ny += 2 * gcy; size = nx * ny;

		// MPI domain parameters //
		mpi_x = mpi_allreduce_comm(x, MPI_MIN, mpi_com.comm);
		mpi_y = mpi_allreduce_comm(y, MPI_MIN, mpi_com.comm);

		mpi_length = mpi_allreduce_comm(length, MPI_SUM, mpi_com.comm) / mpi_com.size_y;
		mpi_width = mpi_allreduce_comm(width, MPI_SUM, mpi_com.comm) / mpi_com.size_x;

		// MPI grid parameters //
		mpi_nx = mpi_allreduce_comm(nx - 2 * gcx, MPI_SUM, mpi_com.comm);
		mpi_ny = mpi_allreduce_comm(ny - 2 * gcy, MPI_SUM, mpi_com.comm);

		mpi_nx = (mpi_nx / mpi_com.size_y) + 2 * gcx;
		mpi_ny = (mpi_ny / mpi_com.size_x) + 2 * gcy;

		mpi_size = mpi_nx * mpi_ny;

		// allocate memory for profile manipulation //
		profile_memory_size = (nx > ny) ? nx : ny;
		allocate(&profile_memory, profile_memory_size);

		return true;
	}
}
