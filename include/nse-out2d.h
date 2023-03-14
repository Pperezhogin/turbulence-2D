#pragma once

#include "unigrid2d.h"
#include "fourier-methods.h"
#include <string>

// *[nse-out2d.h]: full //

namespace nse
{
	// * write tecplot output * //
	template< typename T >
	bool write_tecplot(const char* filename,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename,
		T* U, T* V, const char* u_name, const char* v_name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename, const int idx,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename, const int idx,
		T* U, T* V, const char* u_name, const char* v_name,
		const uniGrid2d< T >& grid, const T sol_time);
	// -------------------------------------------------------------------- //

	// * write tecplot output [sub-domain] * //
	template< typename T >
	bool write_tecplot(const char* filename,
		T* X, const char* name,
		const T xmin, const T xmax,
		const T ymin, const T ymax,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename,
		T* U, T* V, const char* u_name, const char* v_name,
		const T xmin, const T xmax,
		const T ymin, const T ymax,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename, const int idx,
		T* X, const char* name,
		const T xmin, const T xmax,
		const T ymin, const T ymax,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot(const char* filename, const int idx,
		T* U, T* V, const char* u_name, const char* v_name,
		const T xmin, const T xmax,
		const T ymin, const T ymax,
		const uniGrid2d< T >& grid, const T sol_time);
	// -------------------------------------------------------------------- //

	// * write tecplot f(x) * //
	template< typename T >
	bool write_tecplot_x(const char* filename,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_y(const char* filename,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_x(const char* filename, const int idx,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_y(const char* filename, const int idx,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);
	// -------------------------------------------------------------------- //

	// * write tecplot f(x) with normalization * //
	template< typename T >
	bool write_tecplot_x(const char* filename,
		T* X, const char* name,
		const T value, const T length,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_y(const char* filename,
		T* X, const char* name,
		const T value, const T length,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_x(const char* filename, const int idx,
		T* X, const char* name,
		const T value, const T length,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_tecplot_y(const char* filename, const int idx,
		T* X, const char* name,
		const T value, const T length,
		const uniGrid2d< T >& grid, const T sol_time);
	// -------------------------------------------------------------------- //

	// * write binary output * //
    template< typename T >
	bool write_binary_przgn(const char* filename,
	T* Xin, const uniGrid2d< T >& grid, const int idx);

	template< typename T >
	bool write_binary_przgn_filter(std::string filename,
	T* Xin, const int N_coarse,	const uniGrid2d< T >& grid, const int idx);

	template< typename T >
	bool read_binary_przgn(const char* filename,
	T* Xout, const uniGrid2d< T >& grid);

	template< typename T >
	bool read_binary_przgn(const char* filename,
	T* Xout, const uniGrid2d< T >& grid, const int idx);

	template< typename T >
	bool read_series(const char* filename, const int length,
	T* tout, T* Xout, const uniGrid2d< T >& grid);
    
    template< typename T >
	bool write_binary_stamp(const char* filename,
		const double* cpu_stamp, const int cpu_stamp_size,
		const uniGrid2d< T >& grid, const T sol_time, const int sol_idx);
	
	template< typename T >
	bool write_binary(const char* filename,
	T* U, T* V, const uniGrid2d< T >& grid, const int idx);

	template< typename T >
	bool write_binary(const char* filename,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_binary(const char* filename,
		T* U, T* V, const char* u_name, const char* v_name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_binary_stamp(const char* filename, const int idx,
		const double* cpu_stamp, const int cpu_stamp_size,
		const uniGrid2d< T >& grid, const T sol_time, const int sol_idx);

	template< typename T >
	bool write_binary(const char* filename, const int idx,
		T* X, const char* name,
		const uniGrid2d< T >& grid, const T sol_time);

	template< typename T >
	bool write_binary(const char* filename, const int idx,
		T* U, T* V, const char* u_name, const char* v_name,
		const uniGrid2d< T >& grid, const T sol_time);
	// -------------------------------------------------------------------- //

	// * read binary input * //
	template< typename T >
	bool read_binary_stamp(const char* filename,
		double** cpu_stamp, int* cpu_stamp_size,
		const uniGrid2d< T >& grid, T* sol_time, int* sol_idx);

	template< typename T >
	bool read_binary(const char* filename,
		T* X, 
		const uniGrid2d< T >& grid);

	template< typename T >
	bool read_binary(const char* filename,
		T* U, T* V, 
		const uniGrid2d< T >& grid);

	template< typename T >
	bool read_binary_stamp(const char* filename, const int idx,
		double** cpu_stamp, int* cpu_stamp_size,
		const uniGrid2d< T >& grid, T* sol_time, int* sol_idx);

	template< typename T >
	bool read_binary(const char* filename, const int idx,
		T* X,
		const uniGrid2d< T >& grid);

	template< typename T >
	bool read_binary(const char* filename, const int idx,
		T* U, T* V,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //
}
