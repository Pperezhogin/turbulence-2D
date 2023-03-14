#define _CRT_SECURE_NO_DEPRECATE
#include "nse-out2d.h"
#include "str-com.h"

#include <stdio.h>

// * write output * //
template< typename T >
bool nse::write_tecplot(
	const char* filename,
	T* Xin,
	const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];

	grid.mpi_com.gather(X, Xin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {

			int i, j, idx;
			T px, py = grid.mpi_y + grid.dyh;
			for (j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, py += grid.dy) {
				px = grid.mpi_x + grid.dxh;
				for (i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++, px += grid.dx) {
					idx = i * grid.mpi_ny + j;

					fprintf(ptr, "%f\n", X[idx]);
				}
			}

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot(
	const char* filename,
	T* Uin, T* Vin,
	const char* u_name, const char* v_name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *U, *V;
	if (grid.mpi_com.rank == 0) {
		U = new T[grid.mpi_size];
		V = new T[grid.mpi_size];
	}

	grid.mpi_com.gather(U, Uin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	grid.mpi_com.gather(V, Vin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {

			int i, j, idx;
			T px, py = grid.mpi_y + grid.dyh;
			for (j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, py += grid.dy) {
				px = grid.mpi_x + grid.dxh;
				for (i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++, px += grid.dx) {
					idx = i * grid.mpi_ny + j;

					fprintf(ptr, "%f %f\n",
						(T) U[idx],
						(T) V[idx]);
				}
			}

			fclose(ptr);
			status = 1;
		}

		delete[] U; delete[] V;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot(n_filename,
		Xin, name, grid, sol_time);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_tecplot(
	const char* filename, const int idx,
	T* Uin, T* Vin,
	const char* u_name, const char* v_name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot(n_filename,
		Uin, Vin, u_name, v_name, grid, sol_time);

	delete[] n_filename;
	return status;
}

// * write output * //
template< typename T >
bool nse::write_tecplot(
	const char* filename,
	T* Xin,
	const char* name,
	const T xmin, const T xmax,
	const T ymin, const T ymax,
	const uniGrid2d< T >& grid, const T sol_time)
{
	int imin = grid.mpi_locate_x(xmin), imax = grid.mpi_locate_x(xmax),
		jmin = grid.mpi_locate_y(ymin), jmax = grid.mpi_locate_y(ymax);

	if ((imin == -1) && (xmin <= grid.mpi_x)) imin = grid.gcx;
	if ((jmin == -1) && (ymin <= grid.mpi_y)) jmin = grid.gcy;
	if ((imax == -1) && (xmax >= grid.mpi_x + grid.mpi_length)) 
		imax = grid.mpi_nx - grid.gcx - 1;
	if ((jmax == -1) && (ymax >= grid.mpi_y + grid.mpi_width)) 
		jmax = grid.mpi_ny - grid.gcy - 1;

	if ((imin == -1) || (imax == -1) ||
		(jmin == -1) || (jmax == -1) ||
		(imin > imax) || (jmin > jmax)) {
		return false;
	}

	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];

	grid.mpi_com.gather(X, Xin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {

			int i, j, idx;
			T px, py = grid.mpi_y + (jmin - grid.gcy) * grid.dy + grid.dyh;
			for (j = jmin; j <= jmax; j++, py += grid.dy) {
				if (j != jmin)
					fprintf(ptr, "\n");
				px = grid.mpi_x + (imin - grid.gcx) * grid.dx + grid.dxh;
				for (i = imin; i <= imax; i++, px += grid.dx) {
					idx = i * grid.mpi_ny + j;

					fprintf(ptr, "%f %f %f\n", px, py, X[idx]);
				}
			}

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot(
	const char* filename,
	T* Uin, T* Vin,
	const char* u_name, const char* v_name,
	const T xmin, const T xmax,
	const T ymin, const T ymax,
	const uniGrid2d< T >& grid, const T sol_time)
{
	int imin = grid.mpi_locate_x(xmin), imax = grid.mpi_locate_x(xmax),
		jmin = grid.mpi_locate_y(ymin), jmax = grid.mpi_locate_y(ymax);

	if ((imin == -1) && (xmin <= grid.mpi_x)) imin = grid.gcx;
	if ((jmin == -1) && (ymin <= grid.mpi_y)) jmin = grid.gcy;
	if ((imax == -1) && (xmax >= grid.mpi_x + grid.mpi_length)) 
		imax = grid.mpi_nx - grid.gcx - 1;
	if ((jmax == -1) && (ymax >= grid.mpi_y + grid.mpi_width)) 
		jmax = grid.mpi_ny - grid.gcy - 1;

	if ((imin == -1) || (imax == -1) ||
		(jmin == -1) || (jmax == -1) ||
		(imin > imax) || (jmin > jmax)) {
		return false;
	}

	T *U, *V;
	if (grid.mpi_com.rank == 0) {
		U = new T[grid.mpi_size];
		V = new T[grid.mpi_size];
	}

	grid.mpi_com.gather(U, Uin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	grid.mpi_com.gather(V, Vin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {

			int i, j, idx;
			T px, py = grid.mpi_y + (jmin - grid.gcy) * grid.dy + grid.dyh;
			for (j = jmin; j <= jmax; j++, py += grid.dy) {
				px = grid.mpi_x + (imin - grid.gcx) * grid.dx + grid.dxh;
				for (i = imin; i <= imax; i++, px += grid.dx) {
					idx = i * grid.mpi_ny + j;

					fprintf(ptr, "%f %f\n",
						(T) 0.5 * (U[idx] + U[idx + grid.mpi_ny]),
						(T) 0.5 * (V[idx] + V[idx + 1]));
				}
			}

			fclose(ptr);
			status = 1;
		}

		delete[] U; delete[] V;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const T xmin, const T xmax,
	const T ymin, const T ymax,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot(n_filename,
		Xin, name,
		xmin, xmax,
		ymin, ymax,
		grid, sol_time);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_tecplot(
	const char* filename, const int idx,
	T* Uin, T* Vin, const char* u_name, const char* v_name,
	const T xmin, const T xmax,
	const T ymin, const T ymax,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot(n_filename,
		Uin, Vin, u_name, v_name,
		xmin, xmax,
		ymin, ymax,
		grid, sol_time);

	delete[] n_filename;
	return status;
}
// -------------------------------------------------------------------- //

// * write tecplot f(x) * //
template< typename T >
bool nse::write_tecplot_x(
	const char* filename,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_nx];

	grid.mpi_com.gather_x(X, Xin, 0, grid.nx, grid.gcx);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {
			fprintf(ptr, " TITLE = \"nse-2D [F(x)]\"\n");
			fprintf(ptr, " VARIABLES = \"X\", \"%s\"\n", name);
			fprintf(ptr, " ZONE I = %i, DATAPACKING = POINT, SOLUTIONTIME = %f\n",
				grid.mpi_nx - 2 * grid.gcx, sol_time);

			T px = grid.mpi_x + grid.dxh;
			for (int i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++, px += grid.dx)
				fprintf(ptr, "%f %f\n", px, X[i]);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot_y(
	const char* filename,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_ny];

	grid.mpi_com.gather_y(X, Xin, 0, grid.ny, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {
			fprintf(ptr, " TITLE = \"nse-2D [F(y)]\"\n");
			fprintf(ptr, " VARIABLES = \"Y\", \"%s\"\n", name);
			fprintf(ptr, " ZONE I = %i, DATAPACKING = POINT, SOLUTIONTIME = %f\n",
				grid.mpi_ny - 2 * grid.gcy, sol_time);

			T py = grid.mpi_y + grid.dyh;
			for (int j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, py += grid.dy)
				fprintf(ptr, "%f %f\n", py, X[j]);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot_x(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot_x(n_filename,
		Xin, name, grid, sol_time);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_tecplot_y(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot_y(n_filename,
		Xin, name, grid, sol_time);

	delete[] n_filename;
	return status;
}
// -------------------------------------------------------------------- //

// * write tecplot f(x) with normalization * //
template< typename T >
bool nse::write_tecplot_x(
	const char* filename,
	T* Xin, const char* name,
	const T value, const T length,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_nx];

	grid.mpi_com.gather_x(X, Xin, 0, grid.nx, grid.gcx);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {
			fprintf(ptr, " TITLE = \"nse-2D [F(x)]\"\n");
			fprintf(ptr, " VARIABLES = \"X\", \"%s\"\n", name);
			fprintf(ptr, " ZONE I = %i, DATAPACKING = POINT, SOLUTIONTIME = %f\n",
				grid.mpi_nx - 2 * grid.gcx, sol_time);

			const T i_value = (T) 1.0 / value;
			const T i_length = (T) 1.0 / length;

			T px = grid.mpi_x + grid.dxh;
			for (int i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++, px += grid.dx)
				fprintf(ptr, "%f %f\n", px * i_length, X[i] * i_value);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot_y(
	const char* filename,
	T* Xin, const char* name,
	const T value, const T length,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_ny];

	grid.mpi_com.gather_y(X, Xin, 0, grid.ny, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "w");

		if (ptr != NULL) {
			fprintf(ptr, " TITLE = \"nse-2D [F(y)]\"\n");
			fprintf(ptr, " VARIABLES = \"Y\", \"%s\"\n", name);
			fprintf(ptr, " ZONE I = %i, DATAPACKING = POINT, SOLUTIONTIME = %f\n",
				grid.mpi_ny - 2 * grid.gcy, sol_time);

			const T i_value = (T) 1.0 / value;
			const T i_length = (T) 1.0 / length;

			T py = grid.mpi_y + grid.dyh;
			for (int j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, py += grid.dy)
				fprintf(ptr, "%f %f\n", py * i_length, X[j] * i_value);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_tecplot_x(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const T value, const T length,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot_x(n_filename,
		Xin, name, value, length, grid, sol_time);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_tecplot_y(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const T value, const T length,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_tecplot_y(n_filename,
		Xin, name, value, length, grid, sol_time);

	delete[] n_filename;
	return status;
}
// ------------------------------------------------------------------------ //

// * write binary output * //
template< typename T >
bool nse::write_binary_przgn(
	const char* filename,
	T* Xin,
	const uniGrid2d< T >& grid, const int idx)
{
	T *X;
    
    char* n_filename = append_index(filename, idx);
	
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];
	
	grid.mpi_com.gather(X, Xin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(n_filename, "wb");

		if (ptr != NULL) {
		  
			// main data //
			fwrite(X, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_binary_przgn_filter(
	std::string filename,
	T* Xin, const int N_coarse,
	const uniGrid2d< T >& grid, const int idx)
{
	T *X;
	T *Xc;
    
    char* n_filename = append_index(filename.c_str(), idx);
	
	nse::spectral_filter(Xin, Xin, N_coarse, grid);
	
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];
	if (grid.mpi_com.rank == 0) Xc = new T[N_coarse*N_coarse];
	
	grid.mpi_com.gather(X, Xin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		coarse_resolution(Xc, X, N_coarse, grid);

		FILE* ptr = fopen(n_filename, "wb");

		if (ptr != NULL) {
		  
			// main data //
			fwrite(Xc, sizeof(T), N_coarse * N_coarse, ptr);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
		delete[] Xc;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::read_binary_przgn(
	const char* filename,
	T* Xout,
	const uniGrid2d< T >& grid)
{
	T *X;
	
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "rb");

		if (ptr != NULL) {
		  
			// main data //
			fread(X, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

			fclose(ptr);
			status = 1;
		}
	}

	grid.mpi_com.scatter(Xout, X, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	if (grid.mpi_com.rank == 0) delete[] X;
	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::read_binary_przgn(
	const char* filename,
	T* Xout,
	const uniGrid2d< T >& grid,  const int idx)
{
	T *X;
	char* n_filename = append_index(filename, idx);
	
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(n_filename, "rb");

		if (ptr != NULL) {

			printf("I am reading\n");
		  
			// main data //
			fread(X, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

			printf("reading done\n");

			fclose(ptr);
			status = 1;
		}
	}

	grid.mpi_com.scatter(Xout, X, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	if (grid.mpi_com.rank == 0) delete[] X;
	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::read_series(
	const char* filename, const int length,
	T* tout, T* Xout,
	const uniGrid2d< T >& grid)
{
	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "rb");

		if (ptr != NULL) {
		  
			// main data //
			fread(tout, sizeof(T), length, ptr);
			fread(Xout, sizeof(T), length, ptr);

			fclose(ptr);
			status = 1;
		}
	}

	MPI_Bcast(tout, length, mpi_type< T >(), 0, MPI_COMM_WORLD);
	MPI_Bcast(Xout, length, mpi_type< T >(), 0, MPI_COMM_WORLD);
	
	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}


template< typename T >
bool nse::write_binary_stamp(
	const char* filename,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< T >& grid, const T sol_time, const int sol_idx)
{
	if (cpu_stamp_size <= 0) return false;

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "wb");

		if (ptr != NULL) {

			// header data //
			int header[4];
			header[0] = 'n' + 's' + 'e';      // file identifier
			header[1] = 0;                    // uniform grid data flag
			header[2] = 2;                    // dimensions number
			header[3] = sizeof(T);			  // data type size

			fwrite(header, sizeof(int), 4, ptr);

			// domain definition //
			T domain_id[4];
			domain_id[0] = grid.mpi_x;
			domain_id[1] = grid.mpi_y;
			domain_id[2] = grid.mpi_length;
			domain_id[3] = grid.mpi_width;

			fwrite(domain_id, sizeof(T), 4, ptr);

			// grid definition //
			int grid_id[4];
			grid_id[0] = grid.mpi_nx;
			grid_id[1] = grid.mpi_ny;
			grid_id[2] = grid.gcx;
			grid_id[3] = grid.gcy;

			fwrite(grid_id, sizeof(int), 4, ptr);

			// time stamp //
			T out_time = sol_time;
			int out_idx = sol_idx;

			fwrite(&out_time, sizeof(T), 1, ptr);
			fwrite(&out_idx, sizeof(int), 1, ptr);

			// cpu stamp //
			int out_cpu_stamp_size = cpu_stamp_size;
			double *out_cpu_stamp = new double[out_cpu_stamp_size];
			memcpy(out_cpu_stamp, cpu_stamp,
				out_cpu_stamp_size * sizeof(double));

			fwrite(&out_cpu_stamp_size, sizeof(int), 1, ptr);
			fwrite(out_cpu_stamp, sizeof(double),
				out_cpu_stamp_size, ptr);

			delete[] out_cpu_stamp;

			fclose(ptr);
			status = 1;
		}
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_binary(
	const char* filename,
	T* Uin, T* Vin, const uniGrid2d< T >& grid, const int idx)
{
	T *_U;
	T *_V;
	if (grid.mpi_com.rank == 0) _U = new T[grid.mpi_size];
	if (grid.mpi_com.rank == 0) _V = new T[grid.mpi_size];

	
	grid.mpi_com.gather(_U, Uin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	grid.mpi_com.gather(_V, Vin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	
	float *U;
	float *V;
	
	if (grid.mpi_com.rank == 0) {
	  U = new float[grid.mpi_size];
	  V = new float[grid.mpi_size];
	  for ( int i = 0; i < grid.mpi_size; i++ ) {
	      U[i] = float( _U[i] );
	      V[i] = float( _V[i] );
	  }
	}

	
	char* n_filename = app_index(filename, idx);
	
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(n_filename, "wb");
		if (ptr != NULL) {
			fwrite(U, sizeof(float), grid.mpi_nx * grid.mpi_ny, ptr);
			fwrite(V, sizeof(float), grid.mpi_nx * grid.mpi_ny, ptr);
			fclose(ptr);
		}

	}

	return (true);
}

template< typename T >
bool nse::write_binary(
	const char* filename,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *X;
	if (grid.mpi_com.rank == 0) X = new T[grid.mpi_size];

	grid.mpi_com.gather(X, Xin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "wb");

		if (ptr != NULL) {

			// header data //
			int header[4];
			header[0] = 'n' + 's' + 'e';      // file identifier
			header[1] = 0;                    // uniform grid data flag
			header[2] = 2;                    // dimensions number
			header[3] = sizeof(T);			  // data type size

			fwrite(header, sizeof(int), 4, ptr);

			// domain definition //
			T domain_id[4];
			domain_id[0] = grid.mpi_x;
			domain_id[1] = grid.mpi_y;
			domain_id[2] = grid.mpi_length;
			domain_id[3] = grid.mpi_width;

			fwrite(domain_id, sizeof(T), 4, ptr);

			// grid definition //
			int grid_id[4];
			grid_id[0] = grid.mpi_nx;
			grid_id[1] = grid.mpi_ny;
			grid_id[2] = grid.gcx;
			grid_id[3] = grid.gcy;

			fwrite(grid_id, sizeof(int), 4, ptr);

			// time stamp //
			T time = sol_time;

			fwrite(&time, sizeof(T), 1, ptr);

			// field definition //
			int type = 0;               // scalar field
			int name_length = strlen(name);

			fwrite(&type, sizeof(int), 1, ptr);
			fwrite(&name_length, sizeof(int), 1, ptr);
			fwrite(name, sizeof(char), name_length, ptr);

			// main data //
			fwrite(X, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

			fclose(ptr);
			status = 1;
		}

		delete[] X;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_binary(
	const char* filename,
	T* Uin, T* Vin, const char* u_name, const char* v_name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	T *U, *V;
	if (grid.mpi_com.rank == 0) {
		U = new T[grid.mpi_size];
		V = new T[grid.mpi_size];
	}

	grid.mpi_com.gather(U, Uin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	grid.mpi_com.gather(V, Vin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "wb");

		if (ptr != NULL) {

			// header data //
			int header[4];
			header[0] = 'n' + 's' + 'e';      // file identifier
			header[1] = 0;                    // uniform grid data flag
			header[2] = 2;                    // dimensions number
			header[3] = sizeof(T);          // data type size

			fwrite(header, sizeof(int), 4, ptr);

			// domain definition //
			T domain_id[4];
			domain_id[0] = grid.mpi_x;
			domain_id[1] = grid.mpi_y;
			domain_id[2] = grid.mpi_length;
			domain_id[3] = grid.mpi_width;

			fwrite(domain_id, sizeof(T), 4, ptr);

			// grid definition //
			int grid_id[4];
			grid_id[0] = grid.mpi_nx;
			grid_id[1] = grid.mpi_ny;
			grid_id[2] = grid.gcx;
			grid_id[3] = grid.gcy;

			fwrite(grid_id, sizeof(int), 4, ptr);

			// time stamp //
			T time = sol_time;

			fwrite(&time, sizeof(T), 1, ptr);

			// field definition //
			int type = 1;               // vector field
			int name_length[2];
			name_length[0] = strlen(u_name);
			name_length[1] = strlen(v_name);

			fwrite(&type, sizeof(int), 1, ptr);
			fwrite(name_length, sizeof(int), 2, ptr);
			fwrite(u_name, sizeof(char), name_length[0], ptr);
			fwrite(v_name, sizeof(char), name_length[1], ptr);

			// main data //
			fwrite(U, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);
			fwrite(V, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

			fclose(ptr);
			status = 1;
		}

		delete[] U; delete[] V;
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}

template< typename T >
bool nse::write_binary_stamp(
	const char* filename, const int idx,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< T >& grid, const T sol_time, const int sol_idx)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_binary_stamp(n_filename,
		cpu_stamp, cpu_stamp_size,
		grid, sol_time, sol_idx);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_binary(
	const char* filename, const int idx,
	T* Xin, const char* name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_binary(n_filename,
		Xin, name, grid, sol_time);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::write_binary(
	const char* filename, const int idx,
	T* Uin, T* Vin, const char* u_name, const char* v_name,
	const uniGrid2d< T >& grid, const T sol_time)
{
	char* n_filename = append_index(filename, idx);
	bool status = write_binary(n_filename,
		Uin, Vin, u_name, v_name, grid, sol_time);

	delete[] n_filename;
	return status;
}
// ------------------------------------------------------------------------ //

// * read binary input * //
template< typename T >
bool nse::read_binary_stamp(
	const char* filename,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< T >& grid, T* sol_time, int* sol_idx)
{
	T domain_id[4];
	int grid_id[4];
	int status = 0;

	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "rb");

		if (ptr != NULL) {
			int header[4];
			fread(header, sizeof(int), 4, ptr);

			if ((header[0] == 'n' + 's' + 'e') &&
				(header[1] == 0) &&                // uniform data
				(header[2] == 2) &&                // 2D
				(header[3] == sizeof(T)))		// appropriate data type
			{
				fread(domain_id, sizeof(T), 4, ptr);
				fread(grid_id, sizeof(int), 4, ptr);

				fread(sol_time, sizeof(T), 1, ptr);
				fread(sol_idx, sizeof(int), 1, ptr);

				fread(cpu_stamp_size, sizeof(int), 1, ptr);

				(*cpu_stamp) = new double[(*cpu_stamp_size)];
				fread((*cpu_stamp), sizeof(double), (*cpu_stamp_size), ptr);

				status = 1;
			}

			fclose(ptr);
		}
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (status == 1) {
		MPI_Bcast(sol_time, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
		MPI_Bcast(sol_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Bcast(cpu_stamp_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (grid.mpi_com.rank != 0)
			(*cpu_stamp) = new double[(*cpu_stamp_size)];
		MPI_Bcast((*cpu_stamp), (*cpu_stamp_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	return (status == 1);
}

template< typename T >
bool nse::read_binary(
	const char* filename, 
	T* Xin, 
	const uniGrid2d< T >& grid)
{
	T *X;
	T domain_id[4];
	int grid_id[4];
	int status = 0;

	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "rb");

		if (ptr != NULL) {
			int header[4];
			fread(header, sizeof(int), 4, ptr);

			if ((header[0] == 'n' + 's' + 'e') &&
				(header[1] == 0) &&                // uniform data
				(header[2] == 2) &&                // 2D
				(header[3] == sizeof(T)))       // appropriate data type
			{
				fread(domain_id, sizeof(T), 4, ptr);
				fread(grid_id, sizeof(int), 4, ptr);

				T time_stamp;
				fread(&time_stamp, sizeof(T), 1, ptr);

				int field_type;

				fread(&field_type, sizeof(int), 1, ptr);

				if ((grid_id[0] == grid.mpi_nx) && 
					(grid_id[1] == grid.mpi_ny) &&
					(field_type == 0))
				{
					int field_name_length;
					char* field_name;

					fread(&field_name_length, sizeof(int), 1, ptr);

					field_name = new char[field_name_length + 1];
					fread(field_name, sizeof(char), field_name_length, ptr);
					field_name[field_name_length] = '\0';

					X = new T[grid.mpi_nx * grid.mpi_ny];
					fread(X, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

					status = 1;
					delete[] field_name;
				}
				else
				if (field_type == 0)
					status = 2;
			}

			fclose(ptr);
		}
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (status == 1) {	// read status - OK -
		grid.mpi_com.scatter(Xin, X, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

		if (grid.mpi_com.rank == 0)
			delete[] X;
	}
	if (status == 2) {	// - grid-grid(input) interpolation -
		MPI_Bcast(domain_id, 4, mpi_type< T >(), 0, MPI_COMM_WORLD);
		MPI_Bcast(grid_id, 4, MPI_INT, 0, MPI_COMM_WORLD);

		int _gcx = grid_id[2],
			_gcy = grid_id[3];
		int _cx = grid_id[0] - 2 * _gcx,
			_cy = grid_id[1] - 2 * _gcy;
		uniGrid2d< T > grid_interp;
		if (!grid_interp.set(
			grid.mpi_x, grid.mpi_y,
			grid.mpi_length, grid.mpi_width,

			_cx, _cy,
			_gcx, _gcy,

			grid.mpi_com.size_x, grid.mpi_com.size_y)) return false;

		T* X_interp = new T[grid_interp.size];
		null(X_interp, grid_interp.size);

		if (!read_binary(filename, X_interp, grid_interp))
		{
			delete[] X_interp;
			return false;
		}

		grid.c_grid_reinterp(Xin, X_interp, grid_interp);

		delete[] X_interp;
		status = 1;
	}

	return (status == 1);
}

template< typename T >
bool nse::read_binary(
	const char* filename, 
	T* Uin, T* Vin, 
	const uniGrid2d< T >& grid)
{
	T *U, *V;
	T domain_id[4];
	int grid_id[4];
	int status = 0;

	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(filename, "rb");

		if (ptr != NULL) {
			int header[4];
			fread(header, sizeof(int), 4, ptr);

			if ((header[0] == 'n' + 's' + 'e') &&
				(header[1] == 0) &&                // uniform data
				(header[2] == 2) &&                // 2D
				(header[3] == sizeof(T)))       // appropriate data type
			{
				fread(domain_id, sizeof(T), 4, ptr);
				fread(grid_id, sizeof(int), 4, ptr);

				T time_stamp;
				fread(&time_stamp, sizeof(T), 1, ptr);

				int field_type;

				fread(&field_type, sizeof(int), 1, ptr);

				if ((grid_id[0] == grid.mpi_nx) && 
					(grid_id[1] == grid.mpi_ny) &&
					(field_type == 1))
				{
					int field_name_length[2];
					char *field_name[2];

					fread(field_name_length, sizeof(int), 2, ptr);

					field_name[0] = new char[field_name_length[0] + 1];
					field_name[1] = new char[field_name_length[1] + 1];
					fread(field_name[0], sizeof(char), field_name_length[0], ptr);
					fread(field_name[1], sizeof(char), field_name_length[1], ptr);
					field_name[0][field_name_length[0]] = '\0';
					field_name[1][field_name_length[1]] = '\0';

					U = new T[grid.mpi_nx * grid.mpi_ny];
					V = new T[grid.mpi_nx * grid.mpi_ny];
					fread(U, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);
					fread(V, sizeof(T), grid.mpi_nx * grid.mpi_ny, ptr);

					status = 1;
					delete[] field_name[0];
					delete[] field_name[1];
				}
				else
				if (field_type == 1)
					status = 2;

			}

			fclose(ptr);
		}
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (status == 1) {	// read status - OK -
		grid.mpi_com.scatter(Uin, U, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
		grid.mpi_com.scatter(Vin, V, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);

		if (grid.mpi_com.rank == 0) {
			delete[] U;
			delete[] V;
		}
	}
	if (status == 2) {	// - grid-grid(input) interpolation -
		MPI_Bcast(domain_id, 4, mpi_type< T >(), 0, MPI_COMM_WORLD);
		MPI_Bcast(grid_id, 4, MPI_INT, 0, MPI_COMM_WORLD);

		int _gcx = grid_id[2],
			_gcy = grid_id[3];
		int _cx = grid_id[0] - 2 * _gcx,
			_cy = grid_id[1] - 2 * _gcy;
		uniGrid2d< T > grid_interp;
		if (!grid_interp.set(
			grid.mpi_x, grid.mpi_y,
			grid.mpi_length, grid.mpi_width,

			_cx, _cy,
			_gcx, _gcy,

			grid.mpi_com.size_x, grid.mpi_com.size_y)) return false;

		T *U_interp = new T[grid_interp.size],
			*V_interp = new T[grid_interp.size];
		null(U_interp, grid_interp.size);
		null(V_interp, grid_interp.size);

		if (!read_binary(filename, U_interp, V_interp, grid_interp))
		{
			delete[] U_interp;
			delete[] V_interp;
			return false;
		}

		grid.u_grid_reinterp(Uin, U_interp, grid_interp);
		grid.v_grid_reinterp(Vin, V_interp, grid_interp);

		delete[] U_interp;
		delete[] V_interp;
		status = 1;
	}

	return (status == 1);
}

template< typename T >
bool nse::read_binary_stamp(
	const char* filename, const int idx,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< T >& grid, T* sol_time, int* sol_idx)
{
	char* n_filename = append_index(filename, idx);
	bool status = read_binary_stamp(n_filename,
		cpu_stamp, cpu_stamp_size,
		grid, sol_time, sol_idx);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::read_binary(
	const char* filename, const int idx,
	T* Xin,
	const uniGrid2d< T >& grid)
{
	char* n_filename = append_index(filename, idx);
	bool status = read_binary(n_filename,
		Xin, grid);

	delete[] n_filename;
	return status;
}

template< typename T >
bool nse::read_binary(
	const char* filename, const int idx,
	T* Uin, T* Vin,
	const uniGrid2d< T >& grid)
{
	char* n_filename = append_index(filename, idx);
	bool status = read_binary(n_filename,
		Uin, Vin, grid);

	delete[] n_filename;
	return status;
}
// ------------------------------------------------------------------------ //


// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //


// * intialize:  write tecplot output * //
template bool nse::write_tecplot(const char* filename,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename,
	float* U, float* V, const char* u_name, const char* v_name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename,
	double* U, double* V, const char* u_name, const char* v_name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename, const int idx,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename, const int idx,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename, const int idx,
	float* U, float* V, const char* u_name, const char* v_name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename, const int idx,
	double* U, double* V, const char* u_name, const char* v_name,
	const uniGrid2d< double >& grid, const double sol_time);
// ------------------------------------------------------------------------ //

// * initialize: write tecplot output [sub-domain] * //
template bool nse::write_tecplot(const char* filename,
	float* X, const char* name,
	const float xmin, const float xmax,
	const float ymin, const float ymax,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename,
	double* X, const char* name,
	const double xmin, const double xmax,
	const double ymin, const double ymax,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename,
	float* U, float* V, const char* u_name, const char* v_name,
	const float xmin, const float xmax,
	const float ymin, const float ymax,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename,
	double* U, double* V, const char* u_name, const char* v_name,
	const double xmin, const double xmax,
	const double ymin, const double ymax,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename, const int idx,
	float* X, const char* name,
	const float xmin, const float xmax,
	const float ymin, const float ymax,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename, const int idx,
	double* X, const char* name,
	const double xmin, const double xmax,
	const double ymin, const double ymax,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot(const char* filename, const int idx,
	float* U, float* V, const char* u_name, const char* v_name,
	const float xmin, const float xmax,
	const float ymin, const float ymax,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot(const char* filename, const int idx,
	double* U, double* V, const char* u_name, const char* v_name,
	const double xmin, const double xmax,
	const double ymin, const double ymax,
	const uniGrid2d< double >& grid, const double sol_time);
// ------------------------------------------------------------------------ //

// * initialize: write tecplot f(x) * //
template bool nse::write_tecplot_x(const char* filename,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_x(const char* filename,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_y(const char* filename,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_y(const char* filename,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_x(const char* filename, const int idx,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_x(const char* filename, const int idx,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_y(const char* filename, const int idx,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_y(const char* filename, const int idx,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);
// ------------------------------------------------------------------------ //

// * initialize: write tecplot f(x) with normalization * //
template bool nse::write_tecplot_x(const char* filename,
	float* X, const char* name,
	const float value, const float length,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_x(const char* filename,
	double* X, const char* name,
	const double value, const double length,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_y(const char* filename,
	float* X, const char* name,
	const float value, const float length,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_y(const char* filename,
	double* X, const char* name,
	const double value, const double length,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_x(const char* filename, const int idx,
	float* X, const char* name,
	const float value, const float length,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_x(const char* filename, const int idx,
	double* X, const char* name,
	const double value, const double length,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_tecplot_y(const char* filename, const int idx,
	float* X, const char* name,
	const float value, const float length,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_tecplot_y(const char* filename, const int idx,
	double* X, const char* name,
	const double value, const double length,
	const uniGrid2d< double >& grid, const double sol_time);
// ------------------------------------------------------------------------ //

// * intialize: write binary output * //
template bool nse::write_binary_przgn(const char* filename,
	float* Xin, const uniGrid2d< float >& grid, const int idx);
template bool nse::write_binary_przgn(const char* filename,
	double* Xin, const uniGrid2d< double >& grid, const int idx);

template bool nse::write_binary_przgn_filter(std::string filename,
	float* Xin, const int N_coarse,	const uniGrid2d< float >& grid, const int idx);
template bool nse::write_binary_przgn_filter(std::string filename,
	double* Xin, const int N_coarse,	const uniGrid2d< double >& grid, const int idx);

template bool nse::read_binary_przgn(const char* filename,
	float* Xout, const uniGrid2d< float >& grid);
template bool nse::read_binary_przgn(const char* filename,
	double* Xout, const uniGrid2d< double >& grid);

template bool nse::read_binary_przgn(const char* filename,
	float* Xout, const uniGrid2d< float >& grid, const int idx);
template bool nse::read_binary_przgn(const char* filename,
	double* Xout, const uniGrid2d< double >& grid, const int idx);

template bool nse::read_series(const char* filename, const int length,
	float* tout, float* Xout, const uniGrid2d< float >& grid);
template bool nse::read_series(const char* filename, const int length,
	double* tout, double* Xout, const uniGrid2d< double >& grid);

template bool nse::write_binary_stamp(const char* filename,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< float >& grid, const float sol_time, const int sol_idx);
template bool nse::write_binary_stamp(const char* filename,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< double >& grid, const double sol_time, const int sol_idx);

template bool nse::write_binary(const char* filename,
	float* U, float* V, const uniGrid2d< float >& grid, const int idx);
template bool nse::write_binary(const char* filename,
	double* U, double* V, const uniGrid2d< double >& grid, const int idx);

template bool nse::write_binary(const char* filename,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_binary(const char* filename,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_binary(const char* filename,
	float* U, float* V, const char* u_name, const char* v_name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_binary(const char* filename,
	double* U, double* V, const char* u_name, const char* v_name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_binary_stamp(const char* filename, const int idx,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< float >& grid, const float sol_time, const int sol_idxe);
template bool nse::write_binary_stamp(const char* filename, const int idx,
	const double* cpu_stamp, const int cpu_stamp_size,
	const uniGrid2d< double >& grid, const double sol_time, const int sol_idx);

template bool nse::write_binary(const char* filename, const int idx,
	float* X, const char* name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_binary(const char* filename, const int idx,
	double* X, const char* name,
	const uniGrid2d< double >& grid, const double sol_time);

template bool nse::write_binary(const char* filename, const int idx,
	float* U, float* V, const char* u_name, const char* v_name,
	const uniGrid2d< float >& grid, const float sol_time);
template bool nse::write_binary(const char* filename, const int idx,
	double* U, double* V, const char* u_name, const char* v_name,
	const uniGrid2d< double >& grid, const double sol_time);
// ------------------------------------------------------------------------ //

// * intialize: read binary input * //
template bool nse::read_binary_stamp(const char* filename,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< float >& grid, float* sol_time, int* sol_idx);
template bool nse::read_binary_stamp(const char* filename,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< double >& grid, double* sol_time, int* sol_idx);

template bool nse::read_binary( const char* filename, 
	float* X, 
	const uniGrid2d< float >& grid);
template bool nse::read_binary( const char* filename, 
	double* X, 
	const uniGrid2d< double >& grid);

template bool nse::read_binary( const char* filename, 
	float* U, float* V, 
	const uniGrid2d< float >& grid);
template bool nse::read_binary( const char* filename, 
	double* U, double* V, 
	const uniGrid2d< double >& grid);

template bool nse::read_binary_stamp(const char* filename, const int idx,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< float >& grid, float* sol_time, int* sol_idx);
template bool nse::read_binary_stamp(const char* filename, const int idx,
	double** cpu_stamp, int* cpu_stamp_size,
	const uniGrid2d< double >& grid, double* sol_time, int* sol_idx);

template bool nse::read_binary(const char* filename, const int idx,
	float* X,
	const uniGrid2d< float >& grid);
template bool nse::read_binary(const char* filename, const int idx,
	double* X,
	const uniGrid2d< double >& grid);

template bool nse::read_binary(const char* filename, const int idx,
	float* U, float* V,
	const uniGrid2d< float >& grid);
template bool nse::read_binary(const char* filename, const int idx,
	double* U, double* V,
	const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //
