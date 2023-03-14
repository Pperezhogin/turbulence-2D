#include "pois2d-fft.h"

template< typename T >
void nse::fft_mpi_poisson2d_data< T >::init(uniGrid2d< T >&grid)
{
	fftw_mpi_init();

	nx = grid.mpi_nx - 2 * grid.gcx;
	ny = grid.mpi_ny - 2 * grid.gcy;

	if (grid.mpi_com.size_y != 1) {
		if (grid.mpi_com.rank == 0) 
			printf("Error fft: No domain decomposition along y is permitted \n");
		exit(0);
	}

	if (nx % grid.mpi_com.size != 0) {
		if (grid.mpi_com.rank == 0)
			printf("Error fft: nx % MPI Procs should be 0 \n");
		exit(0);
	}
	
	// init size of arrays
	alloc = fftw_mpi_local_size_2d(nx, ny / 2 + 1,
		grid.mpi_com.comm, &width, &start);
	t_alloc = fftw_mpi_local_size_2d(ny / 2 + 1, nx,
		grid.mpi_com.comm, &t_width, &t_start);
	
	// init arrays
	x = fftw_alloc_real(2 * alloc);
	y = fftw_alloc_real(2 * alloc);
	
	ik2i  = fftw_alloc_real(t_alloc);
	k2h   = fftw_alloc_real(t_alloc);
    kmod  = fftw_alloc_real(t_alloc);
    ikmod = fftw_alloc_real(t_alloc);
	
	// init wavenumbers
	init_k(grid);
	
	// init transforms
	f_plan = fftw_mpi_plan_dft_r2c_2d(nx, ny, x, (fftw_complex*)x, 
		grid.mpi_com.comm, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_OUT);
	b_plan = fftw_mpi_plan_dft_c2r_2d(nx, ny, (fftw_complex*)x, x,
		grid.mpi_com.comm, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_IN);
	
	f_plan_y = fftw_mpi_plan_dft_r2c_2d(nx, ny, y, (fftw_complex*)y, 
		grid.mpi_com.comm, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_OUT);
	b_plan_y = fftw_mpi_plan_dft_c2r_2d(nx, ny, (fftw_complex*)y, y,
		grid.mpi_com.comm, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_IN);
}

template< typename T >
void nse::fft_mpi_poisson2d_data< T >::clear()
{
	fftw_mpi_cleanup();
}

template< typename T >
void nse::fft_mpi_poisson2d_data< T >::init_k(uniGrid2d< T >&grid)
{
	// array is transposed so that ky along i dimension, kx along j dimension
	int i, j, idx_fft;
	T i_global; // i is a local index, i_global is a global index
	T kx, ky, dkx, dky;
	T kx_, ky_; // modified wavenumbers, depend on desired numerical scheme
	
	dkx = (T)2.0 * M_PI / grid.mpi_length;
	dky = (T)2.0 * M_PI / grid.mpi_width;
	
	for (i = 0; i < t_width; i++)
	{
		idx_fft = i * nx;
		i_global = i + t_start;
		for (j = 0; j < nx; j++, idx_fft++)
		{	
			kx = j * dkx;
			ky = i_global * dky;
			
			if (i_global > ny / 2)
				ky -= dky * ny;
			
			if (j > nx / 2)
				kx -= dkx * nx;

			kx_ = (T)2.0 * grid.dxi * sin(kx * grid.dxh);
			ky_ = (T)2.0 * grid.dyi * sin(ky * grid.dyh);
            
            ik2i [idx_fft] = (T)1.0 / (kx_ * kx_ + ky_ * ky_);
			k2h  [idx_fft] = kx_ * kx_ + ky_ * ky_;
            kmod [idx_fft] = sqrt(kx * kx + ky * ky);
            ikmod[idx_fft] = (T)1.0 / kmod [idx_fft];
            
			//remove constant from rhs
			if((j == 0) && (i_global == 0)) {
				ik2i  [idx_fft] = (T)0.0;
                ikmod [idx_fft] = (T)0.0;
            }
		}
	}
	
}

template< typename T >
int nse::poisson_fft(T* x, const T* rhs, fft_mpi_poisson2d_data< T >&fft_data,
	const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny / 2 + 1);
		
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
			fft_data.x[idx_fft] = rhs[idx];
		
	}
	
	// forward transform of rhs
	fftw_execute(fft_data.f_plan);
	
	// multiply fourier image by -1/(kx^2+ky^2)
	for (i = 0; i < fft_data.t_width; i++)
	{
		idx_fft = i * fft_data.nx;
		for (j = 0; j < fft_data.nx; j++, idx_fft++)
		{	
			fft_data.x[2 * idx_fft] = - fft_data.ik2i[idx_fft] * fft_data.x[2 * idx_fft];
			fft_data.x[2 * idx_fft + 1] = - fft_data.ik2i[idx_fft] * fft_data.x[2 * idx_fft + 1];
		}
		
	}
	
	// backward transform
	fftw_execute(fft_data.b_plan);
	
	// multiply result by 1/(nx*ny) to get proper norm
	T c_norm = (T)1.0 / (T)fft_data.nx / (T)fft_data.ny;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny /2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
			x[idx] = fft_data.x[idx_fft] * c_norm;		
	}	
		
	return ((fft_data.nx % grid.mpi_com.size_x == 0) && (grid.mpi_com.size_y == 1)) ? 1:-1;
}

template< typename T >
int nse::two_layer_streamfunction(T* psi1, T* psi2, const T* q1, const T* q2, const T kd, fft_mpi_poisson2d_data< T >&fft_data,
	const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny / 2 + 1);
		
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
		{
			fft_data.x[idx_fft] = q1[idx];
			fft_data.y[idx_fft] = q2[idx];
		}
	}
	
	// forward transform of rhs
	fftw_execute(fft_data.f_plan);
	fftw_execute(fft_data.f_plan_y);
	
	T q1q2_Re, q1q2_Im;
	T k2;
	for (i = 0; i < fft_data.t_width; i++)
	{
		idx_fft = i * fft_data.nx;
		for (j = 0; j < fft_data.nx; j++, idx_fft++)
		{	
			q1q2_Re = (fft_data.x[2 * idx_fft]     + fft_data.y[2 * idx_fft]    ) * nse::sqr(kd);
			q1q2_Im = (fft_data.x[2 * idx_fft + 1] + fft_data.y[2 * idx_fft + 1]) * nse::sqr(kd);
			k2 = fft_data.k2h[idx_fft];

			// devision by zero case
			if (k2 < (T)1.e-10)
			{
				fft_data.x[2 * idx_fft]     = (T)0.0;
				fft_data.x[2 * idx_fft + 1] = (T)0.0;
				fft_data.y[2 * idx_fft]     = (T)0.0;
				fft_data.y[2 * idx_fft + 1] = (T)0.0;
				continue;
			}

			fft_data.x[2 * idx_fft]     = -(T)0.5 * ((T)2.0 * k2 * fft_data.x[2 * idx_fft]     + q1q2_Re) / (k2 * (k2 + nse::sqr(kd)));
			fft_data.x[2 * idx_fft + 1] = -(T)0.5 * ((T)2.0 * k2 * fft_data.x[2 * idx_fft + 1] + q1q2_Im) / (k2 * (k2 + nse::sqr(kd)));

			fft_data.y[2 * idx_fft]     = -(T)0.5 * ((T)2.0 * k2 * fft_data.y[2 * idx_fft]     + q1q2_Re) / (k2 * (k2 + nse::sqr(kd)));
			fft_data.y[2 * idx_fft + 1] = -(T)0.5 * ((T)2.0 * k2 * fft_data.y[2 * idx_fft + 1] + q1q2_Im) / (k2 * (k2 + nse::sqr(kd)));
		}
		
	}
	
	// backward transform
	fftw_execute(fft_data.b_plan);
	fftw_execute(fft_data.b_plan_y);

	// multiply result by 1/(nx*ny) to get proper norm
	T c_norm = (T)1.0 / (T)fft_data.nx / (T)fft_data.ny;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny /2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
		{
			psi1[idx] = fft_data.x[idx_fft] * c_norm;		
			psi2[idx] = fft_data.y[idx_fft] * c_norm;
		}
	}	
	
	return ((fft_data.nx % grid.mpi_com.size_x == 0) && (grid.mpi_com.size_y == 1)) ? 1:-1;
}

template< typename T >
int nse::pseudo_poisson_fft(T* x, const T* rhs, fft_mpi_poisson2d_data< T >&fft_data,
	const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny / 2 + 1);
		
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
			fft_data.x[idx_fft] = rhs[idx];
		
	}
	
	// forward transform of rhs
	fftw_execute(fft_data.f_plan);
	
	// multiply fourier image by sqrt(kx^2+ky^2)
	for (i = 0; i < fft_data.t_width; i++)
	{
		idx_fft = i * fft_data.nx;
		for (j = 0; j < fft_data.nx; j++, idx_fft++)
		{	
			fft_data.x[2 * idx_fft    ] =  fft_data.ikmod[idx_fft] * fft_data.x[2 * idx_fft    ];
			fft_data.x[2 * idx_fft + 1] =  fft_data.ikmod[idx_fft] * fft_data.x[2 * idx_fft + 1];
		}
		
	}
	
	// backward transform
	fftw_execute(fft_data.b_plan);
	
	// multiply result by 1/(nx*ny) to get proper norm
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (fft_data.ny /2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++)
			x[idx] = fft_data.x[idx_fft] / (T)fft_data.nx / (T)fft_data.ny;		
	}	
		
	return ((fft_data.nx % grid.mpi_com.size_x == 0) && (grid.mpi_com.size_y == 1)) ? 1:-1;
}

// ------------------------------------------------------------------------ //

// initialize: fft poisson solver

template struct nse::fft_mpi_poisson2d_data< float >;
template struct nse::fft_mpi_poisson2d_data< double >;

template int nse::poisson_fft( 
	float*  x, const float* rhs, fft_mpi_poisson2d_data< float >&fft_data,
	const uniGrid2d< float >&grid);
template int nse::poisson_fft( 
	double*  x, const double* rhs, fft_mpi_poisson2d_data< double >&fft_data,
	const uniGrid2d< double >&grid);

template int nse::two_layer_streamfunction(
float* psi1, float* psi2, const float* q1, const float* q2, const float kd, 
fft_mpi_poisson2d_data< float >&fft_data, const uniGrid2d< float >&grid);
template int nse::two_layer_streamfunction(
double* psi1, double* psi2, const double* q1, const double* q2, const double kd, 
fft_mpi_poisson2d_data< double >&fft_data, const uniGrid2d< double >&grid);

template int nse::pseudo_poisson_fft( 
	float*  x, const float* rhs, fft_mpi_poisson2d_data< float >&fft_data,
	const uniGrid2d< float >&grid);
template int nse::pseudo_poisson_fft( 
	double*  x, const double* rhs, fft_mpi_poisson2d_data< double >&fft_data,
	const uniGrid2d< double >&grid);
