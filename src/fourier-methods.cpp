#include "fourier-methods.h"

// E(k) ~ k^4 exp(-(k/kp)^2)
// abs(psi_k)^2 ~ k exp(-(k/kp)^2)
// abs(omega_k)^2 ~ k^5 exp(-(k/kp)^2)
// this function generates streamfunction on (2pi) square
// with energy per unit area equals:
// 3/8 pi^(3/2) kp^5
// then it is normalized to have unit energy per unit area
template <typename T>
void nse::power_exp_spectra(T* Psi, const T kp, const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft, idx_fft_conj, i_global;
	int kx, ky;
	T k;
	T sigma;
	T phase;
	
	ptrdiff_t nx, ny;
	ptrdiff_t alloc, t_alloc;
	fftw_plan b_plan;
	ptrdiff_t width; // width of data along first direction
	ptrdiff_t start; // start index in global notation along first direction	
	ptrdiff_t t_width;
	ptrdiff_t t_start;
    
    double *streamfunction;
    
	nx = grid.mpi_nx - 2 * grid.gcx;
 	ny = grid.mpi_ny - 2 * grid.gcy;
	
	// init size of array
	alloc = fftw_mpi_local_size_2d(nx, ny / 2 + 1,
		grid.mpi_com.comm, &width, &start);
	t_alloc = fftw_mpi_local_size_2d(ny / 2 + 1, nx,
		grid.mpi_com.comm, &t_width, &t_start);
	
	streamfunction = fftw_alloc_real(2 * alloc);
	
	// init transforms
	b_plan = fftw_mpi_plan_dft_c2r_2d(nx, ny, (fftw_complex*)streamfunction, streamfunction,
		grid.mpi_com.comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
	
	for (i = 0; i < t_width; i++)
	{
		idx_fft = i * nx;
		i_global = i + t_start;
		ky = i_global;
		
		for (j = 0; j < nx; j++, idx_fft++)
		{
			kx = j;			
			if (j > nx / 2)
				kx -= nx;
			
			k = sqrt((T)kx * (T)kx + (T)ky * (T)ky);
			
            // exlude constant, ky = 0 and 2h harmonics
			if (ky > 0.5 && ky < ny/2 - (T)0.5) {
                sigma = sqrt( k * exp(- (k/kp) * (k/kp)));
                phase = uniform_distribution((T)0.0, (T)2.0 * (T)M_PI);
				streamfunction[2 * idx_fft    ] = sigma * cos(phase);
				streamfunction[2 * idx_fft + 1] = sigma * sin(phase);
			} else
            {
                streamfunction[2 * idx_fft    ] = (T)0.0;
                streamfunction[2 * idx_fft + 1] = (T)0.0;
            }
		}
	}
	
	for (i = 0; i < t_width; i++)
	{
        idx_fft = i * nx;
        i_global = i + t_start;
        ky = i_global;
        
        // line ky = 0 should be complex-conjugated, exclude constant and 2h harmonics (nx/2)
        if (ky < (T)0.5) {
            for (j = 0; j < nx / 2; j++, idx_fft++)
            {
                kx = j;			
                k = sqrt((T)kx * (T)kx + (T)ky * (T)ky);
                
                if (kx > (T)0.5) {
                    sigma = (T)2.0 * sqrt( k * exp(- (k/kp) * (k/kp)));
                    phase = uniform_distribution((T)0.0, (T)2.0 * (T)M_PI);
                    streamfunction[2 * idx_fft    ] = sigma * cos(phase);
                    streamfunction[2 * idx_fft + 1] = sigma * sin(phase);
                    
                    idx_fft_conj = idx_fft + nx-j;
                    streamfunction[2 * idx_fft_conj    ] =   sigma * cos(phase);
                    streamfunction[2 * idx_fft_conj + 1] = - sigma * sin(phase);
                }
            }
        }
	}
	
	fftw_execute(b_plan);
	T Amp = 1.0 / sqrt(3./8. * pow(M_PI, 1.5) * pow(kp, 5.));
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (ny / 2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++) {
			Psi[idx] = (T)streamfunction[idx_fft] * Amp;
		}
	}
	
	fftw_destroy_plan(b_plan);
	fftw_free(streamfunction);
}

template <typename T>
T gauss_kernel(T k, T filter_width)
{
	return exp(-(k*k) * filter_width * filter_width / (T)24.);      
}

double spectral_kernel(int kx, int ky, const int N_coarse)
{
	double res = 1.;
	if (kx > N_coarse / 2 || ky > N_coarse / 2) {
		res = 0.;
	}
	return res;
}

template <typename T>
void nse::gauss_filter(T* wc, T* w, const T filter_width, const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft, i_global;
	int kx, ky;
	T k;
	T sigma;
	
	ptrdiff_t nx, ny;
	ptrdiff_t alloc, t_alloc;
	ptrdiff_t width; // width of data along first direction
	ptrdiff_t start; // start index in global notation along first direction	
	ptrdiff_t t_width;
	ptrdiff_t t_start;
    
	nx = grid.mpi_nx - 2 * grid.gcx;
 	ny = grid.mpi_ny - 2 * grid.gcy;
	
	// init size of array
	alloc = fftw_mpi_local_size_2d(nx, ny / 2 + 1,
		grid.mpi_com.comm, &width, &start);
	t_alloc = fftw_mpi_local_size_2d(ny / 2 + 1, nx,
		grid.mpi_com.comm, &t_width, &t_start);

	if (first_call) {
		wf = fftw_alloc_real(2 * alloc);
		// init transforms
		f_plan_filter = fftw_mpi_plan_dft_r2c_2d(nx, ny, wf, (fftw_complex*)wf, 
			grid.mpi_com.comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
		b_plan_filter = fftw_mpi_plan_dft_c2r_2d(nx, ny, (fftw_complex*)wf, wf,
			grid.mpi_com.comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
		first_call = false;
		if (grid.mpi_com.rank == 0) printf("First call of gauss filter\n");
	}

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (ny / 2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++) {
			wf[idx_fft] = (double)w[idx];
		}
	}

	fftw_execute(f_plan_filter);
	
	for (i = 0; i < t_width; i++)
	{
		idx_fft = i * nx;
		i_global = i + t_start;
		ky = i_global;
		
		for (j = 0; j < nx; j++, idx_fft++)
		{
			kx = j;			
			if (j > nx / 2)
				kx -= nx;
			
			k = sqrt((T)kx * (T)kx + (T)ky * (T)ky);
			
			sigma = gauss_kernel(k, filter_width);

			wf[2 * idx_fft    ] *= sigma;
			wf[2 * idx_fft + 1] *= sigma;
		}
	}
	
	fftw_execute(b_plan_filter);

	T c_norm = (T)1.0 / (T)nx / (T)ny;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (ny / 2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++) {
			wc[idx] = (T)wf[idx_fft] * c_norm;
		}
	}
}

template <typename T>
void nse::spectral_filter(T* wc, T* w, const int N_coarse, const uniGrid2d< T >&grid)
{
	int i, j, idx, idx_fft, i_global;
	int kx, ky;
	T sigma;
	
	ptrdiff_t nx, ny;
	ptrdiff_t alloc, t_alloc;
	ptrdiff_t width; // width of data along first direction
	ptrdiff_t start; // start index in global notation along first direction	
	ptrdiff_t t_width;
	ptrdiff_t t_start;
    
	nx = grid.mpi_nx - 2 * grid.gcx;
 	ny = grid.mpi_ny - 2 * grid.gcy;
	
	// init size of array
	alloc = fftw_mpi_local_size_2d(nx, ny / 2 + 1,
		grid.mpi_com.comm, &width, &start);
	t_alloc = fftw_mpi_local_size_2d(ny / 2 + 1, nx,
		grid.mpi_com.comm, &t_width, &t_start);

	if (first_call) {
		wf = fftw_alloc_real(2 * alloc);
		// init transforms
		f_plan_filter = fftw_mpi_plan_dft_r2c_2d(nx, ny, wf, (fftw_complex*)wf, 
			grid.mpi_com.comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
		b_plan_filter = fftw_mpi_plan_dft_c2r_2d(nx, ny, (fftw_complex*)wf, wf,
			grid.mpi_com.comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
		first_call = false;
		if (grid.mpi_com.rank == 0) printf("First call of spectral filter\n");
	}

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (ny / 2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++) {
			wf[idx_fft] = (double)w[idx];
		}
	}

	fftw_execute(f_plan_filter);
	
	for (i = 0; i < t_width; i++)
	{
		idx_fft = i * nx;
		i_global = i + t_start;
		ky = i_global;
		
		for (j = 0; j < nx; j++, idx_fft++)
		{
			kx = j;			
			if (j > nx / 2)
				kx -= nx;
			
			sigma = spectral_kernel(kx, ky, N_coarse);

			wf[2 * idx_fft    ] *= sigma;
			wf[2 * idx_fft + 1] *= sigma;
		}
	}
	
	fftw_execute(b_plan_filter);

	T c_norm = (T)1.0 / (T)nx / (T)ny;
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		idx_fft = (i - grid.gcx) * 2 * (ny / 2 + 1);
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, idx_fft++) {
			wc[idx] = (T)wf[idx_fft] * c_norm;
		}
	}
}

template< typename T >
void nse::coarse_resolution(T* Xc, T* X, const int N_coarse, const uniGrid2d< T >&grid) {
	int ig, jg, idxg;
	int i, idx;
	int offset = (grid.mpi_nx - 2 * grid.gcx) / N_coarse;

	i = 0;
	for (ig = grid.gcx; ig < grid.mpi_nx - grid.gcx; ig += offset)
	{
		idxg = ig * grid.mpi_ny + grid.gcy;
		idx = i * N_coarse;
		for (jg = grid.gcy; jg < grid.mpi_ny - grid.gcy; jg += offset, idxg += offset, idx++) {
			Xc[idx] = X[idxg];
		}
		i++;
	}
}

// ----------------------------------------------------------------------- //
template void nse::power_exp_spectra( 
		float* Psi, const float kp, const uniGrid2d< float >&grid);
template void nse::power_exp_spectra( 
		double* Psi, const double kp, const uniGrid2d< double >&grid);

template void nse::gauss_filter( 
		float* wc, float* w, const float filter_width, const uniGrid2d< float >&grid);
template void nse::gauss_filter( 
		double* wc, double* w, const double filter_width, const uniGrid2d< double >&grid);

template void nse::spectral_filter( 
		float* wc, float* w, const int N_coarse, const uniGrid2d< float >&grid);
template void nse::spectral_filter( 
		double* wc, double* w, const int N_coarse, const uniGrid2d< double >&grid);

template void nse::coarse_resolution(float* Xc, float* X, const int N_coarse, const uniGrid2d< float >&grid);
template void nse::coarse_resolution(double* Xc, double* X, const int N_coarse, const uniGrid2d< double >&grid);

		