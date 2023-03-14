#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include <fftw3-mpi.h>

namespace nse
{
	template< typename T >
	struct fft_mpi_poisson2d_data{
		// 1d-deconposition along the FIRST[x] dimension, i.e. 200x200 -> 25x200, 25x200, ..
		// nx * ny will be transformed to fourier space with transposition to ny * nx wavenumers array
		ptrdiff_t nx, ny;
		
		ptrdiff_t width; // width of data along first direction
		ptrdiff_t start; // start index in global notation along first direction
		// the same for transposed arrays that used for fourier image
		ptrdiff_t t_width; 
		ptrdiff_t t_start; 		
		
		ptrdiff_t alloc;
		ptrdiff_t t_alloc;		
		
		fftw_plan f_plan;
		fftw_plan b_plan;

		fftw_plan f_plan_y;
		fftw_plan b_plan_y;
		
		// x are used for input and output of fourier transform
		double    *x; // x = rhs, rhs -> irhs; ix = - ik2i * irhs; ix -> x
		double    *y;
		
		double *ik2i;  // 1 / (kx^2 + ky^2)
		double *k2h ;  // kx^2 + ky^2
		double *kmod;  // sqrt(kx^2 + ky^2)
		double *ikmod; // 1 / sqrt(kx^2 + ky^2)
		
		public:
			void init(uniGrid2d< T >&grid);
			void clear();
			
		private:
			void init_k(uniGrid2d< T >&grid);
	};
	
	template< typename T >
	int poisson_fft(T* x, const T* rhs, fft_mpi_poisson2d_data< T >&fft_data,
		const uniGrid2d< T >&grid);

	template< typename T >
	int two_layer_streamfunction(T* psi1, T* psi2, const T* q1, const T* q2, const T kd, fft_mpi_poisson2d_data< T >&fft_data,
	const uniGrid2d< T >&grid);
    
    template< typename T >
	int pseudo_poisson_fft(T* x, const T* rhs, fft_mpi_poisson2d_data< T >&fft_data,
		const uniGrid2d< T >&grid);
}
